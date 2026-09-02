// player-modal.js -- popup de detalle de jugador (doble clic en el
// nombre, ver ui.js::dataTable options.doubleClick). Usa el <dialog>
// singleton #player-modal de index.html, mismo espíritu que el
// #chart-tooltip compartido de charts.js.

import { api } from "./api.js";
import { dataTable, el, emptyState, pillToggle, playerPhoto, teamBadge } from "./ui.js";
import { courtShotChart } from "./court.js";

const PER_GAME_STATS = ["MIN", "PTS", "REB", "AST", "STL", "BLK"];

/** Totales de temporada -> por partido (dividido por GP), calculado en
 * el cliente sin otro viaje al backend. Los % de tiro y GP no cambian. */
function toPerGame(seasons) {
  return seasons.map((season) => {
    const row = { ...season };
    for (const key of PER_GAME_STATS) {
      if (row[key] != null && row.GP) row[key] = Math.round((row[key] / row.GP) * 10) / 10;
    }
    return row;
  });
}

function seasonsTable(seasons, mode) {
  const rows = mode === "per_game" ? toPerGame(seasons) : seasons;
  return dataTable(rows, {}, {
    hiddenColumns: ["is_projection"],
    rowClass: (record) => (record.is_projection ? "row-projection" : null),
  });
}

const BIO_LABELS = {
  height: "Altura",
  weight: "Peso",
  age: "Edad",
  country: "País",
  school: "Universidad",
  draft: "Draft",
};

function bioFacts(bio) {
  if (!bio) {
    return el("p", { class: "caption" }, "Bio no disponible para este jugador.");
  }
  const facts = Object.entries(BIO_LABELS)
    .filter(([key]) => bio[key] != null)
    .map(([key, label]) => {
      let value = bio[key];
      if (key === "weight") value = `${value} lb`;
      if (key === "age") value = `${value} años`;
      return el("div", { class: "bio-fact" }, [el("span", { class: "bio-fact-label" }, label), el("span", {}, String(value))]);
    });
  return el("div", { class: "bio-facts" }, facts);
}

function qualityChips(qualities) {
  if (!qualities || qualities.length === 0) return null;
  return el(
    "div",
    { class: "quality-chips" },
    qualities.map((q) => el("span", { class: "quality-chip" }, q))
  );
}

export async function openPlayerModal(playerId, teamId) {
  const dialog = document.getElementById("player-modal");
  dialog.replaceChildren(el("div", { class: "caption", style: "padding: 24px;" }, "Cargando jugador…"));
  dialog.showModal();

  let data;
  try {
    data = await api.player(playerId);
  } catch (err) {
    dialog.replaceChildren(
      el("div", { style: "padding: 24px;" }, [
        el("button", { class: "modal-close", "aria-label": "Cerrar", onclick: () => dialog.close() }, "✕"),
        emptyState(err.message),
      ])
    );
    return;
  }

  let mode = "per_game";
  const seasonsBox = el("div");

  const header = el("div", { class: "card-header-row", style: "margin: 24px 0 10px;" }, [
    el("h3", { style: "margin: 0;" }, "Trayectoria por temporada"),
    data.seasons.length
      ? pillToggle(
          [
            { value: "per_game", label: "Por partido" },
            { value: "totals", label: "Totales" },
          ],
          mode,
          (value) => {
            mode = value;
            seasonsBox.replaceChildren(seasonsTable(data.seasons, mode));
          }
        )
      : null,
  ]);

  seasonsBox.replaceChildren(
    data.seasons.length ? seasonsTable(data.seasons, mode) : el("p", { class: "caption" }, "Sin temporadas registradas.")
  );

  const shotChartBox = el("div");

  dialog.replaceChildren(
    el("div", { class: "detail-modal-body" }, [
      el("button", { class: "modal-close", "aria-label": "Cerrar", onclick: () => dialog.close() }, "✕"),
      el("div", { class: "detail-modal-header" }, [
        playerPhoto(data.player_id, data.name, 96),
        teamBadge(teamId, null, 64),
        el("div", {}, [
          el("h2", { style: "margin: 0 0 4px;" }, data.name),
          el("p", { class: "caption", style: "margin: 0;" }, data.position || "Posición no disponible"),
        ]),
      ]),
      bioFacts(data.bio),
      qualityChips(data.qualities),
      header,
      seasonsBox,
      shotChartBox,
    ])
  );

  loadShotChart(shotChartBox, playerId);
}

/** Se carga aparte (no bloquea el resto del popup): roster_shot_charts.csv
 * puede no existir para jugadores fuera del roster propio, en cuyo caso
 * la sección simplemente no aparece. Toggle Real/Proyectado -- "real" son
 * tiros de verdad de la última temporada jugada; "projected" es un mapa
 * SINTÉTICO (remuestreo del histórico, ver src/shot_chart_projection.py)
 * cuyo conteo de intentos/anotados cuadra exacto con FGA/FG3A/FGM/FG3M ya
 * proyectados -- no una predicción independiente del volumen de tiro.
 */
async function loadShotChart(container, playerId) {
  const chartBox = el("div");
  const headerBox = el("div", { class: "card-header-row", style: "margin: 20px 0 6px;" });

  function renderResult(data, kind) {
    const titleEl = headerBox.querySelector("h3");
    if (!data.shots.length) {
      if (titleEl) titleEl.textContent = "Mapa de tiros";
      chartBox.replaceChildren(
        el(
          "p",
          { class: "caption" },
          kind === "real" ? "Sin mapa de tiros disponible." : "Sin proyección disponible para estimar el mapa de tiros."
        )
      );
      return;
    }
    if (titleEl) titleEl.textContent = `Mapa de tiros -- temporada ${data.season}`;
    chartBox.replaceChildren(courtShotChart(data.shots, { title: `Mapa de tiros de ${playerId}` }));
  }

  async function loadKind(kind) {
    chartBox.replaceChildren(el("p", { class: "caption" }, "Cargando mapa de tiros…"));
    try {
      renderResult(await api.playerShotChart(playerId, kind), kind);
    } catch {
      chartBox.replaceChildren();
    }
  }

  // Primera carga real: si no hay NINGÚN dato (ni real ni proyectado),
  // toda la sección se omite -- pero eso no se sabe sin preguntar
  // primero, así que se decide tras la respuesta inicial.
  let initial;
  try {
    initial = await api.playerShotChart(playerId, "real");
  } catch {
    return;
  }
  let initialKind = "real";
  if (!initial.shots.length) {
    // Puede que solo falte lo real (jugador nuevo sin temporada previa
    // cacheada) pero sí haya proyección -- se comprueba antes de omitir
    // la sección entera.
    try {
      initial = await api.playerShotChart(playerId, "projected");
    } catch {
      return;
    }
    if (!initial.shots.length) return;
    initialKind = "projected";
  }

  headerBox.replaceChildren(
    el("h3", { style: "margin: 0;" }, `Mapa de tiros -- temporada ${initial.season}`),
    pillToggle(
      [
        { value: "real", label: "Real" },
        { value: "projected", label: "Proyectado" },
      ],
      initialKind,
      (value) => loadKind(value)
    )
  );
  container.replaceChildren(headerBox, chartBox);
  renderResult(initial, initialKind);
}

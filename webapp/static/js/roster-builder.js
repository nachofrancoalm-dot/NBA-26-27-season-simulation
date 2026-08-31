// roster-builder.js -- roster hipotético editable del splash (ver
// views/splash.js). Arranca IGUAL que el roster real curado del config
// (mismos 13 player_id de "Mi equipo") -- no un roster vacío -- y deja
// añadir, quitar y sustituir jugadores por cualquier otro de los 30
// equipos reales (src/sandbox_simulation.py), entre un mínimo y un
// máximo (MIN/MAX_ROSTER_SIZE, servidos por /api/sandbox/players).
//
// La única acción de simulación es "Simular la liga completa con este
// roster" (src/league_sandbox.py): sustituye tu roster en los 30 equipos
// reales, simula la liga entera, y deja ese resultado activo
// (hypothetical-league.js) para que Liga y Playoffs / Premios
// individuales lo muestren en vez de los datos reales, hasta que se
// vuelva a pulsar "Volver a los datos reales" en el banner de esas
// pestañas. Antes había TAMBIÉN un botón "Simular este roster" que solo
// corría el Monte Carlo de tu equipo contra un rival proxy genérico
// (sandbox_simulation.py, sin el resto de la liga) -- quitado a
// petición del usuario: sin comparar contra los otros 29 equipos reales,
// ese número aportaba poco frente al de la liga completa. La tabla de
// estadísticas individuales por jugador (antes solo se cargaba tras ESE
// botón) ahora se carga tras simular la liga -- ver runLeagueSimulation.

import { api } from "./api.js";
import { el, teamBadge, emptyState, dataTable, pillToggle } from "./ui.js";
import { setHypotheticalLeague } from "./hypothetical-league.js";

function fmt1(value) {
  return typeof value === "number" ? value.toFixed(1) : "—";
}

function rosterSlot(player, onChangeClick, onRemoveClick) {
  return el("div", { class: "roster-slot" }, [
    teamBadge(player.team_id, player.team_abbreviation, 32),
    el("div", { class: "roster-slot-info" }, [
      el("p", { class: "roster-slot-name" }, player.player_name),
      el(
        "p",
        { class: "roster-slot-meta" },
        [player.team_abbreviation, player.position, `GS/36 ${fmt1(player.game_score_per36)}`].filter(Boolean).join(" · ")
      ),
    ]),
    el("button", { type: "button", class: "roster-slot-change", onclick: onChangeClick }, "Cambiar"),
    onRemoveClick
      ? el(
          "button",
          { type: "button", class: "roster-slot-remove", "aria-label": `Quitar a ${player.player_name}`, onclick: onRemoveClick },
          "✕"
        )
      : null,
  ]);
}

/** Modal de selección: busca por nombre/equipo sobre el pool ya cargado
 * (577 jugadores, filtrado en el cliente -- no hace falta pedirlo por
 * cada tecla) y llama a `onPick(player)` al elegir uno. */
function openPickerModal(pool, excludeIds, onPick) {
  const dialog = document.getElementById("roster-picker-modal");
  const search = el("input", {
    type: "search",
    placeholder: "Buscar jugador o equipo...",
    class: "roster-picker-search",
  });
  const results = el("div", { class: "roster-picker-results" });

  function renderResults(query) {
    const q = query.trim().toLowerCase();
    const matches = pool
      .filter((p) => !excludeIds.has(p.player_id))
      .filter((p) => !q || p.player_name.toLowerCase().includes(q) || p.team_abbreviation.toLowerCase().includes(q))
      .sort((a, b) => (b.game_score_per36 || 0) - (a.game_score_per36 || 0))
      .slice(0, 60);

    if (!matches.length) {
      results.replaceChildren(emptyState("Sin resultados."));
      return;
    }
    results.replaceChildren(
      ...matches.map((p) =>
        el(
          "button",
          {
            type: "button",
            class: "roster-picker-row",
            onclick: () => {
              onPick(p);
              dialog.close();
            },
          },
          [
            teamBadge(p.team_id, p.team_abbreviation, 28),
            el("span", { class: "roster-picker-row-name" }, p.player_name),
            el("span", { class: "roster-picker-row-meta" }, `${p.team_abbreviation} · ${p.position || "?"}`),
            el("span", { class: "roster-picker-row-gs" }, `GS/36 ${fmt1(p.game_score_per36)}`),
          ]
        )
      )
    );
  }

  search.addEventListener("input", () => renderResults(search.value));

  dialog.replaceChildren(
    el("div", { class: "detail-modal-body" }, [
      el("button", { class: "modal-close", "aria-label": "Cerrar", onclick: () => dialog.close() }, "✕"),
      el("h3", { style: "margin: 0 0 14px;" }, "Elegir jugador"),
      search,
      results,
    ])
  );
  renderResults("");
  dialog.showModal();
  search.focus();
}

/** `enter(topKey, subKey)` -- inyectado por splash.js (viene de app.js) --
 * usado para saltar a Liga y Playoffs automáticamente en cuanto termina
 * de simular la liga completa con este roster. */
export function rosterBuilderCard(enter) {
  const card = el("div", { class: "card roster-builder" });
  const slotsBox = el("div", { class: "roster-slots" });
  const countLabel = el("p", { class: "caption roster-count" });
  const addBtn = el("button", { type: "button", class: "roster-slot-add" }, "+ Añadir jugador");
  const statsBox = el("div", { class: "roster-stats" });
  const simulateLeagueBtn = el("button", { type: "button", class: "btn" }, "🏆 Simular la liga completa con este roster");
  const leagueStatusText = el("p", { class: "caption", style: "margin: 8px 0 0;" });
  let pool = [];
  let roster = []; // array de player_id, en orden
  let minSize = 5;
  let maxSize = 15;
  let statsMode = "per_game";
  let lastSimulatedRoster = null; // snapshot del roster de la última simulación -- el toggle de la tabla de jugadores reusa esta lista en vez de re-simular las 2.000 temporadas por un cambio de Totales/Por partido

  function findPlayer(playerId) {
    return pool.find((p) => p.player_id === playerId);
  }

  function renderSlots() {
    const excludeIds = new Set(roster);
    const canRemove = roster.length > minSize;
    const canAdd = roster.length < maxSize;

    slotsBox.replaceChildren(
      ...roster.map((playerId, index) => {
        const player = findPlayer(playerId);
        if (!player) return null;
        return rosterSlot(
          player,
          () => {
            excludeIds.delete(playerId);
            openPickerModal(pool, excludeIds, (picked) => {
              roster[index] = picked.player_id;
              renderSlots();
            });
          },
          canRemove
            ? () => {
                roster.splice(index, 1);
                renderSlots();
              }
            : null
        );
      })
    );

    countLabel.textContent = `${roster.length} jugadores (mínimo ${minSize}, máximo ${maxSize}).`;
    addBtn.disabled = !canAdd;
    addBtn.title = canAdd ? "" : `Ya tienes el máximo de ${maxSize} jugadores.`;
  }

  async function loadRosterStats(mode) {
    statsMode = mode;
    if (!lastSimulatedRoster) return;
    statsBox.replaceChildren(el("p", { class: "caption" }, "Cargando estadísticas por jugador..."));
    try {
      const data = await api.sandboxRosterStats(lastSimulatedRoster, statsMode);
      statsBox.replaceChildren(
        el("div", { class: "card-header-row", style: "margin: 18px 0 6px;" }, [
          el("h3", { style: "margin: 0;" }, "Estadísticas individuales para este roster"),
          pillToggle(
            [
              { value: "per_game", label: "Por partido" },
              { value: "totals", label: "Totales" },
            ],
            statsMode,
            (value) => loadRosterStats(value)
          ),
        ]),
        el(
          "p",
          { class: "caption" },
          "Recalculadas con los minutos que le tocarían en ESTE roster (no los de su equipo real) -- " +
            "así se ve qué produciría cada jugador en el contexto hipotético, no en el suyo."
        ),
        dataTable(data.players, data.glossary, { hiddenColumns: ["player_id"] })
      );
    } catch (err) {
      statsBox.replaceChildren(emptyState(err.message));
    }
  }

  async function runLeagueSimulation() {
    simulateLeagueBtn.disabled = true;
    simulateLeagueBtn.textContent = "Simulando la liga (temporada + playoffs + premios)...";
    leagueStatusText.textContent = "Puede tardar unos segundos -- son 30 equipos, no solo el tuyo.";
    try {
      const data = await api.sandboxLeague(roster);
      setHypotheticalLeague(roster, data);
      leagueStatusText.textContent = "";
      lastSimulatedRoster = [...roster];
      // No se espera a que termine de pintar la tabla antes de saltar de
      // pestaña -- queda lista para cuando el usuario vuelva a Inicio,
      // pero no debe retrasar la transición a Liga y Playoffs.
      loadRosterStats(statsMode);
      // Salta directo a Liga y Playoffs -- ahí ya se ve el banner de
      // "roster hipotético activo" con estos resultados en vez de los
      // reales (ver league.js/awards.js).
      enter("liga-nba", "liga");
    } catch (err) {
      leagueStatusText.textContent = `Error al simular la liga: ${err.message}`;
    } finally {
      simulateLeagueBtn.disabled = false;
      simulateLeagueBtn.textContent = "🏆 Simular la liga completa con este roster";
    }
  }

  simulateLeagueBtn.addEventListener("click", runLeagueSimulation);
  addBtn.addEventListener("click", () => {
    if (roster.length >= maxSize) return;
    openPickerModal(pool, new Set(roster), (picked) => {
      roster.push(picked.player_id);
      renderSlots();
    });
  });

  card.replaceChildren(
    el("div", { class: "card-header-row" }, [el("h2", {}, "🛠️ Tu roster hipotético"), simulateLeagueBtn]),
    el(
      "p",
      { class: "caption" },
      "Arranca igual que el roster real de este proyecto -- añade, quita o cambia cualquier jugador por " +
        "cualquier otro de los 30 equipos. \"Simular la liga completa\" sustituye tu roster en la liga de " +
        "30 equipos reales y lleva ese resultado a Liga y Playoffs / Premios individuales -- los otros 29 " +
        "equipos se quedan con su roster real."
    ),
    slotsBox,
    el("div", { class: "roster-add-row" }, [addBtn, countLabel]),
    leagueStatusText,
    statsBox
  );

  (async () => {
    slotsBox.replaceChildren(el("p", { class: "caption" }, "Cargando catálogo de jugadores..."));
    try {
      const [poolData, defaultData] = await Promise.all([api.sandboxPlayers(), api.sandboxDefaultRoster()]);
      pool = poolData.players;
      minSize = poolData.min_roster_size;
      maxSize = poolData.max_roster_size;
      roster = [...defaultData.player_ids];
      renderSlots();
    } catch (err) {
      slotsBox.replaceChildren(emptyState(err.message));
    }
  })();

  return card;
}

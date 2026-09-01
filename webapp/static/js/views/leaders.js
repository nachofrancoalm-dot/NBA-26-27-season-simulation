import { api } from "../api.js";
import { card, el, dataTable, emptyState, pillToggle, skeleton } from "../ui.js";
import { openPlayerModal } from "../player-modal.js";
import { getScenario, scenarioBar } from "../scenario.js";

/** Rellena un <select> con "(todos)" + los valores distintos presentes
 * en los datos -- nunca una lista fija, así que un país/posición nuevo
 * en el CSV aparece solo, sin tocar este archivo. */
function fillFilterOptions(select, label, values) {
  select.replaceChildren(
    el("option", { value: "" }, `${label} (todos)`),
    ...values.map((v) => el("option", { value: v }, v))
  );
}

export async function render(container) {
  container.replaceChildren(skeleton(["title", "short"]), skeleton());

  const status = await api.status();
  let mode = "per_game";
  let allPlayers = [];
  let glossary = {};

  const teamSelect = el("select", {});
  const positionSelect = el("select", {});
  const countrySelect = el("select", {});
  const tableBox = el("div", { style: "margin-top: 16px;" });
  const modeBox = el("div");

  function applyFilters() {
    const team = teamSelect.value;
    const position = positionSelect.value;
    const country = countrySelect.value;
    const rows = allPlayers.filter(
      (p) =>
        (!team || p.team_abbreviation === team) &&
        (!position || p.position === position) &&
        (!country || p.country === country)
    );
    tableBox.replaceChildren(
      rows.length ? dataTable(rows, glossary, {
        hiddenColumns: ["player_id", "team_id"],
        doubleClick: { column: "player_name", onOpen: (record) => openPlayerModal(record.player_id, record.team_id) },
      }) : emptyState("Ningún jugador cumple los filtros elegidos.")
    );
  }

  async function loadLeaders() {
    tableBox.replaceChildren(skeleton());
    let data;
    try {
      data = await api.leagueLeaders(mode, getScenario());
    } catch (err) {
      tableBox.replaceChildren(emptyState(err.message));
      return;
    }
    allPlayers = data.players;
    glossary = data.glossary;
    const uniqueSorted = (key) => [...new Set(allPlayers.map((p) => p[key]).filter(Boolean))].sort();
    fillFilterOptions(teamSelect, "Equipo", uniqueSorted("team_abbreviation"));
    fillFilterOptions(positionSelect, "Posición", uniqueSorted("position"));
    fillFilterOptions(countrySelect, "País", uniqueSorted("country"));
    applyFilters();
  }

  [teamSelect, positionSelect, countrySelect].forEach((s) => (s.onchange = applyFilters));

  modeBox.replaceChildren(
    pillToggle(
      [
        { value: "per_game", label: "Por partido" },
        { value: "totals", label: "Totales" },
      ],
      mode,
      (value) => {
        mode = value;
        loadLeaders();
      }
    )
  );

  container.replaceChildren(
    scenarioBar(status, () => loadLeaders()),
    card([
      el("h2", {}, "Líderes de estadísticas"),
      el(
        "p",
        { class: "caption" },
        "Los jugadores proyectados de los 30 equipos. Filtra por equipo, posición o país, y haz clic en cualquier cabecera para ordenar. Doble clic en un jugador para ver su detalle."
      ),
      el("div", { class: "card-header-row" }, [
        el("div", { class: "filter-bar" }, [teamSelect, positionSelect, countrySelect]),
        modeBox,
      ]),
      tableBox,
    ])
  );

  await loadLeaders();
}

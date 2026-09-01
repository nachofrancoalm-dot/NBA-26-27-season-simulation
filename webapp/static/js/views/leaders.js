import { api } from "../api.js";
import { card, el, dataTable, emptyState, pillToggle, multiToggle, skeleton } from "../ui.js";
import { openPlayerModal } from "../player-modal.js";
import { getScenario, scenarioBar } from "../scenario.js";

/** Rellena un <select> con "(todos)" + los valores distintos presentes
 * en los datos -- nunca una lista fija, así que un valor nuevo en el CSV
 * aparece solo, sin tocar este archivo. */
function fillFilterOptions(select, label, values) {
  select.replaceChildren(
    el("option", { value: "" }, `${label} (todos)`),
    ...values.map((v) => el("option", { value: v }, v))
  );
}

const POSITION_GROUPS = [
  { value: "G", label: "G" },
  { value: "F", label: "F" },
  { value: "C", label: "C" },
];

/** `position` en los datos es texto libre de CommonPlayerInfo ("F-G",
 * "Guard-Forward", "Center"...) -- no hay PG/SG/SF/PF en la fuente, solo
 * G/F/C. Se reduce cada tramo separado por "-" a su primera letra, así
 * "F-G" cuenta como F Y como G para el filtro (un jugador así debe
 * aparecer al marcar cualquiera de los dos). */
function positionTokens(rawPosition) {
  if (!rawPosition) return [];
  return rawPosition
    .split("-")
    .map((part) => part.trim()[0]?.toUpperCase())
    .filter(Boolean);
}

export async function render(container) {
  container.replaceChildren(skeleton(["title", "short"]), skeleton());

  const status = await api.status();
  let mode = "per_game";
  let allPlayers = [];
  let glossary = {};
  let selectedPositions = new Set();

  const conferenceSelect = el("select", {});
  const teamSelect = el("select", {});
  const countrySelect = el("select", {});
  const positionBox = el("div");
  const tableBox = el("div", { style: "margin-top: 16px;" });
  const modeBox = el("div");

  function applyFilters() {
    const conference = conferenceSelect.value;
    const team = teamSelect.value;
    const country = countrySelect.value;
    const rows = allPlayers.filter(
      (p) =>
        (!conference || p.conference === conference) &&
        (!team || p.team_abbreviation === team) &&
        (!country || p.country === country) &&
        (selectedPositions.size === 0 || positionTokens(p.position).some((t) => selectedPositions.has(t)))
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
    fillFilterOptions(conferenceSelect, "Conferencia", uniqueSorted("conference"));
    fillFilterOptions(teamSelect, "Equipo", uniqueSorted("team_abbreviation"));
    fillFilterOptions(countrySelect, "País", uniqueSorted("country"));
    applyFilters();
  }

  [conferenceSelect, teamSelect, countrySelect].forEach((s) => (s.onchange = applyFilters));

  positionBox.replaceChildren(
    multiToggle(POSITION_GROUPS, selectedPositions, (value) => {
      selectedPositions = value;
      applyFilters();
    })
  );

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
        "Los jugadores proyectados de los 30 equipos. Filtra por conferencia, equipo, posición (varias a la vez) o país, y haz clic en cualquier cabecera para ordenar. Doble clic en un jugador para ver su detalle."
      ),
      el("div", { class: "card-header-row" }, [
        el("div", { class: "filter-bar" }, [conferenceSelect, teamSelect, positionBox, countrySelect]),
        modeBox,
      ]),
      tableBox,
    ])
  );

  await loadLeaders();
}

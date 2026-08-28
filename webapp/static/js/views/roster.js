import { api } from "../api.js";
import { card, el, dataTable, emptyState, pillToggle, skeleton } from "../ui.js";
import { openPlayerModal } from "../player-modal.js";

export async function render(container) {
  let mode = "per_game";

  container.replaceChildren(skeleton(["title", "", "", "short"]));

  const status = await api.status();
  const teamId = status.team.team_id;

  const body = el("div");
  const toggle = pillToggle(
    [
      { value: "per_game", label: "Por partido" },
      { value: "totals", label: "Totales" },
    ],
    mode,
    (value) => {
      mode = value;
      loadAndRender();
    }
  );

  async function loadAndRender() {
    let data;
    try {
      data = await api.roster(mode);
    } catch (err) {
      body.replaceChildren(emptyState(err.message));
      return;
    }
    body.replaceChildren(
      el("p", { class: "caption" }, `${data.players.length} jugadores. Doble clic en un nombre para ver su detalle.`),
      dataTable(data.players, data.glossary, {
        hiddenColumns: ["player_id"],
        doubleClick: { column: "player_name", onOpen: (record) => openPlayerModal(record.player_id, teamId) },
      })
    );
  }

  container.replaceChildren(
    card([
      el("div", { class: "card-header-row" }, [
        el("h2", {}, "Roster: proyección, riesgo y desgaste por jugador"),
        toggle,
      ]),
      body,
    ])
  );

  await loadAndRender();
}

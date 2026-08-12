// team-modal.js -- popup de detalle de equipo (doble clic en la celda de
// equipo de standings/playoffs, ver ui.js::dataTable options.doubleClick).
// Mismo patrón que player-modal.js: <dialog> singleton, reutiliza
// /api/league/team/{abbreviation} (ya existe, Fase 2) y en cascada abre
// el popup de jugador al hacer doble clic en su nombre dentro del roster.

import { api } from "./api.js";
import { dataTable, el, emptyState, pillToggle, statGrid, teamBadge } from "./ui.js";
import { openPlayerModal } from "./player-modal.js";

export async function openTeamModal(abbreviation, teamId) {
  const dialog = document.getElementById("team-modal");
  dialog.replaceChildren(el("div", { class: "caption", style: "padding: 24px;" }, "Cargando equipo…"));
  dialog.showModal();

  let mode = "per_game";
  const body = el("div");

  async function loadAndRender() {
    let data;
    try {
      data = await api.leagueTeam(abbreviation, mode);
    } catch (err) {
      body.replaceChildren(emptyState(err.message));
      return;
    }

    const positionLine =
      data.conference && data.seed
        ? `${data.conference} — Seed ${data.seed}`
        : "Posición en la clasificación no disponible";

    body.replaceChildren(
      el("div", { class: "detail-modal-header" }, [
        teamBadge(data.team_id ?? teamId, abbreviation, 96),
        el("div", {}, [
          el("h2", { style: "margin: 0 0 4px;" }, abbreviation),
          el("p", { class: "caption", style: "margin: 0;" }, positionLine),
        ]),
      ]),
      statGrid([
        ["Victorias medias", data.regular.wins_mean.toFixed(1)],
        ["Playoffs", `${data.playoff.playoff_pct.toFixed(1)}%`],
        ["Semis conf.", `${data.playoff.conf_semis_pct.toFixed(1)}%`],
        ["Finales conf.", `${data.playoff.finals_pct.toFixed(1)}%`],
        ["Campeonato", `${data.playoff.championship_pct.toFixed(1)}%`],
      ]),
      el("div", { style: "display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 20px 0 10px;" }, [
        el("h3", { style: "margin: 0;" }, "Roster proyectado"),
        pillToggle(
          [
            { value: "per_game", label: "Por partido" },
            { value: "totals", label: "Totales" },
          ],
          mode,
          (value) => {
            mode = value;
            loadAndRender();
          }
        ),
      ]),
      data.players.length
        ? dataTable(data.players, data.glossary, {
            hiddenColumns: ["player_id"],
            doubleClick: {
              column: "player_name",
              onOpen: (record) => openPlayerModal(record.player_id, data.team_id ?? teamId),
            },
          })
        : emptyState("Sin roster proyectado disponible.")
    );
  }

  dialog.replaceChildren(
    el("div", { class: "detail-modal-body" }, [el("button", { class: "modal-close", onclick: () => dialog.close() }, "✕"), body])
  );
  await loadAndRender();
}

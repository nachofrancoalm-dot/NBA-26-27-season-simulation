// leaderboard.js -- ranking visual de un premio (MVP, DPOY, ROY, MIP,
// 6.º Hombre): foto real + barra proporcional al valor de temporada, en
// vez de una tabla. La vista previa al pasar el ratón usa
// player-preview.js (tarjeta con foto + stats), no el tooltip de charts.js.

import { el, playerPhoto, teamBadge } from "./ui.js";
import { openPlayerModal } from "./player-modal.js";
import { openTeamModal } from "./team-modal.js";
import { showPlayerPreview, hidePlayerPreview } from "./player-preview.js";

/**
 * `records`: filas ya ordenadas por el backend (ver src/awards_projection.py).
 * `valueKey`: columna usada como longitud de barra. `valueFormat`:
 * formatea el valor a la derecha. `statsFn(record)`: [{label, value}]
 * opcional para la vista previa. `captionFn(record)`: pie de foto de la
 * vista previa, función (no string fijo) porque MIP necesita uno
 * distinto por fila. Clic en cualquier fila abre el popup de detalle.
 */
export function leaderboardChart(
  records,
  { valueKey, valueFormat = (v) => (typeof v === "number" ? v.toFixed(1) : "—"), statsFn, captionFn, teamIds = {} } = {}
) {
  if (!records || !records.length) {
    return el("p", { class: "caption" }, "Sin candidatos.");
  }

  const maxValue = Math.max(...records.map((r) => Math.abs(Number(r[valueKey]) || 0)), 1);

  const rows = records.map((record, index) => {
    const value = Number(record[valueKey]);
    const widthPct = Number.isFinite(value) ? Math.max(4, (Math.abs(value) / maxValue) * 100) : 0;

    const row = el(
      "button",
      {
        type: "button",
        class: "leaderboard-row",
        onclick: () => openPlayerModal(record.player_id, teamIds[record.team_abbreviation]),
      },
      [
        el("span", { class: "leaderboard-rank" }, String(index + 1)),
        playerPhoto(record.player_id, record.player_name, 36),
        el("div", { class: "leaderboard-info" }, [
          el("span", { class: "leaderboard-name" }, record.player_name),
          record.team_abbreviation ? el("span", { class: "leaderboard-team" }, record.team_abbreviation) : null,
        ]),
        // selection_type ("Titular"/"Reserva") solo lo trae el All-Star.
        record.selection_type ? el("span", { class: "leaderboard-tag" }, record.selection_type) : null,
        el("div", { class: "leaderboard-bar-track" }, [el("div", { class: "leaderboard-bar", style: `width: ${widthPct}%;` })]),
        el("span", { class: "leaderboard-value" }, valueFormat(value)),
      ]
    );

    if (statsFn) {
      const preview = () => ({
        photo: playerPhoto(record.player_id, record.player_name, 40),
        name: record.player_name,
        subtitle: record.team_abbreviation,
        caption: captionFn ? captionFn(record) : null,
        stats: statsFn(record),
      });
      row.addEventListener("mousemove", (event) => showPlayerPreview(event, preview()));
      row.addEventListener("mouseleave", hidePlayerPreview);
      row.addEventListener("focus", (event) => showPlayerPreview(event, preview()));
      row.addEventListener("blur", hidePlayerPreview);
    }

    return row;
  });

  return el("div", { class: "leaderboard" }, rows);
}

/**
 * Mismo lenguaje visual que leaderboardChart() pero para EQUIPOS, no
 * jugadores (Entrenador del Año, `data.coy`, premio de equipo -- este
 * proyecto no modela entrenadores). `teamIds`: abreviatura -> team_id.
 */
export function teamLeaderboardChart(records, { valueKey, valueFormat = (v) => (typeof v === "number" ? v.toFixed(1) : "—"), statsFn, captionFn, teamIds = {} } = {}) {
  if (!records || !records.length) {
    return el("p", { class: "caption" }, "Sin candidatos.");
  }

  const maxValue = Math.max(...records.map((r) => Math.abs(Number(r[valueKey]) || 0)), 1);

  const rows = records.map((record, index) => {
    const value = Number(record[valueKey]);
    const widthPct = Number.isFinite(value) ? Math.max(4, (Math.abs(value) / maxValue) * 100) : 0;

    const row = el(
      "button",
      {
        type: "button",
        class: "leaderboard-row",
        onclick: () => openTeamModal(record.team_abbreviation, teamIds[record.team_abbreviation]),
      },
      [
        el("span", { class: "leaderboard-rank" }, String(index + 1)),
        teamBadge(teamIds[record.team_abbreviation], record.team_abbreviation, 36),
        el("div", { class: "leaderboard-info" }, [
          el("span", { class: "leaderboard-name" }, record.team_abbreviation),
          record.conference ? el("span", { class: "leaderboard-team" }, record.conference) : null,
        ]),
        el("div", { class: "leaderboard-bar-track" }, [el("div", { class: "leaderboard-bar", style: `width: ${widthPct}%;` })]),
        el("span", { class: "leaderboard-value" }, valueFormat(value)),
      ]
    );

    if (statsFn) {
      const preview = () => ({
        photo: teamBadge(teamIds[record.team_abbreviation], record.team_abbreviation, 40),
        name: record.team_abbreviation,
        subtitle: record.conference,
        caption: captionFn ? captionFn(record) : null,
        stats: statsFn(record),
      });
      row.addEventListener("mousemove", (event) => showPlayerPreview(event, preview()));
      row.addEventListener("mouseleave", hidePlayerPreview);
      row.addEventListener("focus", (event) => showPlayerPreview(event, preview()));
      row.addEventListener("blur", hidePlayerPreview);
    }

    return row;
  });

  return el("div", { class: "leaderboard" }, rows);
}

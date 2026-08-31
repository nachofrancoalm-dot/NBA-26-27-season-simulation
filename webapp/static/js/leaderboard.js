// leaderboard.js -- ranking visual de un premio (MVP, DPOY, ROY, MIP,
// 6.º Hombre): foto real + barra proporcional al valor de temporada,
// en vez de una tabla. Pensado para complementar (no sustituir del
// todo) los quintetos sobre cancha de court.js -- mismo espíritu de
// "menos tabla, más foto real". La vista previa al pasar el ratón usa
// player-preview.js (tarjeta real con foto + stats), no el tooltip de
// una línea de charts.js -- un jugador tiene más que enseñar que un
// solo número.

import { el, playerPhoto, teamBadge } from "./ui.js";
import { openPlayerModal } from "./player-modal.js";
import { openTeamModal } from "./team-modal.js";
import { showPlayerPreview, hidePlayerPreview } from "./player-preview.js";

/**
 * `records`: filas YA ordenadas por el backend (mvp_score/dpoy_score/
 * season_value/improvement descendente, ver src/awards_projection.py).
 * `valueKey`: columna a usar como longitud de barra.
 * `valueFormat`: formatea el valor mostrado a la derecha de la barra.
 * `statsFn(record)`: [{label, value}] para la vista previa -- opcional
 * (se omite si no aporta nada para ese premio). Qué stats son y de qué
 * temporada (proyectada, real, o una comparación de ambas) lo decide
 * el caller (ver LEADERBOARD_CONFIG en views/awards.js), este módulo
 * solo las pinta.
 * `captionFn(record)`: pie de foto de la vista previa (p.ej.
 * "Temporada proyectada 2026-27") -- función y no un string fijo porque
 * MIP necesita uno DISTINTO por fila (cada jugador tiene su propia
 * temporada real anterior).
 * Clic en cualquier fila abre el popup de detalle del jugador (mismo
 * popup que el resto de la app, ver player-modal.js) -- doble
 * clic/tabla ya no hace falta para "ver más": es la fila entera.
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
        // selection_type ("Titular"/"Reserva") solo lo trae el All-Star
        // (ver compute_all_star_selections) -- en cualquier otro premio
        // el campo no existe y esto simplemente no se pinta.
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
 * jugadores -- Entrenador del Año (`data.coy`, ver
 * awards_projection.compute_coy_candidates) es un premio de equipo
 * (este proyecto no modela entrenadores en absoluto, ver su docstring),
 * así que no encaja en leaderboardChart() (foto de jugador, abre el
 * popup de JUGADOR al hacer clic). `records` necesita `team_abbreviation`
 * y `valueKey`; `teamIds`: abreviatura -> team_id (para el escudo y para
 * abrir el popup de EQUIPO, ver team-modal.js).
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

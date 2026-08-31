// leaderboard.js -- ranking visual de un premio (MVP, DPOY, ROY, MIP,
// 6.º Hombre): foto real + barra proporcional al valor de temporada,
// en vez de una tabla. Pensado para complementar (no sustituir del
// todo) los quintetos sobre cancha de court.js -- mismo espíritu de
// "menos tabla, más foto real". La vista previa al pasar el ratón usa
// player-preview.js (tarjeta real con foto + stats), no el tooltip de
// una línea de charts.js -- un jugador tiene más que enseñar que un
// solo número.

import { el, playerPhoto } from "./ui.js";
import { openPlayerModal } from "./player-modal.js";
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
        el("div", { class: "leaderboard-bar-track" }, [el("div", { class: "leaderboard-bar", style: `width: ${widthPct}%;` })]),
        el("span", { class: "leaderboard-value" }, valueFormat(value)),
      ]
    );

    if (statsFn) {
      const preview = () => ({
        playerId: record.player_id,
        playerName: record.player_name,
        teamAbbreviation: record.team_abbreviation,
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

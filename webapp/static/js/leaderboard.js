// leaderboard.js -- ranking visual de un premio (MVP, DPOY, ROY, MIP,
// 6.º Hombre): foto real + barra proporcional al valor de temporada,
// en vez de una tabla. Pensado para complementar (no sustituir del
// todo) los quintetos sobre cancha de court.js -- mismo espíritu de
// "menos tabla, más foto real". Reutiliza el tooltip compartido
// #chart-tooltip (mismo que charts.js) en vez de duplicar la lógica.

import { el, playerPhoto, showTooltip, hideTooltip } from "./ui.js";
import { openPlayerModal } from "./player-modal.js";

/**
 * `records`: filas YA ordenadas por el backend (mvp_score/dpoy_score/
 * season_value/improvement descendente, ver src/awards_projection.py).
 * `valueKey`: columna a usar como longitud de barra.
 * `valueFormat`: formatea el valor mostrado a la derecha de la barra.
 * `subtitleFn(record)`: texto del tooltip al pasar el ratón (stats
 * rápidas) -- opcional, se omite si no aporta nada para ese premio.
 * Clic en cualquier fila abre el popup de detalle del jugador (mismo
 * popup que el resto de la app, ver player-modal.js) -- doble
 * clic/tabla ya no hace falta para "ver más": es la fila entera.
 */
export function leaderboardChart(records, { valueKey, valueFormat = (v) => (typeof v === "number" ? v.toFixed(1) : "—"), subtitleFn, teamIds = {} } = {}) {
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

    if (subtitleFn) {
      const text = subtitleFn(record);
      row.addEventListener("mousemove", (event) => showTooltip(event, text));
      row.addEventListener("mouseleave", hideTooltip);
      row.addEventListener("focus", (event) => showTooltip(event, text));
      row.addEventListener("blur", hideTooltip);
    }

    return row;
  });

  return el("div", { class: "leaderboard" }, rows);
}

// player-preview.js -- tarjeta de vista previa al pasar el ratón/foco
// sobre un jugador en leaderboard.js o court.js: foto + nombre + equipo
// + un set de stats con su `caption` (p.ej. "Temporada proyectada
// 2026-27"). Este módulo es puramente de presentación -- qué stats
// mostrar y de dónde salen (proyectadas, reales, o una comparación de
// las dos, como en MIP) lo decide siempre el caller, nunca se mezclan
// aquí. Sustituye al #chart-tooltip genérico (texto plano, una línea)
// para este caso concreto: un jugador tiene más que enseñar que un
// solo número.
//
// Un único nodo reutilizado (mismo patrón que #chart-tooltip) en vez de
// crear/destruir un popover por cada hover -- más barato y evita fugas
// de listeners.

import { el, playerPhoto } from "./ui.js";

function previewNode() {
  let node = document.getElementById("player-preview");
  if (!node) {
    node = document.createElement("div");
    node.id = "player-preview";
    node.className = "player-preview";
    document.body.append(node);
  }
  return node;
}

/**
 * `event`: el MouseEvent/FocusEvent que disparó la vista previa (para
 * posicionarla). `player`: {playerId, playerName, teamAbbreviation}.
 * `stats`: [{label, value}] -- ya formateadas por el caller (distintas
 * por premio, ver LEADERBOARD_CONFIG en awards.js). Un `value` puede ser
 * una comparación ya formateada como texto (p.ej. "27.9 → 22.5", MIP:
 * temporada real anterior -> proyectada) -- este módulo no sabe ni le
 * importa qué representa cada stat, solo la pinta. `caption`: string ya
 * formado por el caller (p.ej. "Temporada proyectada 2026-27" o, para
 * MIP, "Real 2025-26 → Proyectada 2026-27"), o null para omitirlo.
 */
export function showPlayerPreview(event, { playerId, playerName, teamAbbreviation, caption, stats = [] }) {
  const node = previewNode();
  // Node.replaceChildren() es el método NATIVO del DOM -- a diferencia
  // de el() (ui.js), NO ignora los `null`: los convierte en un nodo de
  // texto literal "null" (bug real encontrado al probar MIP, que no
  // llevaba caption). Por eso se filtran aquí antes de pasarlos.
  const children = [
    el("div", { class: "player-preview-header" }, [
      playerPhoto(playerId, playerName, 40),
      el("div", { class: "player-preview-heading" }, [
        el("span", { class: "player-preview-name" }, playerName),
        teamAbbreviation ? el("span", { class: "player-preview-team" }, teamAbbreviation) : null,
      ]),
    ]),
    caption ? el("p", { class: "player-preview-caption" }, caption) : null,
    stats.length
      ? el(
          "div",
          { class: "player-preview-stats" },
          stats.map((stat) =>
            el("div", { class: "player-preview-stat" }, [
              el("span", { class: "player-preview-stat-value" }, stat.value),
              el("span", { class: "player-preview-stat-label" }, stat.label),
            ])
          )
        )
      : null,
  ].filter(Boolean);
  node.replaceChildren(...children);

  positionPreview(node, event);
  node.classList.add("visible");
}

export function hidePlayerPreview() {
  const node = document.getElementById("player-preview");
  if (node) node.classList.remove("visible");
}

/** Coloca la tarjeta cerca del cursor/elemento con foco, sin salirse
 * del viewport -- se mide DESPUÉS de pintar (requestAnimationFrame)
 * porque el tamaño depende del contenido (nombre largo, nº de stats). */
function positionPreview(node, event) {
  const margin = 14;
  const anchorX = event.clientX ?? event.target.getBoundingClientRect().left;
  const anchorY = event.clientY ?? event.target.getBoundingClientRect().bottom;
  node.style.left = `${anchorX + margin}px`;
  node.style.top = `${anchorY + margin}px`;

  requestAnimationFrame(() => {
    const rect = node.getBoundingClientRect();
    let left = anchorX + margin;
    let top = anchorY + margin;
    if (rect.right > window.innerWidth - 8) left = anchorX - rect.width - margin;
    if (rect.bottom > window.innerHeight - 8) top = anchorY - rect.height - margin;
    node.style.left = `${Math.max(8, left)}px`;
    node.style.top = `${Math.max(8, top)}px`;
  });
}

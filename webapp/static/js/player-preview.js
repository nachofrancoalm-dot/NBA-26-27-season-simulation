// player-preview.js -- tarjeta de vista previa al pasar el ratón/foco
// sobre un jugador (o un EQUIPO -- ver teamLeaderboardChart() en
// leaderboard.js, Entrenador del Año) en leaderboard.js o court.js:
// foto/insignia + nombre + subtítulo + un set de stats con su
// `caption` (p.ej. "Temporada proyectada 2026-27"). Este módulo es
// puramente de presentación -- qué foto, qué stats mostrar y de dónde
// salen (proyectadas, reales, o una comparación de las dos, como en
// MIP) lo decide siempre el caller (pasa el nodo de foto ya construido
// -- playerPhoto() o teamBadge(), ver ui.js -- en vez de un id/nombre
// que este módulo tendría que interpretar), nunca se mezcla aquí.
// Sustituye al #chart-tooltip genérico (texto plano, una línea) para
// este caso concreto: una entidad tiene más que enseñar que un solo
// número.
//
// Un único nodo reutilizado (mismo patrón que #chart-tooltip) en vez de
// crear/destruir un popover por cada hover -- más barato y evita fugas
// de listeners.

import { el } from "./ui.js";

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
 * posicionarla). `photo`: nodo ya construido (playerPhoto()/teamBadge(),
 * ver ui.js) -- este módulo no sabe si es un jugador o un equipo.
 * `name`: nombre a mostrar en negrita. `subtitle`: línea pequeña debajo
 * (abreviatura de equipo para un jugador, conferencia para un equipo).
 * `stats`: [{label, value}] -- ya formateadas por el caller (distintas
 * por premio, ver LEADERBOARD_CONFIG en awards.js). Un `value` puede ser
 * una comparación ya formateada como texto (p.ej. "27.9 → 22.5", MIP:
 * temporada real anterior -> proyectada) -- este módulo no sabe ni le
 * importa qué representa cada stat, solo la pinta. `caption`: string ya
 * formado por el caller (p.ej. "Temporada proyectada 2026-27" o, para
 * MIP, "Real 2025-26 → Proyectada 2026-27"), o null para omitirlo.
 */
export function showPlayerPreview(event, { photo, name, subtitle, caption, stats = [] }) {
  const node = previewNode();
  // Node.replaceChildren() es el método NATIVO del DOM -- a diferencia
  // de el() (ui.js), NO ignora los `null`: los convierte en un nodo de
  // texto literal "null" (se manifestaba en MIP, que no siempre lleva
  // caption). Por eso se filtran aquí antes de pasarlos.
  const children = [
    el("div", { class: "player-preview-header" }, [
      photo,
      el("div", { class: "player-preview-heading" }, [
        el("span", { class: "player-preview-name" }, name),
        subtitle ? el("span", { class: "player-preview-team" }, subtitle) : null,
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

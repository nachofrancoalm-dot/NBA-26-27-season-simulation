// player-preview.js -- tarjeta de vista previa al pasar el ratón/foco
// sobre un jugador o equipo (leaderboard.js, court.js): foto/insignia +
// nombre + subtítulo + stats + caption. Puramente de presentación -- el
// caller decide qué mostrar y pasa el nodo de foto ya construido.
// Un único nodo reutilizado (mismo patrón que #chart-tooltip).

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
 * `event`: dispara la vista previa (para posicionarla). `photo`: nodo ya
 * construido (playerPhoto()/teamBadge()). `name`/`subtitle`: cabecera.
 * `stats`: [{label, value}] ya formateadas por el caller -- `value`
 * puede ser una comparación como texto (p.ej. "27.9 → 22.5" en MIP).
 * `caption`: string ya formado por el caller, o null para omitirlo.
 */
export function showPlayerPreview(event, { photo, name, subtitle, caption, stats = [] }) {
  const node = previewNode();
  // replaceChildren() nativo NO ignora `null` como el() (ui.js) -- los
  // convierte en el texto literal "null", así que se filtran antes.
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

/** Coloca la tarjeta cerca del cursor sin salirse del viewport -- se mide
 * DESPUÉS de pintar (requestAnimationFrame) porque el tamaño depende del contenido. */
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

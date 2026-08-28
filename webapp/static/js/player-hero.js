// player-hero.js -- gráfico de marca del splash y de la transición entre
// pestañas: la foto REAL de un jugador del roster (cdn.nba.com, mismo
// hotlink en vivo que ya usa playerPhoto() en el popup de jugador -- se
// referencia por URL, nunca se descarga ni se guarda en el repo, igual
// que los escudos de equipo) más un balón animado. Sustituye a un primer
// intento de silueta ilustrada a mano (formas SVG genéricas para
// cabeza/torso/piernas) que quedaba con un brazo apenas visible y una
// pose poco convincente -- una foto real resuelve "un jugador de los
// 76ers" de forma mucho más creíble que reintentar la anatomía a mano, y
// evita además el riesgo de derechos de imagen de usar una foto de stock
// sin licencia sacada de una búsqueda de Google.
//
// `player_id` viene de datos reales del roster (el primer jugador de
// /api/roster, ya ordenado por relevancia -- ver home.js), así que la
// foto cambia sola si el roster del config cambia, sin nada hardcodeado
// aquí sobre qué equipo o jugador es.

import { playerPhoto } from "./ui.js";

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}, children = []) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  for (const child of [].concat(children)) {
    if (child != null) node.append(child);
  }
  return node;
}

function basketballGraphic() {
  return svgEl("svg", { class: "hero-ball", viewBox: "0 0 40 40", "aria-hidden": "true" }, [
    svgEl("g", { class: "db-ball-bounce" }, [
      svgEl("circle", { class: "db-ball", cx: 20, cy: 20, r: 18 }),
      svgEl("path", { class: "db-seam", d: "M 3,20 Q 20,5 37,20" }),
      svgEl("path", { class: "db-seam", d: "M 3,20 Q 20,35 37,20" }),
      svgEl("line", { class: "db-seam", x1: 20, y1: 2, x2: 20, y2: 38 }),
    ]),
  ]);
}

/** Arcos de "velocidad" detrás de la foto -- sugieren movimiento sin
 * intentar dibujar un cuerpo en marcha; mucho más fácil de acertar que
 * una pose anatómica y sigue leyéndose como "dinámico". */
function motionArcs() {
  return svgEl("svg", { class: "hero-motion", viewBox: "0 0 220 220", "aria-hidden": "true" }, [
    svgEl("path", { class: "motion-arc a1", d: "M 6,70 Q 70,45 132,70" }),
    svgEl("path", { class: "motion-arc a2", d: "M 0,112 Q 66,90 126,112" }),
    svgEl("path", { class: "motion-arc a3", d: "M 10,154 Q 62,136 116,154" }),
  ]);
}

/** `<div class="hero-stage">` con la foto del jugador + balón animado +
 * sombra en el suelo, reutilizado por el splash y por la transición entre
 * pestañas (ver hero.css). `size` en px para el círculo de la foto. */
export function playerHeroCard(playerId, playerName, { extraClass = "", size = 176 } = {}) {
  const stage = document.createElement("div");
  stage.className = `hero-stage ${extraClass}`.trim();

  const photoWrap = document.createElement("div");
  photoWrap.className = "hero-photo-bob";
  photoWrap.append(playerPhoto(playerId, playerName, size));

  const ballWrap = document.createElement("div");
  ballWrap.className = "hero-ball-wrap";
  ballWrap.append(basketballGraphic());

  const shadow = document.createElement("div");
  shadow.className = "dribbler-shadow";

  stage.append(motionArcs(), photoWrap, ballWrap, shadow);
  return stage;
}

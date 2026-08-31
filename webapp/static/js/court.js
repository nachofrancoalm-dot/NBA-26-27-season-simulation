// court.js -- mapa de tiros real (LOC_X/LOC_Y de ShotChartDetail, ver
// data_pipeline.fetch_player_shot_chart / build_roster_shot_charts_dataset)
// sobre una media cancha dibujada a mano en SVG, sin librería externa
// (mismo criterio que charts.js: nada de Chart.js/D3 vendorizado). Las
// medidas de la cancha (aro, tablero, zona, línea de 3, semicírculo de
// tiros libres) son las medidas físicas reales de una cancha NBA en
// décimas de pie desde el aro -- el mismo sistema de coordenadas que
// devuelve `ShotChartDetail`, así que los puntos se colocan
// directamente sin reescalar.

import { el, playerPhoto } from "./ui.js";
import { openPlayerModal } from "./player-modal.js";
import { showPlayerPreview, hidePlayerPreview } from "./player-preview.js";

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

/** Líneas de la media cancha -- coordenadas físicas reales (décimas de
 * pie desde el aro, aro en (0,0)), no inventadas. */
function courtLines() {
  const group = svgEl("g", { class: "court-lines" });
  group.append(
    // Límite de la media cancha (línea de banda + línea de medio campo).
    svgEl("rect", { x: -250, y: -47.5, width: 500, height: 470, class: "court-boundary" }),
    // Aro.
    svgEl("circle", { cx: 0, cy: 0, r: 7.5, class: "court-hoop" }),
    // Tablero.
    svgEl("line", { x1: -30, y1: -7.5, x2: 30, y2: -7.5, class: "court-backboard" }),
    // Área restringida (semicírculo bajo el aro).
    svgEl("path", { d: "M -40 0 A 40 40 0 0 0 40 0", class: "court-restricted" }),
    // Zona (rectángulo exterior + interior).
    svgEl("rect", { x: -80, y: -47.5, width: 160, height: 190, class: "court-paint-outer" }),
    svgEl("rect", { x: -60, y: -47.5, width: 120, height: 190, class: "court-paint-inner" }),
    // Semicírculo de tiros libres.
    svgEl("path", { d: "M -60 142.5 A 60 60 0 0 0 60 142.5", class: "court-ft-circle" }),
    // Línea de 3: esquinas rectas + arco.
    svgEl("line", { x1: -220, y1: -47.5, x2: -220, y2: 92.5, class: "court-three" }),
    svgEl("line", { x1: 220, y1: -47.5, x2: 220, y2: 92.5, class: "court-three" }),
    svgEl("path", { d: "M -220 92.5 A 237.5 237.5 0 0 0 220 92.5", class: "court-three" })
  );
  return group;
}

/** `shots`: [{loc_x, loc_y, shot_made, shot_type}, ...] ya en el mismo
 * sistema de coordenadas que las líneas de arriba. */
export function courtShotChart(shots, { title } = {}) {
  const svg = svgEl("svg", {
    viewBox: "-250 -60 500 500",
    class: "court-chart",
    role: "img",
    "aria-label": title || "Mapa de tiros",
  });
  svg.append(courtLines());

  const shotsGroup = svgEl("g", { class: "court-shots" });
  for (const shot of shots) {
    shotsGroup.append(
      svgEl("circle", {
        cx: shot.loc_x,
        cy: shot.loc_y,
        r: 4.2,
        class: shot.shot_made ? "court-shot court-shot-made" : "court-shot court-shot-missed",
      })
    );
  }
  svg.append(shotsGroup);

  const made = shots.filter((s) => s.shot_made).length;
  const pct = shots.length ? ((100 * made) / shots.length).toFixed(1) : "—";

  return el("div", { class: "court-chart-wrap" }, [
    svg,
    el("div", { class: "court-chart-legend" }, [
      el("span", { class: "court-legend-item" }, [el("span", { class: "court-swatch court-swatch-made" }), `Anotado (${made})`]),
      el("span", { class: "court-legend-item" }, [
        el("span", { class: "court-swatch court-swatch-missed" }),
        `Fallado (${shots.length - made})`,
      ]),
      el("span", { class: "caption" }, `${pct}% de acierto sobre ${shots.length} tiros`),
    ]),
  ]);
}

// Posiciones fijas del quinteto clásico 2-2-1 (G-G-F-F-C) sobre la media
// cancha -- NO son las 5 posiciones reales (base/escolta/alero/ala-pívot/
// pívot): `awards_projection._pick_positional_teams` solo distingue 3
// grupos (G, F, C, 2+2+1), así que dentro de cada grupo el orden es
// simplemente "de mayor a menor valor de temporada" (ver
// compute_all_nba_teams/compute_all_defensive_teams), no una posición
// concreta. Colocar al primer G a la izquierda y al segundo a la
// derecha es una decisión puramente visual, no una afirmación de que
// uno es base y el otro escolta.
const LINEUP_SLOT_COORDS = {
  C: [[0, 55]],
  F: [
    [-115, 175],
    [115, 175],
  ],
  G: [
    [-95, 305],
    [95, 305],
  ],
};

function fmt1(value) {
  return typeof value === "number" ? value.toFixed(1) : value;
}

/** Set de stats UNIFICADO con leaderboard.js (a petición del usuario):
 * PPG/RPG/APG/SPG/BPG/FG%/3P%/récord de equipo + el "valor" que ordena
 * el quinteto (season_value en All-NBA, defensive_value en
 * All-Defensive -- awards_projection.compute_all_nba_teams/
 * compute_all_defensive_teams ya incluyen las mismas columnas que
 * MVP/DPOY/ROY/6.º Hombre). Siempre de la temporada proyectada -- los
 * quintetos, a diferencia de MIP, no tienen una "temporada anterior"
 * con la que compararse. */
function lineupStats(record) {
  const stat = (label, key) => (record[key] != null ? { label, value: fmt1(record[key]) } : null);
  const awardValue = record.season_value ?? record.defensive_value;
  const awardLabel = record.season_value != null ? "Valor temporada" : "Valor defensivo";
  return [
    stat("PPG", "PPG"),
    stat("RPG", "RPG"),
    stat("APG", "APG"),
    stat("SPG", "SPG"),
    stat("BPG", "BPG"),
    stat("FG%", "FG%"),
    stat("3P%", "3P%"),
    record.team_record ? { label: "Récord equipo", value: record.team_record } : null,
    awardValue != null ? { label: awardLabel, value: fmt1(awardValue) } : null,
  ].filter(Boolean);
}

/** `records`: 5 filas de un quinteto (all_nba/all_defensive), cada una
 * con player_id, player_name, position_slot ("G"/"F"/"C"), y
 * season_value o defensive_value según el premio. Dibuja la misma
 * media cancha que courtShotChart() (SVG, coordenadas físicas reales)
 * y coloca la foto real de cada jugador (playerPhoto(), mismo hotlink +
 * fallback a iniciales que el resto de la app) encima, vía `<div>`
 * posicionados en % sobre el mismo viewBox -- mezclar HTML dentro de un
 * SVG (`<foreignObject>`) es más frágil entre navegadores que
 * superponer una capa HTML absoluta sobre el SVG. Cada foto es
 * clicable (abre el popup de detalle del jugador, mismo que el resto
 * de la app) y al pasar el ratón muestra un resumen rápido -- así el
 * gráfico basta por sí solo, sin necesitar la tabla de al lado. */
export function courtLineup(records, { title, teamIds = {}, season } = {}) {
  const svg = svgEl("svg", {
    viewBox: "-250 -60 500 500",
    class: "court-chart",
    role: "img",
    "aria-label": title || "Quinteto sobre la cancha",
  });
  svg.append(courtLines());

  const wrap = el("div", { class: "court-chart-wrap court-lineup-wrap" }, [svg]);

  const used = {};
  for (const rec of records) {
    const slot = rec.position_slot;
    const options = LINEUP_SLOT_COORDS[slot] || [[0, 200]];
    const idx = used[slot] || 0;
    used[slot] = idx + 1;
    const [x, y] = options[Math.min(idx, options.length - 1)];

    // viewBox = "-250 -60 500 500" -> % = (coord - min) / tamaño * 100.
    const leftPct = ((x + 250) / 500) * 100;
    const topPct = ((y + 60) / 500) * 100;
    const lastName = (rec.player_name || "?").trim().split(" ").slice(-1)[0];

    const marker = el(
      "button",
      { type: "button", class: "court-lineup-marker", style: `left: ${leftPct}%; top: ${topPct}%;` },
      [playerPhoto(rec.player_id, rec.player_name, 44), el("span", { class: "court-lineup-label" }, lastName)]
    );
    const preview = (event) =>
      showPlayerPreview(event, {
        playerId: rec.player_id,
        playerName: rec.player_name,
        teamAbbreviation: rec.team_abbreviation,
        caption: season ? `Temporada proyectada ${season}` : null,
        stats: lineupStats(rec),
      });
    marker.addEventListener("click", () => openPlayerModal(rec.player_id, teamIds[rec.team_abbreviation]));
    marker.addEventListener("mousemove", preview);
    marker.addEventListener("mouseleave", hidePlayerPreview);
    marker.addEventListener("focus", preview);
    marker.addEventListener("blur", hidePlayerPreview);
    wrap.append(marker);
  }
  return wrap;
}

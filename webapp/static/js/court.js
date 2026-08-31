// court.js -- mapa de tiros real (LOC_X/LOC_Y de ShotChartDetail, ver
// data_pipeline.fetch_player_shot_chart / build_roster_shot_charts_dataset)
// sobre una media cancha dibujada a mano en SVG, sin librería externa
// (mismo criterio que charts.js: nada de Chart.js/D3 vendorizado). Las
// medidas de la cancha (aro, tablero, zona, línea de 3, semicírculo de
// tiros libres) son las medidas físicas reales de una cancha NBA en
// décimas de pie desde el aro -- el mismo sistema de coordenadas que
// devuelve `ShotChartDetail`, así que los puntos se colocan
// directamente sin reescalar.

import { el } from "./ui.js";

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

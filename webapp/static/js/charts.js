// charts.js -- gráficos SVG mínimos, sin dependencias externas. Sigue las
// especificaciones de la skill dataviz de este proyecto: barras finas con
// extremo redondeado, líneas de 2px, gridlines hairline recesivas, un solo
// hue secuencial (azul), tooltip al pasar el ratón, un único eje siempre
// (nunca doble eje Y).

import { showTooltip, hideTooltip } from "./ui.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const WIDTH = 640;
const HEIGHT = 260;
const MARGIN = { top: 16, right: 16, bottom: 28, left: 40 };

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function baseSvg() {
  return svgEl("svg", {
    class: "chart",
    viewBox: `0 0 ${WIDTH} ${HEIGHT}`,
    width: "100%",
    height: "260",
    preserveAspectRatio: "xMidYMid meet",
  });
}

function plotArea() {
  return {
    x0: MARGIN.left,
    x1: WIDTH - MARGIN.right,
    y0: MARGIN.top,
    y1: HEIGHT - MARGIN.bottom,
  };
}

function niceTicks(min, max, count = 4) {
  if (min === max) return [min];
  const step = (max - min) / count;
  return Array.from({ length: count + 1 }, (_, i) => min + step * i);
}

/** Columnas -- histograma de victorias / de percentiles. */
export function columnChart(container, { categories, values, tooltipLabel = "" }) {
  const { x0, x1, y0, y1 } = plotArea();
  const svg = baseSvg();
  const maxValue = Math.max(...values, 1);
  const accent = cssVar("--accent-450");
  const gridline = cssVar("--gridline");
  const baseline = cssVar("--baseline");

  for (const tick of niceTicks(0, maxValue)) {
    const y = y1 - (tick / maxValue) * (y1 - y0);
    svg.append(svgEl("line", { class: "gridline", x1: x0, x2: x1, y1: y, y2: y, stroke: gridline }));
    const label = svgEl("text", { x: x0 - 6, y: y + 3, "text-anchor": "end" });
    label.textContent = Math.round(tick);
    svg.append(label);
  }
  svg.append(svgEl("line", { class: "axis-line", x1: x0, x2: x1, y1: y1, y2: y1, stroke: baseline }));

  const bandWidth = (x1 - x0) / categories.length;
  const barWidth = Math.min(24, bandWidth * 0.7);

  categories.forEach((category, i) => {
    const value = values[i];
    const barHeight = (value / maxValue) * (y1 - y0);
    const cx = x0 + bandWidth * i + bandWidth / 2;
    const barX = cx - barWidth / 2;
    const barY = y1 - barHeight;
    const rect = svgEl("rect", {
      x: barX,
      y: barY,
      width: barWidth,
      height: Math.max(barHeight, 0),
      rx: 4,
      ry: 4,
      fill: accent,
    });
    rect.addEventListener("mousemove", (event) => showTooltip(event, `${category}${tooltipLabel}: ${value}`));
    rect.addEventListener("mouseleave", hideTooltip);
    svg.append(rect);

    if (categories.length <= 20 || i % Math.ceil(categories.length / 20) === 0) {
      const label = svgEl("text", { x: cx, y: y1 + 16, "text-anchor": "middle" });
      label.textContent = category;
      svg.append(label);
    }
  });

  container.replaceChildren(svg);
}

/** Línea única -- serie ordenada (p.ej. Net Rating estimado por temporada simulada). */
export function lineChart(container, { values, yLabel = "" }) {
  const { x0, x1, y0, y1 } = plotArea();
  const svg = baseSvg();
  const min = Math.min(...values);
  const max = Math.max(...values);
  const accent = cssVar("--accent-450");
  const gridline = cssVar("--gridline");
  const baseline = cssVar("--baseline");

  for (const tick of niceTicks(min, max)) {
    const y = y1 - ((tick - min) / (max - min || 1)) * (y1 - y0);
    svg.append(svgEl("line", { class: "gridline", x1: x0, x2: x1, y1: y, y2: y, stroke: gridline }));
    const label = svgEl("text", { x: x0 - 6, y: y + 3, "text-anchor": "end" });
    label.textContent = tick.toFixed(1);
    svg.append(label);
  }
  svg.append(svgEl("line", { class: "axis-line", x1: x0, x2: x1, y1: y1, y2: y1, stroke: baseline }));

  const points = values.map((value, i) => {
    const x = x0 + (i / (values.length - 1 || 1)) * (x1 - x0);
    const y = y1 - ((value - min) / (max - min || 1)) * (y1 - y0);
    return [x, y];
  });

  const path = svgEl("polyline", {
    points: points.map((p) => p.join(",")).join(" "),
    fill: "none",
    stroke: accent,
    "stroke-width": 2,
    "stroke-linejoin": "round",
    "stroke-linecap": "round",
  });
  svg.append(path);

  // Overlay invisible más grueso para facilitar el hover, con tooltip
  // aproximado al punto más cercano en X.
  const hitArea = svgEl("rect", { x: x0, y: y0, width: x1 - x0, height: y1 - y0, fill: "transparent" });
  hitArea.addEventListener("mousemove", (event) => {
    const rect = container.querySelector("svg").getBoundingClientRect();
    const relX = ((event.clientX - rect.left) / rect.width) * WIDTH;
    const idx = Math.round(((relX - x0) / (x1 - x0)) * (values.length - 1));
    const clamped = Math.min(values.length - 1, Math.max(0, idx));
    showTooltip(event, `${yLabel} #${clamped + 1}: ${values[clamped].toFixed(2)}`);
  });
  hitArea.addEventListener("mouseleave", hideTooltip);
  svg.append(hitArea);

  container.replaceChildren(svg);
}

/** Scatter con línea de referencia y=x (guía, no una segunda serie). */
export function scatterChart(container, { points, xLabel = "Real", yLabel = "Simulado" }) {
  const { x0, x1, y0, y1 } = plotArea();
  const svg = baseSvg();
  const allValues = points.flatMap((p) => [p.x, p.y]);
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const accent = cssVar("--accent-450");
  const gridline = cssVar("--gridline");
  const baseline = cssVar("--baseline");
  const surface = cssVar("--surface-card");

  const scaleX = (v) => x0 + ((v - min) / (max - min || 1)) * (x1 - x0);
  const scaleY = (v) => y1 - ((v - min) / (max - min || 1)) * (y1 - y0);

  for (const tick of niceTicks(min, max)) {
    const y = scaleY(tick);
    svg.append(svgEl("line", { class: "gridline", x1: x0, x2: x1, y1: y, y2: y, stroke: gridline }));
    const label = svgEl("text", { x: x0 - 6, y: y + 3, "text-anchor": "end" });
    label.textContent = Math.round(tick);
    svg.append(label);
  }
  svg.append(svgEl("line", { class: "axis-line", x1: x0, x2: x1, y1: y1, y2: y1, stroke: baseline }));

  // Diagonal de referencia y=x -- guía neutra, no una serie de datos.
  svg.append(
    svgEl("line", {
      x1: scaleX(min),
      y1: scaleY(min),
      x2: scaleX(max),
      y2: scaleY(max),
      stroke: baseline,
      "stroke-width": 1,
      "stroke-dasharray": "4 3",
    })
  );

  for (const point of points) {
    const dot = svgEl("circle", {
      cx: scaleX(point.x),
      cy: scaleY(point.y),
      r: 4,
      fill: accent,
      stroke: surface,
      "stroke-width": 2,
    });
    dot.addEventListener("mousemove", (event) =>
      showTooltip(event, `${xLabel}: ${point.x.toFixed(1)} · ${yLabel}: ${point.y.toFixed(1)}`)
    );
    dot.addEventListener("mouseleave", hideTooltip);
    svg.append(dot);
  }

  container.replaceChildren(svg);
}

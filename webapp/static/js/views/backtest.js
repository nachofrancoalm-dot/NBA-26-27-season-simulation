import { api } from "../api.js";
import { card, el, statGrid, dataTable, glossaryExpander, emptyState, skeleton } from "../ui.js";
import { columnChart, scatterChart } from "../charts.js";

export async function render(container) {
  container.replaceChildren(skeleton(["title", "short"]), skeleton());

  const [narrative, sweep] = await Promise.allSettled([api.backtest(), api.backtestSweep()]);

  const cards = [];

  if (narrative.status === "fulfilled") {
    const { cases, glossary, n_extreme_percentile_cases, n_cases } = narrative.value;
    const children = [
      el("h2", {}, "Backtesting contra comparables históricos"),
      dataTable(cases, glossary),
    ];
    if (n_extreme_percentile_cases > 0) {
      children.push(
        el("p", { class: "caption" }, `${n_extreme_percentile_cases} de ${n_cases} casos en percentil extremo (<5% o >95%).`)
      );
    }
    cards.push(card(children));
  } else {
    cards.push(card([el("h2", {}, "Backtesting contra comparables históricos"), emptyState(narrative.reason.message)]));
  }

  if (sweep.status === "fulfilled") {
    const { calibration, percentile_histogram, scatter, cases, glossary } = sweep.value;
    const bias = calibration.mean_error_wins;
    const biasNote =
      bias < -3
        ? `El modelo SOBREESTIMA victorias en promedio (${bias.toFixed(1)} victorias/temporada).`
        : bias > 3
        ? `El modelo SUBESTIMA victorias en promedio (${bias > 0 ? "+" : ""}${bias.toFixed(1)} victorias/temporada).`
        : null;

    cards.push(
      card([
        el("h2", {}, "Backtesting sistemático (30 equipos × varias temporadas)"),
        statGrid([
          ["Casos", calibration.n_cases],
          ["% dentro de P10-P90", `${calibration.pct_within_p10_p90.toFixed(1)}%`],
          ["Percentil medio", calibration.mean_percentile.toFixed(1)],
          ["Error medio (victorias)", `${bias >= 0 ? "+" : ""}${bias.toFixed(1)}`],
          ["Correlación real vs. predicho", calibration.correlation_actual_vs_predicted.toFixed(2)],
        ]),
        biasNote ? el("p", { class: "caption" }, biasNote) : null,
        glossaryExpander(Object.entries(glossary)),
      ])
    );

    cards.push(
      card([
        el("h3", {}, "Distribución de percentiles reales"),
        el("p", { class: "caption" }, "Uniforme = bien calibrado. Concentrado en 0 = sobreestima; en 100 = subestima."),
        el("div", { id: "percentile-histogram" }),
      ])
    );

    cards.push(
      card([
        el("h3", {}, "Victorias reales vs. simuladas (media)"),
        el("div", { id: "backtest-scatter" }),
        el("details", { class: "glossary" }, [
          el("summary", {}, `Ver los ${cases.length} casos individuales`),
          el("div", { class: "glossary-body" }, dataTable(cases)),
        ]),
      ])
    );
  } else {
    cards.push(
      card([el("h2", {}, "Backtesting sistemático (30 equipos × varias temporadas)"), emptyState(sweep.reason.message)])
    );
  }

  container.replaceChildren(...cards);

  if (sweep.status === "fulfilled") {
    const { percentile_histogram, scatter } = sweep.value;
    columnChart(document.getElementById("percentile-histogram"), {
      categories: Object.keys(percentile_histogram),
      values: Object.values(percentile_histogram),
      tooltipLabel: " casos",
    });
    scatterChart(document.getElementById("backtest-scatter"), {
      points: scatter.map((row) => ({ x: row.actual, y: row.simulated })),
      xLabel: "Reales",
      yLabel: "Simuladas",
    });
  }
}

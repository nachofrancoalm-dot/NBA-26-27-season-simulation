import { api } from "../api.js";
import { card, el, statGrid, dataTable, glossaryExpander, emptyState, skeleton } from "../ui.js";
import { columnChart, lineChart } from "../charts.js";

export async function render(container) {
  container.replaceChildren(skeleton(["title", "short"]), skeleton());

  let data;
  try {
    data = await api.simulation();
  } catch (err) {
    container.replaceChildren(card([el("h2", {}, "Distribución de temporadas simuladas"), emptyState(err.message)]));
    return;
  }

  const { summary, wins_histogram, net_rating_sorted, glossary, n_seasons } = data;

  const distributionCard = card([
    el("h2", {}, "Distribución de temporadas simuladas"),
    statGrid([
      ["Victorias medias", summary.mean.toFixed(1)],
      ["P10", summary.p10.toFixed(0)],
      ["Mediana", summary.p50.toFixed(0)],
      ["P90", summary.p90.toFixed(0)],
    ]),
    el("h3", {}, "Victorias por temporada simulada"),
    el("div", { id: "wins-histogram" }),
    el("p", { class: "caption" }, `${n_seasons.toLocaleString("es")} temporadas simuladas.`),
    el("h3", {}, "Net Rating estimado por temporada simulada (ordenado)"),
    el("div", { id: "net-rating-line" }),
    glossaryExpander(Object.entries(glossary)),
  ]);

  const healthyCard = card([
    el("h2", {}, "¿Y si nadie se lesionara?"),
    el(
      "p",
      { class: "caption" },
      "Repite la simulación completa con el riesgo de lesión de todo el roster puesto a cero -- para separar cuánto de la temporada proyectada se pierde por lesiones frente al techo real de talento del equipo."
    ),
    el("button", { class: "btn", id: "simulate-healthy-btn" }, "💪 Simular temporada sin lesiones"),
    el("div", { id: "healthy-result", style: "margin-top: 16px;" }),
  ]);

  const seasonLogCard = card([
    el("h2", {}, "Partidos jugados por jugador en una temporada simulada"),
    el("p", { class: "caption" }, "Una temporada concreta, no la distribución agregada. Cada clic sortea una nueva."),
    el("button", { class: "btn", id: "simulate-season-btn" }, "🩹 Simular partidos de la temporada"),
    el("div", { id: "season-log-result", style: "margin-top: 16px;" }),
  ]);

  container.replaceChildren(distributionCard, healthyCard, seasonLogCard);

  const histCategories = Object.keys(wins_histogram);
  const histValues = Object.values(wins_histogram);
  columnChart(document.getElementById("wins-histogram"), {
    categories: histCategories,
    values: histValues,
    tooltipLabel: " victorias",
  });
  lineChart(document.getElementById("net-rating-line"), { values: net_rating_sorted, yLabel: "Net Rating" });

  document.getElementById("simulate-healthy-btn").addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    button.textContent = "Simulando… (unos segundos)";
    const resultContainer = document.getElementById("healthy-result");
    resultContainer.replaceChildren(el("div", { class: "caption" }, "Simulando 10.000 temporadas sin lesiones…"));
    try {
      const healthy = await api.simulationNoInjuries();
      renderHealthyComparison(resultContainer, summary, healthy.summary);
    } catch (err) {
      resultContainer.replaceChildren(emptyState(err.message));
    } finally {
      button.disabled = false;
      button.textContent = "💪 Simular temporada sin lesiones";
    }
  });

  document.getElementById("simulate-season-btn").addEventListener("click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    button.textContent = "Simulando…";
    const resultContainer = document.getElementById("season-log-result");
    try {
      const result = await api.simulationSeasonLog();
      renderSeasonLog(resultContainer, result);
    } catch (err) {
      resultContainer.replaceChildren(emptyState(err.message));
    } finally {
      button.disabled = false;
      button.textContent = "🩹 Simular partidos de la temporada";
    }
  });
}

function renderHealthyComparison(container, withInjuries, withoutInjuries) {
  const rows = [
    ["Victorias medias", withInjuries.mean, withoutInjuries.mean],
    ["P10", withInjuries.p10, withoutInjuries.p10],
    ["Mediana", withInjuries.p50, withoutInjuries.p50],
    ["P90", withInjuries.p90, withoutInjuries.p90],
  ].map(([metric, real, healthy]) => ({
    Métrica: metric,
    "Con lesiones (real)": real.toFixed(1),
    "Sin lesiones": healthy.toFixed(1),
    Diferencia: `${healthy - real >= 0 ? "+" : ""}${(healthy - real).toFixed(1)}`,
  }));

  container.replaceChildren(
    dataTable(rows),
    el(
      "p",
      { class: "caption" },
      `El riesgo de lesión le cuesta al equipo ${(withoutInjuries.mean - withInjuries.mean).toFixed(1)} victorias de media en la temporada proyectada.`
    )
  );
}

function renderSeasonLog(container, result) {
  const table = dataTable(
    result.players.map(({ player_name, games_played, games_missed }) => ({
      Jugador: player_name,
      "Partidos jugados": games_played,
      "Partidos perdidos": games_missed,
    }))
  );

  const injured = result.players.filter((p) => p.injury_events.length > 0);
  const details =
    injured.length === 0
      ? el("p", { class: "caption" }, "Nadie tuvo ausencias por lesión en esta temporada simulada.")
      : el("div", {}, [
          el("h3", {}, "Detalle de ausencias por jugador"),
          ...injured.map((p) =>
            el("p", {}, [
              el("b", {}, p.player_name),
              ": ",
              p.injury_events
                .map((e) => `partido ${e.start_game}, ${e.length} partido(s) — ${e.category}`)
                .join("; "),
            ])
          ),
        ]);

  container.replaceChildren(table, details);
}

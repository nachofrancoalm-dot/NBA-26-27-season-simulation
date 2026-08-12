import { api } from "../api.js";
import { card, el, statGrid, dataTable, glossaryExpander, emptyState } from "../ui.js";

export async function render(container) {
  container.replaceChildren();
  container.append(el("div", { class: "caption" }, "Cargando campeones…"));

  let data;
  try {
    data = await api.champions();
  } catch (err) {
    container.replaceChildren(card([el("h2", {}, "Campeones reales de la NBA"), emptyState(err.message)]));
    return;
  }

  const cards = [
    card([
      el("h2", {}, "Campeones reales de la NBA"),
      el("p", { class: "caption" }, "Datos reales (no simulados) -- contexto y validación del simulador."),
      statGrid([
        ["Temporadas analizadas", data.n_seasons],
        ["Franquicias campeonas", data.n_distinct_champions],
        ["Más títulos", `${data.most_titles.team_abbreviation} (${data.most_titles.titles})`],
      ]),
    ]),
    card([el("h2", {}, "Camino al título"), dataTable(data.title_paths, data.glossary)]),
    card([
      el("h2", {}, "¿De qué seed salen los campeones?"),
      dataTable(
        data.seed_comparison.map((row) => ({
          Seed: row.seed,
          "% campeones reales": row.real_pct,
          "% campeones simulados": row.simulated_pct,
        }))
      ),
      el("p", { class: "caption" }, "Ningún campeón real salió de un seed peor que el 3 (ver README)."),
    ]),
  ];

  if (data.roster_profiles && data.roster_profiles.length) {
    cards.push(
      card([
        el("h2", {}, "Composición del roster campeón"),
        el("p", { class: "caption" }, "Descriptivo, no predictivo -- muestra demasiado pequeña."),
        dataTable(data.roster_profiles, data.glossary),
      ])
    );
  }

  if (data.seed_trajectories && data.seed_trajectories.length) {
    cards.push(
      card([
        el("h2", {}, "Trayectoria de seed por franquicia"),
        dataTable(data.seed_trajectories),
      ])
    );
  }

  cards.push(card([glossaryExpander(Object.entries(data.glossary))]));

  container.replaceChildren(...cards);
}

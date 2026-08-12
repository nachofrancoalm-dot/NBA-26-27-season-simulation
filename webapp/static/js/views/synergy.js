import { api } from "../api.js";
import { card, el, dataTable, emptyState } from "../ui.js";

export async function render(container) {
  container.replaceChildren();
  container.append(el("div", { class: "caption" }, "Cargando sinergia…"));

  let data;
  try {
    data = await api.synergy();
  } catch (err) {
    container.replaceChildren(card([el("h2", {}, "Sinergia de alineación por pareja de jugadores"), emptyState(err.message)]));
    return;
  }

  container.replaceChildren(
    card([
      el("h2", {}, "Sinergia de alineación por pareja de jugadores"),
      dataTable(data.pairs, data.glossary),
    ])
  );
}

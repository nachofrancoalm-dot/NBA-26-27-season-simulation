import { api } from "../api.js";
import { card, el, dataTable, emptyState, skeleton } from "../ui.js";

export async function render(container) {
  container.replaceChildren(skeleton());

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

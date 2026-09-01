// hypothetical-league.js -- estado compartido "estás viendo tu roster
// hipotético en vez de los datos reales" entre pestañas. roster-builder.js
// lo activa tras simular la liga completa; league.js y awards.js lo leen
// para decidir si muestran los datos reales o el resultado en vivo de
// POST /api/sandbox/league. Vive en memoria de este módulo (singleton de
// ES module, no localStorage ni backend) -- se pierde al recargar, como el resto del estado de navegación.

import { el } from "./ui.js";

let active = null; // { playerIds: number[], result: {standings, playoffs, awards} } | null

export function getHypotheticalLeague() {
  return active;
}

export function setHypotheticalLeague(playerIds, result) {
  active = { playerIds: [...playerIds], result };
}

export function clearHypotheticalLeague() {
  active = null;
}

/** Banner que sustituye a scenarioBar() cuando hay un roster hipotético
 * activo. `onReturn()` limpia el estado y vuelve a renderizar la vista
 * con los datos reales. */
export function hypotheticalBanner(onReturn) {
  return el("div", { class: "banner warning", style: "display: flex; align-items: center; justify-content: space-between; gap: 12px;" }, [
    el(
      "span",
      {},
      "🔮 Viendo la liga con tu roster hipotético en vez de los datos reales -- los otros 29 equipos siguen con su roster real."
    ),
    el("button", { type: "button", class: "btn-ghost-dark", onclick: onReturn }, "Volver a los datos reales"),
  ]);
}

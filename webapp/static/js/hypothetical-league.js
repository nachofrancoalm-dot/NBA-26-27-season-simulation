// hypothetical-league.js -- estado compartido "estás viendo tu roster
// hipotético en vez de los datos reales" entre pestañas. roster-builder.js
// (splash) lo activa tras simular la liga completa; league.js y awards.js
// lo leen al renderizar para decidir si muestran los datos reales
// (precalculados, instantáneos) o el resultado en vivo de
// POST /api/sandbox/league (src/league_sandbox.py).
//
// Vive en memoria de ESTE módulo -- un singleton de ES module, no
// localStorage ni el backend -- así que se pierde al recargar la página,
// igual que el resto del estado de navegación de esta SPA (p.ej. qué
// sub-pestaña estaba activa). No hace falta más: es una vista de sesión,
// no algo que deba sobrevivir a un refresh.

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
 * activo -- mismo hueco visual, pero deja claro que estos NO son los
 * datos reales precalculados. `onReturn()` limpia el estado y vuelve a
 * renderizar la vista con los datos reales. */
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

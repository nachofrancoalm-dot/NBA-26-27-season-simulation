// scenario.js -- estado compartido del escenario de Liga NBA ("con
// lesiones" / "sin lesiones", ver src/league_simulation.py::_apply_scenario).
// Un módulo, no localStorage: el escenario "sin lesiones" vive solo
// mientras dure la sesión del servidor (los CSV se regeneran en cada
// POST /simulate), así que persistirlo entre recargas del navegador
// sería confuso si el server se reinició entre medias. league.js
// ("Liga y Playoffs") y awards.js ("Premios individuales") comparten
// este mismo estado -- cambiar de escenario en uno afecta al otro la
// próxima vez que se visite, mismo espíritu que pillToggle pero a nivel
// de pestaña en vez de por tabla.

import { api } from "./api.js";
import { el } from "./ui.js";

const SCENARIOS = [
  { value: "with_injuries", label: "🏥 Con lesiones" },
  { value: "no_injuries", label: "💪 Sin lesiones" },
];

let currentScenario = "with_injuries";
// Qué escenarios ya se han simulado esta sesión -- evita volver a
// preguntar al backend cada vez que se cambia de sub-pestaña dentro de
// Liga NBA. Arranca con "with_injuries" en true porque ese es el
// resultado ya persistido por el pipeline normal (no necesita botón).
const simulated = { with_injuries: true, no_injuries: false };

export function getScenario() {
  return currentScenario;
}

/** Card con los dos botones de escenario. `onChange()` se llama después
 * de activar un escenario (ya simulado o recién simulado) para que la
 * vista que la usa vuelva a pedir sus datos. */
export function scenarioBar(status, onChange) {
  if (status?.datasets?.league_no_injuries) simulated.no_injuries = true;

  const container = el("div", { class: "segmented", id: "scenario-bar" });
  const statusText = el("span", { class: "caption", style: "margin-left: 12px; margin-bottom: 0;" });

  const buttons = SCENARIOS.map((s) =>
    el(
      "button",
      {
        type: "button",
        "aria-pressed": String(s.value === currentScenario),
        onclick: () => activateScenario(s.value, buttons, statusText, onChange),
      },
      s.label
    )
  );
  container.append(...buttons);

  return el("div", { style: "display: flex; align-items: center; margin-bottom: 16px;" }, [container, statusText]);
}

async function activateScenario(value, buttons, statusText, onChange) {
  if (value === currentScenario && simulated[value]) return;

  buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(SCENARIOS[i].value === value)));

  if (!simulated[value]) {
    statusText.textContent = "Simulando temporada regular + playoffs de los 30 equipos… puede tardar unos 20-30 segundos.";
    buttons.forEach((b) => (b.disabled = true));
    try {
      await api.leagueSimulate(value);
      simulated[value] = true;
      statusText.textContent = "";
    } catch (err) {
      statusText.textContent = `Error al simular: ${err.message}`;
      buttons.forEach((b) => (b.disabled = false));
      buttons.forEach((b, i) => b.setAttribute("aria-pressed", String(SCENARIOS[i].value === currentScenario)));
      return;
    }
    buttons.forEach((b) => (b.disabled = false));
  }

  currentScenario = value;
  onChange();
}

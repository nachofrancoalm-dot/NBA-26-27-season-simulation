import { api } from "./api.js";
import { el, teamBadge } from "./ui.js";
import { applyTeamColors } from "./team-colors.js";
import { playerHeroCard } from "./player-hero.js";
import * as splashView from "./views/splash.js";
import * as rosterView from "./views/roster.js";
import * as simulationView from "./views/simulation.js";
import * as synergyView from "./views/synergy.js";
import * as backtestView from "./views/backtest.js";
import * as leagueView from "./views/league.js";
import * as awardsView from "./views/awards.js";
import * as championsView from "./views/champions.js";
import * as explainerView from "./views/explainer.js";

const TEAM_SUBVIEWS = {
  roster: rosterView,
  simulacion: simulationView,
  sinergia: synergyView,
  backtesting: backtestView,
};

const LEAGUE_SUBVIEWS = {
  liga: leagueView,
  premios: awardsView,
  campeones: championsView,
};

const state = {
  activeSubTab: { "mi-equipo": "roster", "liga-nba": "liga" },
};

function renderHeader(status) {
  const badgeContainer = document.getElementById("header-badge");
  badgeContainer.replaceWith(teamBadge(status.team.team_id, status.team.abbreviation));
  document.getElementById("page-title").textContent = status.team.name;
  document.getElementById("page-subtitle").textContent = `${status.team.season} — Simulador Monte Carlo`;
}

/** selector: botones del grupo. panelPrefix: prefijo COMPLETO del id de
 * panel de ESTE grupo (p.ej. "subtab-team-") -- así un grupo nunca toca
 * los paneles `hidden` del otro grupo. */
function setupTabs(selector, panelPrefix, dataAttr, onSelect) {
  const buttons = document.querySelectorAll(selector);
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      selectTab(buttons, panelPrefix, button.dataset[dataAttr]);
      onSelect(button.dataset[dataAttr]);
    });
  });
}

function selectTab(buttons, panelPrefix, key) {
  buttons.forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === key || b.dataset.subtab === key)));
  document.querySelectorAll(`[id^="${panelPrefix}"]`).forEach((panel) => {
    panel.hidden = panel.id !== `${panelPrefix}${key}`;
  });
}

function renderActiveSubTab(group) {
  const key = state.activeSubTab[group];
  const prefix = group === "mi-equipo" ? "team" : "league";
  const views = group === "mi-equipo" ? TEAM_SUBVIEWS : LEAGUE_SUBVIEWS;
  const container = document.getElementById(`subtab-${prefix}-${key}`);
  return views[key].render(container);
}

/** Solo las TRES secciones reales de la app -- el splash (ver
 * views/splash.js) ya no es una de ellas, vive en su propio overlay
 * (#splash) por encima de .app-shell, no en #top-tabs. */
function renderTopTab(key) {
  if (key === "mi-equipo") return renderActiveSubTab("mi-equipo");
  if (key === "liga-nba") return renderActiveSubTab("liga-nba");
  if (key === "explicador") return explainerView.render(document.getElementById("tab-explicador"));
  return Promise.resolve();
}

const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Overlay de transición al cambiar de pestaña de primer nivel (Mi
 * equipo / Liga NBA / Explicador) Y al entrar desde el splash -- el
 * jugador animado cruza la pantalla mientras la pestaña de destino se
 * renderiza detrás. Con `prefers-reduced-motion` se salta del todo
 * (cambio instantáneo), tal como exige la propia skill de UI/UX de este
 * proyecto. La duración mínima visible (600ms) y la carga real de datos
 * corren en paralelo -- el overlay espera al que tarde más de los dos,
 * nunca a la suma. */
let navTransitionGeneration = 0;

async function withNavTransition(key, doSwitch) {
  if (prefersReducedMotion()) {
    doSwitch();
    return renderTopTab(key);
  }

  // Contador de generación: un clic rápido durante una transición en
  // curso no debe dejar que el `setTimeout` de limpieza de la
  // transición VIEJA borre el contenido que la transición NUEVA acaba
  // de insertar -- cada invocación solo limpia si sigue siendo la más
  // reciente cuando le toca su turno.
  const generation = ++navTransitionGeneration;

  const overlay = document.getElementById("nav-transition");
  overlay.replaceChildren(playerHeroCard(state.featuredPlayerId, state.featuredPlayerName, { extraClass: "dribbler-shadow-off", size: 110 }));
  const track = overlay.querySelector(".hero-stage");
  track.classList.add("nav-transition-track");
  overlay.classList.add("visible");

  doSwitch();
  const minDuration = new Promise((resolve) => setTimeout(resolve, 600));
  await Promise.all([minDuration, renderTopTab(key)]);

  if (generation !== navTransitionGeneration) return;
  overlay.classList.remove("visible");
  setTimeout(() => {
    if (generation === navTransitionGeneration) overlay.replaceChildren();
  }, 250);
}

/** Navegación programática -- usada por los accesos rápidos del splash
 * y por el "volver a Inicio" del logo, para saltar directo a una
 * sub-pestaña concreta con la misma transición que un clic real en la
 * barra de pestañas. */
function navigateTo(topKey, subKey) {
  const topButton = document.querySelector(`#top-tabs [data-tab="${topKey}"]`);
  if (!topButton) return;

  // El estado de la sub-pestaña (y su botón activo) se fija ANTES de
  // arrancar la transición -- withNavTransition llama a renderTopTab
  // de forma síncrona hasta su primer await, y renderActiveSubTab lee
  // state.activeSubTab en ese mismo instante síncrono. Si esta
  // asignación fuera después, renderTopTab leería el valor VIEJO.
  if (subKey) {
    const subSelector = topKey === "mi-equipo" ? "#sub-tabs-team" : "#sub-tabs-league";
    const subPrefix = topKey === "mi-equipo" ? "subtab-team-" : "subtab-league-";
    const subButtons = document.querySelectorAll(`${subSelector} .tab-button`);
    selectTab(subButtons, subPrefix, subKey);
    state.activeSubTab[topKey] = subKey;
  }

  const topButtons = document.querySelectorAll("#top-tabs .tab-button");
  withNavTransition(topKey, () => selectTab(topButtons, "tab-", topKey));
}

function showSplash() {
  document.getElementById("splash").classList.remove("dismissed");
}

function hideSplash() {
  document.getElementById("splash").classList.add("dismissed");
}

/** Entrar a la app desde el splash: lo mismo que navigateTo, más
 * descartar el overlay de splash. Es el `enter` que recibe
 * views/splash.js -- el splash en sí no sabe nada de cómo está montado
 * el shell de pestañas, solo pide "llévame a X". */
function enterApp(topKey, subKey) {
  hideSplash();
  navigateTo(topKey, subKey);
}

async function bootstrap() {
  const status = await api.status();
  renderHeader(status);
  applyTeamColors(status.team.abbreviation);

  document.getElementById("home-link").addEventListener("click", showSplash);

  const topButtons = document.querySelectorAll("#top-tabs .tab-button");
  topButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.tab;
      withNavTransition(key, () => selectTab(topButtons, "tab-", key));
    });
  });

  setupTabs("#sub-tabs-team .tab-button", "subtab-team-", "subtab", (key) => {
    state.activeSubTab["mi-equipo"] = key;
    renderActiveSubTab("mi-equipo");
  });

  setupTabs("#sub-tabs-league .tab-button", "subtab-league-", "subtab", (key) => {
    state.activeSubTab["liga-nba"] = key;
    renderActiveSubTab("liga-nba");
  });

  // El shell de pestañas (por defecto Mi equipo > Roster) se renderiza en
  // paralelo al splash, no después de descartarlo -- así entrar se siente
  // instantáneo (los datos ya están ahí cuando termina la transición) en
  // vez de esperar a que el usuario haga clic para empezar a pedirlos.
  const teamRender = renderActiveSubTab("mi-equipo");

  const rosterResult = await api.roster("per_game").catch(() => null);
  const featured = rosterResult?.players?.[0];
  state.featuredPlayerId = featured?.player_id;
  state.featuredPlayerName = featured?.player_name;

  await Promise.all([teamRender, splashView.render(document.getElementById("splash-content"), enterApp)]);
}

bootstrap().catch((err) => {
  console.error(err);
  document.getElementById("splash-content").replaceChildren(
    el("div", { class: "empty-state" }, `No se pudo cargar la aplicación: ${err.message}`)
  );
});

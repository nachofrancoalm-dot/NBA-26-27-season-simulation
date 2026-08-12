import { api } from "./api.js";
import { el, teamBadge } from "./ui.js";
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
      const key = button.dataset[dataAttr];
      buttons.forEach((b) => b.setAttribute("aria-selected", String(b === button)));
      document.querySelectorAll(`[id^="${panelPrefix}"]`).forEach((panel) => {
        panel.hidden = panel.id !== `${panelPrefix}${key}`;
      });
      onSelect(key);
    });
  });
}

function renderActiveSubTab(group) {
  const key = state.activeSubTab[group];
  const prefix = group === "mi-equipo" ? "team" : "league";
  const views = group === "mi-equipo" ? TEAM_SUBVIEWS : LEAGUE_SUBVIEWS;
  const container = document.getElementById(`subtab-${prefix}-${key}`);
  views[key].render(container);
}

async function bootstrap() {
  const status = await api.status();
  renderHeader(status);

  setupTabs("#top-tabs .tab-button", "tab-", "tab", (key) => {
    if (key === "mi-equipo") renderActiveSubTab("mi-equipo");
    if (key === "liga-nba") renderActiveSubTab("liga-nba");
    if (key === "explicador") explainerView.render(document.getElementById("tab-explicador"));
  });

  setupTabs("#sub-tabs-team .tab-button", "subtab-team-", "subtab", (key) => {
    state.activeSubTab["mi-equipo"] = key;
    renderActiveSubTab("mi-equipo");
  });

  setupTabs("#sub-tabs-league .tab-button", "subtab-league-", "subtab", (key) => {
    state.activeSubTab["liga-nba"] = key;
    renderActiveSubTab("liga-nba");
  });

  renderActiveSubTab("mi-equipo");
}

bootstrap().catch((err) => {
  console.error(err);
  document.getElementById("tab-mi-equipo").replaceChildren(
    el("div", { class: "empty-state" }, `No se pudo cargar la aplicación: ${err.message}`)
  );
});

// api.js -- fetch wrapper compartido por todas las vistas.

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function request(path, options = {}) {
  const response = await fetch(`/api${path}`, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) {
      // sin cuerpo JSON -- nos quedamos con statusText
    }
    throw new ApiError(detail, response.status);
  }
  return response.json();
}

export const api = {
  status: () => request("/status"),
  roster: (mode) => request(`/roster?mode=${mode}`),
  simulation: () => request("/simulation"),
  simulationNoInjuries: () => request("/simulation/no-injuries", { method: "POST" }),
  simulationSeasonLog: () => request("/simulation/season-log", { method: "POST" }),
  synergy: () => request("/synergy"),
  backtest: () => request("/backtest"),
  backtestSweep: () => request("/backtest/sweep"),

  leagueStandings: (scenario = "with_injuries") => request(`/league/standings?scenario=${scenario}`),
  leaguePlayoffs: (scenario = "with_injuries") => request(`/league/playoffs?scenario=${scenario}`),
  leagueTeams: (scenario = "with_injuries") => request(`/league/teams?scenario=${scenario}`),
  leagueTeam: (abbreviation, mode, scenario = "with_injuries") =>
    request(`/league/team/${abbreviation}?mode=${mode}&scenario=${scenario}`),
  leagueBracket: (scenario = "with_injuries") => request(`/league/bracket?scenario=${scenario}`, { method: "POST" }),
  leagueSimulate: (scenario) => request(`/league/simulate?scenario=${scenario}`, { method: "POST" }),

  leagueSimulateSeasonLog: (scenario = "with_injuries") =>
    request(`/league/simulate-season-log?scenario=${scenario}`, { method: "POST" }),
  leagueSchedule: (team, scenario = "with_injuries") =>
    request(`/league/schedule?scenario=${scenario}${team ? `&team=${team}` : ""}`),
  leagueBoxscore: (gameId, scenario = "with_injuries") =>
    request(`/league/boxscore/${gameId}?scenario=${scenario}`),
  leagueHeadToHead: (teamA, teamB, scenario = "with_injuries") =>
    request(`/league/head-to-head?team_a=${teamA}&team_b=${teamB}&scenario=${scenario}`),

  awards: (scenario = "with_injuries") => request(`/awards?scenario=${scenario}`),
  champions: () => request("/champions"),

  player: (playerId) => request(`/player/${playerId}`),

  explainerContext: () => request("/explainer/context"),
  explainerAsk: (question, newsText) =>
    request("/explainer/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, news_text: newsText || null }),
    }),
  explainerSearchNews: (query) => request(`/explainer/search-news?query=${encodeURIComponent(query)}`),

  sandboxPlayers: () => request("/sandbox/players"),
  sandboxDefaultRoster: () => request("/sandbox/default"),
  sandboxSimulate: (playerIds, mcOverrides) =>
    request("/sandbox/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ player_ids: playerIds, mc_overrides: mcOverrides || null }),
    }),
  sandboxRosterStats: (playerIds, mode = "per_game") =>
    request("/sandbox/roster-stats", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ player_ids: playerIds, mode }),
    }),
  sandboxLeague: (playerIds) =>
    request("/sandbox/league", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ player_ids: playerIds }),
    }),
};

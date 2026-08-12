import { api } from "../api.js";
import { card, el, statGrid, dataTable, emptyState, teamBadge, teamCell, pillToggle } from "../ui.js";
import { openPlayerModal } from "../player-modal.js";
import { openTeamModal } from "../team-modal.js";
import { getScenario, scenarioBar } from "../scenario.js";

export async function render(container) {
  container.replaceChildren();
  container.append(el("div", { class: "caption" }, "Cargando liga…"));

  const status = await api.status();
  const scenario = getScenario();

  const [standings, playoffs, teams] = await Promise.allSettled([
    api.leagueStandings(scenario),
    api.leaguePlayoffs(scenario),
    api.leagueTeams(scenario),
  ]);

  const bar = scenarioBar(status, () => render(container));

  if (standings.status === "rejected") {
    container.replaceChildren(
      bar,
      card([el("h2", {}, "Los 30 equipos: temporada regular simulada"), emptyState(standings.reason.message)])
    );
    return;
  }

  const cards = [standingsCard(standings.value)];

  if (playoffs.status === "fulfilled") {
    cards.push(playoffsCard(playoffs.value));
  }

  if (teams.status === "fulfilled") {
    cards.push(teamExplorerCard(teams.value.teams, standings.value.team_ids));
  }

  cards.push(bracketCard());

  container.replaceChildren(bar, ...cards);
  wireBracketButton();
}

function standingsCard(standings) {
  const cols = ["seed", "team_abbreviation", "wins_mean", "wins_p10", "wins_p90", "situacion", "playoff_pct", "championship_pct"];
  const trim = (rows) => rows.map((row) => Object.fromEntries(cols.filter((c) => c in row).map((c) => [c, row[c]])));
  const formatters = { team_abbreviation: (abbr) => teamCell(standings.team_ids[abbr], abbr) };
  const doubleClick = {
    column: "team_abbreviation",
    onOpen: (record) => openTeamModal(record.team_abbreviation, standings.team_ids[record.team_abbreviation]),
  };

  return card([
    el("h2", {}, "Clasificación por conferencia"),
    el(
      "p",
      { class: "caption" },
      "Seeds 1-6 clasifican directo, 7-10 juegan el play-in, 11-15 quedan fuera. Doble clic en un equipo para ver su detalle."
    ),
    el("div", { class: "grid-2" }, [
      el("div", {}, [el("h3", {}, "Este"), dataTable(trim(standings.east), {}, { formatters, doubleClick })]),
      el("div", {}, [el("h3", {}, "Oeste"), dataTable(trim(standings.west), {}, { formatters, doubleClick })]),
    ]),
  ]);
}

function playoffsCard(playoffs) {
  const formatters = { team_abbreviation: (abbr) => teamCell(playoffs.team_ids?.[abbr], abbr) };
  const doubleClick = {
    column: "team_abbreviation",
    onOpen: (record) => openTeamModal(record.team_abbreviation, playoffs.team_ids?.[record.team_abbreviation]),
  };
  const children = [
    el("h2", {}, "Probabilidades de playoffs y campeonato"),
    dataTable(playoffs.teams, playoffs.glossary, { formatters, doubleClick }),
  ];
  if (playoffs.my_team) {
    const row = playoffs.my_team;
    children.push(
      statGrid([
        [`${row.team_abbreviation} — Playoffs`, `${row.playoff_pct.toFixed(1)}%`],
        ["Semis de conferencia", `${row.conf_semis_pct.toFixed(1)}%`],
        ["Finales de conferencia", `${row.finals_pct.toFixed(1)}%`],
        ["Campeonato", `${row.championship_pct.toFixed(1)}%`],
      ])
    );
  }
  return card(children);
}

function teamExplorerCard(teamAbbrevs, teamIds) {
  const detail = el("div", { id: "league-team-detail" });

  const railButtons = teamAbbrevs.map((abbr, i) =>
    el(
      "button",
      {
        type: "button",
        class: "team-rail-item",
        "aria-current": String(i === 0),
        onclick: (event) => {
          document
            .querySelectorAll(".team-rail-item")
            .forEach((b) => b.setAttribute("aria-current", String(b === event.currentTarget)));
          loadTeamDetail(detail, abbr);
        },
      },
      [teamBadge(teamIds[abbr], abbr, 20), abbr]
    )
  );
  const rail = el("div", { class: "team-rail" }, railButtons);

  loadTeamDetail(detail, teamAbbrevs[0]);

  return card([
    el("h2", {}, "Explorar un equipo"),
    el("div", { class: "team-explorer" }, [rail, detail]),
  ]);
}

async function loadTeamDetail(container, abbreviation) {
  let mode = "per_game";
  const scenario = getScenario();
  container.replaceChildren(el("div", { class: "caption" }, "Cargando equipo…"));

  const header = el("div");
  const metricsBox = el("div");
  const tableBox = el("div", { style: "margin-top: 16px;" });

  async function loadRoster() {
    let data;
    try {
      data = await api.leagueTeam(abbreviation, mode, scenario);
    } catch (err) {
      container.replaceChildren(emptyState(err.message));
      return;
    }

    header.replaceChildren(
      el("div", { style: "display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 16px;" }, [
        el("div", { class: "team-identity", style: "gap: 16px;" }, [
          teamBadge(data.team_id, abbreviation, 72),
          el("strong", { style: "font-size: 1.4rem;" }, abbreviation),
        ]),
        pillToggle(
          [
            { value: "per_game", label: "Por partido" },
            { value: "totals", label: "Totales" },
          ],
          mode,
          (value) => {
            mode = value;
            loadRoster();
          }
        ),
      ])
    );

    metricsBox.replaceChildren(
      statGrid([
        ["Victorias medias", data.regular.wins_mean.toFixed(1)],
        ["Playoffs", `${data.playoff.playoff_pct.toFixed(1)}%`],
        ["Semis conf.", `${data.playoff.conf_semis_pct.toFixed(1)}%`],
        ["Finales conf.", `${data.playoff.finals_pct.toFixed(1)}%`],
        ["Campeonato", `${data.playoff.championship_pct.toFixed(1)}%`],
      ])
    );
    tableBox.replaceChildren(
      data.players.length
        ? dataTable(data.players, data.glossary, {
            hiddenColumns: ["player_id"],
            doubleClick: { column: "player_name", onOpen: (record) => openPlayerModal(record.player_id, data.team_id) },
          })
        : emptyState("Sin roster proyectado disponible.")
    );
  }

  container.replaceChildren(header, metricsBox, tableBox);
  await loadRoster();
}

function bracketCard() {
  return card([
    el("h2", {}, "Bracket de playoffs"),
    el("p", { class: "caption" }, "Cada clic sortea un bracket nuevo (play-in, ronda 1, semis y finales)."),
    el("button", { class: "btn", id: "bracket-btn" }, "🎲 Simular un bracket de playoffs"),
    el("div", { id: "bracket-result", style: "margin-top: 16px;" }),
  ]);
}

/** Fila de un equipo dentro de una caja de partido -- seed + insignia +
 * abreviatura, resaltado si ganó esa serie. `seed` puede ser null (usado
 * en play-in, donde ya se ve el seed real en el bracket principal). */
function bracketTeamRow(abbreviation, seed, winner, teamIds) {
  return el("div", { class: `bracket-team${abbreviation === winner ? " winner" : ""}` }, [
    el("span", { class: "seed" }, seed != null ? String(seed) : ""),
    teamBadge(teamIds[abbreviation], abbreviation, 18),
    el("span", { class: "abbr" }, abbreviation || "—"),
  ]);
}

function bracketMatchBox(match, seedByAbbr, teamIds, gridColumn, gridRow) {
  return el(
    "div",
    { class: "bracket-match", style: `grid-column:${gridColumn}; grid-row:${gridRow};` },
    [
      bracketTeamRow(match.team_a, seedByAbbr[match.team_a], match.winner, teamIds),
      bracketTeamRow(match.team_b, seedByAbbr[match.team_b], match.winner, teamIds),
    ]
  );
}

/** Línea "⊐"/"⊏" que conecta dos partidos de una ronda con el partido de
 * la ronda siguiente (misma técnica que un bracket de torneo de verdad:
 * border-top + border-bottom dibujan los dos brazos horizontales, y
 * border-left/right el tramo vertical que los une). `depthClass` fija el
 * alto exacto vía CSS calc() a partir de --leaf-h/--row-gap (ver
 * components.css) para que el conector encaje pixel a pixel entre los
 * centros verticales de los dos partidos que alimenta -- no es una
 * aproximación. */
function bracketConnector(gridColumn, gridRow, side, depthClass) {
  return el("div", {
    class: `bracket-connector side-${side} ${depthClass}`,
    style: `grid-column:${gridColumn}; grid-row:${gridRow};`,
  });
}

/** Rejilla de UNA conferencia: Ronda 1 (4 partidos) -> Semifinales (2) ->
 * Final de conferencia (1, reconstruida en el cliente a partir de los dos
 * ganadores de semis -- el backend solo devuelve el ganador final, no un
 * objeto de partido, ver `_conference_bracket_with_matchups` en
 * league_simulation.py). `side` decide el orden de columnas -- Oeste
 * fluye izquierda->derecha, Este derecha->izquierda (mismas columnas,
 * posiciones en espejo) para que ambas confluyan hacia el centro. */
function conferenceBracketGrid(conf, side, teamIds) {
  const seedByAbbr = Object.fromEntries(conf.seeds_10.map((abbr, i) => [abbr, i + 1]));
  const confFinal = {
    team_a: conf.conf_semis[0].winner,
    team_b: conf.conf_semis[1].winner,
    winner: conf.conference_champion,
  };
  const cols =
    side === "west"
      ? { round1: 1, gutter1: 2, semis: 3, gutter2: 4, final: 5 }
      : { final: 1, gutter2: 2, semis: 3, gutter1: 4, round1: 5 };

  const children = [
    ...conf.round1.map((m, i) => bracketMatchBox(m, seedByAbbr, teamIds, cols.round1, `${i + 1} / span 1`)),
    bracketConnector(cols.gutter1, "1 / 3", side, "depth-1"),
    bracketConnector(cols.gutter1, "3 / 5", side, "depth-1"),
    ...conf.conf_semis.map((m, i) => bracketMatchBox(m, seedByAbbr, teamIds, cols.semis, i === 0 ? "1 / 3" : "3 / 5")),
    bracketConnector(cols.gutter2, "1 / 5", side, "depth-2"),
    bracketMatchBox(confFinal, seedByAbbr, teamIds, cols.final, "1 / 5"),
  ];
  return el("div", { class: `bracket-conf side-${side}` }, children);
}

function bracketRoundLabels(side) {
  const labels =
    side === "west"
      ? ["Ronda 1", "", "Semifinales", "", "Final de conf."]
      : ["Final de conf.", "", "Semifinales", "", "Ronda 1"];
  return el(
    "div",
    { class: `bracket-labels side-${side}` },
    labels.map((label) => el("div", { class: "bracket-label" }, label))
  );
}

function bracketPlayInGame(game, teamIds) {
  return el("div", { class: "bracket-playin-game" }, [
    bracketTeamRow(game.team_a, null, game.winner, teamIds),
    bracketTeamRow(game.team_b, null, game.winner, teamIds),
  ]);
}

function bracketPlayInStrip(name, conf, teamIds) {
  return el("div", {}, [
    el("p", { class: "caption", style: "margin-bottom: 6px; font-weight: 600;" }, name),
    el("div", { class: "bracket-playin-games" }, [
      bracketPlayInGame(conf.play_in.game_7_vs_8, teamIds),
      bracketPlayInGame(conf.play_in.game_9_vs_10, teamIds),
      bracketPlayInGame(conf.play_in.game_elimination, teamIds),
    ]),
  ]);
}

function bracketCenterChampion(result) {
  return el("div", { class: "bracket-final-center" }, [
    el("div", { class: "bracket-trophy" }, "🏆"),
    el("div", { class: "caption", style: "margin: 0;" }, "Campeón NBA"),
    el("div", { class: "team-identity", style: "justify-content: center;" }, [
      teamBadge(result.team_ids[result.nba_champion], result.nba_champion, 48),
      el("strong", { style: "font-size: 1.1rem;" }, result.nba_champion),
    ]),
  ]);
}

function renderBracketResult(result) {
  const west = el("div", {}, [bracketRoundLabels("west"), conferenceBracketGrid(result.west, "west", result.team_ids)]);
  const east = el("div", {}, [bracketRoundLabels("east"), conferenceBracketGrid(result.east, "east", result.team_ids)]);
  const layout = el("div", { class: "bracket-layout" }, [west, bracketCenterChampion(result), east]);

  return el("div", {}, [
    el("div", { class: "bracket-scroll" }, [layout]),
    el("h3", {}, "Play-in"),
    el("div", { class: "grid-2" }, [
      bracketPlayInStrip("Conferencia Este", result.east, result.team_ids),
      bracketPlayInStrip("Conferencia Oeste", result.west, result.team_ids),
    ]),
  ]);
}

function wireBracketButton() {
  const button = document.getElementById("bracket-btn");
  if (!button) return;
  button.addEventListener("click", async () => {
    button.disabled = true;
    button.textContent = "Simulando…";
    const resultContainer = document.getElementById("bracket-result");
    try {
      const result = await api.leagueBracket(getScenario());
      resultContainer.replaceChildren(renderBracketResult(result));
    } catch (err) {
      resultContainer.replaceChildren(emptyState(err.message));
    } finally {
      button.disabled = false;
      button.textContent = "🎲 Simular un bracket de playoffs";
    }
  });
}

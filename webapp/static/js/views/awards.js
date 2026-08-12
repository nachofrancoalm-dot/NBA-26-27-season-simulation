import { api } from "../api.js";
import { card, el, dataTable, glossaryExpander, emptyState } from "../ui.js";
import { openPlayerModal } from "../player-modal.js";
import { openTeamModal } from "../team-modal.js";
import { getScenario, scenarioBar } from "../scenario.js";

/** Doble clic en jugador Y en equipo para cualquier tabla de premios --
 * dataTable() ya soporta varias columnas con doble clic a la vez (ver
 * ui.js). Se omite la entrada de una columna que no exista en esa tabla
 * en particular (p.ej. COY no tiene player_name) sin que falle nada. */
function awardsInteractions(teamIds) {
  return {
    hiddenColumns: ["player_id"],
    doubleClick: [
      {
        column: "player_name",
        onOpen: (record) => openPlayerModal(record.player_id, teamIds[record.team_abbreviation]),
      },
      {
        column: "team_abbreviation",
        onOpen: (record) => openTeamModal(record.team_abbreviation, teamIds[record.team_abbreviation]),
      },
    ],
  };
}

function awardBlock(emoji, title, records, teamIds) {
  return el("div", {}, [
    el("h3", {}, `${emoji} ${title}`),
    records && records.length
      ? dataTable(records, {}, awardsInteractions(teamIds))
      : el("p", { class: "caption" }, "Sin candidatos."),
  ]);
}

export async function render(container) {
  container.replaceChildren();
  container.append(el("div", { class: "caption" }, "Cargando premios…"));

  const status = await api.status();
  const bar = scenarioBar(status, () => render(container));

  let data;
  try {
    data = await api.awards(getScenario());
  } catch (err) {
    container.replaceChildren(bar, card([el("h2", {}, "Premios individuales"), emptyState(err.message)]));
    return;
  }

  const teamIds = data.team_ids || {};
  const cards = [bar];

  const introChildren = [
    el("h2", {}, "Premios individuales"),
    el(
      "p",
      { class: "caption" },
      "Heurísticas sobre la proyección -- NO son una predicción de la votación real de los medios. Doble clic en un jugador o equipo para ver su detalle."
    ),
  ];
  if (data.scope === "own") {
    introChildren.push(el("p", { class: "caption" }, "Calculado solo sobre tu roster (sin datos de los 30 equipos)."));
  }
  cards.push(card(introChildren));

  cards.push(
    card([
      el("div", { class: "grid-2" }, [
        awardBlock("🏆", "MVP", data.mvp, teamIds),
        awardBlock("🛡️", "DPOY", data.dpoy, teamIds),
      ]),
      el("div", { class: "grid-2", style: "margin-top: 16px;" }, [
        awardBlock("🌟", "Rookie del Año", data.roy, teamIds),
        awardBlock("🔥", "Más Mejorado", data.mip, teamIds),
      ]),
      el("div", { class: "grid-2", style: "margin-top: 16px;" }, [
        awardBlock("🎖️", "6.º Hombre", data.sixth_man, teamIds),
        awardBlock("📋", "Entrenador del Año", data.coy, teamIds),
      ]),
      glossaryExpander(Object.entries(data.glossary), "Leyenda de columnas"),
    ])
  );

  cards.push(allStarCard(data, teamIds));
  cards.push(card([allTeamSection("🏀 Quintetos All-NBA", data.all_nba, teamIds)]));
  cards.push(card([allTeamSection("🛡️ Quintetos All-Defensive", data.all_defensive, teamIds)]));
  cards.push(card([glossaryExpander(Object.entries(data.season_awards_glossary), "Leyenda — premios de fin de temporada")]));

  container.replaceChildren(...cards);
}

function allStarCard(data, teamIds) {
  const records = data.all_star || [];
  const conferences = [...new Set(records.map((r) => r.conference).filter(Boolean))];

  const tables =
    conferences.length > 1
      ? el(
          "div",
          { class: "grid-2" },
          conferences.map((conf) =>
            el("div", {}, [
              el("h3", {}, conf),
              dataTable(records.filter((r) => r.conference === conf), {}, awardsInteractions(teamIds)),
            ])
          )
        )
      : records.length
      ? dataTable(records, {}, awardsInteractions(teamIds))
      : el("p", { class: "caption" }, "Sin candidatos.");

  const quota = data.all_star_nationality_quota;
  let banner = null;
  if (quota && quota.checked) {
    if (quota.meets_both) {
      banner = el(
        "div",
        { class: "banner success" },
        `✅ Cuota de nacionalidad cumplida (${quota.us_count} EE.UU. / ${quota.international_count} internacionales).`
      );
    } else if (data.commissioner_picks && data.commissioner_picks.length) {
      banner = el("div", { class: "banner warning" }, [
        `⚠️ Cuota no cumplida de forma natural -- se simuló añadir ${data.commissioner_picks.length} jugador(es) por decisión del comisionado (no por mérito).`,
        dataTable(data.commissioner_picks, {}, awardsInteractions(teamIds)),
      ]);
    } else {
      banner = el("div", { class: "banner warning" }, "⚠️ Cuota de nacionalidad no cumplida y sin candidato disponible para cubrirla.");
    }
  }

  return card([
    el("h2", {}, "⭐ All-Star"),
    el("p", { class: "caption" }, "Titular/Reserva es solo una etiqueta sobre el ranking -- no simula el voto real."),
    tables,
    banner,
  ]);
}

function allTeamSection(title, records, teamIds) {
  if (!records || !records.length) {
    return el("div", {}, [el("h2", {}, title), el("p", { class: "caption" }, "Sin candidatos.")]);
  }
  const teams = [...new Set(records.map((r) => r.team))];
  return el("div", {}, [
    el("h2", {}, title),
    ...teams.map((team) =>
      el("div", { style: "margin-bottom: 10px;" }, [
        el("h3", {}, team),
        dataTable(records.filter((r) => r.team === team), {}, awardsInteractions(teamIds)),
      ])
    ),
  ]);
}

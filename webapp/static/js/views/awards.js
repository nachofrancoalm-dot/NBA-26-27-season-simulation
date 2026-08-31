import { api } from "../api.js";
import { card, el, dataTable, glossaryExpander, emptyState, skeleton } from "../ui.js";
import { openPlayerModal } from "../player-modal.js";
import { openTeamModal } from "../team-modal.js";
import { getScenario, scenarioBar } from "../scenario.js";
import { getHypotheticalLeague, clearHypotheticalLeague, hypotheticalBanner } from "../hypothetical-league.js";
import { courtLineup } from "../court.js";
import { leaderboardChart } from "../leaderboard.js";

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

/** El backend NO redondea PPG/RPG/APG/etc en /api/awards (a diferencia
 * de dataTable(), que sí redondea a 2 decimales para mostrar) -- sin
 * esto, la vista previa mostraría "31.19570399944936" en vez de "31.2". */
function fmt1(value) {
  return typeof value === "number" ? value.toFixed(1) : value;
}

/** Stats rápidas de ofensiva para la vista previa (MVP/ROY/6.º Hombre)
 * -- SIEMPRE de player_df (la temporada proyectada, ver
 * awards_projection.OFFENSIVE_COMPARISON_STATS), nunca mezcladas con
 * temporadas reales. `record.team_record` se añade como stat aparte
 * (no es "del jugador", pero da contexto de si juega en un equipo
 * ganador -- relevante para MVP en particular). */
function offenseStats(record) {
  return [
    record.PPG != null ? { label: "PPG", value: fmt1(record.PPG) } : null,
    record.RPG != null ? { label: "RPG", value: fmt1(record.RPG) } : null,
    record.APG != null ? { label: "APG", value: fmt1(record.APG) } : null,
    record.team_record ? { label: "Récord equipo", value: record.team_record } : null,
  ].filter(Boolean);
}

/** Config de leaderboardChart() por premio -- qué columna manda el
 * largo de la barra (ver los `sort_values(...)` de
 * src/awards_projection.py, la misma columna que ya ordenaba la tabla)
 * y qué stats rápidas mostrar en la vista previa al pasar el ratón. */
const LEADERBOARD_CONFIG = {
  mvp: { valueKey: "mvp_score", statsFn: offenseStats },
  dpoy: {
    valueKey: "dpoy_score",
    statsFn: (r) =>
      [
        r.SPG != null ? { label: "SPG", value: fmt1(r.SPG) } : null,
        r.BPG != null ? { label: "BPG", value: fmt1(r.BPG) } : null,
        r.RPG != null ? { label: "RPG", value: fmt1(r.RPG) } : null,
        r.team_record ? { label: "Récord equipo", value: r.team_record } : null,
      ].filter(Boolean),
  },
  roy: { valueKey: "season_value", statsFn: offenseStats },
  sixth_man: { valueKey: "season_value", statsFn: offenseStats },
  // MIP NO usa la temporada proyectada -- compara el Game Score por-36
  // REAL de las dos últimas temporadas ya jugadas (ver el docstring de
  // compute_mip_candidates: "lo que un jugador YA mejoró", no una
  // proyección). Por eso no lleva `season` global -- se omite el pie
  // "Temporada proyectada X" en la vista previa (ver la llamada a
  // awardBlock más abajo) y en su lugar la propia temporada real
  // (r.latest_season) va como una stat más.
  mip: {
    valueKey: "improvement",
    valueFormat: (v) => (typeof v === "number" ? `+${v.toFixed(1)}` : "—"),
    statsFn: (r) => [
      { label: "GmSc/36 anterior", value: r.previous_game_score_per36?.toFixed(1) },
      { label: "GmSc/36 actual", value: r.latest_game_score_per36?.toFixed(1) },
      { label: "Temporada real", value: r.latest_season },
    ],
    noProjectedSeason: true,
  },
};

/** Ranking visual (foto + barra, ver leaderboard.js) para los premios
 * individuales con una columna de "valor" clara -- reemplaza la tabla
 * a petición del usuario ("más minimalista y visual"). COY sigue en
 * tabla: es un premio de EQUIPO (sin player_name/player_id), no encaja
 * en un ranking de jugadores. */
function awardBlock(emoji, title, records, teamIds, configKey, season) {
  const config = LEADERBOARD_CONFIG[configKey];
  const body =
    records && records.length
      ? config
        ? leaderboardChart(records, { ...config, teamIds, season: config.noProjectedSeason ? null : season })
        : dataTable(records, {}, awardsInteractions(teamIds))
      : el("p", { class: "caption" }, "Sin candidatos.");
  return el("div", {}, [el("h3", {}, `${emoji} ${title}`), body]);
}

export async function render(container) {
  // Este endpoint es el más lento de la app con diferencia (~4s,
  // compute_awards_summary corre pandas sin caché) -- de todas las vistas
  // es donde más importa un esqueleto con movimiento en vez de texto
  // estático, para que quede claro que sigue cargando y no que se quedó
  // colgado.
  container.replaceChildren(skeleton(["title", "short"]), skeleton(["", "", "", ""]));

  const status = await api.status();
  const hypothetical = getHypotheticalLeague();
  const bar = hypothetical
    ? hypotheticalBanner(() => {
        clearHypotheticalLeague();
        render(container);
      })
    : scenarioBar(status, () => render(container));

  let data;
  try {
    // Con un roster hipotético activo, los premios vienen del mismo
    // POST /api/sandbox/league que ya calculó standings/playoffs (ver
    // league.js) -- misma forma exacta que /api/awards, así que el resto
    // de esta función no necesita ramificar por fuente.
    data = hypothetical ? hypothetical.result.awards : await api.awards(getScenario());
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
      "Heurísticas sobre la proyección -- NO son una predicción de la votación real de los medios. " +
        "Clic en un jugador (o pasa el ratón para ver sus stats rápidas) para ver su detalle completo."
    ),
  ];
  if (data.scope === "own") {
    introChildren.push(el("p", { class: "caption" }, "Calculado solo sobre tu roster (sin datos de los 30 equipos)."));
  }
  cards.push(card(introChildren));

  const season = status.team.season;

  cards.push(
    card([
      el("div", { class: "grid-2" }, [
        awardBlock("🏆", "MVP", data.mvp, teamIds, "mvp", season),
        awardBlock("🛡️", "DPOY", data.dpoy, teamIds, "dpoy", season),
      ]),
      el("div", { class: "grid-2", style: "margin-top: 16px;" }, [
        awardBlock("🌟", "Rookie del Año", data.roy, teamIds, "roy", season),
        awardBlock("🔥", "Más Mejorado", data.mip, teamIds, "mip", season),
      ]),
      el("div", { class: "grid-2", style: "margin-top: 16px;" }, [
        awardBlock("🎖️", "6.º Hombre", data.sixth_man, teamIds, "sixth_man", season),
        awardBlock("📋", "Entrenador del Año", data.coy, teamIds),
      ]),
      glossaryExpander(Object.entries(data.glossary), "Leyenda de columnas"),
    ])
  );

  cards.push(allStarCard(data, teamIds));
  cards.push(card([allTeamSection("🏀 Quintetos All-NBA", data.all_nba, teamIds, season)]));
  cards.push(card([allTeamSection("🛡️ Quintetos All-Defensive", data.all_defensive, teamIds, season)]));
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

function allTeamSection(title, records, teamIds, season) {
  if (!records || !records.length) {
    return el("div", {}, [el("h2", {}, title), el("p", { class: "caption" }, "Sin candidatos.")]);
  }
  const teams = [...new Set(records.map((r) => r.team))];
  return el("div", {}, [
    el("h2", {}, title),
    el(
      "p",
      { class: "caption", style: "margin: 0 0 10px;" },
      "El quinteto es 2 bases/escoltas + 2 aleros/ala-pívots + 1 pívot (formato clásico 2-2-1) -- " +
        "la posición sobre la cancha es solo ilustrativa dentro de cada grupo (G/F/C), no una asignación " +
        "real de base vs. escolta o alero vs. ala-pívot. Clic en un jugador (o pasa el ratón) para ver su detalle."
    ),
    el(
      "div",
      { class: "court-lineup-grid" },
      teams.map((team) => {
        const teamRecords = records.filter((r) => r.team === team);
        return el("div", {}, [
          el("h3", { style: "margin: 0 0 8px; text-align: center;" }, team),
          courtLineup(teamRecords, { title: team, teamIds, season }),
        ]);
      })
    ),
  ]);
}

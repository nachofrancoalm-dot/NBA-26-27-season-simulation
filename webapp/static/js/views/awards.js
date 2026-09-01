import { api } from "../api.js";
import { card, el, dataTable, glossaryExpander, emptyState, skeleton } from "../ui.js";
import { openPlayerModal } from "../player-modal.js";
import { openTeamModal } from "../team-modal.js";
import { getScenario, scenarioBar } from "../scenario.js";
import { getHypotheticalLeague, clearHypotheticalLeague, hypotheticalBanner } from "../hypothetical-league.js";
import { courtLineup } from "../court.js";
import { leaderboardChart, teamLeaderboardChart } from "../leaderboard.js";

/** Doble clic en jugador Y en equipo para cualquier tabla de premios --
 * dataTable() soporta varias columnas con doble clic a la vez. */
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

/** El backend no redondea PPG/RPG/APG/etc en /api/awards (a diferencia
 * de dataTable()) -- sin esto se vería "31.19570399944936" en vez de "31.2". */
function fmt1(value) {
  return typeof value === "number" ? value.toFixed(1) : value;
}

/** "anterior → actual" ya formateado, con "—" para el lado que falte --
 * usado solo por MIP (ver `fullStats` con `prevPrefix`). */
function comparisonValue(prev, current) {
  return `${prev != null ? fmt1(prev) : "—"} → ${current != null ? fmt1(current) : "—"}`;
}

/** Set de stats unificado para todos los premios y quintetos: PPG, RPG,
 * APG, SPG, BPG, FG%, 3P%, récord de equipo y el "valor" que ordena ESE
 * premio (distinto nombre/fórmula según el premio, por eso `valueKey`/
 * `valueLabel` se pasan aparte). Sin `prevPrefix`: stats de la temporada
 * proyectada. Con `prevPrefix` ("prev_", solo MIP): comparación "real
 * anterior → proyectada" (ver comparisonValue).
 */
function fullStats(record, { valueKey, valueLabel, prevPrefix } = {}) {
  const stat = (label, key) => {
    const current = record[key];
    const prev = prevPrefix ? record[`${prevPrefix}${key}`] : null;
    if (current == null && prev == null) return null;
    return { label, value: prevPrefix ? comparisonValue(prev, current) : fmt1(current) };
  };

  return [
    stat("PPG", "PPG"),
    stat("RPG", "RPG"),
    stat("APG", "APG"),
    stat("SPG", "SPG"),
    stat("BPG", "BPG"),
    stat("FG%", "FG%"),
    stat("3P%", "3P%"),
    record.team_record ? { label: "Récord equipo", value: record.team_record } : null,
    valueKey && record[valueKey] != null ? { label: valueLabel, value: fmt1(record[valueKey]) } : null,
  ].filter(Boolean);
}

/** Config de leaderboardChart() por premio -- columna que manda el largo
 * de la barra, stats de la vista previa, y pie de foto (`captionFn`).
 * Función de `season` porque MIP necesita un pie distinto por fila
 * (cada jugador tiene su propia `prev_season` real). */
function buildLeaderboardConfig(season) {
  const projectedCaption = () => `Temporada proyectada ${season}`;
  return {
    mvp: { valueKey: "mvp_score", statsFn: (r) => fullStats(r, { valueKey: "mvp_score", valueLabel: "Valor MVP" }), captionFn: projectedCaption },
    dpoy: { valueKey: "dpoy_score", statsFn: (r) => fullStats(r, { valueKey: "dpoy_score", valueLabel: "Valor DPOY" }), captionFn: projectedCaption },
    roy: { valueKey: "season_value", statsFn: (r) => fullStats(r, { valueKey: "season_value", valueLabel: "Valor temporada" }), captionFn: projectedCaption },
    sixth_man: {
      valueKey: "season_value",
      statsFn: (r) => fullStats(r, { valueKey: "season_value", valueLabel: "Valor temporada" }),
      captionFn: projectedCaption,
    },
    // MIP compara la proyectada contra la ÚLTIMA real jugada (prev_*) --
    // distinto del ranking en sí (`improvement`), que usa la penúltima real.
    mip: {
      valueKey: "improvement",
      valueFormat: (v) => (typeof v === "number" ? `+${v.toFixed(1)}` : "—"),
      statsFn: (r) => fullStats(r, { prevPrefix: "prev_" }),
      captionFn: (r) => `Real ${r.prev_season || "?"} → Proyectada ${season}`,
    },
    all_star: {
      valueKey: "season_value",
      statsFn: (r) => fullStats(r, { valueKey: "season_value", valueLabel: "Valor All-Star" }),
      captionFn: projectedCaption,
    },
  };
}

/** COY es un premio de EQUIPO (este proyecto no modela entrenadores) --
 * barra por win_improvement (victorias proyectadas menos reales del año anterior). */
function coyConfig() {
  return {
    valueKey: "win_improvement",
    valueFormat: (v) => (typeof v === "number" ? `+${v.toFixed(1)}` : "—"),
    statsFn: (r) => [
      Number.isFinite(r.prior_wins) ? { label: "Victorias año anterior (real)", value: fmt1(r.prior_wins) } : null,
      Number.isFinite(r.wins_mean) ? { label: "Victorias proyectadas", value: fmt1(r.wins_mean) } : null,
      Number.isFinite(r.win_improvement) ? { label: "Mejora", value: `+${fmt1(r.win_improvement)}` } : null,
    ].filter(Boolean),
  };
}

/** Ranking visual (foto + barra) para premios con columna de "valor"
 * clara. `chartFn`: leaderboardChart (jugadores) o teamLeaderboardChart (COY). */
function awardBlock(emoji, title, records, teamIds, config, chartFn = leaderboardChart) {
  const body =
    records && records.length
      ? config
        ? chartFn(records, { ...config, teamIds })
        : dataTable(records, {}, awardsInteractions(teamIds))
      : el("p", { class: "caption" }, "Sin candidatos.");
  return el("div", {}, [el("h3", {}, `${emoji} ${title}`), body]);
}

export async function render(container) {
  // Endpoint más lento de la app (~4s, compute_awards_summary sin caché)
  // -- el esqueleto con movimiento deja claro que sigue cargando.
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
    // Con roster hipotético activo, los premios vienen del mismo
    // POST /api/sandbox/league (misma forma que /api/awards).
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
  const leaderboardConfig = buildLeaderboardConfig(season);

  cards.push(
    card([
      el("div", { class: "grid-2" }, [
        awardBlock("🏆", "MVP", data.mvp, teamIds, leaderboardConfig.mvp),
        awardBlock("🛡️", "DPOY", data.dpoy, teamIds, leaderboardConfig.dpoy),
      ]),
      el("div", { class: "grid-2", style: "margin-top: 16px;" }, [
        awardBlock("🌟", "Rookie del Año", data.roy, teamIds, leaderboardConfig.roy),
        awardBlock("🔥", "Más Mejorado", data.mip, teamIds, leaderboardConfig.mip),
      ]),
      el("div", { class: "grid-2", style: "margin-top: 16px;" }, [
        awardBlock("🎖️", "6.º Hombre", data.sixth_man, teamIds, leaderboardConfig.sixth_man),
        awardBlock("📋", "Entrenador del Año", data.coy, teamIds, coyConfig(), teamLeaderboardChart),
      ]),
      glossaryExpander(Object.entries(data.glossary), "Leyenda de columnas"),
    ])
  );

  cards.push(allStarCard(data, teamIds, leaderboardConfig.all_star));
  cards.push(card([allTeamSection("🏀 Quintetos All-NBA", data.all_nba, teamIds, season)]));
  cards.push(card([allTeamSection("🛡️ Quintetos All-Defensive", data.all_defensive, teamIds, season)]));
  cards.push(card([glossaryExpander(Object.entries(data.season_awards_glossary), "Leyenda — premios de fin de temporada")]));

  container.replaceChildren(...cards);
}

function allStarCard(data, teamIds, config) {
  const records = data.all_star || [];
  const conferences = [...new Set(records.map((r) => r.conference).filter(Boolean))];

  const leaderboards =
    conferences.length > 1
      ? el(
          "div",
          { class: "grid-2" },
          conferences.map((conf) =>
            el("div", {}, [
              el("h3", {}, conf),
              leaderboardChart(
                records.filter((r) => r.conference === conf),
                { ...config, teamIds }
              ),
            ])
          )
        )
      : records.length
      ? leaderboardChart(records, { ...config, teamIds })
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
    el("p", { class: "caption" }, "Titular/Reserva es solo una etiqueta sobre el ranking -- no simula el voto real. Clic en un jugador (o pasa el ratón) para ver su detalle."),
    leaderboards,
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

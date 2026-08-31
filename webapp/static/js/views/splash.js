// views/splash.js -- pantalla de entrada de marca, ANTERIOR a la app (no
// una pestaña más al lado de "Mi equipo"/"Liga NBA" -- se probó así
// primero y no tenía sentido: una landing page debe ser la puerta de
// entrada, no un contenido que compite al mismo nivel que las secciones
// reales). Vive en el overlay fijo `#splash` de index.html, por encima de
// `.app-shell`; `app.js` la muestra al arrancar y la descarta (con la
// misma transición de "jugador cruzando pantalla" que ya usa la
// navegación entre pestañas) en cuanto el usuario entra por cualquier CTA.
//
// Reutiliza /api/status, /api/team/simulation y /api/league/playoffs --
// los mismos endpoints que ya consumen Roster/Simulación/Liga, así que
// no hay lógica de datos nueva, solo composición visual. Se omite
// /api/awards a propósito (el más lento de la app, ~4s) -- esta pantalla
// debe sentirse instantánea. La foto del hero (staticHeroImage(), ver
// player-hero.js) es un archivo local elegido a mano por el usuario
// (webapp/static/img/embiid.png), no datos del roster -- ya NO hace
// falta pedir /api/roster aquí solo para eso.
//
// rosterBuilderCard() (roster-builder.js) convierte el punto de partida de
// "los 76ers fijos del config" en un roster hipotético editable -- arranca
// igual (mismos 13 player_id), pero cualquier jugador se puede sustituir
// por cualquier otro de los 30 equipos reales y volver a simular en vivo
// (src/sandbox_simulation.py). Es su propia tarjeta autocontenida, no un
// dato más de este módulo -- ver el comentario junto a `kpis` más abajo
// sobre por qué no comparte tira de resultados con el hero.

import { api } from "../api.js";
import { el, skeleton } from "../ui.js";
import { staticHeroImage } from "../player-hero.js";
import { rosterBuilderCard } from "../roster-builder.js";

const QUICK_LINKS = [
  {
    emoji: "🏀",
    title: "Roster y proyecciones",
    desc: "Cada jugador del roster: minutos, riesgo de lesión, desgaste y Game Score proyectado.",
    top: "mi-equipo",
    sub: "roster",
  },
  {
    emoji: "🎲",
    title: "Simulación Monte Carlo",
    desc: "10.000 temporadas simuladas: distribución de victorias, Net Rating y escenario sin lesiones.",
    top: "mi-equipo",
    sub: "simulacion",
  },
  {
    emoji: "🏆",
    title: "Liga y Playoffs",
    desc: "Clasificación de los 30 equipos, probabilidades de título y bracket de playoffs.",
    top: "liga-nba",
    sub: "liga",
  },
  {
    emoji: "🎖️",
    title: "Premios individuales",
    desc: "MVP, DPOY, All-Star y quintetos de fin de temporada -- heurísticas sobre la proyección.",
    top: "liga-nba",
    sub: "premios",
  },
  {
    emoji: "📜",
    title: "Backtesting",
    desc: "El motor validado contra 450 casos históricos reales (30 equipos x 16 temporadas).",
    top: "mi-equipo",
    sub: "backtesting",
  },
  {
    emoji: "🤖",
    title: "Explicador (IA)",
    desc: "Pregunta en lenguaje natural sobre cualquier dato ya calculado en las otras pestañas.",
    top: "explicador",
    sub: null,
  },
];

function fmt1(value) {
  return typeof value === "number" ? value.toFixed(1) : "—";
}

function fmtPct(value) {
  return typeof value === "number" ? `${value.toFixed(1)}%` : "—";
}

function kpi(label, value, sub = null, accent = false) {
  return el("div", { class: "hero-kpi" }, [
    el("p", { class: "label" }, label),
    el("p", { class: `value ${accent ? "accent" : ""}`.trim() }, value),
    sub ? el("p", { class: "sub" }, sub) : null,
  ]);
}

function quickLinkCard(link, enter) {
  return el(
    "button",
    { type: "button", class: "quick-link", onclick: () => enter(link.top, link.sub) },
    [
      el("span", { class: "quick-link-title" }, [link.emoji, " ", link.title]),
      el("span", { class: "quick-link-desc" }, link.desc),
    ]
  );
}

/** `enter(topKey, subKey)`: entra a la app en esa sección, con la
 * transición de navegación normal -- lo inyecta app.js (bootstrap). */
export async function render(container, enter) {
  container.replaceChildren(skeleton(["title", "short"]));

  const [statusResult, simResult, playoffsResult] = await Promise.allSettled([
    api.status(),
    api.simulation(),
    api.leaguePlayoffs(),
  ]);

  if (statusResult.status !== "fulfilled") {
    container.replaceChildren(el("div", { class: "empty-state" }, "No se pudo cargar el estado de la aplicación."));
    return;
  }

  const status = statusResult.value;
  const sim = simResult.status === "fulfilled" ? simResult.value : null;
  const myTeam = playoffsResult.status === "fulfilled" ? playoffsResult.value.my_team : null;

  const hero = el("section", { class: "hero" }, [
    el("div", { class: "hero-copy" }, [
      el("p", { class: "hero-eyebrow" }, `${status.team.season} · Simulador Monte Carlo`),
      el("h1", {}, status.team.name),
      el(
        "p",
        {},
        "Proyección estadística del roster completo, validada por backtesting contra 450 " +
          "temporadas reales de la NBA. Explora el roster, la simulación de temporada, la liga " +
          "completa de 30 equipos y los premios individuales."
      ),
      el("div", { class: "hero-cta-row" }, [
        el("button", { class: "btn", onclick: () => enter("mi-equipo", "roster") }, "Entrar al simulador"),
        el("button", { class: "btn-ghost", onclick: () => enter("liga-nba", "liga") }, "Explorar la liga"),
      ]),
    ]),
    staticHeroImage("/img/embiid.png", "Joel Embiid"),
  ]);

  // Estos KPIs siguen leyendo el resultado YA CALCULADO
  // (simulation_results.csv, 10.000 temporadas, lectura de CSV casi
  // gratis) del roster curado real -- a propósito NO se sustituyen por
  // una tirada en vivo del sandbox al cargar la pantalla, ni se
  // comparten con la tira de resultados de rosterBuilderCard() más
  // abajo. Dos motivos: (1) esta pantalla debe sentirse instantánea, y
  // una simulación en vivo (aunque reducida a 2.000 temporadas) tiene
  // latencia real; (2) el sandbox no modela sinergia de alineación (ver
  // sandbox_simulation.py), así que aunque el roster editado empiece
  // siendo idéntico al curado, correrlo por el motor en vivo daría un
  // número ligeramente distinto al oficial -- mezclarlos en la misma
  // tira sería confuso. rosterBuilderCard tiene su propia tira de
  // resultados, etiquetada como "roster editado", justo para no
  // confundir las dos fuentes.
  const kpis = el("div", { class: "hero-kpis" }, [
    sim
      ? kpi("Victorias medias", fmt1(sim.summary.mean), `P10 ${sim.summary.p10} · P90 ${sim.summary.p90}`, true)
      : kpi("Victorias medias", "—", "Corre la simulación en Mi equipo"),
    kpi(
      "Mediana de victorias",
      sim ? String(sim.summary.p50) : "—",
      sim ? `sobre ${sim.n_seasons.toLocaleString("es")} temporadas` : null
    ),
    myTeam
      ? kpi("Probabilidad de playoffs", fmtPct(myTeam.playoff_pct), null, true)
      : kpi("Probabilidad de playoffs", "—", "Requiere datos de liga (30 equipos)"),
    myTeam ? kpi("Probabilidad de título", fmtPct(myTeam.championship_pct)) : kpi("Probabilidad de título", "—"),
  ]);

  const linksHeading = el("h2", { style: "margin: 4px 0 2px;" }, "Explorar");
  const linksCaption = el("p", { class: "caption" }, "Cada sección tiene su propio detalle, gráficos y glosario.");
  const links = el("div", { class: "quick-links" }, QUICK_LINKS.map((link) => quickLinkCard(link, enter)));

  container.replaceChildren(hero, kpis, rosterBuilderCard(enter), linksHeading, linksCaption, links);
}

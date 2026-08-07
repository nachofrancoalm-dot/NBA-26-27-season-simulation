"""
app.py

Dashboard interactivo (Streamlit) para explorar los resultados ya
generados por el pipeline, sin tener que correr comandos Python sueltos.
Lee únicamente CSV de data/processed/ -- no llama a la API ni corre
simulaciones al vuelo (eso sigue siendo trabajo de data_pipeline.py /
simulation.py / league_simulation.py / backtesting.py, ejecutados por
separado). Toda la lógica de carga/combinación de datos vive en
data_loader.py, testeable; este archivo solo renderiza.

Navegación: un grupo de pestañas de primer nivel ("Resumen", "Mi equipo",
"Liga NBA", "Explicador (IA)"), con sub-pestañas anidadas dentro de "Mi
equipo" y "Liga NBA" -- agrupa temas relacionados en vez de una fila
plana de 7 pestañas. Todos los datasets se cargan una sola vez al
principio del script (Streamlit re-ejecuta todo el archivo en cada
interacción) y se reutilizan tanto en la barra lateral (estado de datos
disponibles) como dentro de cada pestaña.

Uso:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config_loader import load_config  # noqa: E402
from llm_explainer import build_context_snapshot, explain_question  # noqa: E402
from data_loader import (  # noqa: E402
    AWARDS_GLOSSARY,
    BACKTEST_GLOSSARY,
    CALIBRATION_GLOSSARY,
    CHAMPION_GLOSSARY,
    LEAGUE_GLOSSARY,
    LEAGUE_PLAYER_META_COLUMNS,
    ROSTER_STAT_GLOSSARY,
    SEASON_AWARDS_GLOSSARY,
    SIMULATION_GLOSSARY,
    STANDINGS_SITUATION_GLOSSARY,
    SYNERGY_GLOSSARY,
    compute_awards_summary,
    compute_champion_seed_distribution,
    compute_conference_standings,
    compute_win_distribution_summary,
    load_backtest_summary,
    load_backtest_sweep_calibration,
    load_backtest_sweep_summary,
    load_champion_roster_profiles,
    load_champion_seed_trajectories,
    load_champion_title_paths,
    load_league_playoff_summary,
    load_league_player_projections,
    load_league_regular_season_summary,
    load_lineup_synergy_pairs,
    load_roster_overview,
    run_single_bracket_simulation,
    run_single_season_player_log_simulation,
    load_simulation_results,
    select_roster_view,
)


# ---------------------------------------------------------------------
# Helpers de renderizado compartidos entre pestañas
# ---------------------------------------------------------------------

def render_glossary_expander(df_columns, glossary, title="Leyenda de estadísticas"):
    """Tooltips de columna (column_config) + un expander de texto -- reutilizado en varias pestañas."""
    column_config = {
        col: st.column_config.NumberColumn(help=glossary[col]) for col in df_columns if col in glossary
    }
    with st.expander(title):
        for col in df_columns:
            if col in glossary:
                st.markdown(f"**{col}** — {glossary[col]}")
    return column_config


def _series_line(game: dict) -> str:
    winner = game["winner"]
    loser = game["team_b"] if winner == game["team_a"] else game["team_a"]
    return f"**{winner}** venció a {loser}"


def render_conference_bracket(name: str, conf: dict):
    st.markdown(f"#### {name}")
    st.caption("Seeds 1-10 (por victorias medias): " + " → ".join(conf["seeds_10"]))

    with st.container(border=True):
        st.markdown("**Play-in**")
        st.write(_series_line(conf["play_in"]["game_7_vs_8"]) + " (partido único, seed 7)")
        st.write(_series_line(conf["play_in"]["game_9_vs_10"]) + " (partido único)")
        st.write(_series_line(conf["play_in"]["game_elimination"]) + " (partido único, seed 8)")

    with st.container(border=True):
        st.markdown("**Ronda 1** (mejor-de-7)")
        for game in conf["round1"]:
            st.write(_series_line(game))

    with st.container(border=True):
        st.markdown("**Semifinales de conferencia** (mejor-de-7)")
        for game in conf["conf_semis"]:
            st.write(_series_line(game))

    st.success(f"🏆 Campeón de conferencia: **{conf['conference_champion']}**")


def _stat_mode() -> str:
    """Lee el toggle global Totales/Por-partido de la barra lateral."""
    return "per_game" if st.session_state.get("global_stat_mode", "Por partido") == "Por partido" else "totals"


def _award_section(col, emoji: str, title: str, description: str, df):
    with col:
        st.markdown(f"#### {emoji} {title}")
        st.caption(description)
        if df is None:
            st.caption("No disponible (faltan datos de temporada anterior de los 30 equipos).")
        elif df.empty:
            st.caption("No hay candidatos que cumplan el umbral mínimo de minutos.")
        else:
            st.dataframe(df, width="stretch", hide_index=True)


# ---------------------------------------------------------------------
# Configuración de página y carga de datos (una sola vez por rerun)
# ---------------------------------------------------------------------

st.set_page_config(page_title="NBA Superteam Simulator", page_icon="🏀", layout="wide")

config = load_config()
my_team_abbrev = config["team"].get("abbreviation")

roster_overview = load_roster_overview(config)
simulation_results = load_simulation_results(config)
synergy_pairs = load_lineup_synergy_pairs(config)
backtest_summary = load_backtest_summary(config)
backtest_sweep_summary = load_backtest_sweep_summary(config)
backtest_sweep_calibration = load_backtest_sweep_calibration(config)
champion_title_paths = load_champion_title_paths(config)
champion_roster_profiles = load_champion_roster_profiles(config)
champion_seed_trajectories = load_champion_seed_trajectories(config)
regular_season_summary = load_league_regular_season_summary(config)
playoff_summary = load_league_playoff_summary(config)
player_projections = load_league_player_projections(config)
awards = compute_awards_summary(config)
groq_api_key = os.environ.get("GROQ_API_KEY")

DATASET_STATUS = {
    "Roster propio": roster_overview is not None,
    "Simulación Monte Carlo": simulation_results is not None,
    "Sinergia de alineación": synergy_pairs is not None,
    "Backtesting": backtest_summary is not None,
    "Liga (30 equipos)": regular_season_summary is not None,
    "Backtest sweep (450 casos)": backtest_sweep_summary is not None,
    "Campeones reales": champion_title_paths is not None,
    "Explicador (Groq)": bool(groq_api_key),
}

# ---------------------------------------------------------------------
# Barra lateral: identidad del equipo, estado de datos, controles globales
# ---------------------------------------------------------------------

with st.sidebar:
    st.title("🏀 NBA Superteam Sim")
    st.caption(f"{config['team']['name']} — {config['team']['season']}")

    st.radio(
        "Vista de estadísticas (global)",
        ["Por partido", "Totales de temporada"],
        horizontal=False,
        key="global_stat_mode",
        help="Aplica a las tablas de jugadores de 'Mi equipo' y 'Liga NBA'.",
    )

    st.markdown("---")
    st.caption("Datos disponibles")
    for label, ok in DATASET_STATUS.items():
        st.markdown(f"{'✅' if ok else '⬜'} {label}")
    if not DATASET_STATUS["Liga (30 equipos)"]:
        st.caption("Liga NBA y Premios se limitan a tu roster sin `data_pipeline.py --league`.")
    if not DATASET_STATUS["Explicador (Groq)"]:
        st.caption("Copia `.env.example` a `.env` con tu `GROQ_API_KEY` para el Explicador (IA).")

st.title(f"{config['team']['name']} — {config['team']['season']}")
st.caption(
    "Simulador Monte Carlo de rendimiento de un equipo NBA, validado por backtesting "
    "contra casos históricos de superequipos."
)

tab_home, tab_team, tab_league_group, tab_explainer = st.tabs(
    ["🏠 Resumen", "🏀 Mi equipo", "🏆 Liga NBA", "🤖 Explicador (IA)"]
)

# ---------------------------------------------------------------------
# 🏠 Resumen -- landing page con los KPI más relevantes
# ---------------------------------------------------------------------

with tab_home:
    st.subheader("Resumen ejecutivo")

    if simulation_results is None and playoff_summary is None:
        st.warning(
            "Todavía no hay ninguna simulación corrida. Empieza por "
            "`python src/data_pipeline.py` y luego genera al menos "
            "`simulation_results.csv` (ver pestaña **Mi equipo → Simulación**) "
            "para ver KPIs aquí."
        )
    else:
        if simulation_results is not None:
            summary = compute_win_distribution_summary(simulation_results)
            with st.container(border=True):
                st.markdown("**Temporada propia simulada**")
                cols = st.columns(4)
                cols[0].metric("Victorias medias", f"{summary['mean']:.1f}")
                cols[1].metric("P10", f"{summary['p10']:.0f}")
                cols[2].metric("Mediana", f"{summary['p50']:.0f}")
                cols[3].metric("P90", f"{summary['p90']:.0f}")

        if playoff_summary is not None and my_team_abbrev:
            my_team_row = playoff_summary[playoff_summary["team_abbreviation"] == my_team_abbrev]
            if not my_team_row.empty:
                row = my_team_row.iloc[0]
                with st.container(border=True):
                    st.markdown(f"**{my_team_abbrev} frente a los 30 equipos de la NBA**")
                    cols = st.columns(4)
                    cols[0].metric("Playoffs", f"{row['playoff_pct']:.1f}%")
                    cols[1].metric("Semis de conferencia", f"{row['conf_semis_pct']:.1f}%")
                    cols[2].metric("Finales de conferencia", f"{row['finals_pct']:.1f}%")
                    cols[3].metric("Campeonato", f"{row['championship_pct']:.1f}%")

        if awards is not None and not awards["mvp"].empty:
            with st.container(border=True):
                st.markdown("**🏆 Candidato a MVP (heurístico)**")
                top_mvp = awards["mvp"].iloc[0]
                st.write(
                    f"**{top_mvp['player_name']}**"
                    + (f" ({top_mvp['team_abbreviation']})" if "team_abbreviation" in top_mvp else "")
                    + " -- ver el detalle y el resto de premios en **Liga NBA → Premios individuales**."
                )

        st.markdown("---")
        st.markdown(
            "**Guía rápida:** *Mi equipo* tiene el roster, la simulación, la sinergia de "
            "alineación y el backtesting de tu equipo. *Liga NBA* compara contra los otros "
            "29 equipos (si corriste `--league`) y calcula premios individuales heurísticos. "
            "*Explicador (IA)* responde preguntas en lenguaje natural sobre todos estos datos."
        )

# ---------------------------------------------------------------------
# 🏀 Mi equipo -- Roster, Simulación, Sinergia, Backtesting
# ---------------------------------------------------------------------

with tab_team:
    sub_roster, sub_simulation, sub_synergy, sub_backtest = st.tabs(
        ["Roster y proyecciones", "Simulación Monte Carlo", "Sinergia de alineación", "Backtesting"]
    )

    with sub_roster:
        st.subheader("Roster: proyección, riesgo y desgaste por jugador")
        if roster_overview is None:
            st.warning(
                "No se encontró `aging_curve_projection.csv`. Corre primero "
                "`python src/data_pipeline.py` y luego "
                "`python -c \"from src.aging_curve import build_aging_projection_dataset; "
                "from src.config_loader import load_config; "
                "build_aging_projection_dataset(load_config())\"`."
            )
        else:
            roster_view = select_roster_view(
                roster_overview, mode=_stat_mode(), games_per_season=config["simulation"]["games_per_season"]
            )
            column_config = render_glossary_expander(roster_view.columns, ROSTER_STAT_GLOSSARY)
            st.dataframe(roster_view, width="stretch", hide_index=True, column_config=column_config)

    with sub_simulation:
        st.subheader("Distribución de temporadas simuladas")
        if simulation_results is None:
            st.warning(
                "No se encontró `simulation_results.csv`. Corre "
                "`python -c \"from src.simulation import build_simulation_dataset; "
                "from src.config_loader import load_config; "
                "build_simulation_dataset(load_config())\"`."
            )
        else:
            summary = compute_win_distribution_summary(simulation_results)
            with st.container(border=True):
                cols = st.columns(4)
                cols[0].metric("Victorias medias", f"{summary['mean']:.1f}")
                cols[1].metric("P10", f"{summary['p10']:.0f}")
                cols[2].metric("Mediana", f"{summary['p50']:.0f}")
                cols[3].metric("P90", f"{summary['p90']:.0f}")

            st.bar_chart(simulation_results["wins"].value_counts().sort_index())
            st.caption(f"{len(simulation_results):,} temporadas simuladas.".replace(",", " "))

            st.subheader("Net Rating estimado por temporada simulada")
            st.line_chart(simulation_results["net_rating_estimate_mean"].sort_values().reset_index(drop=True))

            render_glossary_expander(simulation_results.columns, SIMULATION_GLOSSARY)

            st.subheader("Partidos jugados por jugador en una temporada simulada")
            st.caption(
                "Una realización concreta (no la distribución agregada de arriba): partidos "
                "jugados/perdidos de cada jugador en UNA temporada simulada, y el detalle de cada "
                "ausencia con una categoría **ilustrativa** según su duración. ⚠️ No es un "
                "diagnóstico real -- nba_api no expone qué lesión sufrió cada jugador, la categoría "
                "es una etiqueta sintética derivada solo de cuántos partidos seguidos faltó (ver "
                "`simulation.DEFAULT_INJURY_TYPE_CATEGORIES`). Cada pulsación sortea una temporada nueva."
            )
            if st.button("🩹 Simular partidos de la temporada"):
                with st.spinner("Simulando temporada..."):
                    st.session_state["player_season_log"] = run_single_season_player_log_simulation(config)

            player_season_log = st.session_state.get("player_season_log")
            if player_season_log is not None:
                games_per_season = config["simulation"]["games_per_season"]
                log_view = player_season_log[["player_name", "games_played", "games_missed"]].copy()
                log_view = log_view.sort_values("games_missed", ascending=False).reset_index(drop=True)
                st.dataframe(
                    log_view,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "player_name": st.column_config.TextColumn("Jugador"),
                        "games_played": st.column_config.NumberColumn(
                            "Partidos jugados", help=f"De {games_per_season} en la temporada."
                        ),
                        "games_missed": st.column_config.NumberColumn("Partidos perdidos"),
                    },
                )

                injured_players = player_season_log[player_season_log["injury_events"].apply(len) > 0]
                if injured_players.empty:
                    st.caption("Nadie tuvo ausencias por lesión en esta temporada simulada.")
                else:
                    st.markdown("**Detalle de ausencias por jugador**")
                    for _, player_row in injured_players.iterrows():
                        events = player_row["injury_events"]
                        lines = "\n".join(
                            f"- Partido {event['start_game']}: {event['length']} partido(s) — {event['category']}"
                            for event in events
                        )
                        st.markdown(f"**{player_row['player_name']}**\n{lines}")

    with sub_synergy:
        st.subheader("Sinergia de alineación por pareja de jugadores")
        if synergy_pairs is None:
            st.warning(
                "No se encontró `lineup_synergy_pairs.csv`. Corre "
                "`python -c \"from src.lineup_synergy import build_lineup_synergy_dataset; "
                "from src.config_loader import load_config; "
                "build_lineup_synergy_dataset(load_config())\"`."
            )
        else:
            column_config = render_glossary_expander(synergy_pairs.columns, SYNERGY_GLOSSARY)
            st.dataframe(synergy_pairs, width="stretch", hide_index=True, column_config=column_config)

    with sub_backtest:
        st.subheader("Backtesting contra comparables históricos")
        if backtest_summary is None:
            st.warning(
                "No se encontró `backtest_summary.csv`. Corre "
                "`python -c \"from src.backtesting import build_backtest_dataset; "
                "from src.config_loader import load_config; "
                "build_backtest_dataset(load_config())\"`."
            )
        else:
            column_config = render_glossary_expander(backtest_summary.columns, BACKTEST_GLOSSARY)
            st.dataframe(backtest_summary, width="stretch", hide_index=True, column_config=column_config)

            low_percentile_cases = backtest_summary[
                (backtest_summary["actual_percentile"] < 5) | (backtest_summary["actual_percentile"] > 95)
            ]
            if not low_percentile_cases.empty:
                st.info(
                    f"{len(low_percentile_cases)} de {len(backtest_summary)} casos caen en un "
                    "percentil extremo (<5% o >95%) -- el modelo sobreestima o subestima "
                    "estos casos de forma significativa. Ver la sección de Backtesting en el "
                    "README para la discusión de por qué (fricción de vestuario no capturada "
                    "por datos de caja de estadísticas)."
                )

        st.markdown("---")
        st.subheader("Backtesting sistemático (30 equipos × varias temporadas)")
        st.caption(
            "Los 4 casos de arriba son superequipos elegidos a mano -- útiles para el caso "
            "narrativo que motivó este proyecto, pero no dicen si el modelo funciona bien EN "
            "GENERAL. Esta sección corre el mismo backtest sobre los 30 equipos NBA en varias "
            "temporadas reales (ver `config['backtest_sweep']`) y resume qué tan bien "
            "calibrado está: si el modelo predice bien, el resultado real debería caer dentro "
            "del rango P10-P90 simulado ~80% de las veces, y no debería haber un sesgo "
            "sistemático a sobreestimar o subestimar victorias."
        )
        if backtest_sweep_calibration is None or backtest_sweep_summary is None:
            st.warning(
                "No se encontró `backtest_sweep_summary.csv`. Corre primero "
                "`python src/data_pipeline.py --backtest-sweep` (ADVERTENCIA: la ingesta más "
                "cara del proyecto, del orden de miles de llamadas a la API, 1.5-3 horas la "
                "primera vez) y luego "
                "`python -c \"from src.backtesting import build_backtest_sweep_dataset; "
                "from src.config_loader import load_config; "
                "build_backtest_sweep_dataset(load_config())\"`."
            )
        else:
            calibration = backtest_sweep_calibration.iloc[0]
            cols = st.columns(5)
            cols[0].metric("Casos", f"{int(calibration['n_cases'])}")
            cols[1].metric("% dentro de P10-P90", f"{calibration['pct_within_p10_p90']:.1f}%")
            cols[2].metric("Percentil medio", f"{calibration['mean_percentile']:.1f}")
            cols[3].metric("Error medio (victorias)", f"{calibration['mean_error_wins']:+.1f}")
            cols[4].metric("Correlación real vs. predicho", f"{calibration['correlation_actual_vs_predicted']:.2f}")
            render_glossary_expander(backtest_sweep_calibration.columns, CALIBRATION_GLOSSARY)

            bias = calibration["mean_error_wins"]
            if bias < -3:
                st.info(
                    f"El modelo SOBREESTIMA victorias en promedio ({bias:+.1f} victorias/temporada) -- "
                    "mismo patrón de 'fricción de vestuario' que los 4 comparables narrativos, pero "
                    "confirmado a escala de liga completa."
                )
            elif bias > 3:
                st.info(
                    f"El modelo SUBESTIMA victorias en promedio ({bias:+.1f} victorias/temporada)."
                )

            st.markdown("##### Distribución de percentiles reales")
            st.caption(
                "Si el modelo estuviera perfectamente calibrado, esta distribución sería "
                "aproximadamente uniforme entre 0 y 100 -- una concentración hacia 0 significa "
                "que el modelo sobreestima sistemáticamente; hacia 100, que subestima."
            )
            # Las etiquetas se pasan como strings ("0-10", "10-20"...): los
            # objetos Interval que devuelve pd.cut por defecto no son
            # serializables por st.bar_chart y lanzan un SchemaValidationError.
            percentile_bins = pd.cut(
                backtest_sweep_summary["actual_percentile"],
                bins=range(0, 101, 10),
                include_lowest=True,
                labels=[f"{low}-{low + 10}" for low in range(0, 100, 10)],
            )
            st.bar_chart(percentile_bins.value_counts().sort_index())

            st.markdown("##### Victorias reales vs. victorias simuladas (media)")
            scatter_df = backtest_sweep_summary[["actual_wins", "simulated_wins_mean"]].rename(
                columns={"actual_wins": "Reales", "simulated_wins_mean": "Simuladas (media)"}
            )
            st.scatter_chart(scatter_df, x="Reales", y="Simuladas (media)")

            with st.expander(f"Ver los {len(backtest_sweep_summary)} casos individuales"):
                st.dataframe(
                    backtest_sweep_summary.sort_values("actual_percentile"),
                    width="stretch", hide_index=True,
                )

# ---------------------------------------------------------------------
# 🏆 Liga NBA -- Liga y Playoffs, Premios individuales
# ---------------------------------------------------------------------

with tab_league_group:
    sub_league, sub_awards, sub_champions = st.tabs(
        ["Liga y Playoffs", "Premios individuales", "Campeones reales"]
    )

    with sub_league:
        st.subheader("Los 30 equipos: temporada regular simulada")

        if regular_season_summary is None or playoff_summary is None:
            st.warning(
                "No se encontraron los CSV de liga completa. Esto requiere primero "
                "`python src/data_pipeline.py --league` (ADVERTENCIA: ~900 llamadas a la "
                "API, 20-30+ min la primera vez) y luego "
                "`python -c \"from src.league_simulation import build_league_simulation_dataset; "
                "from src.config_loader import load_config; "
                "build_league_simulation_dataset(load_config())\"`."
            )
        else:
            st.markdown("##### Clasificación por conferencia")
            st.caption(
                "Victorias medias simuladas de los 30 equipos con sus rosters reales, en un "
                "calendario round-robin (no el calendario oficial, que aún no existe -- ver "
                "ARQUITECTURA.md), ordenadas por conferencia -- seeds 1-6 clasifican directo a "
                "playoffs, 7-10 juegan el play-in, 11-15 quedan fuera (mismo formato real que "
                "se simula)."
            )

            standings = compute_conference_standings(regular_season_summary, playoff_summary)
            standings_display_cols = [
                "seed", "team_abbreviation", "wins_mean", "wins_p10", "wins_p90",
                "situacion", "playoff_pct", "championship_pct",
            ]
            standings_column_config = {
                "seed": st.column_config.NumberColumn(help=LEAGUE_GLOSSARY["seed"]),
                "team_abbreviation": st.column_config.TextColumn("Equipo"),
                "wins_mean": st.column_config.NumberColumn("Victorias", help=LEAGUE_GLOSSARY["wins_mean"], format="%.1f"),
                "wins_p10": st.column_config.NumberColumn("P10", help=LEAGUE_GLOSSARY["wins_p10"], format="%.0f"),
                "wins_p90": st.column_config.NumberColumn("P90", help=LEAGUE_GLOSSARY["wins_p90"], format="%.0f"),
                "situacion": st.column_config.TextColumn("Situación"),
                "playoff_pct": st.column_config.NumberColumn("Playoffs %", help=LEAGUE_GLOSSARY["playoff_pct"], format="%.1f%%"),
                "championship_pct": st.column_config.NumberColumn("Título %", help=LEAGUE_GLOSSARY["championship_pct"], format="%.1f%%"),
            }

            conf_cols = st.columns(2)
            for col, conference_label, conference_key in zip(conf_cols, ["Este", "Oeste"], ["East", "West"]):
                with col:
                    st.markdown(f"**Conferencia {conference_label}**")
                    conf_df = standings[conference_key]
                    display_cols = [c for c in standings_display_cols if c in conf_df.columns]
                    st.dataframe(
                        conf_df[display_cols],
                        width="stretch", hide_index=True,
                        column_config={k: v for k, v in standings_column_config.items() if k in display_cols},
                    )
            st.caption(STANDINGS_SITUATION_GLOSSARY)

            with st.expander("Ver tabla completa de los 30 equipos (sin dividir por conferencia)"):
                st.dataframe(
                    regular_season_summary[["team_abbreviation", "conference", "wins_mean", "wins_p10", "wins_p90"]],
                    width="stretch", hide_index=True,
                )

            st.subheader("Probabilidades de playoffs y campeonato")
            column_config = render_glossary_expander(playoff_summary.columns, LEAGUE_GLOSSARY)
            st.dataframe(playoff_summary, width="stretch", hide_index=True, column_config=column_config)

            my_team_row = playoff_summary[playoff_summary["team_abbreviation"] == my_team_abbrev]
            if not my_team_row.empty:
                row = my_team_row.iloc[0]
                with st.container(border=True):
                    cols = st.columns(4)
                    cols[0].metric(f"{my_team_abbrev} — Playoffs", f"{row['playoff_pct']:.1f}%")
                    cols[1].metric("Semis de conferencia", f"{row['conf_semis_pct']:.1f}%")
                    cols[2].metric("Finales de conferencia", f"{row['finals_pct']:.1f}%")
                    cols[3].metric("Campeonato", f"{row['championship_pct']:.1f}%")

            st.subheader("Explorar un equipo")
            if player_projections is None:
                st.warning(
                    "No se encontró `league_player_projections.csv` (se genera junto con el "
                    "resto de la simulación de liga -- puede que la ingesta se completara "
                    "antes de esta versión del dashboard; vuelve a correr "
                    "`build_league_simulation_dataset`)."
                )
            else:
                team_options = sorted(regular_season_summary["team_abbreviation"].unique().tolist())
                default_index = team_options.index(my_team_abbrev) if my_team_abbrev in team_options else 0
                selected_team = st.selectbox("Equipo", team_options, index=default_index)

                team_regular = regular_season_summary[regular_season_summary["team_abbreviation"] == selected_team]
                team_playoff = playoff_summary[playoff_summary["team_abbreviation"] == selected_team]
                if not team_regular.empty and not team_playoff.empty:
                    reg_row = team_regular.iloc[0]
                    po_row = team_playoff.iloc[0]
                    with st.container(border=True):
                        cols = st.columns(5)
                        cols[0].metric("Victorias medias", f"{reg_row['wins_mean']:.1f}")
                        cols[1].metric("Playoffs", f"{po_row['playoff_pct']:.1f}%")
                        cols[2].metric("Semis conf.", f"{po_row['conf_semis_pct']:.1f}%")
                        cols[3].metric("Finales conf.", f"{po_row['finals_pct']:.1f}%")
                        cols[4].metric("Campeonato", f"{po_row['championship_pct']:.1f}%")

                team_players = player_projections[player_projections["team_abbreviation"] == selected_team]
                team_view = select_roster_view(
                    team_players,
                    mode=_stat_mode(),
                    meta_columns=LEAGUE_PLAYER_META_COLUMNS,
                    games_per_season=config["simulation"]["games_per_season"],
                )
                column_config = render_glossary_expander(
                    team_view.columns, ROSTER_STAT_GLOSSARY, title=f"Leyenda — {selected_team}"
                )
                st.dataframe(
                    team_view.sort_values("game_score_per36", ascending=False),
                    width="stretch", hide_index=True, column_config=column_config,
                )

            st.subheader("Bracket de playoffs")
            st.caption(
                "Una realización concreta de playoffs (no la distribución agregada de arriba) -- "
                "play-in, ronda 1, semis y finales de conferencia, con el emparejamiento y ganador "
                "de cada serie. El seeding usa las victorias medias ya simuladas; cada partido de "
                "play-in y cada serie se sortea de nuevo al pulsar el botón."
            )
            if st.button("🎲 Simular un bracket de playoffs"):
                with st.spinner("Simulando bracket..."):
                    st.session_state["bracket_result"] = run_single_bracket_simulation(config)

            bracket_result = st.session_state.get("bracket_result")
            if bracket_result is not None:
                bracket_cols = st.columns(2)
                with bracket_cols[0]:
                    render_conference_bracket("Conferencia Este", bracket_result["east"])
                with bracket_cols[1]:
                    render_conference_bracket("Conferencia Oeste", bracket_result["west"])

                st.markdown("---")
                st.markdown(
                    f"## 🏆🏆 Campeón de la NBA: **{bracket_result['nba_champion']}** 🏆🏆",
                )

    with sub_awards:
        st.subheader("Premios individuales — heurísticas sobre la proyección")
        st.caption(
            "MVP, DPOY, 6.º Hombre, ROY, MIP y (si hay datos de los 30 equipos) COY, "
            "calculados a partir de las proyecciones y los datos reales ya cargados -- "
            "NO son una predicción de la votación real de los medios, son lecturas "
            "narrativas de los números. Cada fórmula está documentada en "
            "`src/awards_projection.py`. Ver la leyenda más abajo para el detalle de "
            "cada columna."
        )

        if awards is None:
            st.warning(
                "No se encontró `aging_curve_projection.csv` ni `league_player_projections.csv`. "
                "Corre primero el pipeline de tu equipo (o `--league` para los 30 equipos, más "
                "significativo para estos premios) antes de ver esta pestaña."
            )
        else:
            if awards["scope"] == "own":
                st.info(
                    "Calculado solo sobre tu roster propio (no se encontraron datos de los 30 "
                    "equipos) -- útil para ver quién destaca dentro de tu plantilla, pero no es "
                    "comparable con el resto de la liga. Corre `data_pipeline.py --league` para "
                    "un análisis a nivel NBA."
                )

            row1 = st.columns(2)
            _award_section(
                row1[0], "🏆", "MVP",
                "Valor de temporada (Game Score proyectado × minutos) ponderado por el % de "
                "victorias del equipo.",
                awards["mvp"],
            )
            _award_section(
                row1[1], "🛡️", "DPOY (Defensor del Año)",
                "Proxy defensivo por-36 min (robos, tapones, rebote defensivo, faltas) -- el box "
                "score no ve casi nada de la defensa real, tómalo con pinzas.",
                awards["dpoy"],
            )

            row2 = st.columns(2)
            _award_section(
                row2[0], "🌟", "Rookie del Año (ROY)",
                "Mismo valor de temporada que el MVP, solo entre jugadores con una única "
                "temporada registrada (rookies).",
                awards["roy"],
            )
            _award_section(
                row2[1], "🔥", "Más Mejorado (MIP)",
                "Salto de Game Score por-36 REAL entre las dos temporadas más recientes (no usa "
                "la proyección) -- mide la mejora que ya ocurrió.",
                awards["mip"],
            )

            row3 = st.columns(2)
            _award_section(
                row3[0], "🎖️", "6.º Hombre del Año",
                "Mismo valor de temporada que el MVP, solo entre jugadores que salieron del "
                "banquillo la mayoría de partidos la temporada real más reciente.",
                awards["sixth_man"],
            )
            _award_section(
                row3[1], "📋", "Entrenador del Año (COY) — proxy de equipo",
                "Este proyecto no modela entrenadores: es el equipo que más mejoró frente a sus "
                "victorias REALES de la temporada anterior, el proxy más común de la votación real.",
                awards["coy"],
            )

            with st.expander("Leyenda de columnas"):
                for col_name, explanation in AWARDS_GLOSSARY.items():
                    st.markdown(f"**{col_name}** — {explanation}")

            st.markdown("---")
            st.subheader("Premios de fin de temporada")
            st.caption(
                "All-Star, All-NBA y All-Defensive. **All-Star**: sin restricción de partidos "
                "(se vota a mitad de temporada). **All-NBA y All-Defensive**: exigen un mínimo de "
                "**65 partidos jugados** (esperados, según el modelo de riesgo de lesión -- ver "
                "GP en la pestaña Roster), la política real de la NBA desde 2023-24. Formato "
                "clásico 2 bases/escoltas + 2 aleros/ala-pívots + 1 pívot por quinteto -- requiere "
                "`roster_positions.csv` (`python src/data_pipeline.py`, ya incluido en el pipeline "
                "normal) para conocer la posición real de cada jugador."
            )

            st.markdown("#### ⭐ All-Star")
            st.caption(
                "**Titular/Reserva** es solo una etiqueta sobre el ranking heurístico (los 5 de "
                "mayor valor por conferencia = Titular) -- NO simula el voto real (50% fans + 25% "
                "jugadores + 25% medios para titulares, entrenadores para reservas), porque ese "
                "dato no existe en ningún sitio accesible."
            )
            if awards["all_star"] is None or awards["all_star"].empty:
                st.caption("No hay candidatos disponibles.")
            else:
                star_display_cols = [
                    c for c in ["player_name", "team_abbreviation", "country", "selection_type", "season_value"]
                    if c in awards["all_star"].columns
                ]
                if "conference" in awards["all_star"].columns and awards["all_star"]["conference"].nunique() > 1:
                    star_cols = st.columns(2)
                    for col, conference in zip(star_cols, sorted(awards["all_star"]["conference"].dropna().unique())):
                        with col:
                            st.markdown(f"**Conferencia {conference}**")
                            conf_players = awards["all_star"][awards["all_star"]["conference"] == conference]
                            st.dataframe(conf_players[star_display_cols], width="stretch", hide_index=True)
                else:
                    st.dataframe(awards["all_star"][star_display_cols], width="stretch", hide_index=True)

                quota = awards.get("all_star_nationality_quota")
                all_star_final = awards.get("all_star_final")
                commissioner_picks = (
                    all_star_final[all_star_final["commissioner_pick"]]
                    if all_star_final is not None and "commissioner_pick" in all_star_final.columns
                    else None
                )

                if quota and quota["checked"]:
                    if quota["meets_both"]:
                        st.success(
                            f"✅ Cuota de nacionalidad cumplida: {quota['us_count']} de EE.UU. "
                            f"(mínimo 16), {quota['international_count']} internacionales (mínimo 8)."
                            + (f" {quota['unknown_count']} sin nacionalidad conocida." if quota["unknown_count"] else "")
                        )
                    elif commissioner_picks is not None and not commissioner_picks.empty:
                        st.warning(
                            f"⚠️ **La selección natural NO cumplía la cuota real de la NBA** "
                            f"({quota['us_count']} de EE.UU., mínimo 16; "
                            f"{quota['international_count']} internacionales, mínimo 8). "
                            f"Se simuló la intervención del comisionado añadiendo a los siguientes "
                            f"{len(commissioner_picks)} jugador(es) -- **fueron elegidos a dedo para "
                            f"cubrir el cupo de nacionalidad, NO por mérito del ranking natural**. "
                            "En la vida real esta es una decisión discrecional de Adam Silver que "
                            "este proyecto no puede predecir; aquí se aproxima cogiendo al jugador "
                            "no seleccionado de mayor valor de temporada de la nacionalidad que "
                            "faltaba."
                        )
                        st.dataframe(
                            commissioner_picks[[c for c in ["player_name", "conference", "country", "season_value"] if c in commissioner_picks.columns]],
                            width="stretch", hide_index=True,
                        )
                    else:
                        st.warning(
                            f"⚠️ La selección natural NO cumple la cuota real de la NBA "
                            f"({quota['us_count']} de EE.UU., mínimo 16; "
                            f"{quota['international_count']} internacionales, mínimo 8) y no se "
                            "encontró ningún candidato elegible de la nacionalidad que falta para "
                            "cubrir el hueco."
                        )
                elif quota and not quota["checked"]:
                    st.caption(
                        "No se puede verificar la cuota de nacionalidad: falta `country` en los "
                        "datos (corre `data_pipeline.py --league` para los 30 equipos, o el "
                        "pipeline normal para tu roster propio)."
                    )

            all_nba_cols = st.columns(2)
            with all_nba_cols[0]:
                st.markdown("#### 🏀 Quintetos All-NBA")
                if awards["all_nba"] is None or awards["all_nba"].empty:
                    st.caption(
                        "No hay candidatos que cumplan el umbral de 65 partidos y tengan posición "
                        "conocida (¿corriste `data_pipeline.build_roster_positions_dataset`?)."
                    )
                else:
                    for team_name in awards["all_nba"]["team"].unique():
                        st.markdown(f"**{team_name}**")
                        team_df = awards["all_nba"][awards["all_nba"]["team"] == team_name]
                        st.dataframe(
                            team_df[["position_slot", "player_name", "games_played_expected", "season_value"]],
                            width="stretch", hide_index=True,
                        )
            with all_nba_cols[1]:
                st.markdown("#### 🛡️ Quintetos All-Defensive")
                if awards["all_defensive"] is None or awards["all_defensive"].empty:
                    st.caption(
                        "No hay candidatos que cumplan el umbral de 65 partidos y tengan posición "
                        "conocida."
                    )
                else:
                    for team_name in awards["all_defensive"]["team"].unique():
                        st.markdown(f"**{team_name}**")
                        team_df = awards["all_defensive"][awards["all_defensive"]["team"] == team_name]
                        st.dataframe(
                            team_df[["position_slot", "player_name", "games_played_expected", "defensive_value"]],
                            width="stretch", hide_index=True,
                        )

            with st.expander("Leyenda — Premios de fin de temporada"):
                for col_name, explanation in SEASON_AWARDS_GLOSSARY.items():
                    st.markdown(f"**{col_name}** — {explanation}")

    with sub_champions:
        st.subheader("Campeones reales de la NBA — contexto y validación")
        st.caption(
            "Datos REALES (no simulados) de los campeones de las temporadas cubiertas por el "
            "backtest sweep: desde qué seed salieron, a quién eliminaron y cómo estaba compuesto "
            "su roster. Sirve para dos cosas: contexto sobre qué aspecto tiene un campeón, y "
            "**validar el simulador** comparando de qué seeds salen los campeones reales frente "
            "a los simulados."
        )

        if champion_title_paths is None:
            st.warning(
                "No se encontró `champion_title_paths.csv`. Requiere haber corrido "
                "`python src/data_pipeline.py --backtest-sweep` y luego "
                "`python -c \"from src.champion_profiles import build_champion_analysis_dataset; "
                "from src.config_loader import load_config; "
                "build_champion_analysis_dataset(load_config())\"`."
            )
        else:
            n_seasons = len(champion_title_paths)
            n_distinct = champion_title_paths["team_abbreviation"].nunique()
            cols = st.columns(3)
            cols[0].metric("Temporadas analizadas", n_seasons)
            cols[1].metric("Franquicias campeonas distintas", n_distinct)
            repeat = champion_title_paths["team_abbreviation"].value_counts().idxmax()
            repeat_n = champion_title_paths["team_abbreviation"].value_counts().max()
            cols[2].metric("Más títulos", f"{repeat} ({repeat_n})")

            st.markdown("##### Camino al título de cada campeón")
            column_config = render_glossary_expander(
                champion_title_paths.columns, CHAMPION_GLOSSARY, title="Leyenda — camino al título"
            )
            st.dataframe(
                champion_title_paths, width="stretch", hide_index=True, column_config=column_config
            )

            st.markdown("##### Validación: ¿de qué seed salen los campeones?")
            seed_dist = compute_champion_seed_distribution(champion_title_paths)
            real_pct = {int(r.seed): r.pct for r in seed_dist.itertuples()}

            simulated_pct = {}
            if regular_season_summary is not None and playoff_summary is not None:
                standings_for_seeds = compute_conference_standings(regular_season_summary, playoff_summary)
                rows = [
                    {"seed": int(row["seed"]), "champ": row["championship_pct"]}
                    for conf in ("East", "West")
                    for _, row in standings_for_seeds[conf].iterrows()
                ]
                sim = pd.DataFrame(rows).groupby("seed")["champ"].sum()
                if sim.sum() > 0:
                    simulated_pct = (sim / sim.sum() * 100).to_dict()

            comparison = pd.DataFrame(
                {
                    "seed": sorted(set(real_pct) | set(simulated_pct)),
                }
            )
            comparison["% campeones REALES"] = comparison["seed"].map(real_pct).fillna(0.0).round(1)
            comparison["% campeones SIMULADOS"] = comparison["seed"].map(simulated_pct).fillna(0.0).round(1)
            st.dataframe(comparison, width="stretch", hide_index=True)
            st.caption(
                "En las temporadas reales analizadas **ningún** campeón salió de un seed peor que "
                "el 3. Si el simulador reparte títulos entre seeds bajos, está siendo demasiado "
                "aleatorio — ver la discusión de señal/ruido del seeding en el README."
            )

            if champion_roster_profiles is not None:
                st.markdown("##### Composición del roster de cada campeón")
                st.caption(
                    "Reparto de minutos por posición, experiencia y concentración en las estrellas. "
                    "**Descriptivo, no predictivo**: son 15 campeones, muestra demasiado pequeña "
                    "para extraer una 'receta' de título (este proyecto ya se equivocó una vez "
                    "sacando conclusiones fuertes de 4 casos — ver README)."
                )
                column_config = render_glossary_expander(
                    champion_roster_profiles.columns, CHAMPION_GLOSSARY, title="Leyenda — composición de roster"
                )
                st.dataframe(
                    champion_roster_profiles.round(1), width="stretch", hide_index=True,
                    column_config=column_config,
                )

            if champion_seed_trajectories is not None:
                st.markdown("##### Trayectoria de seed por franquicia")
                st.caption(
                    "Puesto de cada franquicia en su conferencia, temporada a temporada "
                    "(1 = mejor récord, 11-15 = fuera de playoffs). Deja ver quién sostuvo un "
                    "nivel alto y quién osciló."
                )
                st.dataframe(champion_seed_trajectories, width="stretch")

# ---------------------------------------------------------------------
# 🤖 Explicador (IA)
# ---------------------------------------------------------------------

with tab_explainer:
    st.subheader("Explicador de resultados en lenguaje natural")
    st.caption(
        "Pregunta lo que quieras sobre los datos ya calculados en las otras pestañas "
        "(roster, riesgo de lesión/desgaste, simulación, backtesting, liga y playoffs). "
        "El modelo (Groq) responde SOLO a partir de esos datos -- no inventa cifras "
        "ni corre simulaciones nuevas."
    )

    if not groq_api_key:
        st.warning(
            "No se encontró la variable de entorno `GROQ_API_KEY`. Copia `.env.example` "
            "a `.env` en la raíz del proyecto y rellena tu API key de "
            "https://console.groq.com/keys para activar esta pestaña."
        )
    else:
        with st.expander("Ver el contexto (snapshot de datos) que recibe el modelo"):
            st.markdown(build_context_snapshot(config))

        if "explainer_history" not in st.session_state:
            st.session_state["explainer_history"] = []

        for turn in st.session_state["explainer_history"]:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])

        question = st.chat_input("Ej: ¿por qué Chicago lidera el Este?")
        if question:
            st.session_state["explainer_history"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Consultando a Groq..."):
                    try:
                        answer = explain_question(question, config, api_key=groq_api_key)
                    except Exception as exc:  # noqa: BLE001 -- mostramos cualquier error de API al usuario
                        answer = f"Error al consultar el modelo: {exc}"
                st.markdown(answer)
            st.session_state["explainer_history"].append({"role": "assistant", "content": answer})

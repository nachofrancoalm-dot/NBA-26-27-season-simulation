"""
pt_defend_signal.py

EXPERIMENTO, no forma parte del pipeline de producción. Investiga si
`PCT_PLUSMINUS` (defensa por tracking -- ver
data_pipeline.fetch_league_pt_defend_stats: cuánto empeora el % de tiro
REAL del rival cuando este jugador es el defensor más cercano, frente a
su % de tiro normal) aporta señal PREDICTIVA que Game Score+NET_RATING
no capturan -- misma pregunta que hustle_stats_signal.py (que dio
resultado NEGATIVO), pero con una métrica de impacto defensivo directo
en vez de actividad/esfuerzo.

BUG REAL en la primera versión de este experimento, encontrado antes de
reportar el resultado: usaba el PCT_PLUSMINUS de la MISMA temporada que
se predecía (R²=0.69, altísimo) -- violación de la regla de NO
LOOK-AHEAD de este proyecto (ver backtesting.py y
advanced_impact.adjusted_game_score_per36, que sí la respetan). Un
equipo que defendió bien DURANTE una temporada correlaciona con su
diferencial DURANTE esa misma temporada casi por definición -- no es una
predicción, es casi tautológico. Reescrito para usar SOLO la temporada
PREVIA de cada jugador (mismo patrón que
advanced_impact.compute_recency_weighted_advanced): "¿la defensa por
tracking del año pasado predice el resultado de este año?", la pregunta
real que hay que responder.

LIMITACIÓN DE DATOS: disponible desde 2013-14 (Second Spectrum) -- para
usar la temporada PREVIA como predictor, el primer caso utilizable es
2014-15 (con datos de tracking de 2013-14).

Uso:
    python scripts/experiments/pt_defend_signal.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config_loader import get_paths, load_config  # noqa: E402
from season_utils import season_start_year  # noqa: E402


def build_team_pt_defend_features(config: dict) -> pd.DataFrame:
    """
    Una fila por caso (equipo-temporada) con `team_pct_plusminus_prior`:
    PCT_PLUSMINUS de la TEMPORADA ANTERIOR de cada jugador del roster de
    ese caso (no look-ahead), agregado a nivel de equipo ponderando por
    los minutos/partido REALES que ese jugador jugó en la temporada del
    caso (de backtest_sweep_player_career_stats.csv -- el peso de "cuánto
    pesa este jugador en el equipo" tiene que venir de la temporada que
    se está prediciendo, no de la anterior).

    Jugador sin temporada previa de tracking (rookie, o su año anterior
    cae antes de 2013-14): se excluye de la ponderación, igual que
    project_historical_player con un rookie sin historial -- no se
    inventa un valor de liga.
    """
    paths = get_paths(config)
    rosters = pd.read_csv(paths["processed"] / "backtest_sweep_rosters.csv")
    player_stats = pd.read_csv(paths["processed"] / "backtest_sweep_player_career_stats.csv")
    pt_defend = pd.read_csv(paths["processed"] / "league_pt_defend_stats.csv")
    pt_defend = pt_defend[pt_defend["D_FGA"] > 0][["PLAYER_ID", "season", "D_FGA", "PCT_PLUSMINUS"]]
    # Un jugador traspasado a mitad de temporada puede tener más de una
    # fila para la misma (PLAYER_ID, season) -- colapsar ponderando por
    # D_FGA, mismo criterio que advanced_impact.compute_recency_weighted_advanced
    # usa para colapsar temporadas partidas por traspaso.
    pt_defend_collapsed = (
        pt_defend.assign(_weighted=pt_defend["PCT_PLUSMINUS"] * pt_defend["D_FGA"])
        .groupby(["PLAYER_ID", "season"])
        .apply(lambda g: g["_weighted"].sum() / g["D_FGA"].sum(), include_groups=False)
        .rename("PCT_PLUSMINUS")
        .reset_index()
    )
    pt_defend_by_player = {
        pid: g.set_index("season")["PCT_PLUSMINUS"] for pid, g in pt_defend_collapsed.groupby("PLAYER_ID")
    }

    # Minutos/partido REALES del jugador en la temporada del caso (el
    # peso para agregar a nivel de equipo) -- mismo criterio que
    # aging_curve_shrinkage.py.
    minutes_lookup = {}
    for _, row in player_stats.iterrows():
        gp = row.get("GP", 0)
        if gp and gp > 0:
            minutes_lookup[(int(row["PLAYER_ID"]), str(row["SEASON_ID"]))] = float(row["MIN"]) / float(gp)

    rows = []
    for (name, season), group in rosters.groupby(["comparable_name", "season"]):
        prior_season_year = season_start_year(season) - 1
        prior_season = f"{prior_season_year}-{str(prior_season_year + 1)[-2:]}"

        weighted_sum, total_weight = 0.0, 0.0
        for player_id in group["PLAYER_ID"].astype(int):
            series = pt_defend_by_player.get(player_id)
            if series is None or prior_season not in series.index:
                continue
            weight = minutes_lookup.get((player_id, season), 0.0)
            if weight <= 0:
                continue
            weighted_sum += series[prior_season] * weight
            total_weight += weight

        if total_weight <= 0:
            continue
        rows.append({
            "comparable_name": name, "season": season,
            "team_pct_plusminus_prior": weighted_sum / total_weight,
            "coverage": total_weight,
        })
    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    paths = get_paths(config)

    pt_defend_path = paths["processed"] / "league_pt_defend_stats.csv"
    if not pt_defend_path.exists():
        raise FileNotFoundError(
            f"No se encontró {pt_defend_path}. Corre "
            "`from data_pipeline import build_league_pt_defend_dataset` primero."
        )

    features = build_team_pt_defend_features(config)
    print(f"Casos con temporada previa de tracking disponible: {len(features)}")

    calibration_features = pd.read_csv(paths["processed"] / "experiment_bayesian_calibration_features.csv")
    df = features.merge(calibration_features, on=["comparable_name", "season"], how="inner")
    print(f"Casos tras cruzar con la métrica ya calibrada: {len(df)}")
    print(f"Temporadas cubiertas: {sorted(df['season'].unique())}")

    import statsmodels.api as sm

    y = df["y_actual_diff_points_pg"]
    x_base = sm.add_constant(df[["x_game_score_vs_baseline"]])
    model_base = sm.OLS(y, x_base).fit()
    print(f"\nR² modelo base (solo Game Score + NET_RATING), sobre este subconjunto: {model_base.rsquared:.4f}")

    raw_corr = df["team_pct_plusminus_prior"].corr(df["y_actual_diff_points_pg"])
    print(f"Correlación cruda team_pct_plusminus_prior vs resultado real: {raw_corr:.3f}")

    x_plus = sm.add_constant(df[["x_game_score_vs_baseline", "team_pct_plusminus_prior"]])
    model_plus = sm.OLS(y, x_plus).fit()
    f_test = model_plus.compare_f_test(model_base)

    print(f"\nR² con team_pct_plusminus_prior añadida: {model_plus.rsquared:.4f} "
          f"(delta: {model_plus.rsquared - model_base.rsquared:.4f})")
    print(f"Coeficiente: {model_plus.params['team_pct_plusminus_prior']:.4f} "
          f"(p-value: {model_plus.pvalues['team_pct_plusminus_prior']:.4f})")
    print(f"F-test p-value (vs. modelo base): {f_test[1]:.4f}")


if __name__ == "__main__":
    main()

"""
contract_year_effect.py

EXPERIMENTO, no forma parte del pipeline de producción. Evalúa la
hipótesis de que un jugador en el último año de contrato ("contract
year"/"walk year") produce más de lo esperado para asegurarse una buena
extensión.

Datos (descargados manualmente de Kaggle, nba_api no expone salarios;
ver data/raw/contract_data/): "NBA Player Stats and Salaries_2010-2025"
(jugador-temporada 2010-2025, salario + stats por partido, Year = año
en que termina la temporada) y "nba_contracts_history" (199 contratos de
138 jugadores con CONTRACT_START/END, muestra parcial de contratos
vigentes/recientes al momento del scrape). Ambas convenciones de año
verificadas a mano contra contratos conocidos.

Método: para contratos de >= 2 temporadas, compara la temporada final
(season_end_year == CONTRACT_END) contra las intermedias del mismo
contrato y jugador, de dos formas: (1) delta pareado por contrato
(GmSc final - media no-final) con t de Student, y (2) regresión OLS con
efectos fijos por jugador (demeaning) controlando por edad y edad², para
aislar el efecto de "es buen jugador" o "le tocó su pico de edad". Game
Score (Hollinger) es la métrica de producción.

Uso:
    python scripts/experiments/contract_year_effect.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config_loader import get_paths, load_config  # noqa: E402

CONTRACT_DATA_SUBDIR = "contract_data"
SALARY_STATS_FILENAME = "NBA Player Stats and Salaries_2010-2025.csv"
CONTRACTS_FILENAME = "nba_contracts_history.csv"

# Duración mínima de contrato (temporadas) para comparar final vs. intermedias.
MIN_CONTRACT_SPAN_SEASONS = 2


def compute_game_score(row: pd.Series) -> float:
    """Game Score (Hollinger) sobre columnas por partido (no totales)."""
    return (
        row["PTS"]
        + 0.4 * row["FG"]
        - 0.7 * row["FGA"]
        - 0.4 * (row["FTA"] - row["FT"])
        + 0.7 * row["ORB"]
        + 0.3 * row["DRB"]
        + row["STL"]
        + 0.7 * row["AST"]
        + 0.7 * row["BLK"]
        - 0.4 * row["PF"]
        - row["TOV"]
    )


def load_season_stats(config: dict) -> pd.DataFrame:
    """Una fila por jugador-temporada con `player`, `season_end_year`, `age`, `game_score`."""
    paths = get_paths(config)
    path = paths["raw"] / CONTRACT_DATA_SUBDIR / SALARY_STATS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}.")

    df = pd.read_csv(path)
    df = df.rename(columns={"Player": "player", "Year": "season_end_year", "Age": "age"})
    df["game_score"] = df.apply(compute_game_score, axis=1)
    return df[["player", "season_end_year", "age", "game_score"]]


def load_contracts(config: dict) -> pd.DataFrame:
    """Una fila por contrato con `player`, `contract_id`, `contract_start`, `contract_end`; solo duración >= MIN_CONTRACT_SPAN_SEASONS."""
    paths = get_paths(config)
    path = paths["raw"] / CONTRACT_DATA_SUBDIR / CONTRACTS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}.")

    df = pd.read_csv(path)
    df = df.rename(columns={"NAME": "player", "CONTRACT_START": "contract_start", "CONTRACT_END": "contract_end"})
    df["contract_start"] = pd.to_numeric(df["contract_start"], errors="coerce")
    df["contract_end"] = pd.to_numeric(df["contract_end"], errors="coerce")
    df = df.dropna(subset=["contract_start", "contract_end"])
    df["contract_start"] = df["contract_start"].astype(int)
    df["contract_end"] = df["contract_end"].astype(int)
    df["span_seasons"] = df["contract_end"] - df["contract_start"]
    df = df[df["span_seasons"] >= MIN_CONTRACT_SPAN_SEASONS].reset_index(drop=True)
    df["contract_id"] = df.index
    return df[["contract_id", "player", "contract_start", "contract_end", "span_seasons"]]


def build_contract_year_panel(config: dict) -> pd.DataFrame:
    """Una fila por (contrato, temporada) con `is_final_year`; descarta contratos sin datos tanto en la temporada final como en al menos una intermedia."""
    stats = load_season_stats(config)
    contracts = load_contracts(config)

    rows = []
    for _, contract in contracts.iterrows():
        seasons = range(int(contract["contract_start"]) + 1, int(contract["contract_end"]) + 1)
        player_stats = stats[stats["player"] == contract["player"]]
        contract_rows = []
        for season in seasons:
            match = player_stats[player_stats["season_end_year"] == season]
            if match.empty:
                continue
            match_row = match.iloc[0]
            contract_rows.append(
                {
                    "contract_id": contract["contract_id"],
                    "player": contract["player"],
                    "season_end_year": season,
                    "age": match_row["age"],
                    "game_score": match_row["game_score"],
                    "is_final_year": season == contract["contract_end"],
                }
            )
        has_final = any(r["is_final_year"] for r in contract_rows)
        has_non_final = any(not r["is_final_year"] for r in contract_rows)
        if has_final and has_non_final:
            rows.extend(contract_rows)

    return pd.DataFrame(rows)


def paired_delta_test(panel: pd.DataFrame):
    """Por contrato: GmSc(final) - media(GmSc no-final). Devuelve
    (deltas_df, resultado del t de Student pareado contra 0)."""
    from scipy import stats as scipy_stats

    rows = []
    for contract_id, group in panel.groupby("contract_id"):
        final = group[group["is_final_year"]]["game_score"]
        non_final = group[~group["is_final_year"]]["game_score"]
        if final.empty or non_final.empty:
            continue
        rows.append(
            {
                "contract_id": contract_id,
                "player": group["player"].iloc[0],
                "final_game_score": final.mean(),
                "non_final_game_score": non_final.mean(),
                "delta": final.mean() - non_final.mean(),
            }
        )
    deltas = pd.DataFrame(rows)
    t_result = scipy_stats.ttest_1samp(deltas["delta"], 0.0)
    return deltas, t_result


def fixed_effects_regression(panel: pd.DataFrame):
    """OLS de game_score ~ is_final_year + age + age^2, con efectos fijos por jugador vía demeaning, para aislar is_final_year de nivel de jugador y edad."""
    import statsmodels.api as sm

    df = panel.copy()
    df["age2"] = df["age"] ** 2
    df["is_final_year"] = df["is_final_year"].astype(float)

    demeaned = df.groupby("player")[["game_score", "age", "age2", "is_final_year"]].transform(lambda s: s - s.mean())
    X = sm.add_constant(demeaned[["age", "age2", "is_final_year"]])
    model = sm.OLS(demeaned["game_score"], X).fit()
    return model


def main() -> None:
    config = load_config()
    paths = get_paths(config)

    panel = build_contract_year_panel(config)
    n_contracts = panel["contract_id"].nunique()
    n_players = panel["player"].nunique()
    print(f"Contratos con temporada final + al menos una intermedia con datos: {n_contracts} ({n_players} jugadores distintos)")

    panel_path = paths["processed"] / "experiment_contract_year_panel.csv"
    panel.to_csv(panel_path, index=False)
    print(f"Guardado: {panel_path}")

    print("\n=== Delta pareado por contrato (GmSc año final - GmSc media resto del contrato) ===")
    deltas, t_result = paired_delta_test(panel)
    positive = (deltas["delta"] > 0).sum()
    print(f"Media del delta: {deltas['delta'].mean():.3f} (mediana {deltas['delta'].median():.3f})")
    print(f"Contratos con año final MEJOR que el resto: {positive}/{len(deltas)} ({100 * positive / len(deltas):.1f}%)")
    print(f"t de Student (H0: delta medio = 0): t={t_result.statistic:.3f}, p={t_result.pvalue:.4f}")

    print("\n=== Regresión con efectos fijos por jugador + control de edad ===")
    model = fixed_effects_regression(panel)
    print(model.summary().tables[1])
    coef = model.params["is_final_year"]
    pvalue = model.pvalues["is_final_year"]
    print(f"\nCoeficiente is_final_year: {coef:.3f} (p={pvalue:.4f})")
    print(f"R²: {model.rsquared:.4f}")


if __name__ == "__main__":
    main()

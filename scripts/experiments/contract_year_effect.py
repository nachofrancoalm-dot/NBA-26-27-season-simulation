"""
contract_year_effect.py

EXPERIMENTO, no forma parte del pipeline de producción. Evalúa si
merece la pena incorporar información salarial/contractual (año de
contrato / agencia libre inminente) al motor de proyección, bajo la
hipótesis de que un jugador en el último año de su contrato ("contract
year"/"walk year") produce más de lo esperado para asegurarse una
buena extensión.

Fuentes de datos (ninguna forma parte del pipeline de nba_api ya
existente -- nba_api no expone salarios ni contratos; ambas descargadas
manualmente desde Kaggle, ver data/raw/contract_data/):
  - "NBA Player Stats and Salaries_2010-2025.csv"
    (kaggle.com/datasets/ratin21/nba-player-stats-and-salaries-2010-2025):
    una fila por jugador-temporada (2010-2025), con salario y
    estadísticas reales POR PARTIDO (no totales). Convención de la
    columna Year = año en que TERMINA la temporada -- verificado a mano
    contra contratos conocidos: LeBron James Year=2011/Team=MIA/
    Salary=14.5M coincide con su salario real de la temporada 2010-11
    (primera en Miami), y Stephen Curry Year=2014/Salary=9.89M coincide
    con el primer año de su extensión de 2013 (temporada 2013-14).
  - "nba_contracts_history.csv"
    (kaggle.com/datasets/jarosawjaworski/current-nba-players-contracts-history):
    199 contratos de 138 jugadores con CONTRACT_START/CONTRACT_END (año
    calendario de inicio/fin de CADA contrato). Solo cubre contratos que
    estaban "vigentes o recientes" en el momento del scrape, NO el
    historial contractual completo de la liga -- por eso el análisis
    solo puede usar esta muestra parcial de contratos, no todos los
    jugador-temporada del otro CSV. Convención verificada con los
    contratos de 1 año de LeBron (CONTRACT_START=2015/END=2016 y
    2016/2017, los conocidos acuerdos "1+1" con opción de jugador que
    firmó en Cleveland precisamente para maximizar renegociación anual
    -- un ejemplo del propio patrón que este experimento intenta medir
    de forma sistemática): CONTRACT_END coincide con la convención
    Year=season_end_year del otro CSV, así que la temporada "de
    contrato" (la última antes de la agencia libre) de un contrato dado
    es season_end_year == CONTRACT_END.

Diseño del experimento (por qué esta comparación y no otra):
  Comparar la producción de un jugador en su año de contrato contra la
  de OTROS jugadores no sirve (cada jugador tiene su propio nivel).
  Comparar contra la temporada anterior/posterior del MISMO jugador
  tampoco basta a secas: la producción sube y baja con la edad de forma
  predecible, así que un año de contrato que cae en la cima de la curva
  de edad de un jugador parecería "efecto contrato" sin serlo. Por eso,
  para contratos de >= 2 temporadas (`MIN_CONTRACT_SPAN_SEASONS`), se
  compara la temporada FINAL (season_end_year == CONTRACT_END,
  is_final_year=True) contra las temporadas INTERMEDIAS del MISMO
  contrato y del MISMO jugador, de dos formas complementarias:
    1) Delta pareado por contrato: GmSc(final) - media(GmSc no-final),
       con test de signos (¿en cuántos contratos el año final es mejor
       que el resto del mismo contrato?) y t de Student pareada.
    2) Regresión con efectos fijos por jugador (demeaning dentro de
       jugador) controlando por edad y edad² -- así el coeficiente de
       is_final_year no puede explicarse por "este jugador es bueno"
       (efecto fijo) ni por "esta temporada le tocó estar en su pico de
       edad" (control de edad).
  Game Score (Hollinger) se usa como métrica de producción compuesta
  porque combina lo que ya está disponible por partido en el CSV
  (PTS, FG, FGA, FT, FTA, ORB, DRB, AST, STL, BLK, TOV, PF) sin
  necesitar re-descargar nada.

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

# Duración mínima de contrato (temporadas) para poder comparar la final
# contra al menos una intermedia del mismo contrato.
MIN_CONTRACT_SPAN_SEASONS = 2


def compute_game_score(row: pd.Series) -> float:
    """Game Score (Hollinger), fórmula estándar, sobre columnas POR
    PARTIDO (no totales -- el CSV de salarios ya viene así)."""
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
    """Devuelve una fila por jugador-temporada con `player`,
    `season_end_year`, `age`, `game_score` (por partido)."""
    paths = get_paths(config)
    path = paths["raw"] / CONTRACT_DATA_SUBDIR / SALARY_STATS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}.")

    df = pd.read_csv(path)
    df = df.rename(columns={"Player": "player", "Year": "season_end_year", "Age": "age"})
    df["game_score"] = df.apply(compute_game_score, axis=1)
    return df[["player", "season_end_year", "age", "game_score"]]


def load_contracts(config: dict) -> pd.DataFrame:
    """Devuelve una fila por contrato con `player`, `contract_id`,
    `contract_start`, `contract_end` -- solo contratos con duración >=
    MIN_CONTRACT_SPAN_SEASONS (necesitan al menos una temporada
    intermedia además de la final para poder comparar)."""
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
    """Una fila por (contrato, temporada real encontrada en el CSV de
    salarios): `player`, `contract_id`, `season_end_year`, `age`,
    `game_score`, `is_final_year`. Solo se quedan los contratos que
    tienen AL MENOS una temporada final y una no-final con datos reales
    -- si falta una de las dos, ese contrato no aporta nada a la
    comparación."""
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
    """OLS de game_score ~ is_final_year + age + age^2, con efectos
    fijos por jugador vía demeaning dentro de cada jugador (resta la
    media del propio jugador a cada variable, incluida la variable
    dependiente) -- el coeficiente de is_final_year queda así aislado de
    "este jugador es bueno" y, con los controles de edad, también de
    "le tocó la temporada en su pico de edad natural"."""
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

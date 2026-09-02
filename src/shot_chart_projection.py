"""
shot_chart_projection.py

Estima el shot chart de la temporada PROYECTADA de un jugador,
remuestreando sus tiros reales de las últimas N temporadas (ponderadas
por recencia, mismo criterio que
aging_curve.compute_recency_weighted_baseline) y ajustando el conteo
exacto para que cuadre con FGA/FG3A/FGM/FG3M ya proyectados por
aging_curve.py -- este módulo no decide CUÁNTO va a tirar un jugador,
solo DÓNDE, a partir de un volumen ya fijado en otro sitio.

INVESTIGACIÓN QUE MOTIVA ESTO: sobre 11 jugadores del roster con 2
temporadas reales de shot chart cacheadas (2024-25 vs 2025-26), la
distancia media entre distribuciones de zona (Restricted Area/Mid-Range/
Corner 3/...) fue 0.094 (0 = idéntica) y el cambio medio en % de tiros
de 3 fue de 6.5 puntos -- señal real suficiente para justificar el
remuestreo en vez de un patrón genérico o aleatorio.

MÉTODO: por jugador, dos bolsas de tiros históricos reales (2PT y 3PT),
cada tiro pesado por recencia de su temporada (0.5 ** (temporadas
atrás / half_life), igual que aging_curve.py). Se remuestrean con
reemplazo, ponderado por ese peso, EXACTAMENTE round(FG3A_projected)
tiros de 3 y round(FGA_projected - FG3A_projected) de 2. De esos tiros
remuestreados se marcan como anotados exactamente FG3M_projected /
(FGM_projected - FG3M_projected), eligiendo cuáles vía el % de acierto
histórico ponderado por recencia de la ZONA (SHOT_ZONE_BASIC) de cada
tiro -- las zonas donde el jugador acierta más tienen más probabilidad
de aparecer en verde, sin dejar de cuadrar el total exacto.

LIMITACIÓN CONOCIDA: si un jugador no tiene ningún tiro histórico de un
tipo (p. ej. un pívot que nunca ha tirado un triple) pero la proyección
le asigna algunos, esos tiros concretos no se pueden ubicar -- no se
inventa una zona sin base real, se devuelven menos tiros de los
pedidos para ese tipo en vez de fabricar una localización.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from aging_curve import DEFAULT_N_SEASONS_LOOKBACK, DEFAULT_RECENCY_HALF_LIFE_SEASONS
from season_utils import season_start_year

THREE_PT_SHOT_TYPE = "3PT Field Goal"
TWO_PT_SHOT_TYPE = "2PT Field Goal"

# Piso de probabilidad para una zona sin aciertos históricos (o sin
# tiros) -- evita que quede con probabilidad exactamente 0 de aparecer
# marcada como anotada, lo que rompería el muestreo ponderado si TODAS
# las zonas presentes tuvieran 0% real (jugador con muestra muy corta).
MIN_ZONE_MAKE_RATE = 0.05

SHOT_CHART_PROJECTION_COLUMNS = ["loc_x", "loc_y", "shot_made", "shot_type"]


def _recency_weighted_shot_pool(shots: pd.DataFrame, n_seasons: int, half_life_seasons: float) -> pd.DataFrame:
    """Filtra `shots` a las últimas n_seasons temporadas presentes y
    añade una columna `_weight` de recencia (0.5 ** (temporadas atrás /
    half_life), temporada más reciente = peso 1.0)."""
    if shots.empty:
        return shots.assign(_weight=pd.Series(dtype=float))

    seasons_present = sorted(shots["season"].astype(str).unique(), key=season_start_year, reverse=True)
    seasons_used = seasons_present[:n_seasons]
    seasons_ago = {season: i for i, season in enumerate(seasons_used)}

    recent = shots[shots["season"].astype(str).isin(seasons_used)].copy().reset_index(drop=True)
    recent["_weight"] = recent["season"].astype(str).map(seasons_ago).apply(lambda ago: 0.5 ** (ago / half_life_seasons))
    return recent


def _resample_attempts(pool: pd.DataFrame, n_attempts: int, rng: np.random.Generator) -> pd.DataFrame:
    """Remuestrea con reemplazo `n_attempts` filas de `pool`, ponderadas
    por `_weight`. Vacío si no hay tiros de ese tipo en el histórico."""
    if n_attempts <= 0 or pool.empty:
        return pool.iloc[0:0]
    weights = pool["_weight"].to_numpy()
    probs = weights / weights.sum()
    chosen = rng.choice(len(pool), size=n_attempts, replace=True, p=probs)
    return pool.iloc[chosen].reset_index(drop=True)


def _zone_make_rates(pool: pd.DataFrame) -> pd.Series:
    """% de acierto ponderado por recencia de cada SHOT_ZONE_BASIC en
    `pool`, con un piso de MIN_ZONE_MAKE_RATE."""
    return (
        pool.groupby("shot_zone_basic")
        .apply(lambda g: np.average(g["shot_made"].astype(float), weights=g["_weight"]), include_groups=False)
        .clip(lower=MIN_ZONE_MAKE_RATE)
    )


def _assign_makes(sampled: pd.DataFrame, type_pool: pd.DataFrame, n_makes: int, rng: np.random.Generator) -> np.ndarray:
    """
    Decide qué `n_makes` de los tiros ya remuestreados (`sampled`) se
    marcan como anotados -- ponderado por el % de acierto histórico de
    la zona de cada tiro (`_zone_make_rates` sobre `type_pool`), no
    uniforme, para que las zonas donde el jugador acierta más tengan más
    probabilidad de salir en verde. El TOTAL de anotados es siempre
    exactamente `n_makes`.
    """
    n_attempts = len(sampled)
    n_makes = max(0, min(n_makes, n_attempts))
    made = np.zeros(n_attempts, dtype=bool)
    if n_makes == 0 or n_attempts == 0:
        return made

    zone_rates = _zone_make_rates(type_pool)
    weights = sampled["shot_zone_basic"].map(zone_rates).fillna(MIN_ZONE_MAKE_RATE).to_numpy()
    weights = weights / weights.sum()
    made_positions = rng.choice(n_attempts, size=n_makes, replace=False, p=weights)
    made[made_positions] = True
    return made


def project_player_shot_chart(
    shots: pd.DataFrame,
    target_fga: float,
    target_fg3a: float,
    target_fgm: float,
    target_fg3m: float,
    n_seasons: int = DEFAULT_N_SEASONS_LOOKBACK,
    half_life_seasons: float = DEFAULT_RECENCY_HALF_LIFE_SEASONS,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Shot chart SINTÉTICO para la temporada proyectada de un jugador:
    remuestrea `shots` (histórico real, columnas season/loc_x/loc_y/
    shot_made/shot_type/shot_zone_basic) para que el conteo de intentos
    y anotados de 2/3 cuadre EXACTO con lo que aging_curve.py ya
    proyectó -- ver docstring del módulo para el método completo.
    `rng` por defecto usa un Generator sin semilla fija (tirada distinta
    cada vez); pásalo con semilla para un resultado reproducible.
    Devuelve columnas loc_x/loc_y/shot_made/shot_type -- vacío si no hay
    ningún tiro histórico de ningún tipo.
    """
    rng = rng if rng is not None else np.random.default_rng()
    pool = _recency_weighted_shot_pool(shots, n_seasons, half_life_seasons)

    n_3pt = round(target_fg3a)
    n_2pt = round(max(target_fga - target_fg3a, 0))
    n_3pt_makes = round(target_fg3m)
    n_2pt_makes = round(max(target_fgm - target_fg3m, 0))

    parts = []
    for shot_type, n_attempts, n_makes in (
        (THREE_PT_SHOT_TYPE, n_3pt, n_3pt_makes),
        (TWO_PT_SHOT_TYPE, n_2pt, n_2pt_makes),
    ):
        type_pool = pool[pool["shot_type"] == shot_type] if not pool.empty else pool
        sampled = _resample_attempts(type_pool, n_attempts, rng)
        if sampled.empty:
            continue
        sampled = sampled.assign(shot_made=_assign_makes(sampled, type_pool, n_makes, rng))
        parts.append(sampled[SHOT_CHART_PROJECTION_COLUMNS])

    if not parts:
        return pd.DataFrame(columns=SHOT_CHART_PROJECTION_COLUMNS)
    return pd.concat(parts, ignore_index=True)

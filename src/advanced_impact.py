"""
Métrica de impacto COMPUESTA: Game Score de caja + estadísticas avanzadas.

POR QUÉ EXISTE
--------------
El resto del proyecto mide el impacto de un jugador con el Game Score de
Hollinger (`aging_curve.compute_game_score_per36`), una métrica de CAJA
puramente ofensiva: solo ve la defensa a través de robos, tapones y
rebote defensivo. Todo lo demás que hace un buen defensor -- contestar
tiros sin taponarlos, rotar, no perder su marca, disuadir penetraciones
-- es literalmente invisible para ella.

El caso que lo dejó claro en este proyecto: los **Knicks 2025-26**
ganaron el título con 53 victorias y el modelo los proyectaba 12º del
Este. Brunson (19.4 GS/36) y Towns (19.0) sí se valoraban como élite,
pero Anunoby (13.5), Bridges (13.1) y Hart (12.9) -- tres especialistas
defensivos -- salían como jugadores del montón. Su Game Score de equipo
quedaba 22º de 30.

QUÉ SE MIDIÓ Y QUÉ SOBREVIVIÓ
------------------------------
Sobre los 480 casos del backtest sweep (16 temporadas x 30 equipos), con
la proyección REAL del pipeline (minutos normalizados a 240, recencia de
3 temporadas, sin look-ahead), contra el diferencial de puntos REAL y
validado dejando fuera una temporada entera cada vez (LOSO):

    modelo                     R fuera de muestra
    Game Score solo                    0.702      <- lo que había
    Game Score + PIE                   0.702      <- no aporta NADA
    Game Score + NET_RATING            0.754      <- integrado
    Game Score + PIE + NET_RATING      0.753      <- PIE resta

**PIE quedó fuera, y esto corrigió una medición anterior de este mismo
proyecto** que lo daba como útil (r=0.624 en solitario). Dos motivos por
los que aquella lectura era engañosa:

1. `corr(PIE de equipo, NET_RATING de equipo) = 0.64`. En solitario PIE
   correlaciona porque va montado en la misma señal que NET_RATING; una
   vez NET_RATING está en el modelo, PIE no añade información.
2. Con las tres variables el coeficiente de PIE sale **negativo**
   (-46.8), que es un artefacto clásico de colinealidad, no un hallazgo:
   nadie va a defender que más cuota de producción prediga PEOR. Un
   coeficiente con el signo al revés es señal de que la variable sobra.

Se conserva el cableado de PIE (`pie_weight`, por defecto 0.0) como
palanca de experimentación, igual que `simulation.apply_star_bonus` se
dejó implementada pero desactivada -- misma convención, mismo motivo:
está medida y descartada, no sin probar.

VALIDACIÓN LATERAL BONITA: la regresión de este ajuste devuelve, para el
Game Score solo, una pendiente de **0.2946** contra el diferencial real.
`game_score_to_net_rating_scale` estaba calibrado a **0.29** por una vía
completamente distinta. Dos caminos independientes al mismo número.

CÓMO SE COMBINAN (y por qué así)
--------------------------------
Como un **ajuste ADITIVO en unidades de Game Score**, no como un z-score
compuesto:

    impacto/36 = game_score/36
                 + k_pie * (PIE      - PIE_medio_de_esa_temporada)
                 + k_net * (NET_RATING - NET_medio_de_esa_temporada)

Tres propiedades que hacen que esta forma encaje sin romper nada de lo
ya calibrado:

1. **Suma cero preservada por construcción.** Las desviaciones se toman
   respecto a la media de liga de ESA temporada, así que el ajuste medio
   sobre los 30 equipos es ~0. La línea base de equipo promedio
   (`backtesting.compute_projected_league_baselines`) no se mueve, y con
   ella tampoco la restricción de que la media de victorias sea games/2
   -- el bug que costó más caro de este proyecto (ver "INFLACIÓN DE ERA"
   en el docstring de simulation.py).
2. **Ajuste de era gratis.** Al centrar por temporada, un PIE de 0.12 en
   2010-11 y uno de 0.12 en 2025-26 se juzgan cada uno contra su propio
   contexto. Es el mismo remedio que ya se aplicó al Game Score.
3. **La escala `game_score_to_net_rating_scale` sigue significando lo
   mismo.** El resultado sigue expresado en puntos de Game Score, así que
   el resto de la tubería (línea base, conversión a net rating, logística)
   no se entera del cambio. Sí hubo que RE-AJUSTAR su valor numérico,
   porque la métrica compuesta tiene más dispersión que el Game Score solo.

Un z-score compuesto habría exigido llevar media y desviación de liga por
temporada de las tres métricas y renormalizar la varianza a mano, con más
sitios donde equivocarse y sin ninguna de estas tres garantías.

Las métricas avanzadas se promedian con la MISMA ponderación por recencia
que el Game Score (`compute_recency_weighted_advanced`, media exponencial
de las últimas N temporadas). Eso no es solo consistencia: promediar 3
temporadas suele significar promediar varios contextos de equipo, lo que
diluye parcialmente el caveat de abajo.

CAVEAT IMPORTANTE -- CONTAMINACIÓN POR CONTEXTO DE EQUIPO
----------------------------------------------------------
`NET_RATING` y `PIE` NO son métricas puramente individuales. El
NET_RATING de un jugador mide cómo iba SU EQUIPO con él en cancha, no lo
que aportaría en otro sitio. Un suplente de un equipo campeón hereda
NET_RATING de campeón; una estrella de un equipo en reconstrucción carga
con el de su equipo.

Consecuencia para este proyecto: predice **muy bien rosters REALES** (que
es lo que valida el backtest sweep), pero transplantarlo a un roster
HIPOTÉTICO -- el caso de uso central -- asume justo lo que el propio
proyecto ya demostró que falla (que juntar piezas buenas produce la suma
de sus partes). En concreto, el roster hipotético del config hereda el
NET_RATING que cada jugador tuvo en OTRO equipo, así que juntar cinco
titulares de equipos ganadores infla la proyección de una forma que este
módulo NO corrige.

**Y aun así el peso va sin encoger.** Se probó encogerlo (0.25, 0.4, 0.5,
0.6, 0.75) y fuera de muestra el R mejora de forma MONÓTONA hasta el peso
completo (0.701 → 0.754): no hay ninguna señal empírica de
sobreajuste que justifique bajarlo. Encogerlo "por si acaso" sería forzar
un parámetro contra la evidencia para que el resultado se vea como uno
espera -- exactamente el error que este proyecto ya cometió una vez (ver
la sobrecorrección de la línea base en CLAUDE.md). La limitación se
documenta; el parámetro no se manipula.

Quien quiera medir el efecto del caveat tiene la palanca: con
`advanced_impact: {enabled: false}` en el YAML se recupera exactamente el
Game Score puro y se pueden comparar las dos proyecciones del mismo
roster hipotético.

TERCERA MÉTRICA INTEGRADA: PCT_PLUSMINUS (defensa por tracking)
-----------------------------------------------------------------
`leaguedashptdefend` (Second Spectrum, `data_pipeline.fetch_league_pt_defend_stats`,
disponible desde 2013-14): PCT_PLUSMINUS = % de tiro REAL del rival
cuando este jugador es el defensor más cercano, menos el % de tiro
NORMAL de esos mismos rivales -- negativo significa que el rival tira
PEOR de lo normal defendido por este jugador. A diferencia de los
hustle stats (`scripts/experiments/hustle_stats_signal.py`, CONTESTED_SHOTS/
DEFLECTIONS/etc., investigados y DESCARTADOS -- miden actividad, no si
esa actividad de verdad impide anotar), esta es la señal de impacto
defensivo más directa que expone `nba_api`.

Investigado en `scripts/experiments/pt_defend_signal.py`. BUG REAL
encontrado y corregido ANTES de reportar el resultado: la primera
versión usaba el PCT_PLUSMINUS de la MISMA temporada que se predecía
(R² inflado a 0.69, casi tautológico). Reescrito para usar solo la
temporada PREVIA de cada jugador (mismo patrón de no-look-ahead que
`compute_recency_weighted_advanced` ya aplicaba a PIE/NET_RATING).
Resultado corregido, validado leave-one-season-out (16 pliegues, 480
casos -- con PCT_PLUSMINUS=NaN para los ~120 casos de temporadas
anteriores a 2013-14, que caen a Game Score+NET_RATING solos, ver más
abajo): R² fuera de muestra 0.512 → 0.528, MAE 2.65 → 2.61 puntos/partido,
mejora real pero modesta.

**Peso derivado con la MISMA técnica que `net_rating_weight`** (ratio de
coeficientes de una regresión conjunta, no una nueva escala inventada):
regresión de DiffPointsPG contra el composite existente (GS+NET_RATING,
ya en unidades de Game Score de equipo) + PCT_PLUSMINUS de equipo da
coeficientes 0.1588 y -60.40 respectivamente -- ratio -380.2 (cuántos
puntos de Game Score de EQUIPO vale 1 punto de PCT_PLUSMINUS), pasado a
por-36-por-jugador: -380.2 * 36/240 = **-57.03**. Signo negativo a
propósito: un defensor bueno tiene PCT_PLUSMINUS más NEGATIVO que la
media, y el ajuste debe ser POSITIVO para él.

**`game_score_to_net_rating_scale` RE-CALIBRADO de 0.172 a 0.1617** al
integrar esta tercera métrica -- el composite cambia (más dispersión),
así que la escala que lo convierte a puntos de diferencial también tiene
que cambiar. Validado LOSO igual que el resto: el scale recalibrado
sale estable en los 16 pliegues (0.159-0.165), escala final sobre los
480 casos completos: 0.1617.

**Tolerancia a métricas parciales (cambio real en
`compute_league_advanced_baselines`/`compute_recency_weighted_advanced`):**
antes de esto, PIE y NET_RATING SIEMPRE coexistían (mismo endpoint, misma
fila) así que un `groupby` que asumiera "todas las métricas presentes en
todas las filas" nunca fallaba en silencio. PCT_PLUSMINUS rompe esa
asunción -- temporadas anteriores a 2013-14, o jugadores sin cobertura de
tracking, tienen esa columna en NaN mientras PIE/NET_RATING sí existen.
Sin arreglar esto, una media ponderada ingenua (`(col*peso).sum()/peso.sum()`)
habría diluido el denominador con el peso de filas SIN dato para esa
métrica, sesgando la media hacia 0 en silencio. Ambas funciones ahora
calculan la media ponderada de CADA métrica sobre sus propias filas no
nulas, no sobre el grupo entero -- si un jugador/temporada no tiene
PCT_PLUSMINUS, simplemente no participa en ESA métrica (PIE/NET_RATING
no se ven afectados), y si NINGUNA temporada reciente de un jugador
tiene el dato, `PCT_PLUSMINUS` se omite del dict devuelto (no se inventa
un 0.0 -- 0.0 significaría "defensor exactamente promedio", un valor,
no una ausencia de dato).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from season_utils import season_start_year

# Pesos por defecto, en "puntos de Game Score/36 por unidad de desviación
# de la métrica respecto a la media de su temporada".
#
# net_rating_weight = 0.42 sale de la regresión sobre los 480 casos:
# coeficientes 0.2127 (Game Score de equipo) y 0.5956 (NET_RATING medio
# del roster) contra el diferencial real, o sea 2.80 puntos de Game Score
# de EQUIPO por punto de NET_RATING; pasado a por-36 por jugador,
# 2.80 * 36/240 = 0.42. No está encogido a mano -- ver el caveat del
# docstring, donde se explica por qué NO se encogió.
#
# pie_weight = 0.0 a propósito: medido, no aporta nada fuera de muestra y
# con NET_RATING presente su coeficiente sale con el signo al revés. El
# cableado se conserva para poder re-probarlo.
DEFAULT_ADVANCED_IMPACT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "pie_weight": 0.0,
    "net_rating_weight": 0.42,
    # Derivado por la misma técnica que net_rating_weight (ratio de
    # coeficientes de una regresión conjunta contra DiffPointsPG real,
    # ver "TERCERA MÉTRICA INTEGRADA" en el docstring del módulo) --
    # negativo a propósito: un defensor bueno tiene PCT_PLUSMINUS más
    # negativo que la media, y el ajuste debe ser positivo para él.
    "pct_plusminus_weight": -57.03,
    # Recencia propia, alineada con aging_curve por defecto.
    "n_seasons_lookback": 3,
    "recency_half_life_seasons": 1.5,
    # Un jugador con muy pocos minutos tiene NET_RATING/PIE ruidosísimos
    # (un suplente con 40 minutos en toda la temporada puede salir +30).
    # Por debajo de este umbral se ignora su fila avanzada y se cae al
    # Game Score puro para ese jugador.
    "min_minutes_for_advanced": 500,
}

ADVANCED_METRICS = ("PIE", "NET_RATING", "PCT_PLUSMINUS")

# Nombre de la columna de minutos TOTALES que este módulo usa como peso y
# como umbral de muestra. No es la columna MIN cruda del endpoint: con
# measure_type="Advanced", `leaguedashplayerstats` devuelve MIN por
# PARTIDO (ver el aviso en data_pipeline.fetch_league_advanced_player_stats),
# así que `load_advanced_stats` la deriva como MIN * GP.
TOTAL_MINUTES_COLUMN = "total_minutes"


def resolve_advanced_impact_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Mezcla `config["advanced_impact"]` sobre los valores por defecto."""
    resolved = dict(DEFAULT_ADVANCED_IMPACT_CONFIG)
    resolved.update(config.get("advanced_impact") or {})
    return resolved


def compute_league_advanced_baselines(
    advanced_stats: pd.DataFrame, min_minutes: int = 500
) -> Dict[str, Dict[str, float]]:
    """
    Media de liga de cada métrica avanzada POR TEMPORADA, ponderada por
    minutos. Devuelve {season: {"PIE": x, "NET_RATING": y}}.

    Es el punto de referencia contra el que se centra cada jugador (ver
    docstring del módulo, propiedades 1 y 2). Ponderar por minutos y no
    contar a todo el que pisó la cancha es lo mismo que hace
    `aging_curve.compute_league_game_score_baseline` -- se busca la media
    del jugador de ROTACIÓN, no la del último del banquillo.

    `advanced_stats` debe tener el esquema de
    `league_advanced_player_stats.csv` (season, PLAYER_ID, MIN + las
    columnas avanzadas), opcionalmente con `PCT_PLUSMINUS` fusionada por
    `merge_pt_defend_stats` -- NaN para temporadas sin datos de tracking
    (anteriores a 2013-14). Cada métrica se promedia sobre sus PROPIAS
    filas no nulas, no sobre el grupo entero: promediar con un
    denominador que incluye el peso de filas sin dato para esa métrica
    sesgaría la media hacia 0 en silencio (ver "TERCERA MÉTRICA
    INTEGRADA" en el docstring del módulo -- PIE/NET_RATING nunca tenían
    este problema porque vienen del mismo endpoint y siempre coexisten).
    """
    if advanced_stats.empty:
        return {}

    eligible = advanced_stats[advanced_stats[TOTAL_MINUTES_COLUMN] >= min_minutes]
    baselines: Dict[str, Dict[str, float]] = {}
    for season, group in eligible.groupby("season"):
        season_baseline: Dict[str, float] = {}
        for metric in ADVANCED_METRICS:
            if metric not in group.columns:
                continue
            metric_rows = group[group[metric].notna()]
            weights = metric_rows[TOTAL_MINUTES_COLUMN]
            if weights.sum() <= 0:
                continue
            season_baseline[metric] = float((metric_rows[metric] * weights).sum() / weights.sum())
        if season_baseline:
            baselines[str(season)] = season_baseline
    return baselines


def compute_recency_weighted_advanced(
    player_advanced: pd.DataFrame,
    target_season: str,
    n_seasons: int = 3,
    half_life: float = 1.5,
    min_minutes: int = 500,
) -> Optional[Dict[str, float]]:
    """
    Media ponderada por recencia de las métricas avanzadas de UN jugador
    sobre sus N temporadas más recientes ANTERIORES a `target_season`.

    REGLA DE NO LOOK-AHEAD: filtra estrictamente a temporadas previas,
    igual que `backtesting.filter_seasons_before`. Devuelve None si el
    jugador no tiene ninguna temporada elegible (rookie, o solo
    temporadas por debajo de `min_minutes`) -- el llamante debe caer al
    Game Score puro en ese caso, no inventar un valor.

    Un jugador traspasado a mitad de temporada aparece con una fila por
    equipo; se colapsan a una sola ponderando por minutos, que es la
    lectura correcta de "cómo le fue esa temporada" (`dedupe_traded_seasons`
    de season_utils no sirve aquí: ese CSV no trae fila 'TOT').

    Cada métrica se promedia sobre las temporadas donde REALMENTE hay
    dato para ella, no sobre las `n_seasons` completas -- un jugador cuyo
    historial reciente cae antes de 2013-14 (o sin cobertura de tracking)
    simplemente no lleva `PCT_PLUSMINUS` en el dict devuelto, en vez de
    heredar un 0.0 inventado (0.0 significaría "defensor exactamente
    promedio", no "sin dato"). PIE/NET_RATING no se ven afectados por
    esto -- siempre coexisten, vienen del mismo endpoint.
    """
    if player_advanced.empty:
        return None

    target_year = season_start_year(target_season)
    eligible = player_advanced[
        (player_advanced["season"].apply(season_start_year) < target_year)
        & (player_advanced[TOTAL_MINUTES_COLUMN] >= min_minutes)
    ].copy()
    if eligible.empty:
        return None

    # Colapsar temporadas partidas por traspaso (varias filas, un año).
    # Cada métrica ponderada solo sobre SUS PROPIAS filas no nulas -- ver
    # el mismo razonamiento en compute_league_advanced_baselines.
    eligible["_year"] = eligible["season"].apply(season_start_year)
    collapsed = []
    for year, group in eligible.groupby("_year"):
        row: Dict[str, float] = {"_year": year}
        for metric in ADVANCED_METRICS:
            if metric not in group.columns:
                continue
            metric_rows = group[group[metric].notna()]
            metric_weights = metric_rows[TOTAL_MINUTES_COLUMN]
            if metric_weights.sum() > 0:
                row[metric] = float((metric_rows[metric] * metric_weights).sum() / metric_weights.sum())
        collapsed.append(row)

    recent = sorted(collapsed, key=lambda r: r["_year"], reverse=True)[:n_seasons]
    seasons_ago = {r["_year"]: target_year - r["_year"] for r in recent}
    result: Dict[str, float] = {}
    for metric in ADVANCED_METRICS:
        rows_with_metric = [r for r in recent if metric in r]
        if not rows_with_metric:
            continue  # ninguna temporada reciente tiene este dato -- no se inventa un valor
        weights = np.array([0.5 ** (seasons_ago[r["_year"]] / half_life) for r in rows_with_metric])
        total = weights.sum()
        if total <= 0:
            continue
        result[metric] = float(sum(r[metric] * w for r, w in zip(rows_with_metric, weights)) / total)
    return result if result else None


def blend_impact_per36(
    game_score_per36: float,
    advanced: Optional[Dict[str, float]],
    season_baseline: Optional[Dict[str, float]],
    impact_config: Dict[str, Any],
) -> float:
    """
    Aplica el ajuste aditivo del docstring del módulo a UN jugador.

    Devuelve `game_score_per36` sin tocar si el ajuste está desactivado,
    si el jugador no tiene métricas avanzadas utilizables, o si falta la
    línea base de su temporada -- degradar al Game Score puro es siempre
    seguro (es lo que el modelo hacía antes), inventar un ajuste no.
    """
    if not impact_config.get("enabled", True) or not advanced or not season_baseline:
        return float(game_score_per36)

    adjustment = 0.0
    for metric, weight_key in (
        ("PIE", "pie_weight"),
        ("NET_RATING", "net_rating_weight"),
        ("PCT_PLUSMINUS", "pct_plusminus_weight"),
    ):
        if metric in advanced and metric in season_baseline:
            adjustment += impact_config[weight_key] * (advanced[metric] - season_baseline[metric])
    return float(game_score_per36 + adjustment)


def adjusted_game_score_per36(
    game_score_per36: float,
    player_advanced: Optional[pd.DataFrame],
    target_season: str,
    baselines: Dict[str, Dict[str, float]],
    impact_config: Dict[str, Any],
) -> float:
    """
    Punto de entrada ÚNICO para los tres motores del proyecto
    (`backtesting.project_historical_player`,
    `aging_curve.build_aging_projection_dataset`,
    `league_simulation.project_team_roster`): encadena
    `compute_recency_weighted_advanced` + `blend_impact_per36` para un
    jugador.

    Existe como función compartida a propósito. Este proyecto ya arrastró
    el mismo bug de normalización de minutos en dos módulos por tenerlo
    duplicado (ver `simulation.normalize_rotation_minutes`); una métrica
    de impacto que se calcule distinto en el backtest que en la
    simulación invalidaría la calibración del backtest en silencio.

    `target_season` es la temporada que se PROYECTA -- las métricas se
    toman solo de temporadas anteriores (no look-ahead).
    """
    if not impact_config.get("enabled", True) or player_advanced is None or player_advanced.empty:
        return float(game_score_per36)

    advanced = compute_recency_weighted_advanced(
        player_advanced,
        target_season,
        n_seasons=impact_config["n_seasons_lookback"],
        half_life=impact_config["recency_half_life_seasons"],
        min_minutes=impact_config["min_minutes_for_advanced"],
    )
    # La línea base de referencia es la de la temporada ANTERIOR a la
    # proyectada: es de donde vienen las métricas del jugador, así que es
    # contra esa población contra la que hay que centrarlas.
    prior_season = _previous_season(target_season)
    return blend_impact_per36(
        game_score_per36, advanced, baselines.get(prior_season), impact_config
    )


def _previous_season(season: str) -> str:
    """'2026-27' -> '2025-26'. Duplicado a propósito de data_pipeline: este
    módulo no debe importar la capa de ingesta (dependencia al revés)."""
    start_year = season_start_year(season) - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def build_advanced_context(
    advanced_stats: pd.DataFrame, config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Empaqueta todo lo que hace falta para ajustar jugadores: las filas
    avanzadas ya AGRUPADAS por PLAYER_ID, las líneas base por temporada y
    la config resuelta. Devuelve None si el ajuste está desactivado o no
    hay datos -- los llamantes tratan None como "usa Game Score puro".

    El groupby precalculado no es cosmético: el backtest sweep proyecta
    ~7.000 jugador-caso, y filtrar el DataFrame completo (>9.000 filas)
    una vez por jugador convertía la pasada en minutos. Agrupar una sola
    vez lo deja en un lookup de diccionario.
    """
    impact_config = resolve_advanced_impact_config(config)
    if not impact_config.get("enabled", True) or advanced_stats is None or advanced_stats.empty:
        return None
    return {
        "by_player": {int(pid): group for pid, group in advanced_stats.groupby("PLAYER_ID")},
        "baselines": compute_league_advanced_baselines(
            advanced_stats, min_minutes=impact_config["min_minutes_for_advanced"]
        ),
        "impact_config": impact_config,
    }


def adjust_with_context(
    game_score_per36: float,
    player_id: int,
    target_season: str,
    context: Optional[Dict[str, Any]],
) -> float:
    """
    Aplica el ajuste a un jugador usando el contexto de
    `build_advanced_context`. Es la firma que usan los tres motores; con
    `context=None` devuelve el Game Score sin tocar.
    """
    if context is None:
        return float(game_score_per36)
    return adjusted_game_score_per36(
        game_score_per36,
        context["by_player"].get(int(player_id)),
        target_season,
        context["baselines"],
        context["impact_config"],
    )


def load_advanced_stats(processed_dir) -> pd.DataFrame:
    """
    Lee `league_advanced_player_stats.csv` (lo genera
    `data_pipeline.build_league_advanced_player_stats_dataset`). Devuelve
    un DataFrame VACÍO si no existe -- todo el proyecto sigue corriendo
    sin él, simplemente con Game Score puro. Es un dataset opcional a
    propósito: quien clone el repo no debería necesitar 19 llamadas más a
    la API para que la simulación funcione.

    Si `league_pt_defend_stats.csv` también existe (opcional, ver
    `data_pipeline.build_league_pt_defend_dataset`), le fusiona
    `PCT_PLUSMINUS` -- ver `merge_pt_defend_stats`. Su ausencia es
    igual de segura: el proyecto sigue con PIE/NET_RATING solos.
    """
    path = processed_dir / "league_advanced_player_stats.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["season"] = df["season"].astype(str)
    df = prepare_advanced_stats(df)

    pt_defend_path = processed_dir / "league_pt_defend_stats.csv"
    if pt_defend_path.exists() and pt_defend_path.stat().st_size > 0:
        df = merge_pt_defend_stats(df, pd.read_csv(pt_defend_path))
    return df


def prepare_advanced_stats(advanced_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Añade `total_minutes` = MIN * GP. Separada de `load_advanced_stats`
    para que los tests puedan construir un DataFrame en memoria y pasarlo
    por el mismo camino que el CSV real, sin tocar disco.
    """
    if advanced_stats.empty:
        return advanced_stats
    df = advanced_stats.copy()
    df[TOTAL_MINUTES_COLUMN] = df["MIN"] * df["GP"]
    return df


def merge_pt_defend_stats(advanced_stats: pd.DataFrame, pt_defend_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la columna `PCT_PLUSMINUS` (defensa por tracking, ver
    `data_pipeline.fetch_league_pt_defend_stats`) a `advanced_stats`,
    cruzando por (PLAYER_ID, season). Colapsa posibles filas duplicadas
    de un jugador traspasado a mitad de temporada ponderando por D_FGA
    (tiros defendidos de cerca -- este endpoint no trae minutos).

    Deja `PCT_PLUSMINUS` en NaN para jugador/temporada sin datos de
    tracking (anteriores a 2013-14, o sin cobertura) -- separada a
    propósito de `prepare_advanced_stats` para poder testear el cruce
    con DataFrames en memoria, mismo patrón.
    """
    if advanced_stats.empty or pt_defend_stats is None or pt_defend_stats.empty:
        return advanced_stats

    pt = pt_defend_stats[pt_defend_stats["D_FGA"] > 0].copy()
    if pt.empty:
        return advanced_stats
    pt["season"] = pt["season"].astype(str)
    pt["_weighted"] = pt["PCT_PLUSMINUS"] * pt["D_FGA"]
    collapsed = (
        pt.groupby(["PLAYER_ID", "season"])
        .apply(lambda g: g["_weighted"].sum() / g["D_FGA"].sum(), include_groups=False)
        .rename("PCT_PLUSMINUS")
        .reset_index()
    )
    return advanced_stats.merge(collapsed, on=["PLAYER_ID", "season"], how="left")

"""
simulation.py

Motor de simulación Monte Carlo: la pieza final que consume las 6 señales
de `src/context/` (injury_model, fatigue_accumulation, schedule_strength,
performance_curve, opponent_weighting, conference_adjustment) más la
proyección individual (`aging_curve.py`) para simular `n_seasons`
temporadas hipotéticas del equipo del config.

LIMITACIÓN DE DATOS / DECISIÓN DE DISEÑO IMPORTANTE
------------------------------------------------------
El calendario real de `config["team"]["season"]` puede no existir
todavía (ver `schedule_strength.py` -- a la fecha de escribir esto,
`team_schedule.csv` solo tenía 2 partidos de pretemporada para 2026-27).
No se puede simular contra un calendario que no existe. En su lugar, cada
partido simulado usa un **calendario sintético representativo**:

- Rival: se muestrea el WinPCT de un rival al azar de la distribución
  real de WinPCT de TODA la liga (`prior_season_standings.csv`), en vez
  de un rival específico en una fecha específica.
- Back-to-back: se sortea con una probabilidad configurable
  (`b2b_probability`, 0.18 por defecto -- aproximación al ~15-20% de
  partidos en back-to-back típico de un calendario NBA moderno de 82
  partidos).

Cuando la NBA publique el calendario completo, `team_schedule.csv` tendrá
82 filas reales y este módulo se puede adaptar para leerlo directamente
en vez de muestrear -- la interfaz (un array de opponent_win_pct /
is_back_to_back por partido) no cambia.

MECÁNICA POR TEMPORADA SIMULADA
----------------------------------
1. **Disponibilidad (lesiones)** -- por jugador, se sortea el nº de
   partidos perdidos esa temporada de una binomial negativa con media
   `risk_score * games_per_season` (de injury_model.py) y se agrupan en
   UN tramo contiguo -- las lesiones reales son una racha, no partidos
   sueltos al azar.
2. **Contribución por partido** -- Game Score por-36 de cada jugador
   (de aging_curve.py) escalado a sus minutos proyectados, con:
   - **prima de "estrella"** opcional y DESACTIVADA por defecto
     (`apply_star_bonus`, ver su docstring y la sección "INVESTIGACIÓN"
     más abajo -- se probó y se descartó como comportamiento por
     defecto, queda como palanca de experimentación).
   - penalización de fatiga en back-to-backs, proporcional a su
     `fatigue_score` (de fatigue_accumulation.py).
   - desgaste progresivo a lo largo de la temporada para jugadores de
     fatiga alta (proporcional al avance de temporada).
   - ruido de partido a partido (varianza natural de rendimiento).
3. **Resultado del partido** -- Game Score de equipo (suma de jugadores
   disponibles) menos un ajuste por fuerza de rival, convertido a
   probabilidad de victoria vía función logística, y sorteado como
   victoria/derrota (Bernoulli). La escala de conversión Game Score ->
   diferencial de puntos y la escala de varianza del resultado son
   aproximaciones documentadas y configurables (`config["monte_carlo"]`)
   -- no hay datos suficientes en este proyecto para calibrarlas con
   una regresión propia.

INVESTIGACIÓN: EQUIPOS CON ESTRELLAS PERO BANQUILLO MEDIOCRE, ¿SUBVALORADOS?
------------------------------------------------------------------------------
Investigado a raíz de que los Knicks (campeones vigentes según el
usuario) proyectaban por debajo de la media de la liga mientras equipos
"en reconstrucción" como Chicago o New Orleans proyectaban arriba.
Diagnóstico con datos reales: Brunson (19.4 GS/36) y Towns (19.0) SÍ se
proyectan como élite -- más altos que cualquier jugador individual de
Chicago -- pero el resto de la rotación de los Knicks (Anunoby, Bridges,
Hart, Shamet: 9.6-13.4) es más floja que la rotación completa de Chicago
(14-18, sin estrellas pero pareja). Se probó `apply_star_bonus()`: los
`star_bonus_top_n` jugadores de mayor valor de temporada de CADA equipo
reciben una prima multiplicativa (`star_bonus_multiplier`) sobre su Game
Score/36 -- la idea de que una estrella eleva las victorias de forma no
lineal (atención defensiva, creación de tiro para otros, cierre de
partidos) es real y está en la literatura de análisis deportivo.

DOS PROBLEMAS AL PROBARLO CON DATOS REALES, POR LOS QUE SE DEJÓ
DESACTIVADA POR DEFECTO (`star_bonus_top_n=0`):

1. **No resuelve el caso que la motivó.** Chicago y New Orleans TAMBIÉN
   tienen su propio "mejor jugador" (Giddey 18.4, Zion 23.5) que recibe
   la misma prima proporcional -- el hueco Knicks-vs-Chicago apenas se
   mueve (con top2/x1.15: 94.68→99.40 vs 100.01→104.31, la diferencia
   pasa de 5.33 a 4.91). Boostear al mejor jugador de CADA equipo no
   favorece de forma diferencial a los equipos "estrella + banquillo
   flojo" frente a los de "sin estrella + banquillo parejo".
2. **Empeora el backtesting** contra los propios comparables históricos
   de este proyecto (Heat 2010-11, Warriors 2016-17, Nets 2020-21, Suns
   2022-23) -- que son, precisamente, equipos de estrellas apiladas.
   `actual_percentile` ya era bajo SIN la prima (todos estos casos caen
   por debajo del percentil 10, ver sección de Backtesting en README:
   "fricción de vestuario no capturada por datos de caja"), y con la
   prima activa cae más todavía (ej. Suns 2022-23: percentil 0.05 → 0.03,
   Heat 2010-11: 7.67 → 2.83). Es decir: los datos de ESTE MISMO proyecto
   ya muestran que apilar estrellas suele rendir POR DEBAJO de la suma
   lineal de talento (el fenómeno real de "fricción de superequipo") --
   añadir una prima que hace las estrellas más valiosas todavía va en la
   dirección CONTRARIA a esa evidencia ya validada.

CONCLUSIÓN: el resultado "Knicks bajo, Chicago/Pelicans alto" no se
identificó como un bug -- es lo que salen los números reales más
recientes de cada jugador (los titulares complementarios de los Knicks
tuvieron una temporada más floja en tasa por-36 que la rotación completa
de Chicago/Pelicans) combinado con una limitación ya documentada del
proyecto (suma lineal de Game Score, sin capturar defensa real ni
encaje de plantilla más allá de `lineup_synergy.py`). `apply_star_bonus()`
se deja implementada y testeada, pero DESACTIVADA por defecto -- útil
como palanca de experimentación (`config["monte_carlo"]`), no como
arreglo por defecto, porque no se sostiene contra los propios datos de
validación del proyecto.

BUG REAL: LÍNEA BASE GENÉRICA INFLABA LAS VICTORIAS DEL EQUIPO PROPIO
------------------------------------------------------------------------
Encontrado al comparar, para el MISMO roster (ya arreglado el mismatch
de league_simulation.py -- ver "BUG REAL" en su docstring), las
victorias medias de "Mi equipo" (motor de este módulo: 50.4) contra las
de "Liga y Playoffs" (motor de league_simulation.py: 37.7) -- una
diferencia de 13 victorias con exactamente los mismos jugadores/minutos.
Causa: este módulo compara el Game Score del equipo propio contra una
línea base GENÉRICA fija (`league_average_game_score_per36=10.0`, el
promedio de Hollinger sobre TODO jugador de la liga, incluida la
"basura" de banquillo que casi no juega -- equipo "promedio" ~66.7 de
Game Score total), mientras que league_simulation.py compara contra el
Game Score REAL de los otros 29 equipos (rotaciones reales de ~10
jugadores, media empírica ~97-98 -- mucho más alta, porque excluye a los
jugadores que casi no juegan). Comparar contra un rival ficticio mucho
más flojo que los rivales reales infla las victorias proyectadas de
CUALQUIER equipo propio, no solo de este caso.

ARREGLADO con `compute_league_average_game_score_per36()`: si
`league_player_projections.csv` existe (se generó `--league`),
`build_simulation_dataset()` recalibra automáticamente la línea base
desde el promedio REAL de los 30 equipos en vez del valor genérico de
Hollinger -- a menos que el usuario haya fijado su propio valor a mano
en `config["monte_carlo"]`, que se respeta. Sin datos de liga completa
(pipeline sin `--league`), se mantiene el valor genérico como fallback
-- sigue siendo una aproximación razonable cuando no hay nada mejor.

SOBRECORRECCIÓN en la primera versión de este fix, encontrada al
probarlo: calibrar con la contribución "de pico" (game_score_per36 *
minutos, sin más) de los 30 equipos dio una línea base de ~97.4 -- y las
victorias del equipo propio se DESPLOMARON a 19.7 (peor que antes de
arreglar nada). Motivo: `sample_injury_absences()` sí resta partidos
perdidos según `risk_score` en la simulación real del equipo propio,
pero la línea base "de pico" no reflejaba esa misma degradación para
los 30 equipos -- comparar un equipo YA penalizado por lesiones contra
una línea base que asume salud perfecta en TODA la liga penaliza de
más, sobre todo a rosters con jugadores de riesgo alto (LeBron viejo,
Embiid propenso a lesión, exactamente el caso que motivó este fix).
Corregido descontando cada contribución por `(1 - risk_score)` -- ver
docstring de `compute_league_average_game_score_per36()` para por qué
ese descuento es exacto en expectativa, no una aproximación arbitraria.
Con el descuento, la línea base calibrada baja a ~70.2 (cerca del valor
genérico de 66.7) y las victorias del equipo propio quedan en ~46.7 --
mucho más cerca de las 37.7 de "Liga y Playoffs" que las 50.4
originales (gap de 12.7 victorias reducido a ~9).

GAP RESIDUAL, honesto: ~9 victorias de diferencia SIGUEN sin cerrarse
del todo, y no se puede cerrar sin unificar los dos motores por
completo (fuera de alcance) -- "Mi equipo" sigue comparando contra un
rival PROXY (un escalar de WinPCT muestreado de la temporada anterior,
convertido a puntos vía `opponent_strength_scale`), mientras que "Liga
y Playoffs" compara contra el Game Score REAL y específico de cada uno
de los 29 rivales simulados esa temporada -- una medida más granular y
correlacionada que un proxy escalar. Es una diferencia de diseño
documentada desde el principio en los docstrings de ambos módulos, no
algo que este fix pretenda eliminar del todo.

NO se aplica a `backtesting.py`: los comparables históricos son de
temporadas pasadas para las que no existe un `league_player_projections.csv`
equivalente (sería una liga completa por cada temporada histórica,
fuera del alcance actual) -- queda con la línea base genérica,
documentado como limitación conocida. (ACTUALIZACIÓN: sí se resolvió
después, con el backtest sweep de 450 casos -- ver el bug de inflación
de era justo abajo.)

BUG REAL: INFLACIÓN DE ERA (el sesgo dominante del modelo)
------------------------------------------------------------------------
Encontrado al correr el backtest sweep de 450 casos (30 equipos x 15
temporadas, 2010-11 a 2024-25) -- el dataset que por fin permitió medir
la calibración con rigor en vez de con 4 casos elegidos a mano. Los
resultados iniciales fueron malísimos y con un patrón inconfundible:

    % de casos dentro del rango P10-P90:  36.7%  (deberia ser ~80%)
    percentil real MEDIANO:                4.5   (deberia ser ~50)
    error medio (real - predicho):       -13.2 victorias

Pero lo revelador fue el desglose por temporada: el modelo predecía una
MEDIA de 49.8 victorias por equipo en 2010-11 y de 66.0 en 2024-25.
Ambas son imposibles: en una liga real de 30 equipos cada victoria es la
derrota de otro, así que la media SIEMPRE es exactamente 41 en 82
partidos. El modelo violaba la restricción de suma cero, y la violación
CRECÍA monotónicamente con el tiempo (+5.8 en 2012-13 -> +25.0 en
2024-25).

CAUSA: el nivel de Game Score de la NBA no es estable en el tiempo. Subió
de ~10.7 por-36 en 2010-11 a ~13.4 en 2024-25 (más ritmo de juego y
revolución del triple). La línea base era FIJA (10.0 = "jugador promedio
de Hollinger"), así que un equipo MEDIO de 2024-25 aparecía +22 puntos de
Game Score por encima de la referencia -- mérito de su época, no suyo. La
correlación entre "inflación de Game Score de la era" y "exceso de
victorias predichas" es **0.926**: el sesgo estaba casi enteramente
explicado por esto.

ARREGLADO con `aging_curve.compute_league_game_score_baseline()`: cada
equipo se compara contra la media de SU PROPIA temporada, no contra una
constante. `backtesting.run_backtest_case()` acepta
`league_baseline_per36` y `build_backtest_sweep_dataset()` lo calcula por
temporada desde los ~350 jugadores/temporada con >=500 minutos del sweep
(que sí son una muestra representativa de la liga; los 4 comparables
narrativos NO lo son -- son 60 jugadores de 4 superequipos --, por eso
ese camino sigue usando el valor genérico).

BUG REAL: MINUTOS DEL ROSTER SIN NORMALIZAR EN EL BACKTEST
------------------------------------------------------------------------
Tercer bug del mismo análisis, y el MISMO que ya se había arreglado en
`league_simulation.project_team_roster` -- pero `backtesting.py` nunca lo
recibió. Sumaba el Game Score de TODO el roster (14-18 jugadores,
283-345 min/partido) en vez de los 240 reales, inflando la fuerza de cada
equipo un 18-43%. Arreglado extrayendo la lógica a
`normalize_rotation_minutes()` (en este módulo), ahora COMPARTIDA por
`backtesting.py` y `league_simulation.py` para que no pueda volver a
divergir -- que un bug idéntico viviera arreglado en un módulo y sin
arreglar en otro es exactamente lo que evita compartir la función.

BUG REAL: ESCALA GAME SCORE -> DIFERENCIAL (~3.5x de sobreestimación)
------------------------------------------------------------------------
`game_score_to_net_rating_scale` valía 1.0 con el comentario "1 punto de
Game Score sobre la línea base ~= 1 punto de diferencial" -- una
suposición que NUNCA se había verificado contra datos. Regresando el
diferencial de puntos REAL (de los game logs con PLUS_MINUS de los 450
casos) contra el Game Score de equipo proyectado, normalizado y relativo
a la media de su temporada, la pendiente empírica es **0.29**, no 1.0. El
modelo amplificaba las diferencias entre equipos ~3.5x, lo que además
explica la sobreconfianza de las distribuciones (net ratings simulados de
±20-30 puntos, cuando en la NBA real el rango es aproximadamente ±12).

RESULTADO DE LAS TRES CORRECCIONES (mismos 450 casos):
    % dentro de P10-P90:   36.7% -> 55.3%   (ideal ~80%)
    percentil MEDIANO:       4.5 -> 30.2    (ideal ~50)
    error medio:           -13.2 -> -3.5 victorias
    error absoluto medio:   15.0 -> 7.75 victorias
    correlación:           0.538 -> 0.690
Y el sesgo de era desapareció: el exceso de victorias pasó de crecer
(+5.8 en 2012-13 a +25.0 en 2024-25) a quedarse plano (+1.6 a +4.6).

REFERENCIAS EMPÍRICAS del mismo análisis, útiles para calibrar a futuro:
- 1 punto de diferencial real = 2.48 victorias en 82 partidos
  (r=0.966 entre diferencial y % de victorias -- el techo teórico de
  cualquier modelo que ya conozca el resultado de la temporada).
- El "baseline tonto" honesto (usar el diferencial de la temporada
  ANTERIOR para predecir la siguiente) da r=0.619 y MAE=7.39 victorias.
  Cualquier modelo de proyección que no supere eso no está aportando.
  El modelo ya lo supera en correlación (0.690) y queda a la par en error
  absoluto (7.75) -- antes de estas correcciones era peor en ambas.
- Escala logística REAL de un partido individual: K = 7.23 puntos de
  diferencial (ajustada sobre los partidos como visitante, para aislar
  la ventaja de campo). Es decir, +4 puntos de diferencial ~= 63.5% de
  probabilidad de ganar un partido.

BUG REAL: FALTABA LA VENTAJA DE CAMPO
------------------------------------------------------------------------
El modelo no modelaba la ventaja de campo EN NINGÚN SITIO. Medida sobre
los game logs reales de las 15 temporadas del sweep:
    temporada regular: +2.41 pts en casa (57.4% de victorias locales)
    playoffs:          +3.98 pts en casa (60.3% de victorias locales)
Añadida en `sample_schedule_context` (mitad de partidos en casa, mitad
fuera -- exactamente 41 y 41, no un sorteo) + `compute_game_net_rating_estimate`
para la temporada regular, y en `league_simulation.simulate_series` con
el formato real 2-2-1-1-1 (el mejor seed es local en los partidos 1, 2,
5 y 7) para los playoffs.

LIMITACIÓN CONOCIDA Y MEDIDA: SEÑAL/RUIDO DEL SEEDING
------------------------------------------------------------------------
Encontrada al validar contra la distribución real de seeds campeones (en
15 temporadas reales, el 60% de los campeones fueron seed 1 y NINGUNO
salió de un seed peor que el 3). El simulador reparte títulos a seeds 4+
un ~25% de las veces. La causa NO es que los partidos sean demasiado
aleatorios -- medido, son incluso más deterministas que la realidad
(escala efectiva 4.25 pts vs 7.23 real). La causa es el seeding:

                    talento (dif. entre equipos)   ruido (por temporada)   señal/ruido
    REAL                   11.27 victorias               4.53                 2.49
    MODELO                  6.52                         7.70                 0.85

El modelo comprime las diferencias de talento a la mitad y casi duplica
el ruido, así que en una temporada simulada cualquiera el seeding es casi
un sorteo -- y un seed 6 que en realidad es un top-3 gana el título sin
que eso sea "una sorpresa".

Las dos mitades tienen causas distintas y ninguna se arregla tocando un
parámetro:
- El talento comprimido es CORRECTO para predecir (la proyección regresa
  a la media, que es lo que minimiza el error -- MAE 7.75). El problema
  es usar esa estimación regresada COMO SI fuera el talento verdadero al
  simular: eso es confundir la distribución predictiva con la simulación
  "plug-in". Arreglarlo bien exige separar incertidumbre de estimación de
  ruido de temporada, un cambio arquitectónico.
- El ruido excesivo por temporada sí se podría bajar, pero estrecharía
  las bandas P10-P90 y EMPEORARÍA la métrica de calibración del
  backtesting (55.3% de casos reales dentro de P10-P90, ideal ~80%), que
  ya se queda corta.
Se deja documentado y medido en vez de forzar los parámetros para que el
resultado "se vea bien" -- que es exactamente el error que este proyecto
ya cometió una vez (ver la sección de Backtesting del README).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import get_paths  # noqa: E402
from lineup_synergy import build_synergy_matrix, compute_game_synergy_adjustment, compute_style_profile  # noqa: E402

TOTAL_TEAM_MINUTES_PER_GAME = 240.0  # 5 posiciones x 48 minutos -- constante del deporte, no del equipo

DEFAULT_MONTE_CARLO_CONFIG: Dict[str, float] = {
    "injury_dispersion": 2.0,  # binomial negativa: menor = más varianza en partidos perdidos
    "b2b_probability": 0.18,
    "b2b_fatigue_penalty": 0.15,  # % de reducción máxima en back-to-back para fatigue_score=1
    "season_fatigue_decay": 0.10,  # % de reducción máxima al final de temporada para fatigue_score=1
    "game_variance_std": 3.0,  # desviación estándar del ruido de Game Score por partido
    # Game Score de equipo NO es directamente un diferencial de puntos: hay
    # que restar la línea base de un equipo "promedio" antes de que la
    # diferencia se pueda leer como net rating. Este valor GENÉRICO (10.0,
    # el promedio de Hollinger sobre TODOS los jugadores de la liga,
    # incluida la basura de banquillo que apenas juega) es solo el
    # fallback cuando no hay datos reales de liga completa disponibles --
    # build_simulation_dataset() lo RECALIBRA automáticamente desde
    # league_player_projections.csv cuando existe (ver
    # compute_league_average_game_score_per36() más abajo y la sección
    # "BUG REAL" en el docstring del módulo) -- un equipo "promedio" de
    # rotación real (solo los ~10 jugadores que de verdad juegan minutos
    # relevantes, no todo jugador de la liga) rinde bastante más que 10
    # de Game Score/36.
    "league_average_game_score_per36": 10.0,
    # CALIBRADO EMPÍRICAMENTE sobre los 450 casos del backtest sweep (30
    # equipos x 15 temporadas): regresión del diferencial de puntos REAL
    # contra el Game Score de equipo proyectado (relativo a la media de su
    # propia temporada) da una pendiente de ~0.18, no 1.0. El valor
    # anterior (1.0, "1 punto de Game Score = 1 punto de diferencial") era
    # una suposición nunca verificada y sobreestimaba la ventaja de un
    # buen equipo por un factor de ~5.5 -- ver "BUG REAL: ESCALA
    # GAME SCORE -> DIFERENCIAL" en el docstring del módulo.
    # RE-CALIBRADO a 0.21 al integrar la métrica de impacto compuesta
    # (src/advanced_impact.py): el ajuste por NET_RATING ensancha la
    # dispersión del talento de equipo, así que cada punto de la métrica
    # vale menos diferencial que cuando era Game Score puro (0.2946
    # medido para el Game Score solo sobre los mismos 480 casos --
    # confirmación independiente del 0.29 anterior). Con
    # `advanced_impact: {enabled: false}` habría que volver a 0.29.
    #
    # RE-CALIBRADO otra vez a 0.172 (scripts/experiments/bayesian_calibration.py,
    # experimento explícitamente fuera de src/ -- ver su docstring): un
    # modelo bayesiano jerárquico sobre los mismos 480 casos, con
    # partial pooling del slope por temporada, da una media posterior de
    # 0.172 (94% HDI [0.157, 0.188], que deja fuera al 0.21 anterior por
    # un margen claro). Validado con leave-one-season-out (16 ajustes,
    # cada temporada predicha SIN haberla visto): el slope recalibrado
    # sale entre 0.169 y 0.175 en los 16 casos -- estable, no un
    # sobreajuste a los 480 casos completos -- y gana en R² fuera de
    # muestra a la constante anterior en 13 de 16 temporadas (R² global
    # 0.488 -> 0.512, error medio 2.70 -> 2.65 puntos/partido). La
    # jerarquía por temporada no aportó lo que se esperaba (las 16
    # pendientes salen casi idénticas entre sí -- el ajuste de era que ya
    # aplica `league_baseline_by_season` parece absorber toda esa
    # variación), así que el resultado se queda en una única constante
    # mejor calibrada, no en un slope por temporada.
    #
    # RE-CALIBRADO otra vez a 0.1617 al integrar PCT_PLUSMINUS (defensa
    # por tracking) en advanced_impact.py -- ver "TERCERA MÉTRICA
    # INTEGRADA" en su docstring. El composite cambia (más dispersión al
    # sumar una tercera señal), así que la escala que lo convierte a
    # puntos también cambia. Validado LOSO igual que la vez anterior: la
    # escala recalibrada sale estable en los 16 pliegues (0.159-0.165),
    # escala final sobre los 480 casos completos: 0.1617. R² fuera de
    # muestra 0.512 -> 0.528, MAE 2.65 -> 2.61 puntos/partido.
    "game_score_to_net_rating_scale": 0.1617,
    "opponent_strength_scale": 20.0,  # cuánto resta/suma un rival top/flojo (WinPCT 1.0 vs 0.0) al diferencial
    "outcome_variance_scale": 12.0,  # dispersión típica de resultado de un partido NBA individual
    # VENTAJA DE CAMPO, medida sobre los game logs reales de las 15
    # temporadas del backtest sweep (columna PLUS_MINUS, separando local
    # de visitante por la columna MATCHUP: "vs." = casa, "@" = fuera):
    #   temporada regular: +2.41 pts en casa (57.4% de victorias locales)
    #   playoffs:          +3.98 pts en casa (60.3% de victorias locales)
    # Son puntos de DIFERENCIAL, así que entran directamente al net rating
    # (mismas unidades), no hace falta escalarlos por
    # game_score_to_net_rating_scale. Ver "BUG REAL: FALTABA LA VENTAJA DE
    # CAMPO" en el docstring del módulo.
    "home_court_advantage": 2.41,
    "playoff_home_court_advantage": 3.98,
    # DESACTIVADA POR DEFECTO -- ver "BUG REAL" más abajo en este docstring
    # para por qué se probó y por qué NO se activó por defecto (empeora el
    # backtesting contra las superestrellas históricas sin corregir de
    # forma significativa el problema que motivó probarla). Sigue expuesta
    # como config opcional para quien quiera experimentar.
    "star_bonus_top_n": 0,  # nº de jugadores (por valor de temporada) que reciben la prima -- ver apply_star_bonus
    "star_bonus_multiplier": 1.0,  # multiplicador de Game Score/36 efectivo para esos jugadores
    # DESACTIVADA POR DEFECTO (0.0) -- ver "INCERTIDUMBRE DE CALIDAD DE
    # EQUIPO" en el docstring del módulo y compute_game_net_rating_estimate.
    # Puntos de diferencial (mismas unidades que home_court_advantage), NO
    # de Game Score -- se suma directamente al net rating ya escalado.
    "team_quality_uncertainty_std": 0.0,
}


def apply_star_bonus(
    game_score_per36: np.ndarray, minutes_projection: np.ndarray, mc_config: Dict[str, float]
) -> np.ndarray:
    """
    Prima de "estrella": multiplica el Game Score/36 efectivo de los
    `star_bonus_top_n` jugadores con más "valor de temporada"
    (game_score_per36 * minutos/36 -- misma métrica que
    src/awards_projection.py usa para MVP) por `star_bonus_multiplier`.

    DESACTIVADA POR DEFECTO (`star_bonus_top_n=0` en
    DEFAULT_MONTE_CARLO_CONFIG) -- ver "INVESTIGACIÓN" en el docstring del
    módulo. Se probó como posible arreglo para equipos "estrellas + banca
    floja" infravalorados frente a equipos "sin estrella + banca pareja",
    pero (a) no resuelve el caso que la motivó -- todos los equipos tienen
    SU propio mejor jugador, así que la prima proporcional apenas cambia
    la diferencia relativa -- y (b) empeora el backtesting contra los
    comparables históricos de este proyecto, que son precisamente equipos
    de estrellas apiladas que en la vida real rindieron POR DEBAJO de la
    suma de su talento ("fricción de superequipo", ya documentada en el
    README). Queda implementada y testeada como palanca de
    experimentación opcional en `config["monte_carlo"]`, no como
    comportamiento por defecto.
    """
    top_n = int(mc_config.get("star_bonus_top_n", 0))
    multiplier = mc_config.get("star_bonus_multiplier", 1.0)
    if top_n <= 0 or multiplier == 1.0 or len(game_score_per36) == 0:
        return game_score_per36

    season_value = game_score_per36 * minutes_projection / 36.0
    star_indices = np.argsort(-season_value)[:top_n]
    boosted = game_score_per36.copy()
    boosted[star_indices] = boosted[star_indices] * multiplier
    return boosted


DEFAULT_ROTATION_SIZE = 10  # tamaño de rotación real NBA -- ver normalize_rotation_minutes


def normalize_rotation_minutes(
    raw_minutes_by_player: Dict[Any, float], rotation_size: int = DEFAULT_ROTATION_SIZE
) -> Dict[Any, float]:
    """
    Escala los minutos/partido REALES de un roster para que la ROTACIÓN
    (los `rotation_size` jugadores con más minutos) sume exactamente
    TOTAL_TEAM_MINUTES_PER_GAME (240 = 5 posiciones x 48 min, lo único
    que existe de verdad en un partido). Los jugadores fuera de esa
    rotación quedan en 0.

    POR QUÉ (mismo bug encontrado dos veces, en dos módulos distintos):
    la suma de minutos/partido reales de un roster COMPLETO no suma 240 --
    a lo largo de 82 partidos rotan 14-20 jugadores, así que suma 280-345.
    Sumar el Game Score de todos ellos infla la fuerza del equipo un
    18-43% frente a un rival medido sobre 240 minutos, y penaliza
    especialmente a los equipos con mucho movimiento de plantilla
    (lesiones, tanking, two-way) que son los que más jugadores acumulan.
    Restringir la normalización a la rotación (en vez de escalar todo el
    roster) evita el efecto colateral de diluir a las estrellas -- ver el
    historial completo en el docstring de
    league_simulation.project_team_roster.

    Extraído aquí tras encontrar el MISMO bug en `backtesting.py`, que
    sumaba el roster entero sin normalizar mientras
    `league_simulation.py` ya lo tenía arreglado. Función compartida para
    que no pueda volver a divergir.
    """
    if not raw_minutes_by_player:
        return {}
    rotation_ids = sorted(raw_minutes_by_player, key=lambda pid: -raw_minutes_by_player[pid])[:rotation_size]
    total_rotation_minutes = sum(raw_minutes_by_player[pid] for pid in rotation_ids)
    if total_rotation_minutes <= 0:
        return {pid: 0.0 for pid in raw_minutes_by_player}

    scale = TOTAL_TEAM_MINUTES_PER_GAME / total_rotation_minutes
    rotation = set(rotation_ids)
    return {
        pid: (raw_minutes_by_player[pid] * scale if pid in rotation else 0.0)
        for pid in raw_minutes_by_player
    }


def load_league_mean_synergy(processed_dir: Path) -> float:
    """
    Media del ajuste de sinergia esperado de los 30 equipos, en puntos de
    net rating (de `league_team_synergy_baseline.csv`, que escribe
    league_simulation.py -- es el único módulo con las 30 matrices de
    sinergia a la vez). 0.0 si el archivo no existe, que deja el
    comportamiento anterior intacto.
    """
    path = processed_dir / "league_team_synergy_baseline.csv"
    if not path.exists() or path.stat().st_size == 0:
        return 0.0
    df = pd.read_csv(path)
    if df.empty or "expected_synergy_net_rating" not in df.columns:
        return 0.0
    return float(df["expected_synergy_net_rating"].mean())


def compute_league_average_game_score_per36(
    player_projections: pd.DataFrame,
    league_mean_synergy_net_rating: float = 0.0,
    game_score_to_net_rating_scale: float = DEFAULT_MONTE_CARLO_CONFIG["game_score_to_net_rating_scale"],
) -> float:
    """
    Recalibra `league_average_game_score_per36` desde los 30 equipos REALES
    ya proyectados (league_player_projections.csv, generado por
    league_simulation.py) en vez de usar el valor genérico de Hollinger
    (10.0 -- el promedio sobre TODO jugador de la liga, incluida la
    "basura" de banquillo que casi no juega). Ver "BUG REAL" en el
    docstring del módulo para el hallazgo que motivó esto: comparar el
    equipo propio contra esa línea base genérica (~66.7 de Game Score
    total) en vez de contra el nivel real de una rotación de 10 jugadores
    (~97-98 de media, empíricamente, porque solo cuenta minutos de
    rotación real, no cualquier minuto de la liga) infla las victorias
    proyectadas del equipo propio frente a lo que la liga real de 30
    equipos dice que debería sacar ese mismo roster.

    `player_projections` debe tener las columnas team_abbreviation,
    game_score_per36, minutes_projection (esquema de
    league_player_projections.csv). Devuelve la tasa por-36 equivalente
    al Game Score total medio de esos 30 equipos.

    OJO CON LA PRIMERA VERSIÓN DE ESTE FIX (sobrecorrección real,
    encontrada al probarlo): usar la contribución "de pico" (sin ajustar
    por disponibilidad) sobreestima la línea base, porque
    `sample_injury_absences()` sí resta partidos perdidos según
    `risk_score` en la simulación real -- comparar un equipo YA
    ajustado por lesiones (el propio) contra una línea base que asume
    salud perfecta en TODOS los 30 equipos penaliza de más,
    especialmente a rosters con jugadores de riesgo alto (LeBron viejo,
    Embiid propenso a lesión). Ver "BUG REAL" en el docstring del
    módulo. Por eso aquí se descuenta cada contribución por
    `(1 - risk_score)` -- la fracción de partidos que el modelo de
    lesiones espera que ese jugador esté disponible en promedio (la
    binomial negativa de `sample_injury_absences` tiene media
    `risk_score * games_per_season`, así que este descuento es exacto
    en expectativa, no una aproximación arbitraria). Si no hay columna
    `risk_score` (proyección sin ese dato), no se aplica descuento.
    """
    contribution = player_projections["game_score_per36"] * player_projections["minutes_projection"] / 36.0
    if "risk_score" in player_projections.columns:
        contribution = contribution * (1 - player_projections["risk_score"].clip(0, 1))

    team_totals = player_projections.assign(_contribution=contribution).groupby("team_abbreviation")["_contribution"].sum()
    average_team_total = float(team_totals.mean())

    # SEGUNDO término no centrado, además de las lesiones de arriba: el
    # ajuste de sinergia que run_monte_carlo suma al net rating es SIEMPRE
    # POSITIVO (medido: +4.4 a +11.9 sobre los 30 equipos). Si la línea
    # base no lo lleva incorporado, el equipo propio cobra ese bonus
    # "gratis" contra una referencia que no lo tiene. Mismo bug que en
    # backtesting.expected_team_game_score_equivalent, donde valía +5.0 de
    # net rating para el equipo promedio. Se divide por la escala porque
    # la simulación lo suma DESPUÉS de convertir a diferencial.
    if league_mean_synergy_net_rating:
        average_team_total += league_mean_synergy_net_rating / game_score_to_net_rating_scale

    return average_team_total / (TOTAL_TEAM_MINUTES_PER_GAME / 36.0)


def compute_expected_games_played(risk_scores: np.ndarray, games_per_season: int) -> np.ndarray:
    """
    Partidos jugados ESPERADOS en la temporada simulada, por jugador:
    games_per_season * (1 - risk_score).

    No es una aproximación -- es la media EXACTA de la binomial negativa
    que usa `sample_injury_absences` (`mean_missed = risk_score *
    games_per_season` es, por construcción de `p_param`, la media de la
    distribución que ahí se sortea). Se deja como fórmula analítica en vez
    de promediar muchas tiradas de Monte Carlo porque el resultado sería
    idéntico salvo ruido de muestreo -- no hace falta pagar ese coste para
    un valor que ya tiene forma cerrada exacta.

    Reemplaza, en el dashboard, al "partidos jugados en la temporada REAL
    más reciente" (dato histórico) por un valor que sí refleja la
    temporada que se está simulando -- ver DEFAULT_INJURY_TYPE_CATEGORIES
    más abajo para la ausencia de tipo de lesión real, mismo principio: se
    muestra lo que el modelo puede respaldar, no un dato prestado de otra
    temporada.
    """
    return games_per_season * (1 - np.clip(risk_scores, 0, 1))


def compute_expected_effective_minutes_per_game(
    minutes_projection: np.ndarray, risk_scores: np.ndarray
) -> np.ndarray:
    """
    Minutos por partido EFECTIVOS de la temporada simulada, por jugador:
    minutes_projection * (1 - risk_score). Es el promedio de minutos a lo
    largo de TODA la temporada, contando como 0 los partidos que se espera
    perder por lesión -- distinto de `minutes_projection` (los minutos
    ASUMIDOS los partidos en que sí juega, un input fijo, no una salida
    del modelo).

    Los minutos NO varían partido a partido en este modelo (ver
    `compute_player_contributions`: son siempre `minutes_projection`
    cuando el jugador está disponible, nunca menos) -- por eso este valor
    es simplemente minutes_projection escalado por la fracción de
    partidos que se espera jugar, no una media de una cantidad que de
    verdad fluctúe en la simulación.
    """
    return minutes_projection * (1 - np.clip(risk_scores, 0, 1))


def sample_injury_absences(
    risk_scores: np.ndarray,
    n_seasons: int,
    games_per_season: int,
    rng: np.random.Generator,
    dispersion: float,
) -> np.ndarray:
    """
    Devuelve un array booleano (n_seasons, games_per_season, n_players):
    True = jugador disponible ese partido esa temporada simulada. Cada
    jugador pierde, en cada temporada, un tramo CONTIGUO de partidos
    (no partidos sueltos al azar) de longitud sorteada de una binomial
    negativa con media risk_score * games_per_season.
    """
    n_players = len(risk_scores)
    available = np.ones((n_seasons, games_per_season, n_players), dtype=bool)
    day_idx = np.arange(games_per_season)

    for p in range(n_players):
        mean_missed = risk_scores[p] * games_per_season
        if mean_missed <= 0:
            continue
        p_param = dispersion / (dispersion + mean_missed)
        games_missed = rng.negative_binomial(dispersion, p_param, size=n_seasons)
        games_missed = np.clip(games_missed, 0, games_per_season)

        max_start = games_per_season - games_missed
        start = (rng.random(n_seasons) * (max_start + 1)).astype(int)

        absent = (day_idx[None, :] >= start[:, None]) & (day_idx[None, :] < (start + games_missed)[:, None])
        available[:, :, p] = ~absent

    return available


# Categorías ILUSTRATIVAS de tipo de lesión, derivadas únicamente de
# cuántos partidos SEGUIDOS falta un jugador en una racha de ausencia
# concreta. NO son un diagnóstico real: nba_api no expone qué lesión
# sufrió cada jugador (ver injury_model.py -- risk_score ya es un proxy
# estadístico, no datos médicos), así que no hay ninguna fuente de la que
# derivar el tipo real. Los puntos de corte son una estimación de este
# proyecto, no una tabla clínica -- expuestos como constante para que
# quede claro en el código y en la UI que es una etiqueta sintética.
DEFAULT_INJURY_TYPE_CATEGORIES: List[Dict[str, Any]] = [
    {"max_games": 3, "label": "Molestia menor (día a día)"},
    {"max_games": 10, "label": "Lesión leve-moderada"},
    {"max_games": 20, "label": "Lesión moderada"},
    {"max_games": None, "label": "Lesión significativa / baja prolongada"},  # None = sin tope
]


def categorize_injury_absence(
    games_missed: int, categories: List[Dict[str, Any]] = DEFAULT_INJURY_TYPE_CATEGORIES
) -> str:
    """Etiqueta ilustrativa (ver DEFAULT_INJURY_TYPE_CATEGORIES) para una
    racha de `games_missed` partidos seguidos."""
    for category in categories:
        if category["max_games"] is None or games_missed <= category["max_games"]:
            return category["label"]
    return categories[-1]["label"]


def _extract_absence_streaks(player_available: np.ndarray) -> List[Dict[str, int]]:
    """
    `player_available`: bool 1D (games_per_season,) de UN jugador en UNA
    temporada. Devuelve una lista de rachas contiguas de ausencia
    (`start_game` 1-indexado, `length`) -- el mismo supuesto de
    "las lesiones son una racha, no partidos sueltos" que ya usa
    sample_injury_absences, aplicado a una tirada concreta.
    """
    streaks = []
    absent = ~player_available
    games_per_season = len(absent)
    i = 0
    while i < games_per_season:
        if absent[i]:
            start = i
            while i < games_per_season and absent[i]:
                i += 1
            streaks.append({"start_game": start + 1, "length": i - start})
        else:
            i += 1
    return streaks


def simulate_single_season_player_log(
    player_ids: list,
    player_names: Dict[int, str],
    risk_scores: np.ndarray,
    games_per_season: int,
    mc_config: Dict[str, float],
    random_seed: int,
) -> pd.DataFrame:
    """
    Simula UNA temporada concreta (no la distribución agregada de
    n_seasons de run_monte_carlo) y devuelve, por jugador, partidos
    jugados/perdidos y el detalle de cada racha de ausencia con su
    categoría ilustrativa (ver DEFAULT_INJURY_TYPE_CATEGORIES).

    Reutiliza `sample_injury_absences` con n_seasons=1 -- "una temporada"
    aquí es exactamente el mismo mecanismo de ausencias que ya alimenta la
    simulación agregada, no un modelo paralelo que podría divergir (mismo
    principio que llevó a extraer `normalize_rotation_minutes` compartida).
    """
    rng = np.random.default_rng(random_seed)
    available = sample_injury_absences(risk_scores, 1, games_per_season, rng, mc_config["injury_dispersion"])[0]

    rows = []
    for player_index, player_id in enumerate(player_ids):
        player_available = available[:, player_index]
        games_played = int(player_available.sum())
        streaks = _extract_absence_streaks(player_available)
        rows.append(
            {
                "player_id": player_id,
                "player_name": player_names.get(player_id, str(player_id)),
                "games_played": games_played,
                "games_missed": games_per_season - games_played,
                "injury_events": [
                    {**streak, "category": categorize_injury_absence(streak["length"])} for streak in streaks
                ],
            }
        )
    return pd.DataFrame(rows)


def sample_schedule_context(
    league_win_pcts: np.ndarray,
    n_seasons: int,
    games_per_season: int,
    rng: np.random.Generator,
    b2b_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calendario sintético representativo -- ver docstring del módulo.
    Devuelve (opponent_win_pct, is_back_to_back, is_home), los tres
    (n_seasons, games_per_season).

    `is_home`: mitad de los partidos en casa, mitad fuera (hecho exacto
    del calendario NBA: 41 y 41 en una temporada de 82). Se reparte
    alternando y luego se baraja por temporada, así el total es
    exactamente la mitad en vez de aproximadamente la mitad -- con un
    sorteo Bernoulli una temporada concreta podría tener 47 partidos en
    casa, lo que introduciría una varianza que en la realidad no existe.
    """
    opponent_win_pct = rng.choice(league_win_pcts, size=(n_seasons, games_per_season))
    is_back_to_back = rng.random((n_seasons, games_per_season)) < b2b_probability
    is_back_to_back[:, 0] = False  # el primer partido de temporada nunca es back-to-back

    is_home = np.zeros((n_seasons, games_per_season), dtype=bool)
    is_home[:, : games_per_season // 2] = True
    is_home = rng.permuted(is_home, axis=1)
    return opponent_win_pct, is_back_to_back, is_home


def compute_player_contributions(
    game_score_per36: np.ndarray,
    minutes_projection: np.ndarray,
    fatigue_scores: np.ndarray,
    available: np.ndarray,
    is_back_to_back: np.ndarray,
    rng: np.random.Generator,
    mc_config: Dict[str, float],
) -> np.ndarray:
    """
    Contribución de Game Score de cada jugador a cada partido de cada
    temporada simulada: (n_seasons, games_per_season, n_players).
    """
    n_seasons, games_per_season, n_players = available.shape
    effective_game_score_per36 = apply_star_bonus(game_score_per36, minutes_projection, mc_config)
    base_contribution = effective_game_score_per36[None, None, :] * (minutes_projection[None, None, :] / 36.0)

    season_progress = np.arange(games_per_season) / games_per_season
    fatigue_decay = 1 - (
        fatigue_scores[None, None, :] * mc_config["season_fatigue_decay"] * season_progress[None, :, None]
    )
    b2b_penalty = 1 - (
        fatigue_scores[None, None, :] * mc_config["b2b_fatigue_penalty"] * is_back_to_back[:, :, None]
    )
    noise = rng.normal(0, mc_config["game_variance_std"], size=(n_seasons, games_per_season, n_players))

    contribution = base_contribution * fatigue_decay * b2b_penalty + noise
    return np.where(available, contribution, 0.0)


def compute_game_net_rating_estimate(
    player_contributions: np.ndarray,
    opponent_win_pct: np.ndarray,
    mc_config: Dict[str, float],
    is_home: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Game Score de equipo (suma de jugadores) MENOS la línea base de un
    equipo "promedio" (ver league_average_game_score_per36 en
    DEFAULT_MONTE_CARLO_CONFIG -- el Game Score de equipo no es
    directamente un diferencial de puntos, solo el exceso sobre un rival
    promedio lo es), menos el ajuste por fuerza de rival, MÁS la ventaja
    de campo. (n_seasons, games_per_season).

    `is_home`: bool array del mismo shape. Suma `home_court_advantage`
    en casa y lo resta fuera -- ambos equipos no pueden jugar en casa, así
    que la ventaja del local es la desventaja del visitante. Si es None
    no se aplica (compatibilidad con llamadas antiguas y con el
    backtesting, que usa calendario real y podría no traer el dato).
    """
    team_game_score = player_contributions.sum(axis=2)
    league_average_team_game_score = (
        mc_config["league_average_game_score_per36"] * TOTAL_TEAM_MINUTES_PER_GAME / 36.0
    )
    opponent_adjustment = (opponent_win_pct - 0.5) * mc_config["opponent_strength_scale"]
    net_rating = (
        team_game_score - league_average_team_game_score
    ) * mc_config["game_score_to_net_rating_scale"] - opponent_adjustment

    if is_home is not None:
        hca = mc_config.get("home_court_advantage", 0.0)
        net_rating = net_rating + np.where(is_home, hca, -hca)
    return net_rating


def compute_win_probabilities(net_rating_estimate: np.ndarray, outcome_variance_scale: float) -> np.ndarray:
    """Probabilidad de victoria vía función logística sobre el diferencial estimado."""
    return 1 / (1 + np.exp(-net_rating_estimate / outcome_variance_scale))


def sample_team_quality_noise(n_seasons: int, std: float, rng: np.random.Generator) -> np.ndarray:
    """
    Un valor por TEMPORADA simulada (no por partido): "en este mundo
    hipotético, ¿el equipo es en realidad algo mejor o peor de lo que dice
    nuestra proyección de talento?" -- química, salud de coincidencia,
    entrenador, nada que un box score capture. Shape (n_seasons, 1) para
    poder sumarse por broadcasting a un array (n_seasons, games_per_season)
    -- MISMO valor en los 82 partidos de una temporada simulada, a
    diferencia del ruido de `compute_player_contributions` (`game_variance_std`),
    que se sortea partido a partido.

    LO QUE ESTO ARREGLA Y LO QUE NO ARREGLA (ver
    scripts/experiments/team_quality_uncertainty.py): con std=0 (default,
    desactivado) esto es una función identidad. Con std>0, ensancha la
    banda P10-P90 de un equipo CONCRETO alrededor de su propia proyección
    -- corrige que solo el 61% de los 480 casos del backtest sweep caían
    dentro de esa banda (debería ser ~80%). NO mueve `wins_mean` (la MEDIA
    de miles de temporadas simuladas) porque es ruido de media cero -- se
    cancela al promediar por la ley de los grandes números. Si lo que se
    busca es más separación entre las victorias medias proyectadas de
    equipos distintos, esto no es la palanca (medido y descartado en
    scripts/experiments/aging_curve_shrinkage.py: el cuello de botella
    está en la propia señal de talento, no en cuánta incertidumbre se
    simula alrededor de ella).
    """
    if std <= 0:
        return np.zeros((n_seasons, 1))
    return rng.normal(0.0, std, size=(n_seasons, 1))


def run_monte_carlo(
    player_ids: list,
    game_score_per36: np.ndarray,
    minutes_projection: np.ndarray,
    risk_scores: np.ndarray,
    fatigue_scores: np.ndarray,
    league_win_pcts: np.ndarray,
    n_seasons: int,
    games_per_season: int,
    mc_config: Dict[str, float],
    random_seed: int,
    synergy_matrix: Optional[np.ndarray] = None,
    fixed_schedule: Optional[tuple] = None,
) -> pd.DataFrame:
    """
    Orquesta una simulación Monte Carlo completa y devuelve un DataFrame
    con una fila por temporada simulada: wins, losses,
    net_rating_estimate_mean, total_games_missed (suma de todos los
    jugadores). `synergy_matrix` (de lineup_synergy.py) es opcional --
    sin ella, el motor suma contribuciones individuales sin modelar
    encaje de alineación.

    `fixed_schedule` -- opcional (opponent_win_pct_1d, is_back_to_back_1d)
    o (opponent_win_pct_1d, is_back_to_back_1d, is_home_1d), cada uno
    shape (games_in_season,). Usado por src/backtesting.py: a diferencia
    de una temporada futura (calendario sintético muestreado, ver
    sample_schedule_context), una temporada histórica YA se jugó, así que
    se puede usar su calendario real en vez de muestrear uno. Cuando se
    pasa, `games_per_season` se ignora -- se deriva de la longitud del
    calendario real. La tercera componente (local/visitante) es opcional
    por compatibilidad: sin ella no se aplica ventaja de campo.
    """
    rng = np.random.default_rng(random_seed)

    if fixed_schedule is not None:
        fixed_opponent_win_pct, fixed_is_back_to_back, *rest = fixed_schedule
        games_per_season = len(fixed_opponent_win_pct)
        opponent_win_pct = np.tile(fixed_opponent_win_pct, (n_seasons, 1))
        is_back_to_back = np.tile(fixed_is_back_to_back, (n_seasons, 1))
        is_home = np.tile(rest[0], (n_seasons, 1)) if rest else None
    else:
        opponent_win_pct, is_back_to_back, is_home = sample_schedule_context(
            league_win_pcts, n_seasons, games_per_season, rng, mc_config["b2b_probability"]
        )

    available = sample_injury_absences(
        risk_scores, n_seasons, games_per_season, rng, mc_config["injury_dispersion"]
    )
    contributions = compute_player_contributions(
        game_score_per36, minutes_projection, fatigue_scores, available, is_back_to_back, rng, mc_config
    )
    net_rating_estimate = compute_game_net_rating_estimate(
        contributions, opponent_win_pct, mc_config, is_home=is_home
    )

    if synergy_matrix is not None:
        synergy_adjustment = compute_game_synergy_adjustment(available, synergy_matrix)
        net_rating_estimate = net_rating_estimate + synergy_adjustment

    team_quality_noise = sample_team_quality_noise(
        n_seasons, mc_config.get("team_quality_uncertainty_std", 0.0), rng
    )
    net_rating_estimate = net_rating_estimate + team_quality_noise

    win_probability = compute_win_probabilities(net_rating_estimate, mc_config["outcome_variance_scale"])

    outcomes = rng.random((n_seasons, games_per_season)) < win_probability

    wins = outcomes.sum(axis=1)
    games_missed_per_season = (~available).sum(axis=(1, 2))

    return pd.DataFrame(
        {
            "season_index": np.arange(n_seasons),
            "wins": wins,
            "losses": games_per_season - wins,
            "net_rating_estimate_mean": net_rating_estimate.mean(axis=1),
            "total_games_missed": games_missed_per_season,
        }
    )


def compute_simulation_results(
    config: Dict[str, Any], risk_scores_override: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Toda la lógica de `build_simulation_dataset` salvo el guardado en
    disco -- extraída para que el frontend web pueda pedir una variante
    "en vivo" (p.ej. `risk_scores_override=zeros` para un modo "sin
    lesiones") sin escribir sobre `simulation_results.csv`, el resultado
    persistido de la configuración real. `risk_scores_override`, si se
    pasa, sustituye por completo los `risk_score` de `injury_risk.csv`
    (mismo orden que `player_ids`) -- todo lo demás (sinergia, línea
    base de liga, fatiga) se calcula exactamente igual que la corrida
    real, así que el resultado sigue siendo comparable.
    """
    paths = get_paths(config)
    required = {
        "aging_curve_projection.csv": "context.aging_curve.build_aging_projection_dataset",
        "injury_risk.csv": "context.injury_model.build_injury_risk_dataset",
        "fatigue_risk.csv": "context.fatigue_accumulation.build_fatigue_dataset",
        "prior_season_standings.csv": "data_pipeline.build_prior_season_standings_dataset",
    }
    for filename, builder in required.items():
        path = paths["processed"] / filename
        if not path.exists():
            raise FileNotFoundError(f"No se encontró {path}. Corre `{builder}` primero.")

    aging = pd.read_csv(paths["processed"] / "aging_curve_projection.csv").set_index("player_id")
    injury = pd.read_csv(paths["processed"] / "injury_risk.csv").set_index("player_id")
    fatigue = pd.read_csv(paths["processed"] / "fatigue_risk.csv").set_index("player_id")
    standings = pd.read_csv(paths["processed"] / "prior_season_standings.csv")

    player_ids = [p["player_id"] for p in config["roster"] if p.get("player_id")]
    missing = [pid for pid in player_ids if pid not in aging.index]
    if missing:
        raise ValueError(
            f"player_id(s) {missing} están en el roster pero no en aging_curve_projection.csv. "
            "Corre el pipeline de proyección para todos los jugadores del roster."
        )

    game_score_per36 = aging.loc[player_ids, "game_score_per36"].to_numpy()
    minutes_projection_by_player = {
        p["player_id"]: p.get("minutes_projection", 0) for p in config["roster"] if p.get("player_id")
    }
    minutes_projection = np.array([minutes_projection_by_player[pid] for pid in player_ids])
    risk_scores = (
        risk_scores_override if risk_scores_override is not None else injury.loc[player_ids, "risk_score"].to_numpy()
    )
    fatigue_scores = fatigue.loc[player_ids, "fatigue_score"].to_numpy()
    league_win_pcts = standings["WinPCT"].to_numpy()

    profiles = {pid: compute_style_profile(aging.loc[pid]) for pid in player_ids}
    syn_cfg = config.get("lineup_synergy", {})
    synergy_matrix = build_synergy_matrix(
        player_ids,
        profiles,
        minutes_projection_by_player,
        usage_threshold=syn_cfg.get("usage_threshold", 18.0),
        usage_clash_weight=syn_cfg.get("usage_clash_weight", 0.05),
        playmaking_spacing_weight=syn_cfg.get("playmaking_spacing_weight", 0.02),
    )

    mc_cfg = {**DEFAULT_MONTE_CARLO_CONFIG, **config.get("monte_carlo", {})}

    # Recalibra la línea base de "equipo promedio" desde los 30 equipos
    # reales si están disponibles (ver compute_league_average_game_score_per36
    # y "BUG REAL" en el docstring del módulo) -- salvo que el usuario haya
    # fijado su propio valor a mano en config["monte_carlo"], que se respeta.
    league_projections_path = paths["processed"] / "league_player_projections.csv"
    if "league_average_game_score_per36" not in config.get("monte_carlo", {}) and league_projections_path.exists():
        league_projections = pd.read_csv(league_projections_path)
        mc_cfg["league_average_game_score_per36"] = compute_league_average_game_score_per36(
            league_projections,
            league_mean_synergy_net_rating=load_league_mean_synergy(paths["processed"]),
            game_score_to_net_rating_scale=mc_cfg["game_score_to_net_rating_scale"],
        )

    n_seasons = config["simulation"]["n_seasons"]
    games_per_season = config["simulation"]["games_per_season"]
    random_seed = config["simulation"]["random_seed"]

    return run_monte_carlo(
        player_ids,
        game_score_per36,
        minutes_projection,
        risk_scores,
        fatigue_scores,
        league_win_pcts,
        n_seasons,
        games_per_season,
        mc_cfg,
        random_seed,
        synergy_matrix=synergy_matrix,
    )


def build_simulation_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Corre `compute_simulation_results` con los `risk_score` reales de
    `injury_risk.csv` y guarda `data/processed/simulation_results.csv`
    (una fila por temporada simulada) -- el resultado "oficial" de la
    configuración actual, el que lee el resto de la app.
    """
    results = compute_simulation_results(config)

    paths = get_paths(config)
    out_path = paths["processed"] / "simulation_results.csv"
    results.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(results)} temporadas simuladas)")
    print(
        f"Wins: media={results['wins'].mean():.1f}, "
        f"p10={results['wins'].quantile(0.1):.0f}, "
        f"mediana={results['wins'].median():.0f}, "
        f"p90={results['wins'].quantile(0.9):.0f}"
    )
    return results


if __name__ == "__main__":
    from config_loader import load_config

    build_simulation_dataset(load_config())

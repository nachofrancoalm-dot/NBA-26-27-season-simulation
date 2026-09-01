"""
lineup_synergy.py

Modelo de sinergia de alineación: ajusta el Game Score de equipo que usa
`simulation.py` según qué tan bien encajan estadísticamente los jugadores
que comparten cancha, en vez de sumar sus contribuciones como si fueran
independientes.

Este roster nunca ha compartido cancha -- es hipotético para 2026-27, así
que no hay minutos jugados juntos que consultar en `leaguedashlineups`.
Por eso la sinergia se DERIVA de los perfiles estadísticos proyectados
por `aging_curve.py` (uso, creación de juego, espaciado, presencia
interior), no de partidos reales. Por la misma razón tampoco usa
`role_expected` de team_config.yaml: ese campo es descriptivo para
lectura humana, no una entrada de cálculo.

Dos efectos modelados, ambos con respaldo en analítica pública de
básquet: `usage_clash` penaliza a dos jugadores de alto "usage"
compartiendo cancha (la eficiencia cae según sube el uso propio, y
concentrar el uso en pocas estrellas beneficia al resto -- "solo hay un
balón"; ver "Discovering the Efficiency Frontier", hooponomics).
`playmaking_spacing_synergy` da un bonus cuando un creador de juego
comparte cancha con un tirador de volumen (efecto "gravedad" del
tirador). Ambos se ponderan por cuánto podrían compartir cancha
(`pair_weight = min(minutes_i, minutes_j) / 48`), una aproximación ya
que no hay datos de rotación real.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_loader import get_paths  # noqa: E402

DEFAULT_USAGE_THRESHOLD = 18.0  # FGA_per36 + 0.44*FTA_per36 + TOV_per36 por encima de esto = "alto uso"
DEFAULT_USAGE_CLASH_WEIGHT = 0.05
DEFAULT_PLAYMAKING_SPACING_WEIGHT = 0.02
DEFAULT_MIN_SYNERGY_ADJUSTMENT = -12.0  # tope de penalización total, evita que domine el Game Score de equipo
DEFAULT_MAX_SYNERGY_ADJUSTMENT = 12.0


def compute_style_profile(projection_row: pd.Series) -> Dict[str, float]:
    """
    Perfil de estilo de UN jugador a partir de su proyección por-36 de
    aging_curve.py: uso, creación de juego, espaciado, presencia interior.
    """
    return {
        "usage": (
            projection_row["FGA_per36_projected"]
            + 0.44 * projection_row["FTA_per36_projected"]
            + projection_row["TOV_per36_projected"]
        ),
        "playmaking": projection_row["AST_per36_projected"],
        "spacing": projection_row["FG3A_per36_projected"],
        "interior": projection_row["BLK_per36_projected"] + projection_row["DREB_per36_projected"],
    }


def compute_usage_clash(usage_i: float, usage_j: float, threshold: float = DEFAULT_USAGE_THRESHOLD) -> float:
    """Producto del exceso de uso sobre el umbral de ambos jugadores -- 0 si
    alguno está por debajo. Cero cuando solo uno de los dos es de alto uso."""
    excess_i = max(0.0, usage_i - threshold)
    excess_j = max(0.0, usage_j - threshold)
    return excess_i * excess_j


def compute_playmaking_spacing_synergy(
    playmaking_i: float, spacing_i: float, playmaking_j: float, spacing_j: float
) -> float:
    """Bonus simétrico: creador de i abre espacio para el tirador de j, y viceversa."""
    return playmaking_i * spacing_j + playmaking_j * spacing_i


def build_synergy_matrix(
    player_ids: List[int],
    profiles: Dict[int, Dict[str, float]],
    minutes_projection: Dict[int, float],
    usage_threshold: float = DEFAULT_USAGE_THRESHOLD,
    usage_clash_weight: float = DEFAULT_USAGE_CLASH_WEIGHT,
    playmaking_spacing_weight: float = DEFAULT_PLAYMAKING_SPACING_WEIGHT,
) -> np.ndarray:
    """
    Matriz simétrica (n_players, n_players) de ajuste de sinergia por
    pareja: playmaking_spacing_weight * synergy - usage_clash_weight *
    clash, ponderado por pair_weight = min(minutes_i, minutes_j) / 48.
    Diagonal en 0 (un jugador no tiene sinergia consigo mismo).
    """
    n = len(player_ids)
    matrix = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            id_a, id_b = player_ids[a], player_ids[b]
            profile_a, profile_b = profiles[id_a], profiles[id_b]

            clash = compute_usage_clash(profile_a["usage"], profile_b["usage"], usage_threshold)
            synergy = compute_playmaking_spacing_synergy(
                profile_a["playmaking"], profile_a["spacing"], profile_b["playmaking"], profile_b["spacing"]
            )
            pair_weight = min(minutes_projection[id_a], minutes_projection[id_b]) / 48.0

            net = pair_weight * (playmaking_spacing_weight * synergy - usage_clash_weight * clash)
            matrix[a, b] = net
            matrix[b, a] = net
    return matrix


def compute_game_synergy_adjustment(
    available: np.ndarray,
    synergy_matrix: np.ndarray,
    min_adjustment: float = DEFAULT_MIN_SYNERGY_ADJUSTMENT,
    max_adjustment: float = DEFAULT_MAX_SYNERGY_ADJUSTMENT,
) -> np.ndarray:
    """
    Ajuste de sinergia por partido: forma cuadrática sobre qué jugadores
    están disponibles ese partido (solo cuenta la sinergia entre jugadores
    que realmente comparten cancha esa noche). `available` es
    (n_seasons, games_per_season, n_players) booleano. Devuelve
    (n_seasons, games_per_season), capado en [min_adjustment, max_adjustment].
    """
    active = available.astype(float)
    raw = np.einsum("sgi,ij,sgj->sg", active, synergy_matrix, active)
    return np.clip(raw, min_adjustment, max_adjustment)


def build_lineup_synergy_dataset(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Lee aging_curve_projection.csv (generado por aging_curve.py), calcula
    la sinergia por pareja de jugadores del roster y guarda
    data/processed/lineup_synergy_pairs.csv (una fila por pareja,
    ordenada de más sinérgica a más conflictiva).
    """
    paths = get_paths(config)
    aging_path = paths["processed"] / "aging_curve_projection.csv"
    if not aging_path.exists():
        raise FileNotFoundError(
            f"No se encontró {aging_path}. Corre "
            "`context.aging_curve.build_aging_projection_dataset` primero."
        )

    aging = pd.read_csv(aging_path).set_index("player_id")
    roster = [p for p in config["roster"] if p.get("player_id")]
    player_ids = [p["player_id"] for p in roster]
    names = {p["player_id"]: p["name"] for p in roster}
    minutes_projection = {p["player_id"]: p.get("minutes_projection", 0) for p in roster}

    profiles = {pid: compute_style_profile(aging.loc[pid]) for pid in player_ids}

    syn_cfg = config.get("lineup_synergy", {})
    usage_threshold = syn_cfg.get("usage_threshold", DEFAULT_USAGE_THRESHOLD)
    usage_clash_weight = syn_cfg.get("usage_clash_weight", DEFAULT_USAGE_CLASH_WEIGHT)
    playmaking_spacing_weight = syn_cfg.get("playmaking_spacing_weight", DEFAULT_PLAYMAKING_SPACING_WEIGHT)

    rows = []
    for a in range(len(player_ids)):
        for b in range(a + 1, len(player_ids)):
            id_a, id_b = player_ids[a], player_ids[b]
            profile_a, profile_b = profiles[id_a], profiles[id_b]
            clash = compute_usage_clash(profile_a["usage"], profile_b["usage"], usage_threshold)
            synergy = compute_playmaking_spacing_synergy(
                profile_a["playmaking"], profile_a["spacing"], profile_b["playmaking"], profile_b["spacing"]
            )
            pair_weight = min(minutes_projection[id_a], minutes_projection[id_b]) / 48.0
            net_pair_score = pair_weight * (
                playmaking_spacing_weight * synergy - usage_clash_weight * clash
            )
            rows.append(
                {
                    "player_a": names[id_a],
                    "player_b": names[id_b],
                    "usage_clash": clash,
                    "playmaking_spacing_synergy": synergy,
                    "pair_weight": pair_weight,
                    "net_pair_score": net_pair_score,
                }
            )

    result_df = pd.DataFrame(rows).sort_values("net_pair_score", ascending=False).reset_index(drop=True)
    out_path = paths["processed"] / "lineup_synergy_pairs.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(result_df)} parejas)")
    return result_df


if __name__ == "__main__":
    from config_loader import load_config

    build_lineup_synergy_dataset(load_config())

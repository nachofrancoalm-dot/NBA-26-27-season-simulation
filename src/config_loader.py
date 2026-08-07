"""
config_loader.py

Punto único de lectura de team_config.yaml. Todos los demás módulos
importan `load_config()` en vez de leer el YAML por su cuenta, así
garantizamos que un cambio de equipo se propaga de forma consistente
a todo el pipeline.
"""

from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "team_config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Carga y devuelve el YAML de configuración como diccionario."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración en {config_path}. "
            "Copia config/team_config.yaml.example o crea uno nuevo."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    _validate_config(config)
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """Chequeos mínimos para fallar rápido si el config está mal formado."""
    required_top_level = ["team", "roster", "historical_comparables", "simulation", "paths"]
    missing = [k for k in required_top_level if k not in config]
    if missing:
        raise ValueError(f"Faltan claves obligatorias en team_config.yaml: {missing}")

    if "team_id" not in config["team"]:
        raise ValueError("config['team'] debe incluir 'team_id'.")

    for i, player in enumerate(config["roster"]):
        if "player_id" not in player:
            raise ValueError(f"Jugador en posición {i} del roster no tiene 'player_id'.")


def resolve_backtest_sweep_cases(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expande config["backtest_sweep"]["seasons"] a una lista de casos
    {name, team_id, season}: los 30 equipos NBA (misma tabla estática de
    franquicias que league_simulation.py / opponent_weighting.py -- el
    team_id de nba_api es estable a través de mudanzas/cambios de nombre,
    así que sirve igual para temporadas históricas) para CADA temporada
    listada. Usado para el backtesting sistemático a gran escala (ver
    src/backtesting.py:build_backtest_sweep_dataset) -- DISTINTO de
    `config["historical_comparables"]` (un puñado de casos narrativos
    elegidos a mano -- "superequipos" conocidos -- que siguen usando el
    pipeline por defecto, barato). Devuelve [] si `backtest_sweep` no
    está definido en el config -- es opt-in, no forma parte del pipeline
    normal (ver la advertencia de coste en data_pipeline.py).
    """
    from context.opponent_weighting import ABBREVIATION_TO_TEAM_ID

    sweep_cfg = config.get("backtest_sweep")
    if not sweep_cfg:
        return []

    seasons = sweep_cfg["seasons"]
    return [
        {"name": f"{abbreviation} {season}", "team_id": team_id, "season": season}
        for season in seasons
        for abbreviation, team_id in ABBREVIATION_TO_TEAM_ID.items()
    ]


def get_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Devuelve las rutas de datos como objetos Path absolutos, creándolas si no existen."""
    raw_dir = PROJECT_ROOT / config["paths"]["raw_data_dir"]
    processed_dir = PROJECT_ROOT / config["paths"]["processed_data_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return {"raw": raw_dir, "processed": processed_dir}

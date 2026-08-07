"""
resolve_player_ids.py

Resuelve player_id a partir del nombre completo del jugador usando el
catálogo ESTÁTICO que trae nba_api (nba_api.stats.static.players).
Esto NO hace ninguna llamada de red -- es una tabla empaquetada en la
librería -- así que es seguro ejecutarlo tantas veces como haga falta
sin preocuparse por rate-limiting.

Uso:
    python scripts/resolve_player_ids.py "LeBron James" "Jaylen Brown"

O bien, para rellenar automáticamente los player_id que falten en
team_config.yaml (busca por 'name' donde 'player_id' sea null):

    python scripts/resolve_player_ids.py --fill-config
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from nba_api.stats.static import players

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "team_config.yaml"


def resolve_name(full_name: str) -> dict | None:
    """Busca coincidencia exacta (case-insensitive) por nombre completo."""
    matches = players.find_players_by_full_name(full_name)
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  [aviso] Varias coincidencias para '{full_name}', usando la primera: "
              f"{[m['full_name'] for m in matches]}")
    return matches[0]


def fill_config() -> None:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    updated = 0
    for player in config["roster"]:
        if player.get("player_id"):
            continue
        match = resolve_name(player["name"])
        if match is None:
            print(f"  [NO ENCONTRADO] '{player['name']}' -> revisa el nombre exacto "
                  f"o si es un rookie muy reciente aún no indexado en nba_api.")
            continue
        player["player_id"] = match["id"]
        print(f"  [OK] {player['name']} -> {match['id']}")
        updated += 1

    if updated:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
        print(f"\n{updated} player_id(s) rellenados en {CONFIG_PATH}")
    else:
        print("\nNada que actualizar -- todos los jugadores ya tenían player_id.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Nombres completos a resolver")
    parser.add_argument("--fill-config", action="store_true",
                         help="Rellena automáticamente los player_id vacíos en team_config.yaml")
    args = parser.parse_args()

    if args.fill_config:
        fill_config()
        return

    if not args.names:
        parser.print_help()
        return

    for name in args.names:
        match = resolve_name(name)
        if match:
            status = "ACTIVO" if match["is_active"] else "inactivo"
            print(f"{name}: player_id={match['id']} ({status})")
        else:
            print(f"{name}: no encontrado")


if __name__ == "__main__":
    main()

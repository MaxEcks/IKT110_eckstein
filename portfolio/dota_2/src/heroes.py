# heroes.py
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

HEROES_PATH = DATA_DIR / "heroes.json"
MAPPING_PATH = DATA_DIR / "hero_index_mapping.json"

# -----------------------------------------------------------
# Load the raw hero metadata (Valve hero_ids, names, api_name)
# heroes.json is a LIST of hero objects
# -----------------------------------------------------------
with open(HEROES_PATH, "r", encoding="utf-8") as f:
    HEROES = json.load(f)

# Build basic lookup maps using Valve's original hero IDs
ID_TO_NAME = {h["id"]: h["name"] for h in HEROES}
NAME_TO_ID = {h["name"].lower(): h["id"] for h in HEROES}
ID_TO_API = {h["id"]: h["api_name"] for h in HEROES}

# -----------------------------------------------------------
# Load compact ML mapping:
# hero_id_to_index : Valve hero_id → compact ML feature index (0..N-1)
# index_to_hero_id : compact index → Valve hero_id
# -----------------------------------------------------------
with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)

HERO_ID_TO_INDEX = {int(k): int(v) for k, v in mapping["hero_id_to_index"].items()}
INDEX_TO_HERO_ID = {int(k): int(v) for k, v in mapping["index_to_hero_id"].items()}

NUM_HEROES_MODEL = len(HERO_ID_TO_INDEX)


# -----------------------------------------------------------
# Public API — these always use Valve hero IDs.
# They DO NOT expose compact indices.
# -----------------------------------------------------------

def hero_name(hero_id: int) -> str:
    """Return readable name for a given Valve hero_id."""
    return ID_TO_NAME.get(hero_id, f"Unknown Hero {hero_id}")

def hero_api_name(hero_id: int) -> str:
    """Return the API-friendly hero name (used for images)."""
    return ID_TO_API.get(hero_id, "unknown")

def hero_id_from_name(name: str) -> int | None:
    """
    Resolve Valve hero_id from a (case-insensitive) hero name.
    Fallback to partial fuzzy matching.
    """
    name_lower = name.lower().strip()

    if name_lower in NAME_TO_ID:
        return NAME_TO_ID[name_lower]

    for canonical in NAME_TO_ID:
        if name_lower in canonical:
            return NAME_TO_ID[canonical]

    return None

# -----------------------------------------------------------
# ML-Mapping Helpers
# -----------------------------------------------------------

def feature_index_from_hero_id(hero_id: int) -> int | None:
    """
    Convert a Valve hero_id → compact ML feature index.
    Returns None if the hero is not part of the ML universe.
    """
    return HERO_ID_TO_INDEX.get(hero_id)

def hero_id_from_feature_index(index: int) -> int | None:
    """
    Convert compact ML feature index → Valve hero_id.
    """
    return INDEX_TO_HERO_ID.get(index)


# -----------------------------------------------------------
# Debug self-test
# -----------------------------------------------------------
if __name__ == "__main__":
    print("=== HEROES MODULE SELF-TEST ===")
    print(f"Loaded {len(HEROES)} heroes (Valve metadata).")
    print(f"ML model supports {NUM_HEROES_MODEL} heroes (compact mapping).")

    print("\nFirst 5 heroes:")
    for h in HEROES[:5]:
        print(f"  Valve ID {h['id']:<3}  Name: {h['name']:<20}")

    print("\nMapping check (first 5 compact indices):")
    for i in range(min(5, NUM_HEROES_MODEL)):
        hid = hero_id_from_feature_index(i)
        print(f"  compact index {i:<3} → Valve ID {hid:<3} → {hero_name(hid)}")

    print("\n=== TEST COMPLETE ===")
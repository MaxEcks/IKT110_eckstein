# heroes.py
import json
from pathlib import Path

HEROES_PATH = Path(__file__).parent.parent / "data" / "heroes.json"

with open(HEROES_PATH, "r") as f:
    HEROES = json.load(f)

# Build lookup maps
ID_TO_NAME = {h["id"]: h["name"] for h in HEROES}
NAME_TO_ID = {h["name"].lower(): h["id"] for h in HEROES}
ID_TO_API = {h["id"]: h["api_name"] for h in HEROES}

def hero_name(hero_id: int) -> str:
    """Return the readable name for a given hero_id."""
    return ID_TO_NAME.get(hero_id, f"Unknown Hero {hero_id}")

def hero_api_name(hero_id: int) -> str:
    """Return the API-friendly/filename-friendly hero name."""
    return ID_TO_API.get(hero_id, "unknown")

def hero_id_from_name(name: str) -> int | None:
    """
    Return hero_id for a given hero name.
    Case-insensitive, tries best-match fallback.
    """
    name_lower = name.lower().strip()

    # Direct match
    if name_lower in NAME_TO_ID:
        return NAME_TO_ID[name_lower]

    # Fallback fuzzy: partial matching
    for canonical in NAME_TO_ID:
        if name_lower in canonical:
            return NAME_TO_ID[canonical]

    return None

if __name__ == "__main__":
    print("=== HEROES MODULE SELF-TEST ===")

    print(f"Total heroes loaded: {len(HEROES)}\n")

    # Show first 5 heroes
    print("First 5 heroes:")
    for h in HEROES[:5]:
        print(f"  ID {h['id']:>3}  Name: {h['name']:<20}  API: {h['api_name']}")
    print()

    # Direct ID → name
    test_ids = [1, 14, 36, 98, 129]
    print("Testing hero_name():")
    for hid in test_ids:
        print(f"  hero_name({hid}) = {hero_name(hid)}")
    print()

    # Name → ID
    test_names = ["Axe", "pudge", "Invoker", "nonexistent"]
    print("Testing hero_id_from_name():")
    for name in test_names:
        print(f"  hero_id_from_name('{name}') = {hero_id_from_name(name)}")
    print()

    # Fuzzy matching
    print("Testing fuzzy matching:")
    fuzzy_tests = ["pudge", "pu", "inv", "queen", "wyvern"]
    for ft in fuzzy_tests:
        print(f"  hero_id_from_name('{ft}') = {hero_id_from_name(ft)}")
    print()

    # API names
    print("Testing hero_api_name():")
    for hid in [1, 65, 106]:
        print(f"  hero_api_name({hid}) = {hero_api_name(hid)}")

    print("\n=== HEROES MODULE TEST COMPLETE ===")
import json
from pathlib import Path

# Path to the original heroes.json file
FILE = Path("portfolio/dota_2/data/heroes.json")

# Load heroes.json (this file is a LIST of hero objects)
with open(FILE, "r", encoding="utf-8") as f:
    heroes = json.load(f)

# ------------------------------------------
# 1) Keep only the heroes within our supported ID range
#    (e.g., ignore new heroes above ID 136)
# ------------------------------------------
MAX_ID = 136
heroes_filtered = [h for h in heroes if h["id"] <= MAX_ID]

# ------------------------------------------
# 2) Sort heroes by their Valve hero_id
#    This defines a stable ordering for our compact index
# ------------------------------------------
heroes_sorted = sorted(heroes_filtered, key=lambda h: h["id"])

# ------------------------------------------
# 3) Build the compact index mapping:
#       compact index 0..N-1   →   specific hero_id
#    This creates:
#       hero_id_to_index: map hero_id       → compact index
#       index_to_hero_id: map compact index → hero_id
# ------------------------------------------
hero_id_to_index = {h["id"]: i for i, h in enumerate(heroes_sorted)}
index_to_hero_id = {i: h["id"] for i, h in enumerate(heroes_sorted)}

mapping = {
    "hero_id_to_index": hero_id_to_index,
    "index_to_hero_id": index_to_hero_id,
}

# Output file for the mapping
OUTFILE = Path("portfolio/dota_2/data/hero_index_mapping.json")

with open(OUTFILE, "w", encoding="utf-8") as f:
    json.dump(mapping, f, indent=2)

print(f"Created compact mapping for {len(heroes_sorted)} heroes.")
from pathlib import Path
import json
import pickle

# Pfade sauber vom Dateistandort ableiten
SCRIPT_DIR = Path(__file__).resolve().parent        # .../dota_2/src
PROJECT_ROOT = SCRIPT_DIR.parent                   # .../dota_2
DATA_DIR = PROJECT_ROOT / "data"

json_path = DATA_DIR / "hero_index_mapping.json"
pkl_path = DATA_DIR / "heroes_mapping.pkl"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(pkl_path, "wb") as f:
    pickle.dump(data, f)

print(f"[OK] wrote {pkl_path}")
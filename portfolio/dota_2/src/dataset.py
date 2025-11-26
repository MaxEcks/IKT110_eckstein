"""
dataset.py
----------
Builds two datasets directly from a zipped Dota GetMatchDetails dump:

1) analysis_dataset.jsonl
   - one JSON object per line
   - minimal fields for answering the 12 analysis questions

2) training_dataset.npz
   - X: hero feature matrix (+1 / -1 / 0)
   - y: radiant_win labels (1/0)

The ZIP is NOT extracted; everything is streamed with zipfile.
"""

import json
import zipfile
from pathlib import Path

import numpy as np


# ========================================
# CONFIGURATION
# ========================================

# Path defaults (can be overridden when calling build_datasets)
DEFAULT_ZIP_PATH = "portfolio/dota_2/data/dota_games.zip"
DEFAULT_OUT_DIR = "portfolio/dota_2/data"

# Limit for number of valid matches to keep
MAX_MATCHES = 2_000_000

# Minimum match duration in seconds (to filter out remakes / non-games)
MIN_DURATION = 600  # 10 minutes

# Lobby types we consider "serious 5v5"
# 0 - Public matchmaking
# 7 - Ranked
ALLOWED_LOBBY_TYPES = {0, 7}

# Game modes we consider "serious 5v5 draft-like"
# 1  - All Pick
# 2  - Captain's Mode
# 3  - Random Draft
# 4  - Single Draft
# 16 - Captains Draft
# 22 - Ranked Matchmaking
ALLOWED_GAME_MODES = {1, 2, 3, 4, 16, 22}

# ========================================
# HERO INDEX MAPPING (compact feature space)
# ========================================

# Mapping file created e.g. by hero_index_mapping.py from heroes.json
MAPPING_PATH = Path(DEFAULT_OUT_DIR) / "hero_index_mapping.json"

with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    _mapping = json.load(f)

# hero_id_to_index: maps Valve hero_id -> compact feature index (0..N-1)
# index_to_hero_id: inverse mapping (not used here, but useful for analysis)
HERO_ID_TO_INDEX = {int(k): int(v) for k, v in _mapping["hero_id_to_index"].items()}
INDEX_TO_HERO_ID = {int(k): int(v) for k, v in _mapping["index_to_hero_id"].items()}

# Number of hero-features = size of compact hero universe
NUM_HEROES = len(HERO_ID_TO_INDEX)


# ========================================
# FILTER LOGIC
# ========================================

def match_passes_filters(m: dict) -> bool:
    """
    Return True if the match (already unwrapped from 'result') passes
    all quality filters and should be used both for analysis and training.
    """

    # 1) Require exactly 10 human players
    if m.get("human_players") != 10:
        return False

    players = m.get("players", [])
    if len(players) != 10:
        return False

    # 2) Valid + unique hero_ids
    hero_ids = [p.get("hero_id", 0) for p in players]
    if any(h <= 0 for h in hero_ids):
        return False
    if len(set(hero_ids)) != 10:
        return False

    # 3) Leaver status: keep ONLY matches where all players have leaver_status == 0
    for p in players:
        if p.get("leaver_status") != 0:
            return False

    # 4) Lobby type: keep only public + ranked matchmaking
    if m.get("lobby_type") not in ALLOWED_LOBBY_TYPES:
        return False

    # 5) Game mode: keep only "serious 5v5" modes
    if m.get("game_mode") not in ALLOWED_GAME_MODES:
        return False

    # 6) Minimum duration
    if m.get("duration", 0) < MIN_DURATION:
        return False

    return True


# ========================================
# ANALYSIS ROW BUILDER (JSONL)
# ========================================

def build_analysis_row(m: dict) -> dict:
    """
    Extract the minimal information needed for all analysis questions (1–11).

    We store:
      - match_id
      - radiant_win (bool)
      - duration (int)
      - game_mode
      - lobby_type
      - human_players
      - heroes_radiant (list of hero_id)
      - heroes_dire (list of hero_id)
      - picks_bans (as given by API; can be None)
    """
    radiant_heroes = [p["hero_id"] for p in m["players"] if p.get("player_slot", 0) < 128]
    dire_heroes    = [p["hero_id"] for p in m["players"] if p.get("player_slot", 0) >= 128]

    row = {
        "match_id": m.get("match_id"),
        "radiant_win": bool(m.get("radiant_win", False)),
        "duration": m.get("duration"),
        "game_mode": m.get("game_mode"),
        "lobby_type": m.get("lobby_type"),
        "human_players": m.get("human_players"),
        "heroes_radiant": radiant_heroes,
        "heroes_dire": dire_heroes,
        "picks_bans": m.get("picks_bans"),
    }
    return row


# ========================================
# TRAINING DATA BUILDER (X, y)
# ========================================

def build_training_xy(m: dict):
    """
    Build (x, y) for model training.

    x: hero feature vector of length NUM_HEROES
       +1 if hero picked by Radiant
       -1 if hero picked by Dire
        0 if not picked

    Uses a compact hero index mapping:
      hero_id  -> HERO_ID_TO_INDEX[hero_id]  -> feature index 0..NUM_HEROES-1

    y: 1.0 if Radiant wins, else 0.0
    """
    x = np.zeros(NUM_HEROES, dtype=np.float32)

    for p in m["players"]:
        hero_id = p.get("hero_id", 0)
        if hero_id <= 0:
            continue

        # Map Valve hero_id to compact feature index.
        # If the hero is not in our mapping (e.g. new hero outside our universe),
        # we simply ignore it.
        idx = HERO_ID_TO_INDEX.get(hero_id)
        if idx is None:
            continue

        if p.get("player_slot", 0) < 128:
            x[idx] = 1.0
        else:
            x[idx] = -1.0

    y = 1.0 if m.get("radiant_win") else 0.0
    return x, y


# ========================================
# MAIN PIPELINE
# ========================================

def build_datasets(
    zip_path: str = DEFAULT_ZIP_PATH,
    out_dir: str = DEFAULT_OUT_DIR,
    max_matches: int = MAX_MATCHES,
):
    """
    Stream over all JSON files in the given ZIP, apply filters, and
    build:

      - analysis_dataset.jsonl  (one JSON object per line)
      - training_dataset.npz    (X, y)

    zip_path : path to dota_games.zip
    out_dir  : output directory (will be created if not exists)
    max_matches : stop after this many valid matches have been processed
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    analysis_path = out_dir_path / "analysis_dataset.jsonl"
    training_path = out_dir_path / "training_dataset.npz"

    X_list = []
    y_list = []
    kept = 0

    with zipfile.ZipFile(zip_path) as z, open(analysis_path, "w") as analysis_file:
        json_files = [name for name in z.namelist() if name.endswith(".json")]

        for name in json_files:
            try:
                with z.open(name) as f:
                    raw = json.load(f)
            except Exception:
                continue  # skip corrupted or unreadable files

            # unwrap GetMatchDetails structure: either { "result": {...} } or flat
            match = raw.get("result", raw)

            if not isinstance(match, dict):
                continue

            # Apply filters
            if not match_passes_filters(match):
                continue

            # ---- Analysis dataset (JSONL) ----
            analysis_row = build_analysis_row(match)
            analysis_file.write(json.dumps(analysis_row) + "\n")

            # ---- Training dataset (NumPy) ----
            x, y = build_training_xy(match)
            X_list.append(x)
            y_list.append(y)

            kept += 1
            if kept >= max_matches:
                break

    if not X_list:
        raise RuntimeError("No matches passed the filters. Check your filter settings or inspect the raw data.")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.float32)

    np.savez(training_path, X=X, y=y)

    print(f"[dataset] Done. Kept {kept} matches.")
    print(f"[dataset] Analysis dataset written to: {analysis_path}")
    print(f"[dataset] Training dataset written to: {training_path}")


# ========================================
# CLI ENTRY POINT
# ========================================

if __name__ == "__main__":
    build_datasets()
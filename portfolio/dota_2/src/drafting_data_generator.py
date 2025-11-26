"""
Build hero_analysis.csv from the raw match dataset.

This computes for each hero:
- winrate
- pickrate
- best teammate synergy
- best matchups (heroes this hero performs well against)
- worst counters (heroes that strongly counter this hero)

IMPORTANT:
- We now ensure that ALL heroes from heroes.json are present,
  even if they never appear in the dataset.
- Heroes with no data get neutral/default stats so that downstream
  drafting logic does not break or produce nonsense.
"""

import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Dict, Any

import pandas as pd

# ============================================
# Paths
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # adjust if needed
DATASET_PATH = PROJECT_ROOT / "data" / "analysis_dataset.jsonl"
HEROES_PATH = PROJECT_ROOT / "data" / "heroes.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "hero_analysis.csv"


# ============================================
# Helpers
# ============================================

def iter_matches(limit: int | None = None):
    """Yield parsed match dicts, optionally limited by `limit`."""
    count = 0
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)
            yield m
            count += 1
            if limit is not None and count >= limit:
                break


def load_all_hero_ids() -> List[int]:
    """
    Load all hero_ids from heroes.json.

    This is written to be robust for two common formats:
    1) List[{"id": ..., "name": ...}, ...]
    2) Dict[str, {"id": ..., "name": ...}, ...]
    """
    with open(HEROES_PATH, "r", encoding="utf-8") as f:
        heroes_data = json.load(f)

    hero_ids: List[int] = []

    if isinstance(heroes_data, list):
        # Expect list of hero objects with an "id" field
        for h in heroes_data:
            if isinstance(h, dict) and "id" in h:
                hero_ids.append(int(h["id"]))
    elif isinstance(heroes_data, dict):
        # Expect dict of name -> hero object with an "id" field
        for obj in heroes_data.values():
            if isinstance(obj, dict) and "id" in obj:
                hero_ids.append(int(obj["id"]))
    else:
        raise ValueError("Unsupported heroes.json format")

    hero_ids = sorted(set(hero_ids))
    return hero_ids


# ============================================
# Counters
# ============================================

hero_games = Counter()
hero_wins = Counter()

pair_games = Counter()
pair_wins = Counter()

matchup_games = Counter()
matchup_wins = Counter()

total_matches = 0


# ============================================
# Parse dataset
# ============================================

for m in iter_matches():
    total_matches += 1
    radiant = m["heroes_radiant"]
    dire = m["heroes_dire"]
    radiant_win = m["radiant_win"]

    # Individual hero stats
    for h in radiant:
        hero_games[h] += 1
        if radiant_win:
            hero_wins[h] += 1

    for h in dire:
        hero_games[h] += 1
        if not radiant_win:
            hero_wins[h] += 1

    # Teammate synergy (pairs in the same team)
    for team in (radiant, dire):
        # Determine if this specific team won the match
        team_won = (team is radiant and radiant_win) or (
            team is dire and not radiant_win
        )
        for h1, h2 in combinations(team, 2):
            if h1 > h2:
                h1, h2 = h2, h1
            pair_games[(h1, h2)] += 1
            if team_won:
                pair_wins[(h1, h2)] += 1

    # Opponent matchups (A vs B)
    # Direction matters: (A, B) is "hero A performance vs hero B"
    for A in radiant:
        for B in dire:
            matchup_games[(A, B)] += 1
            if radiant_win:
                matchup_wins[(A, B)] += 1

    for A in dire:
        for B in radiant:
            matchup_games[(A, B)] += 1
            if not radiant_win:
                matchup_wins[(A, B)] += 1


# ============================================
# Hero-level stats for heroes that appear in the dataset
# ============================================

hero_wr: Dict[int, float] = {
    h: hero_wins[h] / hero_games[h] for h in hero_games
}
hero_pickrate: Dict[int, float] = {
    h: hero_games[h] / total_matches for h in hero_games
}

# Global average winrate across heroes with data
if hero_wr:
    avg_hero_wr = sum(hero_wr.values()) / len(hero_wr)
else:
    # Extremely unlikely, but safe fallback
    avg_hero_wr = 0.5


# ============================================
# Teammate synergy
# ============================================

synergy_list: List[tuple[int, int, int, float, float]] = []

for (h1, h2), g in pair_games.items():
    if g < 30:  # stability threshold for pairs
        continue

    wr_pair = pair_wins[(h1, h2)] / g
    expected_wr = (hero_wr[h1] + hero_wr[h2]) / 2
    synergy = wr_pair - expected_wr

    synergy_list.append((h1, h2, g, wr_pair, synergy))


# ============================================
# Counters & matchups
# ============================================

counter_list: List[tuple[int, int, int, float, float]] = []

for (A, B), g in matchup_games.items():
    if g < 30:  # stability threshold for matchups
        continue

    wr_AB = matchup_wins[(A, B)] / g

    # Interpretation:
    #   hero_wr[A] = overall winrate of A
    #   wr_AB      = winrate of A in games vs B
    #
    # If wr_AB < hero_wr[A], A performs worse vs B than usual → B is a counter.
    # So hero_wr[A] - wr_AB becomes positive and larger for stronger counters.
    counter_score = hero_wr[A] - wr_AB

    counter_list.append((A, B, g, wr_AB, counter_score))


# ============================================
# Assemble hero rows for ALL heroes
# ============================================

all_hero_ids = load_all_hero_ids()
hero_data: List[Dict[str, Any]] = []

for h in all_hero_ids:
    # ----- Base stats (winrate, pickrate) -----
    if h in hero_wr:
        wr = hero_wr[h]
        pr = hero_pickrate[h]
    else:
        # Hero never appeared in the dataset:
        # use neutral/default stats to avoid nonsense.
        wr = avg_hero_wr
        pr = 0.0

    # ----- Best teammate (highest synergy) -----
    h_synergies = [
        (h1, h2, games, wr_pair, syn)
        for h1, h2, games, wr_pair, syn in synergy_list
        if h in (h1, h2)
    ]

    if h_synergies:
        best_pair = max(h_synergies, key=lambda x: x[4])  # highest synergy value
        best_teammate = best_pair[1] if best_pair[0] == h else best_pair[0]
        best_teammate_synergy = round(best_pair[4], 3)
    else:
        best_teammate = None
        best_teammate_synergy = None

    # ----- Counters and matchups -----
    # We only use (A == h) entries, i.e. "performance of h vs B"
    h_counters = [
        (B, round(score, 3))
        for A, B, g, wrAB, score in counter_list
        if A == h
    ]

    # Worst counters: highest counter_score
    worst_sorted = sorted(h_counters, key=lambda x: x[1], reverse=True)[:5]

    # Best matchups: lowest counter_score
    best_sorted = sorted(h_counters, key=lambda x: x[1])[:5]

    hero_data.append(
        {
            "hero_id": h,
            "winrate": round(wr, 3),
            "pickrate": round(pr, 3),
            "best_teammate": best_teammate,
            "best_teammate_synergy": best_teammate_synergy,
            "best_matchups": best_sorted,
            "worst_counters": worst_sorted,
        }
    )

# ============================================
# Save CSV
# ============================================

df = pd.DataFrame(hero_data)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved hero_analysis.csv successfully to: {OUTPUT_PATH}")
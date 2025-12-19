# draft_engine.py

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import pandas as pd

# ------------------------------------------------------
# Load data
# ------------------------------------------------------

# Relative path to hero_analysis.csv (similar to heroes.py style)
HERO_ANALYSIS_PATH = Path(__file__).parent.parent / "data" / "hero_analysis.csv"

df = pd.read_csv(HERO_ANALYSIS_PATH)

# Convert stringified lists of tuples back to Python objects
df["best_matchups"] = df["best_matchups"].apply(ast.literal_eval)
df["worst_counters"] = df["worst_counters"].apply(ast.literal_eval)

VALID_HEROES: set[int] = set(df["hero_id"].astype(int).tolist())


# ------------------------------------------------------
# Helper functions
# ------------------------------------------------------


def validate_team(team: Iterable[Any]) -> Tuple[List[int], List[Any]]:
    """
    Take a list of hero_ids (or strings) from the app and:

      - try to cast everything to int
      - separate valid and invalid hero_ids according to VALID_HEROES

    Returns:
      (valid_ids, invalid_items)
    """
    valid: List[int] = []
    invalid: List[Any] = []

    for x in team:
        try:
            hid = int(x)
        except Exception:
            invalid.append(x)
            continue

        if hid in VALID_HEROES:
            valid.append(hid)
        else:
            invalid.append(x)

    return valid, invalid


# ------------------------------------------------------
# Scoring functions (heuristics)
# ------------------------------------------------------


def calculate_pick_score(
    hero_id: int, my_team: List[int], enemy_team: List[int]
) -> float:
    """
    Heuristic score for how good hero_id is as the next pick for my_team.

    Uses:
      - general winrate
      - best teammate + synergy
      - best_matchups (counters versus enemy heroes)
      - worst_counters (heroes that counter us)
    """
    row = df[df["hero_id"] == hero_id].iloc[0]

    winrate = row["winrate"]

    # Synergy with my current team
    team_synergy_score = 0.0
    for ally in my_team:
        if row["best_teammate"] == ally:
            team_synergy_score += row["best_teammate_synergy"]

    # How well this hero counters the enemy team
    counter_score = 0.0
    for enemy in enemy_team:
        for eid, val in row["best_matchups"]:
            if eid == enemy:
                # In the original analysis, a lower 'val' is better for us,
                # so we invert the sign.
                counter_score += -val

    # How much this hero is countered by the enemy
    countered_by_score = 0.0
    for enemy in enemy_team:
        for eid, val in row["worst_counters"]:
            if eid == enemy:
                countered_by_score += val

    # Final weighted heuristic score
    score = (
        winrate * 0.5
        + team_synergy_score * 1.5
        + counter_score * 2.0
        - countered_by_score * 1.5
    )
    return float(score)


def calculate_ban_score(
    hero_id: int, my_team: List[int], enemy_team: List[int]
) -> float:
    """
    Heuristic score for how important it is to ban hero_id.

    Uses:
      - how strong the hero is against my_team (best_matchups vs our allies)
      - synergy with the enemy team (best_teammate)
      - own winrate as base strength
    """
    row = df[df["hero_id"] == hero_id].iloc[0]
    winrate = row["winrate"]

    # How strong this hero is against our allies
    strong_against_us = 0.0
    for ally in my_team:
        for eid, val in row["best_matchups"]:
            if eid == ally:
                strong_against_us += -val

    # Synergy with the enemy team
    enemy_synergy = 0.0
    for enemy in enemy_team:
        if row["best_teammate"] == enemy:
            enemy_synergy += row["best_teammate_synergy"]

    # Base strength from winrate
    base_strength = winrate * 0.4

    ban_score = strong_against_us * 2.0 + enemy_synergy * 1.5 + base_strength
    return float(ban_score)


# ------------------------------------------------------
# Recommendation API for the web app
# ------------------------------------------------------


def recommend_pick(
    raw_my_team: Iterable[Any],
    raw_enemy_team: Iterable[Any],
    raw_my_ban,
    raw_enemy_ban,
    top_n: int = 5,
) -> Tuple[List[Tuple[int, float]], Dict[str, List[Any]]]:
    """
    Return a sorted list of recommended picks:
      [(hero_id, score), ...] of length top_n

    Also returns a warnings dict with invalid IDs, e.g.:
      {
        "invalid_my_team": [...],
        "invalid_enemy_team": [...],
      }
    """
    my_team, my_invalid = validate_team(raw_my_team)
    my_ban, my_invalid_ban = validate_team(raw_my_ban)

    enemy_team, enemy_invalid = validate_team(raw_enemy_team)
    enemy_ban, enemy_invalid_ban = validate_team(raw_enemy_ban)

    warnings: Dict[str, List[Any]] = {}
    if my_invalid:
        warnings["invalid_my_team"] = my_invalid
    if enemy_invalid:
        warnings["invalid_enemy_team"] = enemy_invalid

    scores: Dict[int, float] = {}
    for hero_id in df["hero_id"]:
        hid = int(hero_id)
        # Do not recommend heroes that are already in my team
        if hid in my_team or hid in enemy_team or hid in my_ban or hid in enemy_ban:
            continue
        scores[hid] = calculate_pick_score(hid, my_team, enemy_team)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n], warnings


def recommend_ban(
    raw_my_team: Iterable[Any],
    raw_enemy_team: Iterable[Any],
    raw_my_ban,
    raw_enemy_ban,
    top_n: int = 5,
) -> Tuple[List[Tuple[int, float]], Dict[str, List[Any]]]:
    """
    Return a sorted list of recommended bans:
      [(hero_id, score), ...] of length top_n

    Also returns a warnings dict with invalid IDs.
    """
    my_team, my_invalid = validate_team(raw_my_team)
    my_ban, my_invalid_ban = validate_team(raw_my_ban)

    enemy_team, enemy_invalid = validate_team(raw_enemy_team)
    enemy_ban, enemy_invalid_ban = validate_team(raw_enemy_ban)

    warnings: Dict[str, List[Any]] = {}
    if my_invalid:
        warnings["invalid_my_team"] = my_invalid
    if enemy_invalid:
        warnings["invalid_enemy_team"] = enemy_invalid

    scores: Dict[int, float] = {}
    for hero_id in df["hero_id"]:
        hid = int(hero_id)
        if hid in my_team or hid in enemy_team or hid in my_ban or hid in enemy_ban:
            continue
        scores[hid] = calculate_ban_score(hid, my_team, enemy_team)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n], warnings


# ------------------------------------------------------
# Small module test
# ------------------------------------------------------

# if __name__ == "__main__":
#    # Example teams (just for quick local testing)
#    example_my_team = [36, 99]  # replace with real hero_ids
#    example_enemy_team = [98, 33]
#
#    print("=== Drafting Engine Smoke Test ===")
#    print(f"My team:      {example_my_team}")
#    print(f"Enemy team:   {example_enemy_team}")
#    print()
#
#    top_picks, pick_warnings = recommend_pick(
#        example_my_team, example_enemy_team, top_n=5
#    )
#    top_bans, ban_warnings = recommend_ban(example_my_team, example_enemy_team, top_n=5)
#
#    print("Top pick recommendations (hero_id, score):")
#    for hero_id, score in top_picks:
#        print(f"  {hero_id:3d}  ->  {score:.4f}")
#
#    print()
#    print("Top ban recommendations (hero_id, score):")
#    for hero_id, score in top_bans:
#        print(f"  {hero_id:3d}  ->  {score:.4f}")
#
#    if pick_warnings or ban_warnings:
#        print()
#        print("Warnings:")
#        if pick_warnings:
#            print("  Pick warnings:", pick_warnings)
#        if ban_warnings:
#            print("  Ban warnings:", ban_warnings)

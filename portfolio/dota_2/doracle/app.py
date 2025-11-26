# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    url_for,
)

# ------------------------------------------------------------------------------
# Paths & Python import setup
# ------------------------------------------------------------------------------

# This file lives in: project_root/doracle/app.py
PACKAGE_DIR = Path(__file__).resolve().parent              # .../doracle
PROJECT_ROOT = PACKAGE_DIR.parent                          # project_root
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"

# Make src/ importable (so we can use draft_engine, model, etc.)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from draft_engine import recommend_pick, recommend_ban  # type: ignore
# If you later want ML winrate:
# from model import predict_from_draft  # type: ignore


# ------------------------------------------------------------------------------
# Configs
# ------------------------------------------------------------------------------

frontend_port = 5000
app = Flask(__name__, static_folder="static", template_folder="templates")

app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
debug_mode = True


# ------------------------------------------------------------------------------
# Template renders
# ------------------------------------------------------------------------------

@app.route("/")
def index():
    """Main drafting page."""
    return render_template("index.html", heroes=get_heroes())


@app.route("/explore")
def explore():
    """Optional explore page (hero stats, counters, etc.)."""
    return render_template("explore.html", heroes=get_heroes())


# ------------------------------------------------------------------------------
# Drafting APIs (hooked to your heuristic engine)
# ------------------------------------------------------------------------------

@app.route("/suggest1", methods=["POST"])
def suggest1():
    """
    Suggest picks based on the current draft situation.

    Expected JSON body:
    {
      "my_team":   [hero_id, ...],
      "enemy_team": [hero_id, ...],
      "top_n": 5   # optional, default 5
    }

    Returns:
    {
      "picks": [[hero_id, score], ...],
      "warnings": { ... }
    }
    """
    data: Dict[str, Any] = request.get_json(force=True) or {}

    my_team: List[Any] = data.get("my_team", [])
    enemy_team: List[Any] = data.get("enemy_team", [])
    top_n: int = int(data.get("top_n", 5))

    ranked, warnings = recommend_pick(my_team, enemy_team, top_n=top_n)
    return jsonify({"picks": ranked, "warnings": warnings})


@app.route("/suggest2", methods=["POST"])
def suggest2():
    """
    Suggest bans based on the current draft situation.

    Expected JSON body:
    {
      "my_team":   [hero_id, ...],
      "enemy_team": [hero_id, ...],
      "top_n": 5   # optional, default 5
    }

    Returns:
    {
      "bans": [[hero_id, score], ...],
      "warnings": { ... }
    }
    """
    data: Dict[str, Any] = request.get_json(force=True) or {}

    my_team: List[Any] = data.get("my_team", [])
    enemy_team: List[Any] = data.get("enemy_team", [])
    top_n: int = int(data.get("top_n", 5))

    ranked, warnings = recommend_ban(my_team, enemy_team, top_n=top_n)
    return jsonify({"bans": ranked, "warnings": warnings})


@app.route("/suggest3", methods=["POST"])
def suggest3():
    """
    Placeholder for a third feature.

    You can later turn this into:
      - combined pick + ban suggestion,
      - ML winrate evaluation of the current draft,
      - or something else.

    For now, it just echoes the payload.
    """
    data = request.get_json(force=True) or {}
    data["todo"] = (
        "Implement a third feature here (e.g. ML winrate, combined suggestion, etc.)."
    )
    return jsonify(data)


# ------------------------------------------------------------------------------
# Explore / stats API
# ------------------------------------------------------------------------------

@app.route("/stats/<int:heroid>", methods=["GET"])
def get_hero_stats(heroid: int):
    """
    Return hero stats for /explore page based on hero_analysis.csv.

    Also injects correct image URLs:
      static/img/avatar-sb/<HERO_ID>.png

    Response example:
    {
      "hero_id": 1,
      "hero": "Anti-Mage",
      "hero_image": "/static/img/avatar-sb/1.png",
      "win_rate": 0.51,
      "pick_rate": 0.12,
      "best_teammate_id": 13,
      "best_teammate": "Puck",
      "best_teammate_image": "/static/img/avatar-sb/13.png",
      "best_teammate_synergy": 0.034,
      "best_matchups": [
        {"hero_id": X, "name": "...", "score": ..., "image": "/static/img/avatar-sb/X.png"},
        ...
      ],
      "worst_counters": [
        {"hero_id": Y, "name": "...", "score": ..., "image": "/static/img/avatar-sb/Y.png"},
        ...
      ]
    }
    """
    df = get_hero_analysis()
    heroes = get_heroes()

    row = df[df["hero_id"] == heroid]
    if row.empty:
        return jsonify({"error": f"Hero {heroid} not found in hero_analysis.csv"}), 404

    row = row.iloc[0]

    hero_name = heroes.get(heroid, {}).get("name", f"Hero {heroid}")
    hero_image = url_for("static", filename=f"img/avatar-sb/{heroid}.png")

    best_teammate_id = row["best_teammate"]
    if pd.isna(best_teammate_id):
        best_teammate_id = None
        best_teammate_name = None
        best_teammate_synergy = None
        best_teammate_image = None
    else:
        best_teammate_id = int(best_teammate_id)
        best_teammate_name = heroes.get(best_teammate_id, {}).get(
            "name", f"Hero {best_teammate_id}"
        )
        best_teammate_synergy = row["best_teammate_synergy"]
        best_teammate_image = url_for(
            "static", filename=f"img/avatar-sb/{best_teammate_id}.png"
        )

    def _expand_list_of_pairs(
        pairs: list[tuple[int, float]],
    ) -> list[dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for hid, score in pairs:
            hid_int = int(hid)
            out.append(
                {
                    "hero_id": hid_int,
                    "name": heroes.get(hid_int, {}).get("name", f"Hero {hid_int}"),
                    "score": float(score),
                    "image": url_for(
                        "static", filename=f"img/avatar-sb/{hid_int}.png"
                    ),
                }
            )
        return out

    best_matchups = _expand_list_of_pairs(row["best_matchups"])
    worst_counters = _expand_list_of_pairs(row["worst_counters"])

    hero_stats = {
        "hero_id": heroid,
        "hero": hero_name,
        "hero_image": hero_image,
        "win_rate": float(row["winrate"]),
        "pick_rate": float(row["pickrate"]),
        "best_teammate_id": best_teammate_id,
        "best_teammate": best_teammate_name,
        "best_teammate_image": best_teammate_image,
        "best_teammate_synergy": (
            float(best_teammate_synergy) if best_teammate_synergy is not None else None
        ),
        "best_matchups": best_matchups,
        "worst_counters": worst_counters,
    }

    return jsonify(hero_stats)


# ------------------------------------------------------------------------------
# Helpers (heroes + hero_analysis)
# ------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_heroes() -> Dict[int, Dict[str, Any]]:
    """
    Load heroes.json from project_root/data and return a mapping:
      hero_id -> hero_dict

    We also inject an 'image' field with the relative static path:
      img/avatar-sb/<HERO_ID>.png

    The function is written to support two common formats:
      1) List[{"id": ..., "name": ...}, ...]
      2) Dict[str, {"id": ..., "name": ...}, ...]
    """
    heroes_path = DATA_DIR / "heroes.json"
    with open(heroes_path, "r", encoding="utf-8") as fp:
        raw = json.load(fp)

    by_id: Dict[int, Dict[str, Any]] = {}

    if isinstance(raw, list):
        for h in raw:
            if isinstance(h, dict) and "id" in h:
                hid = int(h["id"])
                h["image"] = f"img/avatar-sb/{hid}.png"
                by_id[hid] = h
    elif isinstance(raw, dict):
        # e.g. {"antimage": {"id": 1, "name": "Anti-Mage", ...}, ...}
        for obj in raw.values():
            if isinstance(obj, dict) and "id" in obj:
                hid = int(obj["id"])
                obj["image"] = f"img/avatar-sb/{hid}.png"
                by_id[hid] = obj
    else:
        raise ValueError("Unsupported heroes.json format")

    return by_id


@lru_cache(maxsize=1)
def get_hero_analysis() -> pd.DataFrame:
    """
    Load hero_analysis.csv from project_root/data.
    The DataFrame contains:
      hero_id, winrate, pickrate, best_teammate,
      best_teammate_synergy, best_matchups, worst_counters
    """
    path = DATA_DIR / "hero_analysis.csv"
    df = pd.read_csv(path)

    # Convert stringified lists back to Python objects
    for col in ("best_matchups", "worst_counters"):
        if df[col].dtype == object and isinstance(df[col].iloc[0], str):
            # hero_analysis was generated via json.dumps or repr([...])
            # json.loads is safer if the strings are valid JSON.
            try:
                df[col] = df[col].apply(json.loads)
            except Exception:
                df[col] = df[col].apply(eval)

    return df


# ------------------------------------------------------------------------------
# Dev entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    app.jinja_env.cache = {}
    app.run(debug=debug_mode, host="127.0.0.1", port=frontend_port)

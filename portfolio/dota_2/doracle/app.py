# -------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------

from __future__ import annotations

import json
import pickle
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, url_for

# -------------------------------------------------------------------------------
# Paths & Python import setup
# -------------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent          # .../portfolio/dota_2/doracle
PROJECT_ROOT = PACKAGE_DIR.parent                      # .../portfolio/dota_2

DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"

# ensure src is importable
sp = str(SRC_DIR)
if sp not in sys.path:
    sys.path.insert(0, sp)

from draft_engine import recommend_pick, recommend_ban  # type: ignore

# -------------------------------------------------------------------------------
# Configs
# -------------------------------------------------------------------------------

frontend_port = 5000
app = Flask(__name__, static_folder="static", template_folder="templates")

app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True
debug_mode = True

# -------------------------------------------------------------------------------
# ML Prediction Helpers
# -------------------------------------------------------------------------------


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_single_value(theta, input_vector):
    bias = theta[0]
    weight = theta[1:]
    linear = bias + np.dot(input_vector, weight)
    return sigmoid(linear)


def get_theta():
    theta_path = DATA_DIR / "theta_model.npz"
    data = np.load(theta_path)
    return data["theta"]


# -------------------------------------------------------------------------------
# Template renders
# -------------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html", heroes=get_heroes())


@app.route("/explore")
def explore():
    return render_template("explore.html", heroes=get_heroes())


# -------------------------------------------------------------------------------
# Drafting APIs
# -------------------------------------------------------------------------------


@app.route("/radiant_pick", methods=["POST"])
def radiant_pick():
    data = request.get_json(force=True) or {}
    radiant = data.get("radiant", [])
    dire = data.get("dire", [])
    radiant_ban = data.get("radiant_ban", [])
    dire_ban = data.get("dire_ban", [])
    top_n = int(data.get("top_n", 5))
    ranked, warnings = recommend_pick(radiant, dire, radiant_ban, dire_ban, top_n=top_n)
    return jsonify({"picks": ranked, "warnings": warnings})


@app.route("/radiant_ban", methods=["POST"])
def radiant_ban():
    data = request.get_json(force=True) or {}
    radiant = data.get("radiant", [])
    dire = data.get("dire", [])
    radiant_ban = data.get("radiant_ban", [])
    dire_ban = data.get("dire_ban", [])
    top_n = int(data.get("top_n", 7))
    ranked, warnings = recommend_ban(radiant, dire, radiant_ban, dire_ban, top_n=top_n)
    return jsonify({"bans": ranked, "warnings": warnings})


@app.route("/dire_pick", methods=["POST"])
def dire_pick():
    data = request.get_json(force=True) or {}
    radiant = data.get("radiant", [])
    dire = data.get("dire", [])
    radiant_ban = data.get("radiant_ban", [])
    dire_ban = data.get("dire_ban", [])
    top_n = int(data.get("top_n", 5))
    ranked, warnings = recommend_pick(dire, radiant, dire_ban, radiant_ban, top_n=top_n)
    return jsonify({"picks": ranked, "warnings": warnings})


@app.route("/dire_ban", methods=["POST"])
def dire_ban():
    data = request.get_json(force=True) or {}
    radiant = data.get("radiant", [])
    dire = data.get("dire", [])
    radiant_ban = data.get("radiant_ban", [])
    dire_ban = data.get("dire_ban", [])
    top_n = int(data.get("top_n", 7))
    ranked, warnings = recommend_ban(dire, radiant, dire_ban, radiant_ban, top_n=top_n)
    return jsonify({"bans": ranked, "warnings": warnings})


@app.route("/stats/<int:hero_id>")
def hero_stats(hero_id: int):
    df = get_hero_analysis()
    hero_row = df[df["hero_id"] == hero_id]
    if hero_row.empty:
        return jsonify({"error": "Hero not found"}), 404
    hero_data: Dict[str, Any] = hero_row.to_dict(orient="records")[0]  # type: ignore
    return jsonify(hero_data)


# -------------------------------------------------------------------------------
# Winner Prediction API (neu)
# -------------------------------------------------------------------------------


@app.route("/predict_winner", methods=["POST"])
def predict_winner():
    data = request.get_json(force=True) or {}
    radiant_team: List[int] = data.get("radiant", [])
    dire_team: List[int] = data.get("dire", [])

    # Feature vector length must match the training setup
    # (In our case: number of heroes in heroes.json)
    num_heroes = len(get_heroes())
    x = np.zeros(num_heroes, dtype=float)

    # Load hero-id → feature-index mapping using an absolute path
    # (robust against different working directories)
    mapping_path = DATA_DIR / "heroes_mapping.pkl"
    with open(mapping_path, "rb") as f:
        mapping = pickle.load(f)

    hero_id_to_index = mapping["hero_id_to_index"]

    # The mapping uses 1-based indices → convert to 0-based feature indices
    for hid in radiant_team:
        idx = hero_id_to_index.get(str(hid), hero_id_to_index.get(hid))
        if idx is not None:
            x[int(idx) - 1] = 1.0

    for hid in dire_team:
        idx = hero_id_to_index.get(str(hid), hero_id_to_index.get(hid))
        if idx is not None:
            x[int(idx) - 1] = 1.0

    theta = get_theta()
    prob_radiant = float(predict_single_value(theta, x))
    prob_dire = float(1.0 - prob_radiant)

    return jsonify({"radiant": prob_radiant, "dire": prob_dire})



# -------------------------------------------------------------------------------
# Helpers (heroes + hero_analysis)
# -------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_heroes() -> Dict[int, Dict[str, Any]]:
    heroes_path = DATA_DIR / "heroes.json"
    with open(heroes_path, "r", encoding="utf-8") as fp:
        raw = json.load(fp)

    by_id: Dict[int, Dict[str, Any]] = {}
    if isinstance(raw, list):
        for h in raw:
            if "id" in h:
                hid = int(h["id"])
                h["image"] = f"img/avatar-sb/{hid}.png"
                by_id[hid] = h
    elif isinstance(raw, dict):
        for obj in raw.values():
            if "id" in obj:
                hid = int(obj["id"])
                obj["image"] = f"img/avatar-sb/{hid}.png"
                by_id[hid] = obj
    else:
        raise ValueError("Unsupported heroes.json format")
    return by_id


@lru_cache(maxsize=1)
def get_hero_analysis() -> pd.DataFrame:
    path = DATA_DIR / "hero_analysis.csv"
    df = pd.read_csv(path)
    for col in ("best_matchups", "worst_counters"):
        if df[col].dtype == object and isinstance(df[col].iloc[0], str):
            try:
                df[col] = df[col].apply(json.loads)
            except Exception:
                df[col] = df[col].apply(eval)
    return df


# -------------------------------------------------------------------------------
# Dev entry point
# -------------------------------------------------------------------------------

if __name__ == "__main__":
    app.jinja_env.cache = {}
    app.run(debug=debug_mode, host="127.0.0.1", port=frontend_port)

# -------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np  # <-- neu für ML Prediction
import pandas as pd
from flask import Flask, render_template, request, jsonify, url_for

# -------------------------------------------------------------------------------
# Paths & Python import setup
# -------------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
    top_n = int(data.get("top_n", 5))
    ranked, warnings = recommend_pick(radiant, dire, top_n=top_n)
    return jsonify({"picks": ranked, "warnings": warnings})

@app.route("/radiant_ban", methods=["POST"])
def radiant_ban():
    data = request.get_json(force=True) or {}
    radiant = data.get("radiant", [])
    dire = data.get("dire", [])
    top_n = int(data.get("top_n", 5))
    ranked, warnings = recommend_ban(radiant, dire, top_n=top_n)
    return jsonify({"bans": ranked, "warnings": warnings})

@app.route("/dire_pick", methods=["POST"])
def dire_pick():
    data = request.get_json(force=True) or {}
    radiant = data.get("radiant", [])
    dire = data.get("dire", [])
    top_n = int(data.get("top_n", 5))
    ranked, warnings = recommend_pick(dire, radiant, top_n=top_n)
    return jsonify({"picks": ranked, "warnings": warnings})

@app.route("/dire_ban", methods=["POST"])
def dire_ban():
    data = request.get_json(force=True) or {}
    radiant = data.get("radiant", [])
    dire = data.get("dire", [])
    top_n = int(data.get("top_n", 5))
    ranked, warnings = recommend_ban(dire, radiant, top_n=top_n)
    return jsonify({"bans": ranked, "warnings": warnings})

# -------------------------------------------------------------------------------
# Winner Prediction API (neu)
# -------------------------------------------------------------------------------

@app.route("/predict_winner", methods=["POST"])
def predict_winner():
    data = request.get_json(force=True) or {}
    radiant_team: List[int] = data.get("radiant", [])
    dire_team: List[int] = data.get("dire", [])

    num_heroes = len(get_heroes())
    x = np.zeros(num_heroes)
    for hid in radiant_team:
        x[hid - 1] = 1
    for hid in dire_team:
        x[hid - 1] = -1

    theta = get_theta()
    prob_radiant = predict_single_value(theta, x)
    prob_dire = 1 - prob_radiant

    return jsonify({"radiant": float(prob_radiant), "dire": float(prob_dire)})

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

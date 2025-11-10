# app_dashboard.py
# KKD - Real Estate Dashboard (Price -> Days on Market)
# Dark UI, robust feature alignment, metrics in KPIs, numeric price in dcc.Store

import json, pickle, re
from datetime import datetime
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

# =========================
# Load required pickle files
# =========================
with open("./pickle/parameters.pkl", "rb") as f:
    PARAMS = pickle.load(f)  # {col: {"mean":..., "std":...}, ...}

with open("./pickle/theta_price_model.pkl", "rb") as f:
    THETA_PRICE = pickle.load(f)  # [bias, w1, ...]

with open("./pickle/theta_days_model.pkl", "rb") as f:
    THETA_DAYS = pickle.load(f)   # [bias, w1, ...]

with open("./pickle/price_features.pkl", "rb") as f:
    PRICE_FEATS = pickle.load(f)  # list[str] in training order

with open("./pickle/soldtime_features.pkl", "rb") as f:
    DAYS_FEATS = pickle.load(f)   # list[str] in training order

with open("./pickle/district_linkage.pkl", "rb") as f:
    DISTRICT_LINK = pickle.load(f)  # {district_id: {..., "schools":[...] } }

# agent_names.pkl can be either a list or a dict bundle
with open("./pickle/agent_names.pkl", "rb") as f:
    AGENT_NAMES = pickle.load(f)

with open("./pickle/metrics.pkl", "rb") as f:
    METRICS = pickle.load(f)

# =========================
# Helpers (norm, vec, pred)
# =========================
def _mu(col):  return float(PARAMS[col]["mean"])
def _sd(col):  return float(PARAMS[col]["std"])

def _z(val, col):
    sd = _sd(col)
    return 0.0 if sd == 0 else (float(val) - _mu(col)) / sd

def _denorm(val, col):
    return float(val) * _sd(col) + _mu(col)

def _predict(theta, x_vec):
    b = float(theta[0])
    w = np.asarray(theta[1:], dtype=float)
    x = np.asarray(x_vec, dtype=float)
    return b + np.dot(x, w)

def vectorize(sample_dict, feature_list):
    """z-scale values using PARAMS for features present; keep raw for one-hots (not in PARAMS)."""
    out = []
    for col in feature_list:
        raw = sample_dict.get(col, 0.0)
        if col in PARAMS:
            out.append(_z(raw, col))
        else:
            out.append(float(raw))
    return np.array(out, dtype=float)

# -------------------------
# Feature sanitizers & alignment
# -------------------------
def _sanitize_feats(feats):
    """Fix known typos like 'price_per_m2bathrooms'."""
    fixed = []
    for col in feats:
        if col == "price_per_m2bathrooms":
            if "price_per_m2" not in fixed:
                fixed.append("price_per_m2")
            if "bathrooms" not in feats and "bathrooms" not in fixed:
                fixed.append("bathrooms")
        else:
            fixed.append(col)
    return fixed

PRICE_FEATS = _sanitize_feats(PRICE_FEATS)
DAYS_FEATS  = _sanitize_feats(DAYS_FEATS)

def _align_feats_to_theta(feats, theta, label="MODEL"):
    """Ensure len(feats) == len(theta)-1 by dropping baseline dummies deterministically."""
    target_len = len(theta) - 1
    f = list(feats)

    if len(f) == target_len:
        return f

    # 1) Drop advertising baseline if present (often 'adv_no')
    if len(f) > target_len and "adv_no" in f and (("adv_regular" in f) or ("adv_premium" in f)):
        f.remove("adv_no")

    # 2) Drop one month baseline (January) to go from 12 -> 11
    month_cols = [c for c in f if c.startswith("month_")]
    if len(f) > target_len and len(month_cols) == 12 and "month_January" in f:
        f.remove("month_January")

    # 3) Drop one agent baseline if still too long
    if len(f) > target_len:
        agent_cols = [c for c in f if c.startswith("agent_")]
        for c in agent_cols:
            if len(f) <= target_len:
                break
            f.remove(c)

    # 4) Drop columns not in PARAMS (often one-hots) until lengths match
    if len(f) > target_len:
        for c in list(f):
            if len(f) <= target_len:
                break
            if c not in PARAMS:
                f.remove(c)

    # 5) Final trim if still too long
    if len(f) > target_len:
        f = f[:target_len]

    return f

ALIGNED_PRICE_FEATS = _align_feats_to_theta(PRICE_FEATS, THETA_PRICE, "PRICE")
ALIGNED_DAYS_FEATS  = _align_feats_to_theta(DAYS_FEATS,  THETA_DAYS,  "DAYS")

# -------------------------
# Metrics formatting helpers
# -------------------------
def _fmt_int(x):
    try: return f"{float(x):,.0f}"
    except: return "—"

def _fmt_1(x):
    try: return f"{float(x):.1f}"
    except: return "—"

def _metrics_line(kind: str):
    """Return a compact metrics string for 'price' or 'days' from METRICS pickle."""
    if not METRICS:
        return ""
    if kind == "price":
        r2   = METRICS.get("R^2_price"); rmse = METRICS.get("RMSE_price"); mae = METRICS.get("MAE_price")
        if r2 is None: return ""
        return f"R² {r2:.2f} • RMSE {_fmt_int(rmse)} NOK • MAE {_fmt_int(mae)} NOK"
    if kind == "days":
        r2   = METRICS.get("R^2_days"); rmse = METRICS.get("RMSE_days"); mae = METRICS.get("MAE_days")
        if r2 is None: return ""
        return f"R² {r2:.2f} • RMSE {_fmt_1(rmse)} days • MAE {_fmt_1(mae)} days"
    return ""

# -------------------------
# UI constants
# -------------------------
COLOR_OPTIONS = ["black", "blue", "gray", "green", "red", "unknown", "white"]
MONTHS_FULL   = ["January","February","March","April","May","June","July",
                 "August","September","October","November","December"]

# ==============
# Dash App Setup (Dark theme + cards)
# ==============
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "KKD - Real Estate Dashboard"

def kpi(title, value, sub=None, color="light"):
    return dbc.Card(
        dbc.CardBody([
            html.Div(title, className="text-muted small"),
            html.Div(value, className=f"display-6 text-{color}"),
            html.Div(sub or "", className="small text-secondary")
        ]),
        className="shadow-sm border-0", style={"borderRadius":"1rem"}
    )

def card(title, body):
    return dbc.Card(
        dbc.CardBody([html.H5(title, className="mb-3"), body]),
        className="mb-4 shadow-sm border-0", style={"borderRadius":"1rem"}
    )

# --- Layout -------------------------------------------------------------------
app.layout = dbc.Container([
    html.H2("KKD - Real Estate Dashboard", className="text-center my-4"),

    # Store for numeric price (used by Days prediction)
    dcc.Store(id="store_price"),

    # PRICE SECTION
    dbc.Row([
        dbc.Col(card("Section 1 - Price prediction (NOK)", html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("District"),
                    dcc.Dropdown(
                        id="district_id",
                        options=[{"label": f"District {d}", "value": d} for d in DISTRICT_LINK.keys()],
                        placeholder="Select a district...", style={"color":"black"}
                    ),
                    html.Br(),
                    html.Label("School"),
                    dcc.Dropdown(id="school_id", placeholder="Select a school...", style={"color":"black"}),
                    html.Br(),
                    html.Label("Condition rating (0-10)"),
                    dcc.Slider(id="condition_rating", min=0, max=10, step=0.5, value=6,
                               marks={0:"0",5:"5",10:"10"}, tooltip={"placement":"bottom"}),
                    html.Br(),
                    html.Label("Bathrooms"),
                    dcc.Input(id="bathrooms", type="number", value=2, style={"width":"100%"}),
                    html.Br(), html.Label("Kitchens"),
                    dcc.Input(id="kitchens", type="number", value=1, style={"width":"100%"}),
                    html.Br(), html.Label("Rooms"),
                    dcc.Input(id="rooms", type="number", value=3, style={"width":"100%"}),
                    html.Br(), html.Label("Lot width"),
                    dcc.Input(id="lot_w", type="number", value=50, style={"width":"100%"}),
                    html.Br(), html.Label("Size (m²)"),
                    dcc.Input(id="size", type="number", value=120, style={"width":"100%"}),
                    html.Br(), html.Label("External storage (m²)"),
                    dcc.Input(id="external_storage_m2", type="number", value=5, style={"width":"100%"}),
                    html.Br(), html.Label("Storage rating (0-10)"),
                    dcc.Slider(id="storage_rating", min=0, max=10, step=0.5, value=5,
                               marks={0:"0",5:"5",10:"10"}, tooltip={"placement":"bottom"}),
                ], width=6),
                dbc.Col([
                    html.Label("Year built"),
                    dcc.Input(id="year_built", type="number", value=1990, style={"width":"100%"}),
                    html.Br(), html.Label("Remodeled year"),
                    dcc.Input(id="remodeled_year", type="number", value=2015, style={"width":"100%"}),
                    html.Br(), html.Label("Fireplace"),
                    dbc.Checklist(id="fireplace", options=[{"label":"Yes","value":1}], value=[]),
                    html.Br(), html.Label("Parking"),
                    dbc.Checklist(id="parking", options=[{"label":"Yes","value":1}], value=[]),
                    html.Br(), html.Label("Sun factor (0-1)"),
                    dcc.Slider(id="sun_factor", min=0, max=1, step=0.01, value=0.6,
                               marks={0:"0",0.5:"0.5",1:"1.0"}, tooltip={"placement":"bottom"}),
                    html.Br(), html.Label("Exterior color"),
                    dcc.Dropdown(
                        id="color", options=[{"label": c.capitalize(), "value": c} for c in COLOR_OPTIONS],
                        value="blue", style={"color":"black"}
                    ),
                    html.Br(),
                    dbc.Button("Predict price", id="btn_price", color="success", className="w-100"),
                    html.Div(id="out_price", className="mt-3"),
                ], width=6)
            ])
        ])), width=12)
    ]),

    # DAYS SECTION
    dbc.Row([
        dbc.Col(card("Section 2 - Days on market prediction", html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Month listed"),
                    dcc.Dropdown(
                        id="month_listed",
                        options=[{"label":m,"value":m} for m in MONTHS_FULL],
                        value="November", style={"color":"black"}
                    ),
                    html.Br(), html.Label("Advertising"),
                    dbc.RadioItems(
                        id="ad_type",
                        options=[{"label":"None","value":"none"},
                                 {"label":"Regular","value":"regular"},
                                 {"label":"Premium","value":"premium"}],
                        value="none", inline=True
                    ),
                    html.Br(), html.Label("Agent"),
                    dcc.Dropdown(
                        id="agent_name",
                        options=[{"label":a,"value":a} for a in AGENT_NAMES],
                        value=AGENT_NAMES[0] if AGENT_NAMES else None,
                        style={"color":"black"}
                    ),
                    html.Br(),
                    dbc.Button("Predict days", id="btn_days", color="primary", className="w-100"),
                    html.Div(id="out_days", className="mt-3"),
                ], width=6),
                dbc.Col([
                    dbc.Row([
                        dbc.Col(kpi("Predicted price", "—", "NOK", color="success"), id="kpi_price", width=6),
                        dbc.Col(kpi("Predicted days",  "—", "days", color="info"),  id="kpi_days",  width=6),
                    ])
                ], width=6)
            ])
        ])), width=12)
    ]),
], fluid=True)

# =========================
# Callbacks
# =========================

# District -> School options
@app.callback(
    Output("school_id", "options"),
    Input("district_id", "value"),
)
def cb_schools(district_id):
    if not district_id or district_id not in DISTRICT_LINK:
        return []
    schools = DISTRICT_LINK[district_id].get("schools", [])
    return [{"label": f"{s.get('name','School')} ({s['id']})", "value": s["id"]} for s in schools]

# Price prediction (NOK) + store numeric price
@app.callback(
    Output("out_price", "children"),
    Output("kpi_price", "children"),
    Output("store_price", "data"),   # numeric NOK
    Input("btn_price", "n_clicks"),
    State("district_id", "value"), State("school_id", "value"),
    State("condition_rating", "value"),
    State("bathrooms", "value"), State("kitchens", "value"), State("rooms", "value"),
    State("lot_w", "value"), State("size", "value"), State("external_storage_m2", "value"),
    State("storage_rating", "value"),
    State("year_built", "value"), State("remodeled_year", "value"),
    State("fireplace", "value"), State("parking", "value"),
    State("sun_factor", "value"), State("color", "value"),
)
def cb_price(n,
             district_id, school_id,
             condition_rating,
             bathrooms, kitchens, rooms,
             lot_w, size, external_storage_m2,
             storage_rating,
             year_built, remodeled_year,
             fireplace, parking,
             sun_factor, color):
    if not n:
        return "", kpi("Predicted price", "—", "NOK", color="success"), None

    if not district_id or district_id not in DISTRICT_LINK:
        return "Select district & school first.", kpi("Predicted price", "—", "NOK", color="success"), None
    schools = DISTRICT_LINK[district_id].get("schools", [])
    school = next((s for s in schools if s["id"] == school_id), None)
    if school is None:
        return "Select district & school first.", kpi("Predicted price", "—", "NOK", color="success"), None

    now = datetime.now()
    district_cr = DISTRICT_LINK[district_id].get("crime_rating")
    district_pt = DISTRICT_LINK[district_id].get("public_transport_rating")
    school_rating   = school["rating"]
    school_capacity = school["capacity"]
    school_age      = now.year - int(school["built_year"])

    has_fireplace = 1 if fireplace else 0
    has_parking   = 1 if parking else 0
    color_onehot  = {f"color_{c}": (1 if color == c else 0) for c in COLOR_OPTIONS}

    sample_price = {
        "bathrooms": bathrooms,
        "condition_rating": condition_rating,
        "external_storage_m2": external_storage_m2,
        "kitchens": kitchens,
        "lot_w": lot_w,
        "rooms": rooms,
        "size": size,
        "storage_rating": storage_rating,
        "school_age": school_age,
        "house_age": now.year - int(year_built) if year_built else 0,
        "remodel_age": now.year - int(remodeled_year) if remodeled_year else 0,
        "fireplace": has_fireplace,
        "parking": has_parking,
        "district_crime_rating": district_cr,
        "district_public_transport_rating": district_pt,
        "school_rating": school_rating,
        "school_capacity": school_capacity,
        "sun_factor": float(sun_factor) if sun_factor is not None else 0.0,
        **color_onehot,
    }

    x_price  = vectorize(sample_price, ALIGNED_PRICE_FEATS)
    yhat_z   = _predict(THETA_PRICE, x_price)
    yhat_nok = _denorm(yhat_z, "price")

    text = f"Predicted price: NOK {yhat_nok:,.0f}"
    sub  = _metrics_line("price")

    out_text = html.Div([
        html.Div(text, style={"fontWeight": "600", "color": "#2ecc71"}),
        html.Div(sub,  style={"opacity": 0.8, "fontSize": "0.9rem"})
    ])
    out_kpi  = kpi("Predicted price", f"{yhat_nok:,.0f}", sub or "NOK", color="success")

    return out_text, out_kpi, float(yhat_nok)

# Days on Market prediction (uses numeric price from store)
@app.callback(
    Output("out_days", "children"),
    Output("kpi_days", "children"),
    Input("btn_days", "n_clicks"),
    State("store_price", "data"),   # numeric price from cb_price
    State("month_listed", "value"),
    State("ad_type", "value"),
    State("agent_name", "value"),
    State("district_id", "value"),
    State("school_id", "value"),
    State("bathrooms", "value"),
    State("condition_rating", "value"),
    State("kitchens", "value"),
    State("lot_w", "value"),
    State("rooms", "value"),
    State("size", "value"),
    State("sun_factor", "value"),
    State("year_built", "value"),
    State("remodeled_year", "value"),
)
def cb_days(n, pred_price,
            month, ad_type, agent_name,
            district_id, school_id,
            bathrooms, condition_rating, kitchens, lot_w, rooms, size, sun_factor,
            year_built, remodeled_year):
    if not n:
        return "", kpi("Predicted days", "—", "days", color="info")

    if pred_price is None:
        return "Predict price first.", kpi("Predicted days", "—", "days", color="info")

    if not district_id or district_id not in DISTRICT_LINK:
        return "Select district & school first.", kpi("Predicted days", "—", "days", color="info")
    schools = DISTRICT_LINK[district_id].get("schools", [])
    school = next((s for s in schools if s["id"] == school_id), None)
    if school is None:
        return "Select district & school first.", kpi("Predicted days", "—", "days", color="info")

    district_cr = DISTRICT_LINK[district_id].get("crime_rating")
    district_pt = DISTRICT_LINK[district_id].get("public_transport_rating")
    school_rating   = school["rating"]
    school_capacity = school["capacity"]

    # Derived price features
    mean_price     = _mu("price")
    price_relative = pred_price / mean_price if mean_price else 1.0
    price_per_m2   = pred_price / float(size) if size not in (None, 0) else _mu("price_per_m2")

    # One-hots only if present in aligned feature list
    month_onehots = {f"month_{m}": (1 if m == month else 0)
                     for m in MONTHS_FULL if f"month_{m}" in ALIGNED_DAYS_FEATS}
    adv = {
        "adv_no":       1 if ad_type == "none"    else 0,
        "adv_regular":  1 if ad_type == "regular" else 0,
        "adv_premium":  1 if ad_type == "premium" else 0,
    }
    adv = {k: v for k, v in adv.items() if k in ALIGNED_DAYS_FEATS}
    agent_onehots = {
        f"agent_{a}": (1 if a == agent_name else 0)
        for a in AGENT_NAMES if f"agent_{a}" in ALIGNED_DAYS_FEATS
    }

    now = datetime.now()
    house_age   = now.year - int(year_built)     if year_built     else 0
    remodel_age = now.year - int(remodeled_year) if remodeled_year else 0

    sample_days = {
        "price": pred_price,
        "price_relative": price_relative,
        "price_per_m2": price_per_m2,
        "bathrooms": bathrooms,
        "condition_rating": condition_rating,
        "kitchens": kitchens,
        "lot_w": lot_w,
        "rooms": rooms,
        "house_age": house_age,
        "remodel_age": remodel_age,
        "sun_factor": float(sun_factor) if sun_factor is not None else 0.0,
        "district_crime_rating": district_cr,
        "district_public_transport_rating": district_pt,
        "school_rating": school_rating,
        "school_capacity": school_capacity,
        **month_onehots,
        **adv,
        **agent_onehots,
    }

    # Vectorize with aligned features
    for col in ALIGNED_DAYS_FEATS:
        sample_days.setdefault(col, 0.0)

    x_days = vectorize(sample_days, ALIGNED_DAYS_FEATS)
    if x_days.shape[0] != (len(THETA_DAYS) - 1):
        msg = f"[ERROR] Feature/weight mismatch: x={x_days.shape[0]} vs w={len(THETA_DAYS)-1}"
        return msg, kpi("Predicted days", "—", "days", color="danger")

    yhat_z = _predict(THETA_DAYS, x_days)
    yhat_d = _denorm(yhat_z, "days_on_market")

    text = f"Predicted days on market: {yhat_d:.1f} days"
    sub  = _metrics_line("days")

    out_text = html.Div([
        html.Div(text, style={"fontWeight": "600", "color": "#8bd3ff"}),
        html.Div(sub,  style={"opacity": 0.8, "fontSize": "0.9rem"})
    ])
    out_kpi  = kpi("Predicted days", f"{yhat_d:.1f}", sub or "days", color="info")

    return out_text, out_kpi

# ==========
# Run app
# ==========
if __name__ == "__main__":
    app.run(debug=True)
"""
Knut Knut Transport — Flask UI (Route Recommendation)
-----------------------------------------------------

Purpose
  Serve a small web UI that recommends the fastest route based on a pre-trained
  Fortuna model bundle stored in 'fast_and_furious_AI.pkl'.

How it works
  - Reads the model bundle (pickle) and retrieves z-score scaler (mu, sigma).
  - Converts user input (HH, MM) to minutes since midnight and scales it.
  - Predicts duration for all four routes (linear/quadratic as trained), picks
    the minimum, and renders a modern, readable result page (HTML/CSS only).

Endpoints
  - GET /                → simple form for departure time
  - GET /get_best_route  → computes predictions & shows recommendation

Assumptions about the model bundle
  - Keys:
      Scaling: { "mu": float, "sigma": float }
      ACE|ACD|BCE|BCD: { "theta": array-like, "scaler": bool }
  - Model mapping (must match training):
      ACD → polynomial_model
      ACE → linear_model
      BCD → polynomial_model
      BCE → linear_model
  - Display order aligns predictions with roads:
      est_travel_times = [ACD, ACE, BCD, BCE]
      roads            = ["A->C->D", "A->C->E", "B->C->D", "B->C->E"]

Enhancements & adjustments (with assistance from ChatGPT — GPT-5 Thinking)
  - Modernized UI: lightweight responsive HTML/CSS (no external libs), clearer layout.
  - ETA shown (adds predicted minutes to departure).
  - Transparent table of predicted times for all routes.
  - Savings vs. a randomly sampled route + simple cost estimate (NOK/min).
  - Robust model path resolution and readable error messages instead of a blank 500.
  - Sanity checks for the pickle structure (required keys present).
  - Guard against sigma == 0 to avoid division by zero.
  - Basic input validation for hour/minute.
  - Corrected savings sign: saved_minutes = random_time - best_time (≥ 0).
  - Small helpers: minutes_since_midnight, add_minutes.
  - Tip to cache the bundle for performance if needed.

Configuration
  - MODEL_PATH points to the pickle next to this file by default.
  - Use app.run(debug=True) during local development to see tracebacks.

Version
  __version__ = "1.0"
"""
__version__ = "1.0"

import random
import pickle
from pathlib import Path
from datetime import timedelta

from flask import Flask, request, abort

app = Flask(__name__)

# ---- Model path (robust relativ zu dieser Datei) --------------------
MODEL_PATH = Path(__file__).resolve().parent / "fast_and_furious_AI.pkl"

# ---------------------------- Helpers --------------------------------
def minutes_since_midnight(hour, mins):
    """Convert hour+minute (strings or ints) to total minutes since midnight."""
    return int(hour) * 60 + int(mins)

def add_minutes(h, m, delta_min):
    """Return HH:MM after adding delta_min to (h:m)."""
    total = (int(h) * 60 + int(m) + int(delta_min)) % (24 * 60)
    hh = total // 60
    mm = total % 60
    return f"{hh:02d}:{mm:02d}"

def read_model_data():
    """
    Load the trained model bundle (pickle).
    Raises a clear RuntimeError if file not found or not loadable.
    """
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found at: {MODEL_PATH}")
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load pickle: {e}")
    return model

def sanity_check_model_bundle(model: dict):
    """
    Validate the minimal structure we rely on.
    Raises RuntimeError with a clear message if something is missing.
    """
    required_top = ["Scaling", "ACE", "ACD", "BCE", "BCD"]
    for k in required_top:
        if k not in model:
            raise RuntimeError(f"Missing key in model bundle: '{k}'")
    scaling = model["Scaling"]
    if "mu" not in scaling or "sigma" not in scaling:
        raise RuntimeError("Scaling must contain 'mu' and 'sigma'")
    # Ensure each route has theta and scaler flag
    for r in ["ACE", "ACD", "BCE", "BCD"]:
        if "theta" not in model[r] or "scaler" not in model[r]:
            raise RuntimeError(f"Route '{r}' must have 'theta' and 'scaler'")

# Simple, explainable models (must match your training setup)
def linear_model(x, theta, scaler=False):
    if scaler:
        return (x * theta[0] + theta[1]) * theta[2]
    return x * theta[0] + theta[1]

def polynomial_model(x, theta, scaler=False):
    if scaler:
        return (theta[0] * (x**2) + theta[1] * x + theta[2]) * theta[3]
    return theta[0] * (x**2) + theta[1] * x + theta[2]

# ---------------------------- Core route logic -------------------------
def get_the_best_route_as_a_text_informatic(dep_hour, dep_min):
    """
    Predict travel time for each route at the given departure time,
    choose the minimum, and render a modernized HTML view.
    """
    roads = ["A->C->D", "A->C->E", "B->C->D", "B->C->E"]

    # Load + sanity check model bundle
    model = read_model_data()
    sanity_check_model_bundle(model)

    # z-score scaling parameters for departure time (guard sigma)
    mu = float(model["Scaling"]["mu"])
    sigma = float(model["Scaling"]["sigma"]) if float(model["Scaling"]["sigma"]) != 0 else 1.0

    # Prepare scaled input (minutes since midnight -> z-score)
    dep_time_min = minutes_since_midnight(dep_hour, dep_min)
    t = (dep_time_min - mu) / sigma

    # Predict per route (order must match 'roads')
    ACD = polynomial_model(t, model["ACD"]["theta"], scaler=model["ACD"]["scaler"])
    ACE = linear_model(t, model["ACE"]["theta"], scaler=model["ACE"]["scaler"])
    BCD = polynomial_model(t, model["BCD"]["theta"], scaler=model["BCD"]["scaler"])
    BCE = linear_model(t, model["BCE"]["theta"], scaler=model["BCE"]["scaler"])

    est_travel_times = [ACD, ACE, BCD, BCE]  # aligned to 'roads'

    # Pick the best route
    best_index = est_travel_times.index(min(est_travel_times))
    best_road = roads[best_index]
    best_time = round(float(est_travel_times[best_index]))

    # Compare vs. a random route (at the same departure)
    random_road = random.choice(roads)
    random_time = round(float(est_travel_times[roads.index(random_road)]))

    saved_minutes = max(0, random_time - best_time)
    cost_per_minute = 1100 / 60.0
    cost_saving = round(saved_minutes * cost_per_minute, 2)
    eta = add_minutes(dep_hour, dep_min, best_time)

    # small table for transparency
    table_rows = ""
    for r, t_est in zip(roads, est_travel_times):
        table_rows += f"""
            <tr>
              <td>{r}</td>
              <td>{round(float(t_est))} min</td>
            </tr>"""

    # Modernized HTML (same wie zuvor)
    out = f"""
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Knut Knut Transport — Route Recommendation</title>
      <style>
        :root {{
          --bg: #0f172a; --panel: #111827; --muted: #94a3b8; --text: #e5e7eb;
          --accent: #22c55e; --accent2: #38bdf8; --card: #1f2937; --border: #334155;
        }}
        body {{ margin:0; padding:24px; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans;
               background: radial-gradient(1200px 600px at 10% -10%, #1e293b 10%, transparent 40%), var(--bg); color: var(--text); }}
        .container {{ max-width: 860px; margin: 0 auto; }}
        .header {{ display:flex; align-items:center; justify-content:space-between; margin-bottom: 18px; }}
        .brand {{ font-weight: 700; letter-spacing: 0.4px; font-size: 20px; color: var(--muted); }}
        .card {{ background: linear-gradient(180deg, var(--card), #111827); border: 1px solid var(--border);
                 border-radius: 16px; padding: 20px; box-shadow: 0 10px 24px rgba(0,0,0,0.35); margin-bottom: 16px; }}
        .title {{ margin: 0 0 12px 0; font-size: 22px; letter-spacing: .2px; }}
        .grid {{ display:grid; gap:12px; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); }}
        .stat {{ border: 1px solid var(--border); border-radius:12px; padding:12px; background: #0b1220; }}
        .stat .label {{ color: var(--muted); font-size: 12px; }}
        .stat .value {{ font-size: 22px; font-weight: 700; }}
        .best {{ color: var(--accent); }}
        .table {{ width:100%; border-collapse: collapse; margin-top: 8px; }}
        .table th, .table td {{ text-align:left; padding: 8px 10px; border-bottom: 1px solid var(--border); }}
        .btn {{ display:inline-block; margin-top:10px; padding:10px 14px; background: var(--accent2); color:#0b1220; font-weight:700;
                text-decoration:none; border-radius:10px; border:1px solid #0ea5e9; }}
        .muted {{ color: var(--muted); }}
        .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }}
      </style>
    </head>
    <body>
      <div class="container">

        <div class="header">
          <div class="brand">Knut Knut Transport · Smart Route</div>
          <div class="muted mono">AI model: Fortuna (quadratic/linear)</div>
        </div>

        <div class="card">
          <h2 class="title">Route Recommendation</h2>
          <div class="grid">
            <div class="stat">
              <div class="label">Departure</div>
              <div class="value mono">{int(dep_hour):02d}:{int(dep_min):02d}</div>
            </div>
            <div class="stat">
              <div class="label">Best route</div>
              <div class="value best">{best_road}</div>
            </div>
            <div class="stat">
              <div class="label">Estimated travel time</div>
              <div class="value mono">{best_time} min</div>
            </div>
            <div class="stat">
              <div class="label">ETA (arrival)</div>
              <div class="value mono">{eta}</div>
            </div>
          </div>

          <h3 style="margin-top:16px;">Predicted times by route</h3>
          <table class="table mono">
            <thead><tr><th>Route</th><th>Estimated time</th></tr></thead>
            <tbody>
              {table_rows}
            </tbody>
          </table>

          <h3 style="margin-top:16px;">Savings (vs. random route)</h3>
          <div class="grid">
            <div class="stat">
              <div class="label">Random route sampled</div>
              <div class="value mono">{random_road}</div>
            </div>
            <div class="stat">
              <div class="label">Saved time</div>
              <div class="value mono">{saved_minutes} min</div>
            </div>
            <div class="stat">
              <div class="label">Estimated cost saving</div>
              <div class="value mono">{cost_saving} NOK</div>
            </div>
          </div>

          <a class="btn" href="/">← Back</a>
          <div class="muted" style="margin-top:8px;">Tip: try different minutes to see the policy switch.</div>
        </div>

      </div>
    </body>
    </html>
    """
    return out

# ---------------------------- Web UI routes ----------------------------------
@app.route('/')
def get_departure_time():
    return """
    <html>
    <head><meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Knut Knut Transport — Select departure</title></head>
    <body style="background:#0f172a;color:#e5e7eb;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans;">
      <div style="max-width:640px;margin:0 auto;padding:24px;">
        <div style="background:linear-gradient(180deg,#1f2937,#111827);border:1px solid #334155;border-radius:16px;padding:20px;">
          <h3 style="margin-top:0;">Knut Knut Transport — Choose departure</h3>
          <form action="/get_best_route" method="get">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
              <div>
                <label for="hour" style="display:block;margin:8px 0 4px;color:#94a3b8;">Hour</label>
                <select name="hour" id="hour" required style="width:100%;padding:10px 12px;border-radius:10px;border:1px solid #334155;background:#0b1220;color:#e5e7eb;">
                  <option value="06">06</option><option value="07">07</option><option value="08">08</option>
                  <option value="09">09</option><option value="10">10</option><option value="11">11</option>
                  <option value="12">12</option><option value="13">13</option><option value="14">14</option>
                  <option value="15">15</option><option value="16">16</option>
                </select>
              </div>
              <div>
                <label for="mins" style="display:block;margin:8px 0 4px;color:#94a3b8;">Minutes</label>
                <input type="number" name="mins" id="mins" min="0" max="59" step="1" value="00" required
                       style="width:100%;padding:10px 12px;border-radius:10px;border:1px solid #334155;background:#0b1220;color:#e5e7eb;">
              </div>
            </div>
            <button type="submit" style="display:inline-block;margin-top:14px;padding:10px 14px;background:#38bdf8;color:#0b1220;font-weight:700;text-decoration:none;border-radius:10px;border:1px solid #0ea5e9;">Get best route →</button>
          </form>
        </div>
      </div>
    </body>
    </html>
    """

@app.route("/get_best_route")
def get_route():
    try:
        departure_h = request.args.get('hour', '07')
        departure_m = request.args.get('mins', '00')
        # Basic input validation
        if not departure_h.isdigit() or not departure_m.isdigit():
            abort(400, description="Invalid input: hour/min must be digits.")
        if not (0 <= int(departure_h) <= 23 and 0 <= int(departure_m) <= 59):
            abort(400, description="Invalid time range.")
        return get_the_best_route_as_a_text_informatic(departure_h, departure_m)
    except Exception as e:
        # Render a readable error page instead of a blank 500
        return f"<h3>Internal error</h3><pre>{e}</pre><p><a href='/'>Back</a></p>", 500

# ---------------------------- Entrypoint -------------------------------------
if __name__ == '__main__':
    print("<starting>")
    app.run(debug=True)
    print("<done>")

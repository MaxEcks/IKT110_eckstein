# agent_anti_crowd_simple.py
# Einfacher, erklärbarer Auktions-Bot:
# - Marktpreis (Gold/Punkt) = Median aus Vor-Runde (EMA geglättet)
# - Burn-Plan: von Anfang an sinnvoll ausgeben (gold/remaining), leicht konträr (früh etwas mehr)
# - Auswahl per Softmax statt Top-EV (unvorhersagbar, anti-crowd)
# - Aggro-Puls: lognormal (seltene starke Pushes), plus kleiner Jitter
# - Optionales Live-Plot (ein einziges Fenster) mit Matplotlib: LIVE_PLOT=1

import os, random, statistics, math, collections
from typing import Dict, Any
from dnd_auction_game import AuctionGameClient

# -------------------- Konfig (einfach anpassbar via ENV) --------------------
AVG = {2:1.5,3:2.0,4:2.5,6:3.5,8:4.5,10:5.5,12:6.5,20:10.5}

EMA_ALPHA      = float(os.getenv("EMA_ALPHA", "0.15"))   # Glättung cp
HARD_CAP       = int(os.getenv("HARD_CAP",   "4000"))    # Kappe pro Auktion
EPSILON        = float(os.getenv("EPSILON",  "0.10"))    # ±Jitter (10%)
SELECT_K       = int(os.getenv("SELECT_K",   "3"))       # wie viele Auktionen wir pro Runde ziehen
SOFTMAX_TAU    = float(os.getenv("SOFTMAX_TAU","0.7"))   # Softmax-Temperatur (kleiner = spitzer)
EARLY_BOOST    = float(os.getenv("EARLY_BOOST","1.15"))  # früh: leicht >1x burn vs. gold/remaining
ENDGAME_RATIO  = float(os.getenv("ENDGAME_RATIO","0.10"))# letzte 10%: Flush
PULSE_SIGMA    = float(os.getenv("PULSE_SIGMA","0.25"))  # Lognormal-Puls Stärke (Median=1, sigma≈0.25)
LIVE_PLOT      = os.getenv("LIVE_PLOT", "0") == "1"      # 1 = Matplotlib-Liveplot aktivieren

# -------------------- kleine Utils --------------------
def softmax_choice(items, scores, k, tau):
    # items: list of a_id, scores: list of floats (höher = besser)
    # gibt bis zu k einzigartige items gemäß Softmax zurück
    if not items:
        return []
    scaled = [s / max(1e-9, tau) for s in scores]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]  # stabil
    total = sum(exps)
    if total <= 0:
        return []
    probs = [x/total for x in exps]
    # stochastisch ohne Zurücklegen ziehen
    chosen = []
    pool_items = items[:]
    pool_probs = probs[:]
    for _ in range(min(k, len(pool_items))):
        r = random.random()
        acc = 0.0
        idx = 0
        for i, p in enumerate(pool_probs):
            acc += p
            if r <= acc:
                idx = i
                break
        chosen.append(pool_items.pop(idx))
        pool_probs.pop(idx)
        s = sum(pool_probs)
        if s > 0:
            pool_probs = [p/s for p in pool_probs]
        else:
            break
    return chosen

def lognormal_pulse(sigma):
    # Median = 1.0 bei mu=0.0; sigma steuert die Seltenheit starker Pushes
    return random.lognormvariate(0.0, max(1e-9, sigma))

def clamp_int(x, lo, hi):
    return int(max(lo, min(hi, x)))

# -------------------- Optional: ein einziges Live-Plot-Fenster --------------------
class LivePlot:
    def __init__(self):
        self.enabled = False
        if LIVE_PLOT:
            try:
                import matplotlib.pyplot as plt
                self.plt = plt
                self.enabled = True
                self.fig, (self.ax_money, self.ax_cp) = plt.subplots(2, 1, figsize=(7, 6))
                self.fig.canvas.manager.set_window_title("Auction Bot Live")
                self.rounds = []
                self.money_left = []
                self.points = []
                self.cp_hist = []
                self.plt.ion()
                self.fig.tight_layout()
            except Exception:
                self.enabled = False

    def update(self, rnd, money_left, points, cp):
        if not self.enabled:
            return
        self.rounds.append(rnd)
        self.money_left.append(money_left)
        self.points.append(points)
        self.cp_hist.append(cp)

        # clear+draw (ein Fenster, dynamisch)
        self.ax_money.cla()
        self.ax_money.set_title("Money left & Points")
        self.ax_money.set_xlabel("round")
        self.ax_money.plot(self.rounds, self.money_left, label="money left")
        self.ax_money.plot(self.rounds, self.points, label="points")
        self.ax_money.legend(loc="best")

        self.ax_cp.cla()
        self.ax_cp.set_title("cp_overall (median, EMA)")
        self.ax_cp.set_xlabel("round")
        self.ax_cp.plot(self.rounds, self.cp_hist, label="cp_overall")
        self.ax_cp.legend(loc="best")

        self.fig.tight_layout()
        self.plt.pause(0.001)

# -------------------- Agent --------------------
class MarketAgent:
    def __init__(self):
        self.cp = 30.0  # Startschätzer Gold/Punkt
        self.plot = LivePlot()
        self.last_bids = {}
        # einfache Stats für Plot
        self.money_left = 0
        self.points = 0

    def update_cp(self, prev_auctions: dict):
        samples = []
        for a in prev_auctions.values():
            r = int(a.get("reward", 0))
            bids = a.get("bids", [])
            if r > 0 and bids:
                win = int(bids[0]["gold"])
                samples.append(win / r)
        if samples:
            est = statistics.median(samples)
            self.cp = (1-EMA_ALPHA)*self.cp + EMA_ALPHA*est

    def decide(self, agent_id, current_round, states, auctions, prev_auctions, bank_state):
        # 1) cp updaten + Punkte/Money für Plot (wenn prev vorhanden)
        if prev_auctions:
            self.update_cp(prev_auctions)

        me = states[agent_id]
        gold = int(me["gold"])
        pts  = int(me.get("points", 0))
        self.money_left = gold
        self.points = pts

        # 2) Rundeninfos
        rem = len(bank_state.get("gold_income_per_round", []))
        total = current_round + rem
        # progress nutzen (0..1), aber konträr: früh leicht mehr verbrennen
        progress = (current_round / total) if total > 0 else 0.0

        # 3) Burn-Plan: baseline + early boost, Endgame flush
        baseline = gold if rem == 0 else gold / rem
        early_factor = 1.0 + (EARLY_BOOST - 1.0) * max(0.0, 1.0 - progress)  # früh ~EARLY_BOOST, später ~1.0
        target_spend = baseline * early_factor

        if rem <= max(1, int(ENDGAME_RATIO * total)):
            target_spend = gold  # flush

        round_budget = clamp_int(target_spend, 1, gold) if gold > 0 else 0
        if round_budget <= 0 or not auctions:
            # Plot updaten & nichts bieten
            self.plot.update(current_round, self.money_left, self.points, self.cp)
            return {}

        # 4) Auktionen scoren & per Softmax auswählen
        a_ids, scores, evs = [], [], {}
        for a_id, a in auctions.items():
            die = int(a["die"])
            ev  = a["num"] * AVG[die] + a["bonus"]
            if ev <= 0:
                continue
            score = ev / max(1.0, self.cp)
            a_ids.append(a_id)
            scores.append(score)
            evs[a_id] = ev

        if not a_ids:
            self.plot.update(current_round, self.money_left, self.points, self.cp)
            return {}

        chosen = softmax_choice(a_ids, scores, max(1, SELECT_K), SOFTMAX_TAU)

        # 5) Gebote bestimmen: fair * Puls, cap durch Budget/HARD_CAP
        per_share = max(1, round(round_budget / len(chosen)))
        bids = {}
        for a_id in chosen:
            ev = evs[a_id]
            fair = ev * self.cp
            pulse = lognormal_pulse(PULSE_SIGMA)  # median 1.0, selten >1 Push
            base  = fair * pulse
            # Jitter ±EPSILON
            jitter = random.uniform(1.0 - EPSILON, 1.0 + EPSILON)
            bid = int(base * jitter)
            # Caps
            bid = clamp_int(bid, 1, min(HARD_CAP, per_share, gold))
            if bid > 0:
                bids[a_id] = bid
                gold -= bid
                if gold <= 0:
                    break

        # 6) Live-Plot aktualisieren (ein Fenster)
        self.plot.update(current_round, self.money_left, self.points, self.cp)
        return bids

# -------------------- API-Hook --------------------
def make_bid(agent_id, current_round, states, auctions, prev_auctions, bank_state):
    # Persistente Instanz ohne global:
    if not hasattr(make_bid, "_agent"):
        make_bid._agent = MarketAgent()
    return make_bid._agent.decide(agent_id, current_round, states, auctions, prev_auctions, bank_state)

# -------------------- Standalone-Start --------------------
if __name__ == "__main__":
    host = "localhost"
    agent_name = "{}_{}".format(os.path.basename(__file__), random.randint(1, 1000))
    player_id = "Maximilian Eckstein"
    port = 8095

    game = AuctionGameClient(host=host, agent_name=agent_name, player_id=player_id, port=port)
    try:
        game.run(make_bid)
    except KeyboardInterrupt:
        print("<interrupt - shutting down>")
    print("<game is done>")

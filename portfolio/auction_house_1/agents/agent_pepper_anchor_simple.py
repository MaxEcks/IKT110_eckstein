# agent_pepper_anchor_simple.py
# Test-Variante: "Pepper & Anchor"
# - 1x Anchor (beste EV/cp) bekommt den Hauptteil des Budgets
# - alle anderen positiven EV Auktionen bekommen kleine Pepper-Bids
# - cp (Gold/Punkt) = Median aus Vor-Runde (EMA geglättet)
# - Budget ~ gold/remaining, Endgame-Flush
# - Persistente Agent-Instanz via hasattr(make_bid, "_agent")

import os, random, statistics
from dnd_auction_game import AuctionGameClient

# -------------------- einfache Konfig (per ENV überschreibbar) --------------------
AVG = {2:1.5,3:2.0,4:2.5,6:3.5,8:4.5,10:5.5,12:6.5,20:10.5}

EMA_ALPHA      = float(os.getenv("EMA_ALPHA",      "0.15"))  # Glättung cp
HARD_CAP       = int(os.getenv("HARD_CAP",        "4000"))   # Kappe pro Auktion
ENDGAME_RATIO  = float(os.getenv("ENDGAME_RATIO",  "0.10"))   # letzte 10%: Flush

# Budget-Aufteilung pro Runde
ANCHOR_FRAC    = float(os.getenv("ANCHOR_FRAC",    "0.55"))   # Anteil fürs Anchor-Gebot
PEPPER_FRAC    = float(os.getenv("PEPPER_FRAC",    "0.45"))   # Anteil für Pepper-Gesamtbudget

# Pepper-Bid-Größe (fix + Jitter)
PEPPER_MIN     = int(os.getenv("PEPPER_MIN",       "40"))
PEPPER_MAX     = int(os.getenv("PEPPER_MAX",       "160"))
PEPPER_EPS     = float(os.getenv("PEPPER_EPS",     "0.10"))   # ±10% Jitter

# kleiner Jitter auch auf dem Anchor
ANCHOR_EPS     = float(os.getenv("ANCHOR_EPS",     "0.10"))   # ±10%

# -------------------- kleine Utils --------------------
def clamp_int(x, lo, hi):
    return int(max(lo, min(hi, int(x))))

# -------------------- Agent --------------------
class PepperAnchorAgent:
    def __init__(self):
        self.cp = 30.0  # Startschätzer Gold/Punkt

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
            self.cp = (1 - EMA_ALPHA) * self.cp + EMA_ALPHA * est

    def decide(self, agent_id, current_round, states, auctions, prev_auctions, bank_state):
        # 1) cp updaten
        if prev_auctions:
            self.update_cp(prev_auctions)

        me   = states[agent_id]
        gold = int(me["gold"])

        # 2) Rundenfortschritt & Budget bestimmen
        rem   = len(bank_state.get("gold_income_per_round", []))
        total = current_round + rem
        endgame = (total > 0) and (rem <= max(1, int(ENDGAME_RATIO * total)))

        if gold <= 0 or not auctions:
            return {}

        # Baseline: gleichmäßig verbrennen
        baseline = gold if rem == 0 else gold / rem
        round_budget = gold if endgame else clamp_int(baseline, 1, gold)

        if round_budget <= 0:
            return {}

        # 3) Auktionen scoren (EV/cp)
        scored = []
        for a_id, a in auctions.items():
            die = int(a["die"])
            ev  = a["num"] * AVG[die] + a["bonus"]
            if ev <= 0:
                continue
            score = ev / max(1.0, self.cp)
            scored.append((score, ev, a_id))

        if not scored:
            return {}

        scored.sort(reverse=True)
        anchor_score, anchor_ev, anchor_id = scored[0]

        # 4) Budget aufteilen
        anchor_budget = clamp_int(round_budget * ANCHOR_FRAC, 1, round_budget)
        pepper_budget = clamp_int(round_budget * PEPPER_FRAC, 0, round_budget - anchor_budget)

        bids = {}

        # 5) Anchor-Gebot: fair ~ EV * cp, begrenzt durch anchor_budget & HARD_CAP
        fair_anchor = anchor_ev * self.cp
        anchor_bid  = int(fair_anchor * random.uniform(1.0 - ANCHOR_EPS, 1.0 + ANCHOR_EPS))
        anchor_bid  = clamp_int(anchor_bid, 1, min(HARD_CAP, anchor_budget, gold))
        if anchor_bid > 0:
            bids[anchor_id] = anchor_bid
            gold -= anchor_bid

        # 6) Pepper: kleine Gebote auf alle anderen positiven EV Auktionen, bis Pepper-Budget aufgebraucht
        if pepper_budget > 0 and gold > 0:
            remaining = [(s, ev, a_id) for (s, ev, a_id) in scored[1:]]  # ohne Anchor
            # leichte Zufallsreihenfolge, damit nicht immer gleiche Reihenfolge
            random.shuffle(remaining)
            to_spend = min(pepper_budget, gold)
            for _, ev, a_id in remaining:
                if to_spend <= 0 or gold <= 0:
                    break
                # fixe Pepper-Größe mit ±Jitter
                base = random.randint(PEPPER_MIN, PEPPER_MAX)
                bid  = int(base * random.uniform(1.0 - PEPPER_EPS, 1.0 + PEPPER_EPS))
                bid  = clamp_int(bid, 1, min(HARD_CAP, gold, to_spend))
                if bid <= 0:
                    continue
                bids[a_id] = bid
                gold -= bid
                to_spend -= bid

        return bids

# -------------------- API-Hook (persistente Instanz, ohne global) --------------------
def make_bid(agent_id, current_round, states, auctions, prev_auctions, bank_state):
    if not hasattr(make_bid, "_agent"):
        make_bid._agent = PepperAnchorAgent()
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
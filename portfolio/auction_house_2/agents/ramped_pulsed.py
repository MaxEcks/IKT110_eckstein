# agent_market_ramped_pulsed_simple.py
"""
Pulsed Ramp Agent (3-Block-Rampe + optional Pepper-Budget)
Kompatibel zur neuen dnd_auction_game API:
make_bid(agent_id, states, auctions, prev_auctions, pool_gold, prev_pool_buys)
Rückgabeformat:
{"bids": {...}, "pool": 0}
"""

import random
import statistics
from dnd_auction_game import AuctionGameClient

# ------------------------------------------------------------
# Basiskonfiguration (keine ENV-Variablen mehr)
# ------------------------------------------------------------
AVG = {2:1.5,3:2.0,4:2.5,6:3.5,8:4.5,10:5.5,12:6.5,20:10.5}

EMA_ALPHA   = 0.15
HARD_CAP    = 4000
EPSILON     = 0.10
TOP_K       = 6

ENDGAME_RATIO = 0.10

# Pulsblöcke
EARLY_END = 0.30
MID_END   = 0.65
LATE_END  = 0.90

# Spend-Frac pro Block
S_EARLY_START = 0.28
S_EARLY_END   = 0.38

S_MID_START   = 0.35
S_MID_END     = 0.55

S_LATE_START  = 0.50
S_LATE_END    = 0.75

# Aggro pro Block
A_EARLY_START = 0.95
A_EARLY_END   = 1.05

A_MID_START   = 1.00
A_MID_END     = 1.15

A_LATE_START  = 1.05
A_LATE_END    = 1.22

# Pepper
PEPPER_FRAC = 0.08
PEPPER_MIN  = 40
PEPPER_MAX  = 160
PEPPER_EPS  = 0.10

# Falls wir die echte Rundenzahl nicht kennen:
ASSUMED_TOTAL_ROUNDS = 100


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def clamp_int(x, lo, hi):
    return int(max(lo, min(hi, int(x))))

def lerp(a, b, t):
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return a + (b - a) * t

def piecewise_linear(progress, blocks):
    prev_end = 0.0
    prev_val = None
    for end_p, start_v, end_v in blocks:
        width = max(1e-9, end_p - prev_end)
        if progress <= end_p:
            block_t = (progress - prev_end) / width
            return lerp(start_v, end_v, block_t)
        prev_end = end_p
        prev_val = end_v
    return prev_val


# ------------------------------------------------------------
# Pulsed Ramp Agent
# ------------------------------------------------------------
class PulsedRampedAgent:
    def __init__(self):
        self.cp = 30.0
        self.round_counter = 0

        self.spend_blocks = [
            (EARLY_END, S_EARLY_START, S_EARLY_END),
            (MID_END,   S_MID_START,   S_MID_END),
            (LATE_END,  S_LATE_START,  S_LATE_END),
        ]
        self.aggro_blocks = [
            (EARLY_END, A_EARLY_START, A_EARLY_END),
            (MID_END,   A_MID_START,   A_MID_END),
            (LATE_END,  A_LATE_START,  A_LATE_END),
        ]

    # Marktpreis lernen
    def update_cp(self, prev_auctions):
        samples = []
        for a in prev_auctions.values():
            r = int(a.get("reward", 0))
            bids = a.get("bids", [])
            if r > 0 and bids:
                samples.append(int(bids[0]["gold"]) / r)
        if samples:
            est = statistics.median(samples)
            self.cp = (1 - EMA_ALPHA) * self.cp + EMA_ALPHA * est

    # Hauptlogik
    def decide(self, agent_id, states, auctions, prev_auctions, pool_gold, prev_pool_buys):
        # Neue Rundenzahl hochzählen
        self.round_counter += 1
        current_round = self.round_counter

        # cp update
        if prev_auctions:
            self.update_cp(prev_auctions)

        me = states[agent_id]
        gold = int(me["gold"])

        if gold <= 0 or not auctions:
            return {}

        # Fortschritt 0..1
        total = ASSUMED_TOTAL_ROUNDS
        progress = min(1.0, current_round / total)

        # Endgame
        endgame = progress >= (1.0 - ENDGAME_RATIO)

        # Spend & Aggro aus Pulsblöcken
        spend_frac = piecewise_linear(progress, self.spend_blocks)
        aggression = piecewise_linear(progress, self.aggro_blocks)

        # Rundenbudget
        if endgame:
            round_budget = gold
        else:
            round_budget = clamp_int(gold * spend_frac, 1, gold)

        # Auktionen scoren
        scored = []
        for a_id, a in auctions.items():
            ev = a["num"] * AVG[a["die"]] + a["bonus"]
            if ev > 0:
                scored.append((ev / max(1.0, self.cp), ev, a_id))
        if not scored:
            return {}

        scored.sort(reverse=True)

        targets = scored[:TOP_K]
        others = scored[TOP_K:]

        # Pepper-Budget
        pepper_budget = 0
        if PEPPER_FRAC > 0 and not endgame:
            pepper_budget = int(round_budget * PEPPER_FRAC)

        main_budget = max(0, round_budget - pepper_budget)
        bids = {}

        # --------------------------
        # 1) Hauptziele
        # --------------------------
        if main_budget > 0 and targets:
            per_main = max(1, main_budget // len(targets))
            remaining = gold
            for _, ev, a_id in targets:
                fair = ev * self.cp * aggression
                bid = int(min(fair, HARD_CAP, per_main, remaining))
                bid = int(bid * random.uniform(1.0 - EPSILON, 1.0 + EPSILON))
                bid = clamp_int(bid, 1, remaining)

                if bid > 0:
                    bids[a_id] = bid
                    remaining -= bid

            gold = remaining

        # --------------------------
        # 2) Pepper-Bids
        # --------------------------
        if pepper_budget > 0 and gold > 0 and others:
            temp = others[:]
            random.shuffle(temp)
            to_spend = min(pepper_budget, gold)

            for _, ev, a_id in temp:
                if to_spend <= 0 or gold <= 0:
                    break
                base = random.randint(PEPPER_MIN, PEPPER_MAX)
                pbid = int(base * random.uniform(1.0 - PEPPER_EPS, 1.0 + PEPPER_EPS))
                pbid = clamp_int(pbid, 1, min(HARD_CAP, gold, to_spend))
                if pbid <= 0:
                    continue
                if a_id in bids:
                    continue

                bids[a_id] = pbid
                gold -= pbid
                to_spend -= pbid

        return bids


# ------------------------------------------------------------
# API-Hook (neue Signatur)
# ------------------------------------------------------------
def make_bid(agent_id, states, auctions, prev_auctions, pool_gold, prev_pool_buys):
    if not hasattr(make_bid, "_agent"):
        make_bid._agent = PulsedRampedAgent()

    bids = make_bid._agent.decide(agent_id, states, auctions, prev_auctions, pool_gold, prev_pool_buys)
    return {"bids": bids, "pool": 0}   # pool vorerst deaktiviert


# ------------------------------------------------------------
# Standalone-Start
# ------------------------------------------------------------
if __name__ == "__main__":

    host = "localhost"
    agent_name = "Wolf_of_Wall_Street_Pulsed"
    player_id = "Maximilian Eckstein"
    port = 8000

    game = AuctionGameClient(host=host, agent_name=agent_name, player_id=player_id, port=port)
    try:
        game.run(make_bid)
    except KeyboardInterrupt:
        print("<interrupt>")
    print("<game done>")
# agent_wolf_pulsed_cap_pool.py
"""
Pulsed Wolf Agent with:
- Piecewise (pulsed) ramp for spending and aggression
- Cap as percentage of own gold, tied to the pulses
- cp (market price) with stronger weight to follow current prices
- Simple, structured logic (no plots)
- Configurable assumed total rounds
- Conservative, occasional pool usage when profitable and safe

Compatible with dnd_auction_game pool API:
    make_bid(agent_id, states, auctions, prev_auctions, pool_gold, prev_pool_buys)
Return format:
    {"bids": {...}, "pool": points_for_pool}
"""

import os
import random
import statistics
from typing import Dict, Any

from dnd_auction_game import AuctionGameClient


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Expected value per die
AVG = {
    2: 1.5,
    3: 2.0,
    4: 2.5,
    6: 3.5,
    8: 4.5,
    10: 5.5,
    12: 6.5,
    20: 10.5,
}

# Market price learning
EMA_ALPHA = 0.25          # higher weight on recent rounds -> track fast price growth
CP_MIN    = 1.0           # safety bounds for cp
CP_MAX    = 1e9

# Bidding behaviour
EPSILON   = 0.10          # random +- jitter on bids
TOP_K     = 6             # number of main target auctions
ENDGAME_RATIO = 0.10      # last 10% of rounds -> full flush

# Assumed total rounds (can be overridden via env var)
ASSUMED_TOTAL_ROUNDS = int(os.environ.get("ASSUMED_TOTAL_ROUNDS", "500"))

# ---- Pulses (4 blocks) ----
# Each block: (end_progress, spend_start, spend_end, cap_start, cap_end, aggr_start, aggr_end)
# - spend_*: fraction of gold to spend per round
# - cap_*:   fraction of gold allowed per auction
# - aggr_*:  multiplicative factor on cp * EV

PULSE_BLOCKS = [
    # end,   spend_start, spend_end,  cap_start, cap_end,   aggr_start, aggr_end
    (0.25,   0.03,        0.06,      0.02,      0.04,      0.95,       1.00),  # early
    (0.50,   0.06,        0.08,      0.04,      0.06,      1.00,       1.10),  # mid1
    (0.80,   0.08,        0.10,      0.06,      0.08,      1.10,       1.20),  # mid2
    (0.90,   0.10,        0.12,      0.08,      0.10,      1.20,       1.30),  # late (before endgame)
]

# How much we are willing to overpay relative to cp * EV
FAIR_MULT = 1.5

# ------------------------------------------------------------
# Pool strategy (very conservative)
# ------------------------------------------------------------

POOL_RATIO_THRESHOLD = 1.20   # pool_price (gold/point) must be 20% better than cp
POOL_MAX_SURPLUS_FRAC = 0.05  # max 5% of surplus points
POOL_POINTS_ABS_MAX   = 40    # never invest more than 40 points per round
MIN_POINTS_KEEP       = 300   # keep at least this many points
LEAD_POINTS_MIN       = 200   # only use pool if we lead by >= 200 points


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def clamp_int(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(x))))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b by t ∈ [0, 1]."""
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return a + (b - a) * t


def piecewise_pulse(progress: float):
    """
    Map game progress ∈ [0,1] to:
      spend_frac, cap_frac, aggression
    using the configured PULSE_BLOCKS.
    """
    prev_end = 0.0
    last_vals = (PULSE_BLOCKS[-1][1], PULSE_BLOCKS[-1][3], PULSE_BLOCKS[-1][5])

    for end_p, s_start, s_end, c_start, c_end, a_start, a_end in PULSE_BLOCKS:
        width = max(1e-9, end_p - prev_end)
        if progress <= end_p:
            t = (progress - prev_end) / width
            spend = lerp(s_start, s_end, t)
            cap = lerp(c_start, c_end, t)
            aggr = lerp(a_start, a_end, t)
            return spend, cap, aggr
        prev_end = end_p
        last_vals = (s_end, c_end, a_end)

    # after last block
    return last_vals


# ------------------------------------------------------------
# Pulsed Wolf Agent
# ------------------------------------------------------------

class PulsedWolfAgent:
    def __init__(self):
        self.cp = 30.0          # gold per point
        self.cp_pool = None     # gold per point in pool (approx.)
        self.round_counter = 0

    # ----- Learn market price cp from previous auctions -----

    def update_cp(self, prev_auctions: Dict[str, Any]):
        samples = []
        for a in prev_auctions.values():
            reward = int(a.get("reward", 0))
            bids = a.get("bids", [])
            if reward > 0 and bids:
                win_gold = int(bids[0]["gold"])
                samples.append(win_gold / reward)
        if not samples:
            return
        est = statistics.median(samples)
        cp_new = (1.0 - EMA_ALPHA) * self.cp + EMA_ALPHA * est
        self.cp = float(min(CP_MAX, max(CP_MIN, cp_new)))

    # ----- Rough pool price estimate (gold per point) -----

    def update_cp_pool(self, pool_gold: int, prev_pool_buys: Dict[str, Any]):
        if pool_gold <= 0 or not prev_pool_buys:
            return
        total_points = sum(int(v) for v in prev_pool_buys.values())
        if total_points <= 0:
            return
        sample = pool_gold / total_points
        if self.cp_pool is None:
            self.cp_pool = float(sample)
        else:
            self.cp_pool = (1.0 - EMA_ALPHA) * self.cp_pool + EMA_ALPHA * float(sample)

    # ----- Decide how many points to invest in the pool -----

    def decide_pool_points(
        self,
        agent_id: str,
        states: Dict[str, Any],
        points: int,
        progress: float,
        pool_gold: int,
    ) -> int:
        # Endgame: keep all points
        if progress >= (1.0 - ENDGAME_RATIO):
            return 0
        if pool_gold <= 0:
            return 0
        if self.cp_pool is None or self.cp <= 0:
            return 0

        # Pool must be significantly better than auctions
        ratio = self.cp_pool / self.cp
        if ratio < POOL_RATIO_THRESHOLD:
            return 0

        # Compute our lead
        others_points = [int(st.get("points", 0)) for aid, st in states.items() if aid != agent_id]
        if not others_points:
            return 0
        max_other = max(others_points)
        lead = points - max_other
        if lead < LEAD_POINTS_MIN:
            return 0

        if points <= MIN_POINTS_KEEP:
            return 0

        surplus = points - MIN_POINTS_KEEP
        max_by_frac = int(surplus * POOL_MAX_SURPLUS_FRAC)
        pool_bid = min(max_by_frac, POOL_POINTS_ABS_MAX)
        if pool_bid <= 0:
            return 0
        return pool_bid

    # ----- Main decision -----

    def decide(
        self,
        agent_id: str,
        states: Dict[str, Any],
        auctions: Dict[str, Any],
        prev_auctions: Dict[str, Any],
        pool_gold: int,
        prev_pool_buys: Dict[str, Any],
    ) -> (Dict[str, int], int):

        # 1) Round + progress
        self.round_counter += 1
        current_round = self.round_counter

        total_rounds = max(1, ASSUMED_TOTAL_ROUNDS)
        progress = current_round / total_rounds
        if progress < 0.0:
            progress = 0.0
        if progress > 1.0:
            progress = 1.0

        # 2) Update cp & cp_pool from last round
        if prev_auctions:
            self.update_cp(prev_auctions)
        if prev_pool_buys:
            self.update_cp_pool(pool_gold, prev_pool_buys)

        # 3) Read own state
        me = states[agent_id]
        gold = int(me["gold"])
        points = int(me.get("points", 0))

        if gold <= 0 or not auctions:
            return {}, 0

        endgame = progress >= (1.0 - ENDGAME_RATIO)

        # 4) Get pulsed parameters
        spend_frac, cap_frac, aggression = piecewise_pulse(progress)

        # 5) Round budget
        if endgame:
            round_budget = gold   # full flush in endgame
        else:
            round_budget = clamp_int(gold * spend_frac, 1, gold)

        # 6) Score auctions by value_ratio = EV / cp
        scored = []
        for a_id, a in auctions.items():
            die = int(a["die"])
            num = int(a["num"])
            bonus = int(a["bonus"])
            ev = num * AVG[die] + bonus
            if ev <= 0:
                continue
            value_ratio = ev / max(1.0, self.cp)
            scored.append((value_ratio, ev, a_id))

        if not scored:
            return {}, 0

        scored.sort(reverse=True)
        targets = scored[:TOP_K]

        bids: Dict[str, int] = {}

        # 7) Main bids on targets
        if round_budget > 0 and targets:
            per_share = max(1, round_budget // len(targets))
            remaining_gold = gold

            for _, ev, a_id in targets:
                fair_price = ev * self.cp * aggression
                fair_price *= FAIR_MULT  # allow some overpaying relative to cp

                # Cap linked to own gold; in endgame we ignore cap_frac
                if endgame:
                    cap_by_gold = remaining_gold  # effectively only limited by remaining gold + budget
                else:
                    cap_by_gold = cap_frac * remaining_gold

                cap_by_budget = per_share

                max_bid = min(fair_price, cap_by_gold, cap_by_budget, remaining_gold)
                bid = int(max(1, max_bid))

                # Add jitter
                bid = int(bid * random.uniform(1.0 - EPSILON, 1.0 + EPSILON))
                bid = clamp_int(bid, 1, remaining_gold)

                if bid <= 0:
                    continue

                bids[a_id] = bid
                remaining_gold -= bid
                if remaining_gold <= 0:
                    break

            gold = remaining_gold  # remaining gold after main bids

        # 8) Pool decision (conservative, based on lead and pool quality)
        pool_points = self.decide_pool_points(
            agent_id=agent_id,
            states=states,
            points=points,
            progress=progress,
            pool_gold=pool_gold,
        )

        return bids, pool_points


# ------------------------------------------------------------
# API hook
# ------------------------------------------------------------

def make_bid(
    agent_id: str,
    states: Dict[str, Any],
    auctions: Dict[str, Any],
    prev_auctions: Dict[str, Any],
    pool_gold: int,
    prev_pool_buys: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Entry point for dnd_auction_game.
    """
    if not hasattr(make_bid, "_agent"):
        make_bid._agent = PulsedWolfAgent()

    bids, pool_points = make_bid._agent.decide(
        agent_id=agent_id,
        states=states,
        auctions=auctions,
        prev_auctions=prev_auctions,
        pool_gold=pool_gold,
        prev_pool_buys=prev_pool_buys,
    )
    return {"bids": bids, "pool": pool_points}


# ------------------------------------------------------------
# Standalone start
# ------------------------------------------------------------

if __name__ == "__main__":
    host = "localhost"
    agent_name = "Wolf_of_Wall_Street_Pulsed_Cap_Pool"
    player_id = "Maximilian Eckstein"
    port = 8000

    game = AuctionGameClient(
        host=host,
        agent_name=agent_name,
        player_id=player_id,
        port=port,
    )
    try:
        game.run(make_bid)
    except KeyboardInterrupt:
        print("<interrupt>")
    print("<game done>")
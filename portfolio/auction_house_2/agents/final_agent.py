# agent_dr_stockmann_autark_pool.py
"""
Dr. Stockmann - autonomous, learning Wolf with safe pool logic
and step-limited cp adaptation.

Core behaviour:
---------------
- Auction logic:
  * Wolf-style ramp:
      - cp learned from last round's winners
      - spend_frac ramp from start to end
      - aggression ramp from start to end
      - Top-K=3 auctions chosen by expected value (EV)
  * No static HARD_CAP.
    Instead, a dynamic per-auction cap:

        fair_price = EV * cp_safe * aggression

        per_auction_cap = min(
            CAP_EV_MULT   * fair_price,   # at most 2x "fair value"
            CAP_GOLD_FRAC * current_gold, # at most 30% of our gold
            share_cap,                    # our share of the round budget
            remaining_gold
        )

- cp:
  * Estimated as EMA of median(winning_bid / reward_points).
  * Step-limited: per round cp can only move by factors
      CP_STEP_DOWN <= cp_new / cp_old <= CP_STEP_UP
    then clamped to [CP_MIN, CP_MAX].
  * This allows cp to grow with the market, but prevents single-round explosions.

- Pool logic:
  * cp_pool is estimated ONLY when the pool actually decreases
    between rounds and prev_pool_buys is non-empty.
    This protects us from the current auction_house.py bug where
    the pool never decreases and payouts are capped to 1 gold.
  * cp_pool_sample ≈ (pool_before - pool_after) / total_points_bid_prev
  * We invest a small, capped fraction of our surplus points into the
    pool only if:
        cp_pool is known AND
        cp_pool / cp > POOL_RATIO_THRESHOLD AND
        we are not in the endgame.
"""

import os
from typing import Dict, Any

import numpy as np
from dnd_auction_game import AuctionGameClient


# -------------------- Dice EV table --------------------

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


# -------------------- Hyperparameters --------------------

EMA_ALPHA        = 0.20   # smoothing for cp and cp_pool
EPSILON          = 0.10   # random +- jitter on bids
TOP_K            = 3      # number of auctions to target per round (focus)

# Spending and aggression ramps (Wolf-style)
SPEND_FRAC_START = 0.30
SPEND_FRAC_END   = 0.95

AGGR_START       = 0.95
AGGR_END         = 1.20

ENDGAME_RATIO    = 0.10   # last 10 % of rounds = gold flush

# Assumed total number of rounds (for ramping); override via env var if needed.
ASSUMED_TOTAL_ROUNDS = int(os.environ.get("ASSUMED_TOTAL_ROUNDS", "500"))

# cp handling: clamp + step-limit to avoid insane jumps
CP_MIN       = 3.0
CP_MAX       = 5000.0
CP_STEP_UP   = 1.30   # per round cp can increase by at most +30%
CP_STEP_DOWN = 0.70   # per round cp can decrease to at most 70% of previous

# Dynamic auction cap components
CAP_EV_MULT   = 2.0   # we pay at most 2x fair price EV * cp
CAP_GOLD_FRAC = 0.30  # at most 30 % of current gold per auction

# Safety / buffer: minimum "burn" per bid (slightly higher now)
MIN_PER_BID_FLOOR_FRAC = 0.05   # ~5 % of gold / remaining rounds

# Pool strategy (conservative, bug-safe)
POOL_RATIO_THRESHOLD = 1.20   # cp_pool must be 20 % better than cp
POOL_POINTS_FRAC     = 0.10   # max 10 % of surplus points to pool
POOL_POINTS_ABS_MAX  = 50     # never bid more than 50 points on pool
MIN_POINTS_KEEP      = 100    # always keep at least this many points


# ---------------- Utility helpers ----------------

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b by t ∈ [0,1]."""
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return a + (b - a) * t


def clamp_int(x: float, lo: int, hi: int) -> int:
    """Clamp x to integer range [lo, hi] and return int."""
    return int(max(lo, min(hi, int(x))))


# ---------------- Optional live plotting ----------------

class LivePlot:
    """
    Optional live plot; disabled automatically on headless servers.
    Shows gold, points and cp over rounds.
    """
    def __init__(self):
        self.enabled = False
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FuncFormatter

            self.plt = plt
            self.enabled = True

            self.fig, (self.ax_money, self.ax_cp) = plt.subplots(2, 1, figsize=(7, 6))
            self.ax_points = self.ax_money.twinx()

            try:
                self.fig.canvas.manager.set_window_title("Dr. Stockmann - Autark Wolf (Pool)")
            except Exception:
                pass

            self.rounds = []
            self.money_left = []
            self.points = []
            self.cp_hist = []

            self.initial_gold = None
            self.format_thousands = FuncFormatter(lambda x, _: f"{int(x / 1_000)}k")

            self.plt.ion()
            self.fig.tight_layout()

        except Exception:
            self.enabled = False

    def update(self, rnd: int, money_left: int, points: int, cp: float):
        if not self.enabled:
            return

        if self.initial_gold is None:
            self.initial_gold = max(1, money_left)

        self.rounds.append(rnd)
        self.money_left.append(money_left)
        self.points.append(points)
        self.cp_hist.append(cp)

        # Money & points
        self.ax_money.cla()
        self.ax_points.cla()

        self.ax_money.set_title("Gold (left) & Points (right)")
        self.ax_money.set_xlabel("Round")
        self.ax_money.plot(self.rounds, self.money_left, label="gold")
        self.ax_money.set_ylabel("Gold")
        self.ax_money.yaxis.set_major_formatter(self.format_thousands)

        self.ax_points.plot(self.rounds, self.points, linestyle="--", label="points")
        self.ax_points.set_ylabel("Points")

        self.ax_money.legend(loc="upper right")
        self.ax_points.legend(loc="upper left")

        # cp
        self.ax_cp.cla()
        self.ax_cp.set_title("Market price cp (EMA, step-limited)")
        self.ax_cp.set_xlabel("Round")
        self.ax_cp.set_ylabel("Gold per point")
        self.ax_cp.plot(self.rounds, self.cp_hist, label="cp")
        self.ax_cp.legend(loc="best")

        self.fig.tight_layout()
        self.plt.pause(0.001)


# ---------------- Main agent ----------------

class MarketAgent:
    def __init__(self):
        self.cp = 30.0   # initial market estimate (gold per point)

        self.cp_pool = None          # pool price estimate (points -> gold)
        self._last_pool_gold = None  # pool value at previous round

        self._plot = LivePlot()
        self._round_counter = 0

        self._last_gold = 0
        self._last_points = 0

    # ----- learn cp from previous auctions (step-limited) -----

    def update_cp(self, prev_auctions: Dict[str, Any]):
        """
        Update cp from previous winners:

            samples = winning_bid / reward_points

        cp_raw  := EMA(self.cp, median(samples))
        cp_step := limited so that:
             CP_STEP_DOWN <= cp_step / cp_old <= CP_STEP_UP

        Finally clamped to [CP_MIN, CP_MAX].
        """
        samples = []
        for a in prev_auctions.values():
            reward = int(a.get("reward", 0))
            bids = a.get("bids", [])
            if reward > 0 and bids:
                win_gold = int(bids[0]["gold"])
                samples.append(win_gold / reward)

        if not samples:
            return

        est = float(np.median(np.array(samples, dtype=np.float64)))

        # EMA update
        cp_old = self.cp
        cp_raw = (1 - EMA_ALPHA) * cp_old + EMA_ALPHA * est

        # Step-limit: cp cannot jump arbitrarily in one round
        upper = cp_old * CP_STEP_UP
        lower = cp_old * CP_STEP_DOWN
        cp_limited = min(upper, max(lower, cp_raw))

        # Global clamp to avoid completely insane values
        self.cp = float(max(CP_MIN, min(CP_MAX, cp_limited)))

    # ----- learn cp_pool (only if pool actually decreases) -----

    def update_cp_pool(self, pool_gold: int, prev_pool_buys: Dict[str, Any]):
        """
        Estimate cp_pool only in rounds where:
          - we know the previous pool value (_last_pool_gold not None),
          - prev_pool_buys is non-empty,
          - pool has *decreased* (pool_gold < _last_pool_gold).

        Then we approximate:
            payout ≈ _last_pool_gold - pool_gold
            cp_pool_sample ≈ payout / total_points
        """
        if self._last_pool_gold is None:
            return
        if not prev_pool_buys:
            return

        prev_pool = int(self._last_pool_gold)
        curr_pool = int(pool_gold)

        if curr_pool >= prev_pool:
            # nothing paid out (or bugged pool never decreases)
            return

        total_points = int(sum(int(v) for v in prev_pool_buys.values()))
        if total_points <= 0:
            return

        payout = prev_pool - curr_pool
        sample = payout / total_points

        if self.cp_pool is None:
            self.cp_pool = float(sample)
        else:
            self.cp_pool = (1 - EMA_ALPHA) * self.cp_pool + EMA_ALPHA * float(sample)

    # ----- decide pool points (conservative, bug-safe) -----

    def decide_pool_points(self, points: int, progress: float, pool_gold: int) -> int:
        """
        Decide how many points to invest into the pool.

        Conditions:
          - not in endgame (progress < 1 - ENDGAME_RATIO)
          - cp_pool known and cp_pool / cp > POOL_RATIO_THRESHOLD
          - keep MIN_POINTS_KEEP points as safety buffer
          - cap by POOL_POINTS_FRAC and POOL_POINTS_ABS_MAX

        If the auction_house pool bug is present (pool never decreases),
        cp_pool will never be learned and this function returns 0.
        """
        if progress >= (1.0 - ENDGAME_RATIO):
            return 0

        if pool_gold <= 0:
            return 0
        if self.cp_pool is None or self.cp <= 0:
            return 0

        ratio = self.cp_pool / max(self.cp, 1e-9)
        if ratio < POOL_RATIO_THRESHOLD:
            return 0

        if points <= MIN_POINTS_KEEP:
            return 0

        surplus = points - MIN_POINTS_KEEP
        max_by_frac = int(surplus * POOL_POINTS_FRAC)
        max_by_abs  = POOL_POINTS_ABS_MAX
        pool_bid = min(max_by_frac, max_by_abs)

        if pool_bid <= 0:
            return 0

        return pool_bid

    # ----- main decision -----

    def decide(
        self,
        agent_id: str,
        states: Dict[str, Any],
        auctions: Dict[str, Any],
        prev_auctions: Dict[str, Any],
        pool_gold: int,
        prev_pool_buys: Dict[str, Any],
    ) -> (Dict[str, int], int):

        # 1) Round counter
        self._round_counter += 1
        current_round = self._round_counter

        # 2) Update cp from previous round
        if prev_auctions:
            self.update_cp(prev_auctions)

        cp_safe = self.cp  # already step-limited & clamped

        # 3) Update cp_pool (bug-safe)
        if prev_pool_buys:
            self.update_cp_pool(pool_gold=pool_gold, prev_pool_buys=prev_pool_buys)

        # 4) Read our own state
        me = states[agent_id]
        gold = int(me["gold"])
        points = int(me.get("points", 0))
        self._last_gold = gold
        self._last_points = points

        # 5) Progress and endgame detection
        total = max(1, ASSUMED_TOTAL_ROUNDS)
        progress = current_round / total
        progress = 0.0 if progress < 0 else 1.0 if progress > 1 else progress

        rem_rounds = max(1, total - current_round)
        endgame = current_round >= int((1.0 - ENDGAME_RATIO) * total)

        # 6) Spending & aggression ramps
        spend_frac = lerp(SPEND_FRAC_START, SPEND_FRAC_END, progress)
        aggression = lerp(AGGR_START, AGGR_END, progress)

        # 7) Round budget
        if endgame:
            round_budget = gold  # true flush
        else:
            round_budget = clamp_int(gold * spend_frac, 1, gold)

        bids: Dict[str, int] = {}

        # 8) Auction bidding
        if auctions and round_budget > 0 and gold > 0:
            scored = []
            for a_id, a in auctions.items():
                die = int(a["die"])
                ev = a["num"] * AVG[die] + a["bonus"]
                if ev <= 0:
                    continue
                value_ratio = ev / max(1.0, cp_safe)
                scored.append((value_ratio, ev, a_id))

            if scored:
                scored.sort(reverse=True)
                targets = scored[: max(1, TOP_K)]

                targets_left = len(targets)
                per_share = max(1, round(round_budget / targets_left))

                # Minimum burn per round (slightly higher now)
                burn_min = int(MIN_PER_BID_FLOOR_FRAC * gold / max(1, rem_rounds))
                per_bid_floor = max(1, burn_min)

                remaining_gold = gold

                for _, ev, a_id in targets:
                    fair_price = ev * cp_safe * aggression

                    value_cap = CAP_EV_MULT * fair_price
                    gold_cap  = CAP_GOLD_FRAC * gold
                    share_cap = per_share

                    per_auction_cap = min(value_cap, gold_cap, share_cap, remaining_gold)

                    if per_auction_cap < 1:
                        continue

                    bid = int(per_auction_cap * float(
                        np.random.uniform(1.0 - EPSILON, 1.0 + EPSILON)
                    ))

                    bid = max(bid, per_bid_floor)
                    bid = clamp_int(bid, 1, remaining_gold)

                    if bid > 0:
                        bids[a_id] = bid
                        remaining_gold -= bid
                        if remaining_gold <= 0:
                            break

        # 9) Pool bidding (safe & conservative)
        pool_points = self.decide_pool_points(
            points=points,
            progress=progress,
            pool_gold=pool_gold,
        )

        # 10) Remember current pool for next cp_pool estimation
        self._last_pool_gold = int(pool_gold)

        # 11) Live plot update
        self._plot.update(current_round, gold, points, self.cp)

        return bids, pool_points


# ---------------- API hook ----------------

def make_bid(
    agent_id: str,
    states: Dict[str, Any],
    auctions: Dict[str, Any],
    prev_auctions: Dict[str, Any],
    pool_gold: int,
    prev_pool_buys: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Callback entry point for dnd_auction_game.

    Must return:
        {"bids": {auction_id: gold}, "pool": points_for_pool}
    """
    if not hasattr(make_bid, "_agent"):
        make_bid._agent = MarketAgent()

    bids, pool_points = make_bid._agent.decide(
        agent_id=agent_id,
        states=states,
        auctions=auctions,
        prev_auctions=prev_auctions,
        pool_gold=pool_gold,
        prev_pool_buys=prev_pool_buys,
    )

    return {"bids": bids, "pool": pool_points}


# ---------------- Standalone launcher ----------------

if __name__ == "__main__":
    host = "localhost"
    agent_name = "Dr_Stockmann_Autark_Wolf_Pool_v2"
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
        print("<interrupt - shutting down>")
    print("<game is done>")
# agent_market_ramped_simple.py
"""
Linear ramp: early conservative, later aggressive with final flush
- Market price (cp) via EMA-smoothed median of previous round
- Budget ramp: SPEND_FRAC from START to END over the course of the game
- Aggression ramp: AGGR from START to END (multiplies market price)
- Endgame flush: in the last X% of rounds, spend everything reasonable
- A live matplotlib plot starts automatically (if display available)
- Persistent agent instance via hasattr(make_bid, "_agent")

Adapted for new dnd_auction_game API:
  make_bid(agent_id, states, auctions, prev_auctions, pool_gold, prev_pool_buys)
and return format:
  {"bids": {...}, "pool": 0}
"""

import os
import random
from typing import Dict, Any

import numpy as np

from dnd_auction_game import AuctionGameClient

# -------------------- Configuration --------------------
# Average roll values per dice type (expected value)
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

EMA_ALPHA        = 0.20   # smoothing factor for market price (cp)
HARD_CAP         = 500    # bid cap per auction (avoid overspending)
EPSILON          = 0.10   # random +- jitter
TOP_K            = 4      # auctions to target per round

SPEND_FRAC_START = 0.30   # fraction of gold spent at game start
SPEND_FRAC_END   = 0.95   # fraction of gold spent at game end
AGGR_START       = 0.95   # starting aggression (× cp)
AGGR_END         = 1.20   # ending aggression (× cp)
ENDGAME_RATIO    = 0.10   # last 10% of rounds trigger full flush

# Assumed total number of rounds (needed since new API does not pass current_round)
ASSUMED_TOTAL_ROUNDS = int(os.environ.get("ASSUMED_TOTAL_ROUNDS", "500"))


# -------------------- Small utility functions --------------------
def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b by t ∈ [0,1]."""
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return a + (b - a) * t


def clamp_int(x: float, lo: int, hi: int) -> int:
    """Clamp x to integer range [lo, hi]."""
    return int(max(lo, min(hi, int(x))))


# -------------------- Live plot (optional) --------------------
class LivePlot:
    def __init__(self):
        self.enabled = False
        try:
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FuncFormatter

            self.format_thousands = FuncFormatter(lambda x, _: f"{int(x / 1_000)}k")
            self.plt = plt
            self.enabled = True
            self.fig, (self.ax_money, self.ax_cp) = plt.subplots(2, 1, figsize=(7, 6))
            self.ax_points = self.ax_money.twinx()  # second y-axis for points

            try:
                self.fig.canvas.manager.set_window_title("Auction Bot Live")
            except Exception:
                pass

            self.rounds = []
            self.money_left = []
            self.points = []
            self.cp_hist = []

            self.initial_gold = None
            self.plt.ion()
            self.fig.tight_layout()
        except Exception:
            self.enabled = False

    def update(self, rnd: int, money_left: int, points: int, cp: float):
        if not self.enabled:
            return

        # remember initial gold once
        if self.initial_gold is None:
            self.initial_gold = max(1, money_left)

        self.rounds.append(rnd)
        self.money_left.append(money_left)
        self.points.append(points)
        self.cp_hist.append(cp)

        # ---- Top subplot: Money (left y) & Points (right y)
        self.ax_money.cla()
        self.ax_points.cla()

        self.ax_money.set_title("Money left (left) & Points (right)")
        self.ax_money.set_xlabel("Round")

        # Left y: money, scaled to thousands for readability
        self.ax_money.plot(self.rounds, self.money_left, label="money left")
        self.ax_money.set_ylabel("Gold")
        self.ax_money.yaxis.set_major_formatter(self.format_thousands)

        # Right y: points (separate axis so scales don't fight)
        self.ax_points.plot(self.rounds, self.points, linestyle="--", label="points")
        self.ax_points.set_ylabel("Points")

        # Optional: add legends
        self.ax_money.legend(loc="upper right")
        self.ax_points.legend(loc="upper left")

        # ---- Bottom subplot: cp
        self.ax_cp.cla()
        self.ax_cp.set_title("Marketprice (Median, EMA)")
        self.ax_cp.set_xlabel("Round")
        self.ax_cp.set_ylabel("Marketprice (gold per point)")
        self.ax_cp.plot(self.rounds, self.cp_hist, label="cp_overall")
        self.ax_cp.legend(loc="best")

        self.fig.tight_layout()
        self.plt.pause(0.001)


# -------------------- Market agent --------------------
class MarketAgent:
    def __init__(self):
        self.cp = 30.0           # initial market estimate (gold per point)
        self._plot = LivePlot()  # try to open live plot
        self._last_points = 0
        self._last_gold = 0
        self._round_counter = 0  # since new API does not provide current_round

    def update_cp(self, prev_auctions: Dict[str, Any]):
        """Update market price estimate (cp) using previous round results."""
        samples = []
        for a in prev_auctions.values():
            r = int(a.get("reward", 0))
            bids = a.get("bids", [])
            if r > 0 and bids:
                win = int(bids[0]["gold"])  # winning bid is first (index 0)
                samples.append(win / r)

        if samples:
            est = float(np.median(np.array(samples, dtype=np.float64)))  # robust median
            # EMA ... exponential moving average
            self.cp = (1 - EMA_ALPHA) * self.cp + EMA_ALPHA * est

    def decide(
        self,
        agent_id: str,
        states: Dict[str, Any],
        auctions: Dict[str, Any],
        prev_auctions: Dict[str, Any],
        pool_gold: int,
        prev_pool_buys: Dict[str, Any],
    ) -> Dict[str, int]:
        """
        Main bidding logic.

        NOTE: The new API does not give current_round or bank_state anymore.
        We internally keep a round counter and assume ASSUMED_TOTAL_ROUNDS
        for the ramping logic. Pool is currently ignored (no pool bidding).
        """
        # Increase internal round counter
        self._round_counter += 1
        current_round = self._round_counter

        # 1) Update market price from last round
        if prev_auctions:
            self.update_cp(prev_auctions)

        # read own state
        me = states[agent_id]
        gold = int(me["gold"])
        pts = int(me.get("points", 0))
        self._last_gold = gold
        self._last_points = pts

        # 2) Determine round progress and endgame condition (approximate)
        total = max(1, ASSUMED_TOTAL_ROUNDS)
        progress = current_round / total
        progress = 0.0 if progress < 0 else 1.0 if progress > 1 else progress

        rem = max(1, total - current_round)
        endgame = current_round >= int((1.0 - ENDGAME_RATIO) * total)

        # 3) Compute ramp factors
        spend_frac = lerp(SPEND_FRAC_START, SPEND_FRAC_END, progress)
        aggression = lerp(AGGR_START, AGGR_END, progress)

        # 4) Round budget
        if endgame:
            round_budget = gold  # flush all remaining gold
        else:
            round_budget = clamp_int(gold * spend_frac, 1, gold)

        if round_budget <= 0 or not auctions:
            self._plot.update(current_round, self._last_gold, self._last_points, self.cp)
            return {}

        # 5) Score auctions (expected value / cp) and pick top K
        scored = []
        for a_id, a in auctions.items():
            die = int(a["die"])
            ev = a["num"] * AVG[die] + a["bonus"]  # expected points
            if ev <= 0:
                continue
            value_ratio = ev / max(1.0, self.cp)
            scored.append((value_ratio, ev, a_id))  # sort by value_ratio

        if not scored:
            self._plot.update(current_round, self._last_gold, self._last_points, self.cp)
            return {}

        scored.sort(reverse=True)
        targets = scored[: max(1, TOP_K)]

        # 6) Allocate bids for each target auction
        bids: Dict[str, int] = {}
        per_share = max(1, round(round_budget / len(targets)))

        # Floor ensures consistent burn rate across remaining rounds
        burn_min = gold // max(1, rem) if rem > 0 else gold
        per_bid_floor = max(1, burn_min // max(1, len(targets)))

        remaining_gold = gold
        for _, ev, a_id in targets:
            fair = ev * self.cp * aggression
            bid = int(min(fair, HARD_CAP, per_share))
            bid = max(bid, per_bid_floor)

            # small random jitter +- EPSILON to avoid ties
            bid = int(bid * float(np.random.uniform(1.0 - EPSILON, 1.0 + EPSILON)))
            bid = clamp_int(bid, 1, remaining_gold)

            if bid > 0:
                bids[a_id] = bid
                remaining_gold -= bid
                if remaining_gold <= 0:
                    break

        # 7) Update live plot
        self._plot.update(current_round, self._last_gold, self._last_points, self.cp)
        return bids


# -------------------- API hook (persistent instance) --------------------
def make_bid(
    agent_id: str,
    states: Dict[str, Any],
    auctions: Dict[str, Any],
    prev_auctions: Dict[str, Any],
    pool_gold: int,
    prev_pool_buys: Dict[str, Any],
) -> Dict[str, Any]:
    """
    New callback signature for dnd_auction_game (pool version).

    Must return:
      {"bids": {...}, "pool": points_for_pool}

    We currently do NOT use the pool mechanic => pool = 0.
    """
    if not hasattr(make_bid, "_agent"):
        make_bid._agent = MarketAgent()

    bids = make_bid._agent.decide(agent_id, states, auctions, prev_auctions, pool_gold, prev_pool_buys)
    points_for_pool = 0  # bid 1 point for the pool
    return {"bids": bids, "pool": points_for_pool}


# -------------------- Standalone start --------------------
if __name__ == "__main__":
    host = "localhost"
    # agent_name = "{}_{}".format(os.path.basename(__file__), random.randint(1, 1000))
    agent_name = "Wolf_of_Wall_Street"
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
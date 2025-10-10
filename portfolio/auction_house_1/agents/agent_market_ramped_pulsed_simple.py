# agent_market_ramped_pulsed_simple.py
# Mehrere kleine Rampen ("Puls-Rampen") statt einer großen + optionales Pepper-Budget
# - cp via Median(Preis/Punkt) aus Vor-Runde (EMA)
# - 3 Blöcke: Early -> Mid -> Late (piecewise linear für Spend & Aggro)
# - Endgame: Flush
# - Top-K Hauptziele nach EV/cp + kleiner Jitter
# - Optional: kleine Pepper-Gebote auf weitere Auktionen
# - Persistente Instanz ohne 'global' via hasattr(make_bid, "_agent")

import os, random, statistics

from dnd_auction_game import AuctionGameClient

# -------------------- Basiskonfig --------------------
AVG = {2:1.5,3:2.0,4:2.5,6:3.5,8:4.5,10:5.5,12:6.5,20:10.5}

# Lernen / Bieten
EMA_ALPHA   = float(os.getenv("EMA_ALPHA",   "0.15"))   # Glättung für cp
HARD_CAP    = int(os.getenv("HARD_CAP",     "4000"))    # Kappe pro Auktion
EPSILON     = float(os.getenv("EPSILON",    "0.10"))    # ±10% Jitter
TOP_K       = int(os.getenv("TOP_K",        "3"))       # Hauptziele pro Runde

# Endgame
ENDGAME_RATIO = float(os.getenv("ENDGAME_RATIO", "0.10"))  # letzte 10%: Flush

# -------------------- Puls-Rampen (pro Anteil des Spiels) --------------------
# Du kannst diese 3 Blöcke über ENV grob schieben, Defaults sind gut:
EARLY_END = float(os.getenv("EARLY_END", "0.30"))  # Ende Block 1 (30%)
MID_END   = float(os.getenv("MID_END",   "0.65"))  # Ende Block 2 (65%)
LATE_END  = float(os.getenv("LATE_END",  "0.90"))  # Ende Block 3 (90%), Rest = Endgame

# Spend-Frac je Block: Start->End
S_EARLY_START = float(os.getenv("S_EARLY_START", "0.28"))
S_EARLY_END   = float(os.getenv("S_EARLY_END",   "0.38"))

S_MID_START   = float(os.getenv("S_MID_START",   "0.35"))
S_MID_END     = float(os.getenv("S_MID_END",     "0.55"))

S_LATE_START  = float(os.getenv("S_LATE_START",  "0.50"))
S_LATE_END    = float(os.getenv("S_LATE_END",    "0.75"))

# Aggro (multipliziert cp) je Block: Start->End
A_EARLY_START = float(os.getenv("A_EARLY_START", "0.95"))
A_EARLY_END   = float(os.getenv("A_EARLY_END",   "1.05"))

A_MID_START   = float(os.getenv("A_MID_START",   "1.00"))
A_MID_END     = float(os.getenv("A_MID_END",     "1.15"))

A_LATE_START  = float(os.getenv("A_LATE_START",  "1.05"))
A_LATE_END    = float(os.getenv("A_LATE_END",    "1.22"))

# -------------------- Optional: Pepper-Budget --------------------
# Kleiner Anteil der Rundenkasse für Mini-Gebote auf weitere Auktionen
PEPPER_FRAC = float(os.getenv("PEPPER_FRAC", "0.08"))   # 0.00 = aus
PEPPER_MIN  = int(os.getenv("PEPPER_MIN",  "40"))
PEPPER_MAX  = int(os.getenv("PEPPER_MAX", "160"))
PEPPER_EPS  = float(os.getenv("PEPPER_EPS","0.10"))     # ±10%

# -------------------- Utils --------------------
def clamp_int(x, lo, hi):
    return int(max(lo, min(hi, int(x))))

def lerp(a, b, t):
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return a + (b - a) * t

def piecewise_linear(progress, blocks):
    """
    blocks: Liste [(end_p, start_val, end_val), ...] mit end_p in aufsteigender Reihenfolge.
    Gibt den linear interpolierten Wert für progress (0..1) zurück.
    """
    prev_end = 0.0
    prev_val = None
    for end_p, start_v, end_v in blocks:
        width = max(1e-9, end_p - prev_end)
        # Fortschritt innerhalb des Blocks (0..1)
        if progress <= end_p:
            # Wenn progress im ersten Block liegt, interpolieren von start_v -> end_v
            block_t = (progress - prev_end) / width
            return lerp(start_v, end_v, block_t)
        prev_end = end_p
        prev_val = end_v
    # Falls progress > letztes end_p: gib den letzten Endwert zurück
    return prev_val if prev_val is not None else blocks[-1][2]

# -------------------- Agent --------------------
class PulsedRampedAgent:
    def __init__(self):
        self.cp = 30.0  # Startschätzer Gold/Punkt

        # Blöcke vorbereiten (Spend/Aggro)
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
        # 1) Marktpreis-Update
        if prev_auctions:
            self.update_cp(prev_auctions)

        me = states[agent_id]
        gold = int(me["gold"])
        if gold <= 0 or not auctions:
            return {}

        # 2) Fortschritt & Endgame
        rem   = len(bank_state.get("gold_income_per_round", []))
        total = current_round + rem
        progress = (current_round / total) if total > 0 else 0.0
        endgame = (total > 0) and (rem <= max(1, int(ENDGAME_RATIO * total)))

        # 3) Spend/Aggro aus Puls-Blöcken lesen
        spend_frac = piecewise_linear(progress, self.spend_blocks)
        aggression = piecewise_linear(progress, self.aggro_blocks)

        # 4) Rundenbudget
        if endgame:
            round_budget = gold  # Flush
        else:
            round_budget = clamp_int(gold * spend_frac, 1, gold)

        # 5) Auktionen scoren (EV/cp)
        scored = []
        for a_id, a in auctions.items():
            die = int(a["die"])
            ev  = a["num"] * AVG[die] + a["bonus"]
            if ev <= 0:
                continue
            value_ratio = ev / max(1.0, self.cp)
            scored.append((value_ratio, ev, a_id))
        if not scored:
            return {}

        scored.sort(reverse=True)

        # 6) Aufteilen in Hauptziele (Top-K) + Rest (für Pepper)
        k = max(1, TOP_K)
        targets = scored[:k]
        others  = scored[k:]

        # Pepper-Budget
        pepper_budget = 0
        if PEPPER_FRAC > 0 and not endgame:
            pepper_budget = clamp_int(round_budget * PEPPER_FRAC, 0, round_budget)

        main_budget = max(0, round_budget - pepper_budget)
        if main_budget <= 0 and pepper_budget <= 0:
            return {}

        # 7) Hauptgebote (Top-K) – fair * Aggro, gecappt durch main_budget/len
        bids = {}
        if main_budget > 0 and targets:
            per_main = max(1, round(main_budget / len(targets)))
            remaining_gold = gold
            for _, ev, a_id in targets:
                fair = ev * self.cp * aggression
                bid  = int(min(fair, HARD_CAP, per_main, remaining_gold))
                # kleiner Jitter
                bid  = int(bid * random.uniform(1.0 - EPSILON, 1.0 + EPSILON))
                bid  = clamp_int(bid, 1, remaining_gold)
                if bid > 0:
                    bids[a_id] = bid
                    remaining_gold -= bid
                    if remaining_gold <= 0:
                        pepper_budget = 0  # nichts mehr übrig
                        break
            gold = remaining_gold

        # 8) Pepper-Bids (klein & breit) – auf weitere positive EV Auktionen
        if pepper_budget > 0 and gold > 0 and others:
            # etwas zufällig durchmischen
            tmp = others[:]
            random.shuffle(tmp)
            to_spend = min(pepper_budget, gold)
            for _, ev, a_id in tmp:
                if to_spend <= 0 or gold <= 0:
                    break
                base = random.randint(PEPPER_MIN, PEPPER_MAX)
                pbid = int(base * random.uniform(1.0 - PEPPER_EPS, 1.0 + PEPPER_EPS))
                pbid = clamp_int(pbid, 1, min(HARD_CAP, to_spend, gold))
                if pbid <= 0:
                    continue
                # Falls Hauptgebot auf identischer Auktion existiert, addieren wir nicht (einfach halten)
                if a_id in bids:
                    continue
                bids[a_id] = pbid
                gold -= pbid
                to_spend -= pbid

        return bids

# -------------------- API-Hook (persistente Instanz) --------------------
def make_bid(agent_id, current_round, states, auctions, prev_auctions, bank_state):
    if not hasattr(make_bid, "_agent"):
        make_bid._agent = PulsedRampedAgent()
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

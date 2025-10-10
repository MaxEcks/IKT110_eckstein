# agent_baseline_mem.py
# Dein ursprünglicher Bot – nur so angepasst, dass die Agent-Instanz
# über alle Runden erhalten bleibt (ohne global).

import os, random, statistics
from dnd_auction_game import AuctionGameClient

AVG = {2:1.5,3:2.0,4:2.5,6:3.5,8:4.5,10:5.5,12:6.5,20:10.5}

AGGRESSION   = float(os.getenv("AGGRESSION", "1.0"))   # multipliziert den Preis
SPEND_FRAC   = float(os.getenv("SPEND_FRAC", "0.25"))  # max. Anteil deines Golds pro Runde
HARD_CAP     = int(os.getenv("HARD_CAP", "4000"))      # absolute Kappe pro Auktion
EPSILON      = float(os.getenv("EPSILON", "0.12"))     # Zufallsanteil
TOP_K        = int(os.getenv("TOP_K", "3"))            # auf wie viele Auktionen verteilen
EMA_ALPHA    = float(os.getenv("EMA_ALPHA", "0.15"))   # Glättung für clearing price

class MarketAgent:
    def __init__(self):
        self.cp = 30.0  # Startvermutung: Gold pro Punkt (wird online gelernt)

    def update_cp(self, prev_auctions:dict):
        samples=[]
        for a in prev_auctions.values():
            r = a.get("reward", 0)
            bids = a.get("bids", [])
            if r and r>0 and bids:
                win = bids[0]["gold"]
                samples.append(win / max(1, r))
        if samples:
            est = statistics.median(samples)  # robust
            self.cp = (1-EMA_ALPHA)*self.cp + EMA_ALPHA*est

    def decide(self, agent_id, current_round, states, auctions, prev_auctions, bank_state):
        me = states[agent_id]
        gold = me["gold"]
        self.update_cp(prev_auctions)

        # Budget für diese Runde
        round_budget = min(gold, int(gold * SPEND_FRAC))

        scored=[]
        for a_id,a in auctions.items():
            ev = a["num"]*AVG[a["die"]] + a["bonus"]
            if ev <= 0:
                continue
            value_ratio = ev / max(1.0, self.cp)
            scored.append( (value_ratio, ev, a_id) )

        # Top-K nach bestem EV/Preis
        scored.sort(reverse=True)
        targets = scored[:TOP_K]

        bids={}
        if not targets:
            return bids

        per = max(1, round(round_budget / len(targets)))
        for _, ev, a_id in targets:
            base = ev * self.cp * AGGRESSION
            bid = int(min(base, HARD_CAP, per))
            # Unvorhersagbar machen
            jitter = int(random.uniform(-EPSILON, EPSILON) * max(1,bid))
            bid = max(1, bid + jitter)
            # kleine Chance blind zu „hässlichen“ Items zu springen
            if random.random() < (EPSILON/4):
                bid = max(1, int(bid * random.uniform(0.6, 1.4)))
            bids[a_id]=bid

        return bids

def make_bid(agent_id, current_round, states, auctions, prev_auctions, bank_state):
    # Persistente Instanz ohne 'global':
    if not hasattr(make_bid, "_agent"):
        make_bid._agent = MarketAgent()
    return make_bid._agent.decide(agent_id, current_round, states, auctions, prev_auctions, bank_state)

if __name__ == "__main__":
    host = "localhost"
    agent_name = "{}_{}".format(os.path.basename(__file__), random.randint(1, 1000))
    player_id = "Maximilian Eckstein"
    port = 8095

    game = AuctionGameClient(host=host,
                                agent_name=agent_name,
                                player_id=player_id,
                                port=port)
    try:
        game.run(make_bid)
    except KeyboardInterrupt:
        print("<interrupt - shutting down>")

    print("<game is done>")
rounds = 100
interest = 0.10
gold_per_round = 1000
total_gold = 0

for round in range(1, rounds + 1):
    total_gold += gold_per_round + int(total_gold * (interest))
    print(f"After round {round}, total gold: {total_gold}")
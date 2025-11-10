import json
import pickle

def load_jsonl(path):
    """Load a .jsonl file (one JSON object per line)."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# Load raw data
agents = load_jsonl("./data/agents.jsonl")

# Collect agent names (as simple list)
agent_names = [a["name"] for a in agents if "name" in a]

# Save to pickle
with open("./pickle/agent_names.pkl", "wb") as f:
    pickle.dump(agent_names, f)

# Example check
print(f"[OK] Saved {len(agent_names)} agent names.")
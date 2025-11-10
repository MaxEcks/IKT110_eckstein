import json
import pickle

def load_jsonl(path):
    """Load a .jsonl file (one JSON object per line)."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# Load raw data
schools = load_jsonl("./data/schools.jsonl")
districts = load_jsonl("./data/districts.jsonl")

data = {}
current_year = 2025

# Initialize district entries with raw fields and empty school list
for d in districts:
    data[d["id"]] = {
        "crime_rating": d["crime_rating"],
        "public_transport_rating": d["public_transport_rating"],
        "schools": [],
    }

# Add schools to their districts (and compute school_age)
for s in schools:
    district_id = s["district_id"]
    school_age = None
    if "built_year" in s and s["built_year"]:
        try:
            year = int(s["built_year"])
            if 0 < year <= current_year:
                school_age = current_year - year
        except (ValueError, TypeError):
            pass

    s["school_age"] = school_age

    if district_id in data:
        data[district_id]["schools"].append(s)

# Save linkage to pickle
with open("./pickle/district_linkage.pkl", "wb") as f:
    pickle.dump(data, f)

# Example check
print(f"[OK] Saved linkage with {len(data)} districts.")
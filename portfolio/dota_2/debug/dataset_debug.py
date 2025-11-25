import json
import zipfile

ZIP_PATH = "portfolio/dota_2/data/dota_games.zip"

with zipfile.ZipFile(ZIP_PATH) as z:
    names = [n for n in z.namelist() if n.endswith(".json")]

    print("Total JSON files:", len(names))
    print("First 5 files:", names[:5])

    for name in names[:5]:
        print("\n---", name, "---")
        with z.open(name) as f:
            raw = json.load(f)

        print("top-level keys:", list(raw.keys()))

        m = raw.get("result", raw)
        print("match_id:", m.get("match_id"))
        print("human_players:", m.get("human_players"))
        print("duration:", m.get("duration"))
        print("game_mode:", m.get("game_mode"))
        print("lobby_type:", m.get("lobby_type"))
        if "players" in m:
            print("num players:", len(m["players"]))
            print("hero_ids:", [p.get("hero_id") for p in m["players"]])
            print("leaver_status:", [p.get("leaver_status") for p in m["players"]])
        else:
            print("NO PLAYERS FIELD (in result)")
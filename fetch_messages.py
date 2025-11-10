# fetch_messages.py
import requests, json, os

API_BASE = "https://november7-730026606190.europe-west1.run.app"
OUT_DIR = "data"
OUT_FILE = os.path.join(OUT_DIR, "messages.json")

def fetch_all_messages(batch=100):
    os.makedirs(OUT_DIR, exist_ok=True)
    skip = 0
    all_items = []
    while True:
        params = {"skip": skip, "limit": batch}
        r = requests.get(f"{API_BASE}/messages/", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])
        if not items:
            break
        all_items.extend(items)
        print(f"fetched {len(items)} messages (skip={skip})")
        skip += len(items)
        # safety: if server returns total and we've got them all, stop:
        total = data.get("total")
        if total is not None and len(all_items) >= total:
            break
    with open(OUT_FILE, "w", encoding="utf8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)
    print(f"saved {len(all_items)} messages -> {OUT_FILE}")
    return all_items

if __name__ == "__main__":
    fetch_all_messages()

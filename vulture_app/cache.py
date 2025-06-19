# vulture_app/cache.py
import json
from datetime import datetime, timedelta, timezone

CACHE_FILE = "cache.json"
CACHE_EXPIRY_HOURS = 48

def is_expired(entry: dict) -> bool:
    try:
        t = datetime.fromisoformat(entry["timestamp"])
    except Exception:
        return True
    return datetime.now(timezone.utc) - t > timedelta(hours=CACHE_EXPIRY_HOURS)

def load_cache() -> dict:
    try:
     with open(CACHE_FILE, 'r') as f:
        return json.load(f)
    except FileNotFoundError:
        return {}
    
def prune_cache() -> dict:
    raw = load_cache()
    valid = {pid: e for pid, e in raw.items() if not is_expired(e)}
    save_cache(valid)
    return valid

def save_cache(cache: dict) -> None:
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)
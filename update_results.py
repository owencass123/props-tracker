"""
update_results.py — fills in Actual Ks, Over Result, Under Result for any
rows in data/props.csv that don't yet have results, using the MLB Stats API.
Runs after scraper.py in the GitHub Actions workflow.
"""

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

DATA_FILE = Path("data/props.csv")
BASE = "https://statsapi.mlb.com/api/v1"

# ── HTTP session ──────────────────────────────────────────────────────────────

def make_session():
    s = requests.Session()
    retries = Retry(total=4, backoff_factor=0.5,
                    status_forcelist=(429, 500, 502, 503, 504),
                    allowed_methods=frozenset(["GET"]))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers["User-Agent"] = "props-tracker/1.0"
    return s

SESSION = make_session()

# ── name helpers ──────────────────────────────────────────────────────────────

def clean_name(name):
    if not isinstance(name, str):
        return ""
    name = re.sub(r"\s*\(.*?\)", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return ''.join(c for c in unicodedata.normalize('NFKD', name) if not unicodedata.combining(c))

# ── MLB API ───────────────────────────────────────────────────────────────────

@lru_cache(maxsize=256)
def get_game_pks(date_obj):
    try:
        r = SESSION.get(f"{BASE}/schedule",
                        params={"sportId": 1, "date": date_obj.strftime("%m/%d/%Y")},
                        timeout=20)
        r.raise_for_status()
        return [int(g["gamePk"]) for d in r.json().get("dates", []) for g in d.get("games", [])]
    except Exception:
        return []

@lru_cache(maxsize=1024)
def get_boxscore(game_pk):
    try:
        r = SESSION.get(f"{BASE}/game/{game_pk}/boxscore", timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}

def find_ks(date_ts, player_name):
    player_clean = clean_name(player_name)
    if not player_clean:
        return None
    for offset in (-1, 0, 1):
        day = date_ts + pd.Timedelta(days=offset)
        for gpk in get_game_pks(day):
            data = get_boxscore(gpk)
            for side in ("home", "away"):
                players = data.get("teams", {}).get(side, {}).get("players", {}) or {}
                for _, p in players.items():
                    full = clean_name((p.get("person") or {}).get("fullName", ""))
                    if full.lower() != player_clean.lower():
                        continue
                    ks = ((p.get("stats") or {}).get("pitching") or {}).get("strikeOuts")
                    if ks is not None:
                        try:
                            return int(ks)
                        except Exception:
                            pass
    return None

# ── result logic ──────────────────────────────────────────────────────────────

def parse_line(x):
    if pd.isna(x):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(x))
    return float(m.group(1)) if m else None

def decide(actual, line, side):
    if actual is None or line is None:
        return "NoStat"
    if abs(actual - line) < 1e-9:
        return "Push"
    if side == "over":
        return "Win" if actual > line else "Loss"
    return "Win" if actual < line else "Loss"

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not DATA_FILE.exists():
        print("No data file yet — skipping results update.")
        return

    df = pd.read_csv(DATA_FILE, dtype=str)

    # Add result columns if missing
    for col in ("Actual Ks", "Over Result", "Under Result"):
        if col not in df.columns:
            df[col] = ""

    # Only process rows with a valid date and no result yet
    needs_update = df[
        df["Over Result"].fillna("").isin(["", "NoStat"]) &
        df["Date"].notna() &
        (df["Date"].str.strip() != "")
    ].index

    print(f"Rows needing result lookup: {len(needs_update)}")

    cache = {}  # (player, date) → actual_ks
    for idx in needs_update:
        row = df.loc[idx]
        try:
            date_ts = pd.to_datetime(row["Date"], errors="coerce")
            if pd.isna(date_ts):
                continue
            date_ts = date_ts.normalize()
            key = (clean_name(str(row["Player"])), str(date_ts.date()))
            if key not in cache:
                cache[key] = find_ks(date_ts, str(row["Player"]))
            actual = cache[key]
            df.at[idx, "Actual Ks"]    = actual if actual is not None else ""
            df.at[idx, "Over Result"]  = decide(actual, parse_line(row.get("Over Line",  "")), "over")
            df.at[idx, "Under Result"] = decide(actual, parse_line(row.get("Under Line", "")), "under")
        except Exception as e:
            print(f"  Row {idx} error: {e}")

    df.to_csv(DATA_FILE, index=False)
    print(f"✅ Results updated → {DATA_FILE}")

if __name__ == "__main__":
    main()

"""
update_results.py — fills in Actual Ks, Over Result, Under Result for any
rows in data/props.csv that don't yet have results, using the MLB Stats API.
Runs after scraper.py in the GitHub Actions workflow.
"""

import re
import unicodedata
from datetime import timedelta
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
    s.headers["User-Agent"] = "Mozilla/5.0 (compatible; props-tracker/1.0)"
    return s

SESSION = make_session()

# ── plain-dict caches (avoids lru_cache + pd.Timestamp hashability edge cases) ─

_game_pks_cache = {}
_boxscore_cache = {}

# ── name helpers ──────────────────────────────────────────────────────────────

def clean_name(name):
    if not isinstance(name, str):
        return ""
    name = re.sub(r"\s*\(.*?\)", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return ''.join(c for c in unicodedata.normalize('NFKD', name) if not unicodedata.combining(c))

# ── MLB API ───────────────────────────────────────────────────────────────────

def get_game_pks(date_str):
    """date_str: 'MM/DD/YYYY'"""
    if date_str in _game_pks_cache:
        return _game_pks_cache[date_str]
    try:
        r = SESSION.get(f"{BASE}/schedule",
                        params={"sportId": 1, "date": date_str},
                        timeout=20)
        r.raise_for_status()
        pks = [int(g["gamePk"])
               for d in r.json().get("dates", [])
               for g in d.get("games", [])]
        print(f"  📅  {date_str}: {len(pks)} games found: {pks}")
        _game_pks_cache[date_str] = pks
        return pks
    except Exception as e:
        print(f"  ❌  get_game_pks({date_str}) failed: {type(e).__name__}: {e}")
        _game_pks_cache[date_str] = []
        return []

def get_boxscore(game_pk):
    if game_pk in _boxscore_cache:
        return _boxscore_cache[game_pk]
    try:
        r = SESSION.get(f"{BASE}/game/{game_pk}/boxscore", timeout=20)
        r.raise_for_status()
        data = r.json()
        _boxscore_cache[game_pk] = data
        return data
    except Exception as e:
        print(f"  ❌  get_boxscore({game_pk}) failed: {type(e).__name__}: {e}")
        _boxscore_cache[game_pk] = {}
        return {}

def find_ks(date_str, player_name):
    """
    date_str: 'MM/DD/YYYY'
    Returns int strikeout count or None.
    Searches date ±1 day to handle late-night games.
    """
    player_clean = clean_name(player_name)
    if not player_clean:
        return None

    # Build list of dates to search: day-1, day, day+1
    from datetime import datetime
    try:
        base_date = datetime.strptime(date_str, "%m/%d/%Y").date()
    except ValueError:
        print(f"  ❌  find_ks: could not parse date '{date_str}'")
        return None

    dates_to_try = [
        (base_date + timedelta(days=offset)).strftime("%m/%d/%Y")
        for offset in (-1, 0, 1)
    ]

    for search_date in dates_to_try:
        pks = get_game_pks(search_date)
        for gpk in pks:
            data = get_boxscore(gpk)
            if not data:
                continue
            for side in ("home", "away"):
                players = (data.get("teams") or {}).get(side, {}).get("players", {}) or {}
                for _, p in players.items():
                    person = p.get("person") or {}
                    full = clean_name(person.get("fullName", ""))
                    if full.lower() != player_clean.lower():
                        continue
                    pitching = ((p.get("stats") or {}).get("pitching") or {})
                    ks = pitching.get("strikeOuts")
                    print(f"  🔍  {full} in game {gpk} ({search_date}): strikeOuts={ks}")
                    if ks is not None:
                        try:
                            return int(ks)
                        except Exception:
                            pass

    print(f"  ❌  find_ks: no result for '{player_clean}' around {date_str}")
    return None

# ── startup self-test ─────────────────────────────────────────────────────────

def run_self_test():
    """
    Validates that the API pipeline works end-to-end.
    Uses a known completed game: Michael King, 04/08/2026, expected 4 Ks.
    """
    print("── Self-test: Michael King, 04/08/2026 ──")
    result = find_ks("04/08/2026", "Michael King")
    if result == 4:
        print(f"✅ Self-test PASSED: Michael King = {result} Ks")
    elif result is not None:
        print(f"⚠️  Self-test WARNING: expected 4, got {result}")
    else:
        print("❌ Self-test FAILED: got None — API or name-matching issue detected")
    print("────────────────────────────────────────")

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
    run_self_test()

    if not DATA_FILE.exists():
        print("No data file yet — skipping results update.")
        return

    try:
        df = pd.read_csv(DATA_FILE, dtype=str, on_bad_lines='warn')
    except Exception as e:
        print(f"❌ Failed to read {DATA_FILE}: {e}")
        raise

    print(f"Loaded {len(df)} rows from {DATA_FILE}")

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

    # Show unique player/date combos being processed
    unique_keys = set()
    for idx in needs_update:
        row = df.loc[idx]
        unique_keys.add((str(row.get("Player", "")), str(row.get("Date", ""))))
    print(f"Unique player/date combos: {len(unique_keys)}")
    for k in sorted(unique_keys)[:10]:
        print(f"  • {k[0]} | {k[1]}")
    if len(unique_keys) > 10:
        print(f"  ... and {len(unique_keys) - 10} more")

    cache = {}  # (player_clean, date_str) → actual_ks
    updated = 0
    for idx in needs_update:
        row = df.loc[idx]
        try:
            date_str = str(row.get("Date", "")).strip()
            if not date_str:
                continue
            # Validate date is parseable before caching
            from datetime import datetime
            try:
                datetime.strptime(date_str, "%m/%d/%Y")
            except ValueError:
                print(f"  Row {idx}: unparseable date '{date_str}' — skipping")
                continue

            player_clean = clean_name(str(row.get("Player", "")))
            key = (player_clean, date_str)
            if key not in cache:
                cache[key] = find_ks(date_str, str(row.get("Player", "")))
            actual = cache[key]
            df.at[idx, "Actual Ks"]    = actual if actual is not None else ""
            df.at[idx, "Over Result"]  = decide(actual, parse_line(row.get("Over Line",  "")), "over")
            df.at[idx, "Under Result"] = decide(actual, parse_line(row.get("Under Line", "")), "under")
            updated += 1
        except Exception as e:
            print(f"  Row {idx} error: {e}")

    df.to_csv(DATA_FILE, index=False)
    print(f"✅ Results updated ({updated} rows written) → {DATA_FILE}")

if __name__ == "__main__":
    main()

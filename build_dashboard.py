"""
build_dashboard.py — reads data/props.csv and generates docs/index.html,
a fully self-contained interactive dashboard for win-rate analysis.
"""

import json
import re
from pathlib import Path

import pandas as pd
import numpy as np

DATA_FILE  = Path("data/props.csv")
OUTPUT     = Path("docs/index.html")
OUTPUT.parent.mkdir(exist_ok=True)

# ── load & clean data ─────────────────────────────────────────────────────────

def load_data():
    if not DATA_FILE.exists() or DATA_FILE.stat().st_size == 0:
        return pd.DataFrame()

    df = pd.read_csv(DATA_FILE, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    def to_float(x):
        if pd.isna(x): return np.nan
        s = str(x).strip().replace("%", "").replace("+", "")
        try: return float(s)
        except: return np.nan

    def to_int(x):
        if pd.isna(x): return np.nan
        s = str(x).strip()
        try: return int(s)
        except:
            try: return int(s.replace("+",""))
            except: return np.nan

    def parse_line(x):
        if pd.isna(x): return np.nan
        m = re.search(r"(\d+(?:\.\d+)?)", str(x))
        return float(m.group(1)) if m else np.nan

    df["Over EV%"]    = df["Over EV%"].apply(to_float)
    df["Under EV%"]   = df["Under EV%"].apply(to_float)
    df["Over Odds"]   = df["Over Odds"].apply(to_int)
    df["Under Odds"]  = df["Under Odds"].apply(to_int)
    df["Over Line"]   = df["Over Line"].apply(parse_line)
    df["Under Line"]  = df["Under Line"].apply(parse_line)

    for col in ("Actual Ks", "Over Result", "Under Result", "Scrape Date"):
        if col not in df.columns:
            df[col] = ""

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df

# ── aggregate per (Player, Date, Sportsbook, Side) ───────────────────────────

def extract_game_time(matchup):
    """Pull the game start time from matchup string (e.g. '7:05 PM') for sorting."""
    if not matchup:
        return ""
    m = re.search(r'(\d{1,2}:\d{2}\s*[AP]M)', str(matchup), re.IGNORECASE)
    return m.group(1).strip().upper() if m else ""

def time_to_minutes(t):
    """Convert '08:55 AM' / '01:40 PM' to total minutes since midnight for sorting."""
    if not isinstance(t, str) or not t.strip():
        return 9999
    m = re.match(r'(\d{1,2}):(\d{2})\s*(AM|PM)', t.strip(), re.IGNORECASE)
    if not m:
        return 9999
    h, mn, ap = int(m.group(1)), int(m.group(2)), m.group(3).upper()
    if ap == 'PM' and h != 12:
        h += 12
    elif ap == 'AM' and h == 12:
        h = 0
    return h * 60 + mn

def build_records(df):
    if df.empty:
        return []

    records = []
    group_cols = ["Player", "Matchup", "Sportsbook", "Date"]

    for (player, matchup, book, date_val), grp in df.groupby(group_cols, dropna=False):
        # Deduplicate rows with the same Time value — each scraper run re-appends
        # the full odds history, so identical (Time, Over Odds, Under Odds) rows
        # accumulate across scrape sessions.
        grp = grp.drop_duplicates(subset=["Time", "Over Odds", "Under Odds"])
        # Sort by proper 24h time (string sort breaks on AM/PM boundary)
        grp = grp.assign(_sort_key=grp["Time"].apply(time_to_minutes))
        grp = grp.sort_values("_sort_key", na_position="last").drop(columns="_sort_key")

        for side in ("Over", "Under"):
            ev_col     = f"{side} EV%"
            odds_col   = f"{side} Odds"
            line_col   = f"{side} Line"
            result_col = f"{side} Result"

            ev_vals   = grp[ev_col].dropna()
            line_vals = grp[line_col].dropna()

            # Closing line = the most recent non-null line value
            line_val = float(line_vals.iloc[-1]) if len(line_vals) > 0 else None

            # Only use rows that share the closing line for first/last odds.
            # If the line changed (e.g. o6.5 → o5.5), earlier rows at the old
            # line are irrelevant to the current market and skew movement.
            if line_val is not None:
                closing_line_str = line_vals.iloc[-1]  # original string e.g. "o5.5"
                same_line_mask = grp[line_col] == closing_line_str
                odds_vals = grp.loc[same_line_mask, odds_col].dropna()
            else:
                odds_vals = grp[odds_col].dropna()

            ev_cur     = float(ev_vals.iloc[-1])   if len(ev_vals)   > 0 else None
            first_odds = float(odds_vals.iloc[0])  if len(odds_vals) > 0 else None
            last_odds  = float(odds_vals.iloc[-1]) if len(odds_vals) > 0 else None

            # movement — adjusted for the +/- discontinuity at even money
            # e.g. +110 → -110 = 20 pts (not 220), because both are 10 pts from even
            movement = None
            if first_odds is not None and last_odds is not None:
                if first_odds > 0 and last_odds < 0:
                    # crossed from + to - (odds shortened past even money)
                    movement = -((first_odds - 100) + (abs(last_odds) - 100))
                elif first_odds < 0 and last_odds > 0:
                    # crossed from - to + (odds lengthened past even money)
                    movement = (abs(first_odds) - 100) + (last_odds - 100)
                else:
                    movement = last_odds - first_odds

            def in_favor(fo, lo):
                if fo is None or lo is None: return None
                if fo < 0 and lo < 0: return lo < fo
                if fo > 0 and lo > 0: return abs(lo) < abs(fo)
                if fo >= 0 and lo < 0: return True
                if fo < 0 and lo >= 0: return False
                return None

            mov_favor = in_favor(first_odds, last_odds)

            result_vals = grp[result_col].dropna()
            result = ""
            for v in result_vals:
                v = str(v).strip()
                if v.lower() in ("win", "loss", "push"):
                    result = v.capitalize()
                    break
            if not result:
                for v in result_vals:
                    v = str(v).strip()
                    if v:
                        result = v
                        break

            actual_ks = grp["Actual Ks"].dropna()
            actual_ks_val = None
            if len(actual_ks) > 0:
                try: actual_ks_val = float(actual_ks.iloc[-1])
                except: pass

            date_str = pd.Timestamp(date_val).strftime("%m/%d/%Y") if pd.notna(date_val) else ""

            records.append({
                "player":      str(player),
                "matchup":     str(matchup),
                "book":        str(book),
                "side":        side,
                "date":        date_str,
                "gameTime":    extract_game_time(str(matchup)),
                "ev":          ev_cur,
                "firstOdds":   first_odds,
                "lastOdds":    last_odds,
                "movement":    round(movement, 1) if movement is not None else None,
                "movFavor":    mov_favor,
                "line":        line_val,
                "result":      result,
                "actualKs":    actual_ks_val,
            })

    return records

# ── consensus records (one per player/date/side) ──────────────────────────────

def _avg(vals):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return sum(v) / len(v) if v else None

def _in_favor(fo, lo):
    if fo is None or lo is None: return None
    if fo < 0 and lo < 0:   return lo < fo
    if fo > 0 and lo > 0:   return abs(lo) < abs(fo)
    if fo >= 0 and lo < 0:  return True
    if fo < 0 and lo >= 0:  return False
    return None

def build_consensus_records(records):
    """
    Collapse individual book records → one consensus record per
    (player, date, side).  Rules:
      • Find the line used by the most sportsbooks.
      • If two lines tie in book count → skip (no grade).
      • Only include groups with ≥ 3 books at the dominant line.
      • Average EV%, firstOdds, lastOdds, movement across those books.
      • Re-derive Win/Loss from actual Ks vs dominant line.
    """
    from collections import Counter

    groups = {}
    for r in records:
        key = (r["player"], r["date"], r["side"])
        groups.setdefault(key, []).append(r)

    consensus = []
    for (player, date, side), recs in groups.items():
        # Count books per line (ignore null lines)
        line_counts = Counter(r["line"] for r in recs if r["line"] is not None)
        if not line_counts:
            continue

        max_count = max(line_counts.values())
        dominant_lines = [l for l, c in line_counts.items() if c == max_count]

        # Tie between lines → skip
        if len(dominant_lines) > 1:
            continue

        dominant_line = dominant_lines[0]
        dom_recs = [r for r in recs if r["line"] == dominant_line]

        # Need ≥ 3 books
        if len(dom_recs) < 3:
            continue

        avg_ev     = _avg([r["ev"]         for r in dom_recs])
        avg_first  = _avg([r["firstOdds"]  for r in dom_recs])
        avg_last   = _avg([r["lastOdds"]   for r in dom_recs])
        avg_mov    = _avg([r["movement"]   for r in dom_recs])
        mov_favor  = _in_favor(avg_first, avg_last)

        actual_ks = next((r["actualKs"] for r in dom_recs if r["actualKs"] is not None), None)

        if actual_ks is not None:
            if abs(actual_ks - dominant_line) < 1e-9:
                result = "Push"
            elif side == "Over":
                result = "Win" if actual_ks > dominant_line else "Loss"
            else:
                result = "Win" if actual_ks < dominant_line else "Loss"
        else:
            result = ""   # pending

        first = dom_recs[0]
        consensus.append({
            "player":    player,
            "matchup":   first["matchup"],
            "date":      date,
            "gameTime":  first["gameTime"],
            "side":      side,
            "line":      dominant_line,
            "bookCount": len(dom_recs),
            "ev":        avg_ev,
            "firstOdds": avg_first,
            "lastOdds":  avg_last,
            "movement":  round(avg_mov, 1) if avg_mov is not None else None,
            "movFavor":  mov_favor,
            "result":    result,
            "actualKs":  actual_ks,
        })

    return consensus

# ── HTML template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Props Tracker</title>
<style>
  :root{--bg:#0f1117;--card:#1a1d27;--border:#2a2d3a;--accent:#4ade80;--accent2:#60a5fa;--warn:#f59e0b;--red:#f87171;--text:#e2e8f0;--sub:#8892a4;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:14px;padding-bottom:40px;}
  h1{font-size:1.4rem;font-weight:700;color:#fff;}
  h2{font-size:1rem;font-weight:600;color:var(--text);margin-bottom:10px;}
  header{background:var(--card);border-bottom:1px solid var(--border);padding:16px 20px;display:flex;align-items:center;gap:12px;}
  .badge{background:#252836;border:1px solid var(--border);border-radius:6px;padding:3px 10px;font-size:12px;color:var(--sub);}
  .main{max-width:960px;margin:0 auto;padding:16px;}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:20px;}
  .card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:14px 16px;}
  .card .label{font-size:11px;color:var(--sub);text-transform:uppercase;letter-spacing:.05em;}
  .card .value{font-size:1.6rem;font-weight:700;margin-top:4px;}
  .card .value.green{color:var(--accent);}
  .card .value.blue{color:var(--accent2);}
  .card .value.yellow{color:var(--warn);}
  .section{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:16px;}
  .filters{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:16px;align-items:flex-end;}
  .filter-group{display:flex;flex-direction:column;gap:4px;}
  .filter-group label{font-size:11px;color:var(--sub);text-transform:uppercase;letter-spacing:.05em;}
  select,input[type=range]{background:#252836;border:1px solid var(--border);border-radius:6px;color:var(--text);padding:6px 10px;font-size:13px;outline:none;}
  input[type=range]{padding:4px 0;width:140px;accent-color:var(--accent);}
  .range-val{font-size:12px;color:var(--accent);font-weight:600;min-width:40px;}
  table{width:100%;border-collapse:collapse;}
  th{text-align:left;font-size:11px;color:var(--sub);text-transform:uppercase;letter-spacing:.05em;padding:8px 10px;border-bottom:1px solid var(--border);}
  td{padding:8px 10px;border-bottom:1px solid #1e2130;font-size:13px;}
  tr:last-child td{border-bottom:none;}
  tr:hover td{background:#1e2232;}
  .win{color:var(--accent);font-weight:600;}
  .loss{color:var(--red);}
  .pct{font-weight:700;font-size:1.1rem;}
  .pct.high{color:var(--accent);}
  .pct.mid{color:var(--warn);}
  .pct.low{color:var(--red);}
  .n{color:var(--sub);font-size:11px;}
  .tabs{display:flex;gap:4px;margin-bottom:16px;flex-wrap:wrap;}
  .tab{padding:7px 14px;border-radius:6px;border:1px solid var(--border);background:transparent;color:var(--sub);cursor:pointer;font-size:13px;}
  .tab.active{background:var(--accent);color:#000;border-color:var(--accent);font-weight:600;}
  .tab-panel{display:none;}
  .tab-panel.active{display:block;}
  .matrix-wrap{overflow-x:auto;}
  .combo-label{font-size:12px;color:var(--sub);}
  .pending{color:var(--sub);font-style:italic;}
  /* ── Player Summary tab ── */
  .pc{background:#1e2130;border:1px solid var(--border);border-radius:8px;margin-bottom:14px;overflow:hidden;}
  .pc-hdr{display:flex;justify-content:space-between;align-items:center;padding:10px 16px;background:#252836;border-bottom:1px solid var(--border);}
  .pc-name{font-weight:700;font-size:14px;color:#fff;}
  .pc-date{font-size:12px;color:var(--sub);}
  .pc-line-block{border-bottom:1px solid var(--border);padding:10px 16px;}
  .pc-line-block:last-child{border-bottom:none;}
  .pc-line-label{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--sub);margin-bottom:8px;}
  .pc-side-row{display:flex;flex-wrap:wrap;align-items:center;gap:10px;padding:6px 0;}
  .pc-side-row+.pc-side-row{border-top:1px solid #252836;}
  .side-pill{border-radius:4px;padding:2px 10px;font-size:11px;font-weight:700;min-width:52px;text-align:center;}
  .side-pill.over{background:rgba(74,222,128,.15);color:var(--accent);}
  .side-pill.under{background:rgba(96,165,250,.15);color:var(--accent2);}
  .pc-stat{font-size:13px;color:var(--sub);white-space:nowrap;}
  .pc-stat b{color:var(--text);}
  .pc-stat-sep{color:#3a3d4a;font-size:12px;}
  .expand-btn{margin-top:6px;padding:4px 10px;font-size:11px;color:var(--accent2);background:none;border:1px solid #2a3a4a;border-radius:4px;cursor:pointer;display:inline-flex;align-items:center;gap:5px;}
  .expand-btn:hover{background:#1e2a3a;color:#fff;}
  .book-grid{display:none;margin-top:8px;border:1px solid var(--border);border-radius:6px;overflow:hidden;}
  .book-grid.open{display:block;}
  .book-grid table{width:100%;border-collapse:collapse;font-size:12px;}
  .book-grid th{background:#1a1d27;color:var(--sub);text-align:left;padding:5px 10px;font-size:10px;text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--border);}
  .book-grid td{padding:5px 10px;border-bottom:1px solid #1a1d27;}
  .book-grid tr:last-child td{border-bottom:none;}
  .book-grid tr:hover td{background:#252836;}
  @media(max-width:600px){.cards{grid-template-columns:1fr 1fr;}.filters{flex-direction:column;}.pc-side-row{gap:6px;}}
</style>
</head>
<body>

<header>
  <div>
    <h1>⚾ Props Tracker</h1>
    <div style="margin-top:4px;font-size:12px;color:var(--sub)">MLB Strikeout Props — Win Rate Analysis</div>
  </div>
  <div style="margin-left:auto;text-align:right;">
    <div class="badge" id="last-updated">Updated: __UPDATED__</div>
  </div>
</header>

<div class="main">

  <!-- Global filters -->
  <div class="section">
    <h2>Filters</h2>
    <div class="filters">
      <div class="filter-group">
        <label>Side</label>
        <select id="f-side">
          <option value="both">Both</option>
          <option value="Over">Over</option>
          <option value="Under">Under</option>
        </select>
      </div>
      <div class="filter-group">
        <label>Sportsbook</label>
        <select id="f-book"><option value="all">All Books</option></select>
      </div>
      <div class="filter-group">
        <label>Date From</label>
        <input type="date" id="f-date-from">
      </div>
      <div class="filter-group">
        <label>Date To</label>
        <input type="date" id="f-date-to">
      </div>
      <div class="filter-group">
        <label>EV% Min &ge; <span id="ev-min-label" class="range-val">-100%</span></label>
        <input type="range" id="f-ev-min" min="-100" max="100" step="1" value="-100">
      </div>
      <div class="filter-group">
        <label>EV% Max &le; <span id="ev-max-label" class="range-val">+100%</span></label>
        <input type="range" id="f-ev-max" min="-100" max="100" step="1" value="100">
      </div>
      <div class="filter-group">
        <label>Movement</label>
        <select id="f-mov">
          <option value="all">Any</option>
          <option value="favor">In Favor</option>
          <option value="against">Against</option>
          <option value="none">No Movement</option>
        </select>
      </div>
      <div class="filter-group">
        <label>Results only</label>
        <select id="f-results">
          <option value="all">All rows</option>
          <option value="graded" selected>Graded only</option>
        </select>
      </div>
    </div>
    <div class="cards" id="summary-cards">
      <div class="card"><div class="label">Total Bets</div><div class="value blue" id="s-total">—</div></div>
      <div class="card"><div class="label">Wins</div><div class="value green" id="s-wins">—</div></div>
      <div class="card"><div class="label">Losses</div><div class="value" style="color:var(--red)" id="s-losses">—</div></div>
      <div class="card"><div class="label">Win Rate</div><div class="value green" id="s-winrate">—</div></div>
      <div class="card"><div class="label">Avg EV%</div><div class="value yellow" id="s-ev">—</div></div>
    </div>
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <button class="tab active" onclick="showTab('ev')">EV% Breakdown</button>
    <button class="tab" onclick="showTab('mov')">Line Movement</button>
    <button class="tab" onclick="showTab('combo')">Combined</button>
    <button class="tab" onclick="showTab('book')">By Sportsbook</button>
    <button class="tab" onclick="showTab('player')">By Player</button>
    <button class="tab" onclick="showTab('raw')">Raw Data</button>
  </div>

  <!-- EV% Breakdown -->
  <div id="tab-ev" class="tab-panel active section">
    <h2>Win Rate by EV% Threshold</h2>
    <div id="ev-table"></div>
  </div>

  <!-- Line Movement -->
  <div id="tab-mov" class="tab-panel section">
    <h2>Win Rate by Line Movement</h2>
    <div id="mov-table"></div>
  </div>

  <!-- Combined EV% + Movement -->
  <div id="tab-combo" class="tab-panel section">
    <h2>Combined: EV% Threshold + Movement Direction</h2>
    <div id="combo-table"></div>
  </div>

  <!-- By Sportsbook -->
  <div id="tab-book" class="tab-panel section">
    <h2>Win Rate by Sportsbook — Movement Direction</h2>
    <p style="font-size:12px;color:var(--sub);margin-bottom:14px">
      Each book's movement is its own opening → closing odds. Result is the consensus grade for that player/date/side.
      Only bets where the consensus was graded (Win/Loss) and the book had data are included.
    </p>
    <div id="book-table"></div>
  </div>

  <!-- By Player -->
  <div id="tab-player" class="tab-panel section">
    <h2>By Player</h2>
    <div id="player-content"></div>
  </div>

  <!-- Raw Data -->
  <div id="tab-raw" class="tab-panel section">
    <h2>Raw Data <span style="font-size:12px;font-weight:400;color:var(--sub)">(consensus — 1 graded bet per player/date/side)</span></h2>
    <div style="overflow-x:auto;"><table id="raw-table">
      <thead><tr>
        <th>Player</th><th>Date</th><th>Side</th>
        <th>Avg EV%</th><th>Avg Open</th><th>Avg Close</th><th>Avg Move</th>
        <th>Line</th><th>Books</th><th>Actual Ks</th><th>Result</th>
      </tr></thead>
      <tbody id="raw-body"></tbody>
    </table></div>
  </div>

</div>

<script>
// RAW = one consensus record per (player, date, side) — used for all stats/grading
// BOOK_RECORDS = one record per (player, book, date, side) — used for By Player details
const RAW = __CONSENSUS__;
const BOOK_RECORDS = __BOOK_DATA__;

// ── helpers ───────────────────────────────────────────────────────────────────
function pctClass(p){ return p>=55?'high':p>=45?'mid':'low'; }
function fmtPct(p){ if(isNaN(p)||p===null) return '<span class="n">N/A</span>'; return '<span class="pct '+pctClass(p)+'">'+p.toFixed(1)+'%</span>'; }
function fmtN(n){ return '<span class="n">(n='+n+')</span>'; }
function fmtOdds(o){ if(o===null||o===undefined) return '—'; return o>0?'+'+o:String(o); }
function winRate(rows){
  const graded = rows.filter(r=>r.result==='Win'||r.result==='Loss');
  if(!graded.length) return {rate:NaN,wins:0,losses:0,n:0};
  const wins = graded.filter(r=>r.result==='Win').length;
  return {rate:wins/graded.length*100, wins, losses:graded.length-wins, n:graded.length};
}
function parseDate(s){
  if(!s) return null;
  const [m,d,y] = s.split('/');
  return new Date(y,m-1,d);
}
function toInputDate(s){
  if(!s) return '';
  const [m,d,y]=s.split('/');
  return `${y}-${m.padStart(2,'0')}-${d.padStart(2,'0')}`;
}

// ── populate book filter ──────────────────────────────────────────────────────
const books = [...new Set(RAW.map(r=>r.book))].sort();
const bookSel = document.getElementById('f-book');
books.forEach(b=>{ const o=document.createElement('option'); o.value=b; o.textContent=b; bookSel.appendChild(o); });

// ── EV range labels ───────────────────────────────────────────────────────────
const evMinSlider = document.getElementById('f-ev-min');
const evMaxSlider = document.getElementById('f-ev-max');
const evMinLabel  = document.getElementById('ev-min-label');
const evMaxLabel  = document.getElementById('ev-max-label');
function fmtEvLabel(v){ return (v>=0?'+':'')+v+'%'; }
evMinSlider.addEventListener('input',()=>{ evMinLabel.textContent=fmtEvLabel(evMinSlider.value); refresh(); });
evMaxSlider.addEventListener('input',()=>{ evMaxLabel.textContent=fmtEvLabel(evMaxSlider.value); refresh(); });

// ── date range defaults ───────────────────────────────────────────────────────
const dates = RAW.map(r=>r.date).filter(Boolean).sort();
if(dates.length){
  document.getElementById('f-date-from').value = toInputDate(dates[0]);
  document.getElementById('f-date-to').value   = toInputDate(dates[dates.length-1]);
}

// ── filter ────────────────────────────────────────────────────────────────────
function getFiltered(){
  const side     = document.getElementById('f-side').value;
  const book     = document.getElementById('f-book').value;
  const evMin    = parseFloat(evMinSlider.value);
  const evMax    = parseFloat(evMaxSlider.value);
  const movF     = document.getElementById('f-mov').value;
  const resF     = document.getElementById('f-results').value;
  const dateFrom = document.getElementById('f-date-from').value;
  const dateTo   = document.getElementById('f-date-to').value;

  return RAW.filter(r=>{
    if(side!=='both' && r.side!==side) return false;
    if(book!=='all' && r.book!==book) return false;
    if(r.ev===null || r.ev<evMin || r.ev>evMax) return false;
    if(movF==='favor'  && r.movFavor!==true)  return false;
    if(movF==='against'&& r.movFavor!==false) return false;
    if(movF==='none'   && r.movement!==0 && r.movement!==null) return false;
    if(resF==='graded' && r.result!=='Win' && r.result!=='Loss') return false;
    if(dateFrom){
      const d=parseDate(r.date);
      if(!d||d<new Date(dateFrom)) return false;
    }
    if(dateTo){
      const d=parseDate(r.date);
      if(!d||d>new Date(dateTo+'T23:59:59')) return false;
    }
    return true;
  });
}

// ── summary cards ─────────────────────────────────────────────────────────────
function updateCards(rows){
  const {rate,wins,losses,n} = winRate(rows);
  const evVals = rows.map(r=>r.ev).filter(v=>v!==null&&!isNaN(v));
  const avgEv  = evVals.length ? evVals.reduce((a,b)=>a+b,0)/evVals.length : NaN;
  document.getElementById('s-total').textContent   = rows.length;
  document.getElementById('s-wins').textContent    = wins;
  document.getElementById('s-losses').textContent  = losses;
  document.getElementById('s-winrate').textContent = isNaN(rate)?'—':rate.toFixed(1)+'%';
  document.getElementById('s-ev').textContent      = isNaN(avgEv)?'—':(avgEv>=0?'+':'')+avgEv.toFixed(1)+'%';
}

// ── EV% table ─────────────────────────────────────────────────────────────────
function buildEvTable(base){
  const posThresholds = [0,5,10,15,20,25,30,40,50];
  const negThresholds = [0,-5,-10,-15,-20,-25,-30,-40,-50];
  let html='<table><thead><tr><th>EV% Threshold</th><th>Bets (graded)</th><th>Wins</th><th>Losses</th><th>Win Rate</th></tr></thead><tbody>';
  html+=`<tr><td colspan="5" style="color:var(--accent);font-size:11px;text-transform:uppercase;letter-spacing:.05em;padding-top:10px">Positive EV (Overs/Unders priced favorably)</td></tr>`;
  posThresholds.forEach(t=>{
    const rows = base.filter(r=>r.ev!==null&&r.ev>=t&&(r.result==='Win'||r.result==='Loss'));
    const {rate,wins,losses,n} = winRate(rows);
    html+=`<tr><td>EV% &ge; +${t}%</td><td>${n}</td><td class="win">${wins}</td><td class="loss">${losses}</td><td>${fmtPct(rate)}</td></tr>`;
  });
  html+=`<tr><td colspan="5" style="color:var(--red);font-size:11px;text-transform:uppercase;letter-spacing:.05em;padding-top:10px">Negative EV (Fading the book)</td></tr>`;
  negThresholds.forEach(t=>{
    const rows = base.filter(r=>r.ev!==null&&r.ev<=t&&(r.result==='Win'||r.result==='Loss'));
    const {rate,wins,losses,n} = winRate(rows);
    html+=`<tr><td>EV% &le; ${t}%</td><td>${n}</td><td class="win">${wins}</td><td class="loss">${losses}</td><td>${fmtPct(rate)}</td></tr>`;
  });
  html+='</tbody></table>';
  document.getElementById('ev-table').innerHTML=html;
}

// ── movement table ────────────────────────────────────────────────────────────
function buildMovTable(base){
  const cats = [
    {label:'Any Movement',  fn:r=>r.movement!==null},
    {label:'Moved In Favor',fn:r=>r.movFavor===true},
    {label:'Moved Against', fn:r=>r.movFavor===false},
    {label:'No Movement',   fn:r=>r.movement===0},
    {label:'|Move| 1–5',    fn:r=>r.movement!==null&&Math.abs(r.movement)>=1&&Math.abs(r.movement)<=5},
    {label:'|Move| 6–10',   fn:r=>r.movement!==null&&Math.abs(r.movement)>=6&&Math.abs(r.movement)<=10},
    {label:'|Move| 11–20',  fn:r=>r.movement!==null&&Math.abs(r.movement)>=11&&Math.abs(r.movement)<=20},
    {label:'|Move| 21+',    fn:r=>r.movement!==null&&Math.abs(r.movement)>=21},
  ];
  let html='<table><thead><tr><th>Movement Category</th><th>Bets (graded)</th><th>Wins</th><th>Losses</th><th>Win Rate</th></tr></thead><tbody>';
  cats.forEach(({label,fn})=>{
    const rows = base.filter(r=>fn(r)&&(r.result==='Win'||r.result==='Loss'));
    const {rate,wins,losses,n} = winRate(rows);
    html+=`<tr><td>${label}</td><td>${n}</td><td class="win">${wins}</td><td class="loss">${losses}</td><td>${fmtPct(rate)}</td></tr>`;
  });
  html+='</tbody></table>';
  document.getElementById('mov-table').innerHTML=html;
}

// ── combo table ───────────────────────────────────────────────────────────────
function buildComboTable(base){
  const movs = [{label:'In Favor',fn:r=>r.movFavor===true},{label:'Against',fn:r=>r.movFavor===false},{label:'Any',fn:()=>true}];

  // Positive EV section
  const posTs = [0,5,10,15,20,25];
  let html='<div class="matrix-wrap">';
  html+=`<div style="color:var(--accent);font-size:11px;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px">Positive EV &ge; threshold</div>`;
  html+='<table><thead><tr><th>EV% &ge;</th>';
  movs.forEach(m=>html+=`<th>Move: ${m.label}</th>`);
  html+='</tr></thead><tbody>';
  posTs.forEach(t=>{
    html+=`<tr><td><b>+${t}%</b></td>`;
    movs.forEach(({fn})=>{
      const rows=base.filter(r=>r.ev!==null&&r.ev>=t&&fn(r)&&(r.result==='Win'||r.result==='Loss'));
      const {rate,n}=winRate(rows);
      html+=`<td>${fmtPct(rate)} ${fmtN(n)}</td>`;
    });
    html+='</tr>';
  });
  html+='</tbody></table>';

  // Negative EV section
  const negTs = [0,-5,-10,-15,-20,-25];
  html+=`<div style="color:var(--red);font-size:11px;text-transform:uppercase;letter-spacing:.05em;margin:16px 0 6px">Negative EV &le; threshold</div>`;
  html+='<table><thead><tr><th>EV% &le;</th>';
  movs.forEach(m=>html+=`<th>Move: ${m.label}</th>`);
  html+='</tr></thead><tbody>';
  negTs.forEach(t=>{
    html+=`<tr><td><b>${t}%</b></td>`;
    movs.forEach(({fn})=>{
      const rows=base.filter(r=>r.ev!==null&&r.ev<=t&&fn(r)&&(r.result==='Win'||r.result==='Loss'));
      const {rate,n}=winRate(rows);
      html+=`<td>${fmtPct(rate)} ${fmtN(n)}</td>`;
    });
    html+='</tr>';
  });
  html+='</tbody></table></div>';

  document.getElementById('combo-table').innerHTML=html;
}

// ── book table ────────────────────────────────────────────────────────────────
function buildBookTable(base){
  // Build a lookup: player|date|side → consensus result
  const resultMap={};
  base.forEach(r=>{
    if(r.result==='Win'||r.result==='Loss')
      resultMap[r.player+'|'+r.date+'|'+r.side]=r.result;
  });

  // Apply date/side filters from global controls to BOOK_RECORDS
  const side     = document.getElementById('f-side').value;
  const dateFrom = document.getElementById('f-date-from').value;
  const dateTo   = document.getElementById('f-date-to').value;

  // Enrich book records with consensus result
  const enriched = BOOK_RECORDS.map(r=>{
    const result = resultMap[r.player+'|'+r.date+'|'+r.side] || null;
    return {...r, result};
  }).filter(r=>{
    if(r.result!=='Win'&&r.result!=='Loss') return false;  // only graded
    if(side!=='both'&&r.side!==side) return false;
    if(dateFrom){ const d=parseDate(r.date); if(!d||d<new Date(dateFrom)) return false; }
    if(dateTo){   const d=parseDate(r.date); if(!d||d>new Date(dateTo+'T23:59:59')) return false; }
    return true;
  });

  // Group by book
  const bks=[...new Set(enriched.map(r=>r.book))].sort();

  const thStyle='padding:8px 12px;text-align:center;';
  const tdStyle='padding:7px 12px;text-align:center;';

  let html='<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;">';
  html+=`<thead><tr>
    <th style="padding:8px 12px;text-align:left;">Sportsbook</th>
    <th style="${thStyle}">Moved In Favor</th><th style="${thStyle}">W</th><th style="${thStyle}">L</th><th style="${thStyle}">Win Rate</th>
    <th style="padding:8px 4px;color:var(--border)">|</th>
    <th style="${thStyle}">Moved Against</th><th style="${thStyle}">W</th><th style="${thStyle}">L</th><th style="${thStyle}">Win Rate</th>
    <th style="padding:8px 4px;color:var(--border)">|</th>
    <th style="${thStyle}">No Movement</th><th style="${thStyle}">W</th><th style="${thStyle}">L</th><th style="${thStyle}">Win Rate</th>
  </tr></thead><tbody>`;

  bks.forEach(b=>{
    const rows=enriched.filter(r=>r.book===b);

    const favor  = rows.filter(r=>r.movFavor===true);
    const against= rows.filter(r=>r.movFavor===false);
    const none   = rows.filter(r=>r.movement===0||r.movement===null);

    function stats(arr){
      const w=arr.filter(r=>r.result==='Win').length;
      const l=arr.filter(r=>r.result==='Loss').length;
      const n=w+l;
      const rate=n>0?w/n*100:NaN;
      return {w,l,n,rate};
    }

    function cells(arr){
      const {w,l,n,rate}=stats(arr);
      const rateHtml=n>0?fmtPct(rate):`<span class="n">—</span>`;
      return `<td style="${tdStyle}">${n}</td><td style="${tdStyle}" class="win">${w}</td><td style="${tdStyle}" class="loss">${l}</td><td style="${tdStyle}">${rateHtml}</td>`;
    }

    const sep=`<td style="padding:0 4px;color:var(--border);text-align:center">|</td>`;
    html+=`<tr>
      <td style="padding:7px 12px;font-weight:600">${b}</td>
      ${cells(favor)}${sep}${cells(against)}${sep}${cells(none)}
    </tr>`;
  });

  // Totals row
  const favor  = enriched.filter(r=>r.movFavor===true);
  const against= enriched.filter(r=>r.movFavor===false);
  const none   = enriched.filter(r=>r.movement===0||r.movement===null);
  function totCells(arr){
    const w=arr.filter(r=>r.result==='Win').length;
    const l=arr.filter(r=>r.result==='Loss').length;
    const n=w+l;
    const rate=n>0?w/n*100:NaN;
    return `<td style="${tdStyle};font-weight:600">${n}</td><td style="${tdStyle}" class="win">${w}</td><td style="${tdStyle}" class="loss">${l}</td><td style="${tdStyle}">${n>0?fmtPct(rate):'—'}</td>`;
  }
  const sep=`<td style="padding:0 4px;color:var(--border);text-align:center">|</td>`;
  html+=`<tr style="border-top:2px solid var(--border)">
    <td style="padding:7px 12px;font-weight:700;color:var(--sub);font-size:11px;text-transform:uppercase">All Books</td>
    ${totCells(favor)}${sep}${totCells(against)}${sep}${totCells(none)}
  </tr>`;

  html+='</tbody></table></div>';
  document.getElementById('book-table').innerHTML=html;
}

// ── raw table ─────────────────────────────────────────────────────────────────
function gameTimeTo24h(t){
  if(!t) return '99:99';
  const m=t.match(/(\d{1,2}):(\d{2})\s*(AM|PM)/i);
  if(!m) return '99:99';
  let h=parseInt(m[1]); const min=m[2]; const ap=m[3].toUpperCase();
  if(ap==='PM'&&h!==12) h+=12;
  if(ap==='AM'&&h===12) h=0;
  return String(h).padStart(2,'0')+':'+min;
}
function buildRawTable(rows){
  const MAX=300;
  // Sort: newest date first → game time ascending → player ascending
  const sorted=[...rows].sort((a,b)=>{
    const da=parseDate(a.date), db=parseDate(b.date);
    if(da&&db){ const diff=db-da; if(diff!==0) return diff; }
    const ta=gameTimeTo24h(a.gameTime), tb=gameTimeTo24h(b.gameTime);
    if(ta!==tb) return ta<tb?-1:1;
    return (a.player||'').localeCompare(b.player||'');
  });
  const tbody=document.getElementById('raw-body');
  tbody.innerHTML='';
  sorted.slice(0,MAX).forEach(r=>{
    const tr=document.createElement('tr');
    const res=r.result==='Win'?'<span class="win">Win</span>':r.result==='Loss'?'<span class="loss">Loss</span>':r.result==='Push'?'<span style="color:var(--warn)">Push</span>':r.result?'<span class="pending">'+r.result+'</span>':'<span class="n">Pending</span>';
    const mov=r.movement===null?'—':(r.movement>=0?'+':'')+r.movement+(r.movFavor===true?' ↑':r.movFavor===false?' ↓':'');
    const displayName=(r.player||'').replace(/\s*\([^)]*\)/g,'').trim();
    tr.innerHTML=`<td>${displayName}</td><td>${r.date||'—'}</td><td>${r.side}</td>`+
      `<td>${r.ev!==null?(r.ev>=0?'+':'')+r.ev.toFixed(1)+'%':'—'}</td>`+
      `<td>${fmtAvgOdds(r.firstOdds)}</td><td>${fmtAvgOdds(r.lastOdds)}</td><td>${mov}</td>`+
      `<td>${r.line!==null?r.line:'—'}</td><td>${r.bookCount||'—'}</td><td>${r.actualKs!==null&&r.actualKs!==undefined?r.actualKs:'—'}</td><td>${res}</td>`;
    tbody.appendChild(tr);
  });
  if(sorted.length>MAX){
    const tr=document.createElement('tr');
    tr.innerHTML=`<td colspan="11" style="text-align:center;color:var(--sub)">Showing ${MAX} of ${sorted.length} consensus bets — use filters to narrow</td>`;
    tbody.appendChild(tr);
  }
}

// ── player summary ────────────────────────────────────────────────────────────
function fmtAvgOdds(v){
  if(v===null||v===undefined||isNaN(v)) return '—';
  const r=Math.round(v); return r>0?'+'+r:String(r);
}
function fmtAvgMov(v){
  if(v===null||v===undefined||isNaN(v)) return '—';
  const r=Math.round(v*10)/10; return (r>=0?'+':'')+r;
}
function fmtAvgEv(v){
  if(v===null||v===undefined||isNaN(v)) return '—';
  return (v>=0?'+':'')+v.toFixed(1)+'%';
}

function buildPlayerTable(base){
  // base = filtered consensus records (one per player/date/side)
  // For book details, look up matching entries in BOOK_RECORDS

  // Build a lookup map from BOOK_RECORDS: player+date+side+line → [book records]
  const bookMap={};
  BOOK_RECORDS.forEach(r=>{
    const k=r.player+'|'+r.date+'|'+r.side+'|'+(r.line!=null?r.line:'');
    if(!bookMap[k]) bookMap[k]=[];
    bookMap[k].push(r);
  });

  // Group consensus records: player → date → [records]
  const map={};
  base.forEach(r=>{
    if(!map[r.player]) map[r.player]={};
    if(!map[r.player][r.date]) map[r.player][r.date]=[];
    map[r.player][r.date].push(r);
  });

  let html='';
  Object.keys(map).sort().forEach(player=>{
    const sortedDates=Object.keys(map[player]).sort((a,b)=>parseDate(b)-parseDate(a));
    sortedDates.forEach(date=>{
      const recs=map[player][date]; // consensus records for this player/date
      if(!recs.length) return;

      // Collect unique lines across both sides
      const lineSet=new Set(recs.map(r=>r.line!=null?String(r.line):'__'));
      const lines=[...lineSet].filter(l=>l!=='__').sort((a,b)=>parseFloat(a)-parseFloat(b));
      if(lineSet.has('__')) lines.push('__');

      const displayName=player.replace(/\\s*\\([^)]*\\)/g,'').trim();
      html+=`<div class="pc">`;
      html+=`<div class="pc-hdr"><span class="pc-name">${displayName}</span><span class="pc-date">${date}</span></div>`;

      lines.forEach(lk=>{
        const lineVal=lk==='__'?null:parseFloat(lk);
        const lineRecs=recs.filter(r=>(r.line!=null?String(r.line):'__')===lk);
        if(!lineRecs.length) return;

        html+=`<div class="pc-line-block">`;
        if(lines.length>1) html+=`<div class="pc-line-label">Line: ${lk==='__'?'—':lk}</div>`;

        ['Over','Under'].forEach(side=>{
          const cr=lineRecs.find(r=>r.side===side);
          if(!cr) return;

          // Result badge
          let resBadge='';
          if(cr.result==='Win')  resBadge=`<span class="win" style="margin-left:auto;font-size:12px">Win ✓</span>`;
          else if(cr.result==='Loss') resBadge=`<span class="loss" style="margin-left:auto;font-size:12px">Loss ✗</span>`;
          else if(cr.result==='Push') resBadge=`<span style="color:var(--warn);margin-left:auto;font-size:12px">Push</span>`;
          else if(cr.actualKs!==null&&cr.actualKs!==undefined) resBadge=`<span class="n" style="margin-left:auto;font-size:12px">—</span>`;
          else resBadge=`<span class="pending" style="margin-left:auto;font-size:12px">Pending</span>`;

          const ksStr=cr.actualKs!==null&&cr.actualKs!==undefined?` · ${cr.actualKs} K`:'';
          const uid=('pd_'+player+'_'+date+'_'+lk+'_'+side).replace(/[^a-zA-Z0-9]/g,'_');

          html+=`<div class="pc-side-row">`;
          html+=`<span class="side-pill ${side.toLowerCase()}">${side}</span>`;
          html+=`<span class="pc-stat">EV: <b>${fmtAvgEv(cr.ev)}</b></span>`;
          html+=`<span class="pc-stat-sep">|</span>`;
          html+=`<span class="pc-stat">Open: <b>${fmtAvgOdds(cr.firstOdds)}</b></span>`;
          html+=`<span class="pc-stat-sep">|</span>`;
          html+=`<span class="pc-stat">Close: <b>${fmtAvgOdds(cr.lastOdds)}</b></span>`;
          html+=`<span class="pc-stat-sep">|</span>`;
          html+=`<span class="pc-stat">Move: <b>${fmtAvgMov(cr.movement)}</b></span>`;
          html+=`<span class="pc-stat-sep">|</span>`;
          html+=`<span class="pc-stat"><b>${cr.bookCount}</b> books${ksStr}</span>`;
          html+=resBadge;
          html+=`</div>`;

          // Book detail dropdown
          const bookKey=player+'|'+date+'|'+side+'|'+(cr.line!=null?cr.line:'');
          const bookRecs=(bookMap[bookKey]||[]).sort((a,b)=>(a.book||'').localeCompare(b.book||''));
          if(bookRecs.length){
            html+=`<button class="expand-btn" onclick="togglePD('${uid}')"><span id="${uid}_arrow">▶</span> Book Details</button>`;
            html+=`<div class="book-grid" id="${uid}">`;
            html+=`<table><thead><tr><th>Book</th><th>EV%</th><th>Open</th><th>Close</th><th>Move</th></tr></thead><tbody>`;
            bookRecs.forEach(r=>{
              const mov=r.movement!==null?(r.movement>=0?'+':'')+r.movement:'—';
              const evStr=r.ev!==null?(r.ev>=0?'+':'')+r.ev.toFixed(1)+'%':'—';
              html+=`<tr><td>${r.book}</td><td>${evStr}</td><td>${fmtOdds(r.firstOdds)}</td><td>${fmtOdds(r.lastOdds)}</td><td>${mov}</td></tr>`;
            });
            html+=`</tbody></table></div>`;
          }
        });
        html+=`</div>`; // pc-line-block
      });
      html+=`</div>`; // pc
    });
  });

  if(!html) html='<p style="color:var(--sub);text-align:center;padding:20px">No data matches current filters.</p>';
  document.getElementById('player-content').innerHTML=html;
}

function togglePD(uid){
  const el=document.getElementById(uid);
  const arrow=document.getElementById(uid+'_arrow');
  const open=el.classList.toggle('open');
  if(arrow) arrow.textContent=open?'▼':'▶';
}

// ── main refresh ──────────────────────────────────────────────────────────────
function getFilteredForDisplay(){
  // Used for Raw Data and By Player tabs:
  // - Ignores the results filter (shows pending/ungraded rows)
  // - Allows null EV through (today's games may not have EV scraped yet)
  // - Still respects date, side, book filters
  const side     = document.getElementById('f-side').value;
  const dateFrom = document.getElementById('f-date-from').value;
  const dateTo   = document.getElementById('f-date-to').value;
  return RAW.filter(r=>{
    if(side!=='both' && r.side!==side) return false;
    if(dateFrom){ const d=parseDate(r.date); if(!d||d<new Date(dateFrom)) return false; }
    if(dateTo){   const d=parseDate(r.date); if(!d||d>new Date(dateTo+'T23:59:59')) return false; }
    return true;
  });
}

function refresh(){
  const filtered=getFiltered();
  updateCards(filtered);
  buildEvTable(filtered);
  buildMovTable(filtered);
  buildComboTable(filtered);
  buildBookTable(filtered);
  const display=getFilteredForDisplay();
  buildPlayerTable(display);
  buildRawTable(display);
}

// ── tabs ──────────────────────────────────────────────────────────────────────
function showTab(id){
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  event.target.classList.add('active');
}

// ── wire up listeners ─────────────────────────────────────────────────────────
['f-side','f-book','f-mov','f-results','f-date-from','f-date-to'].forEach(id=>{
  document.getElementById(id).addEventListener('change', refresh);
});


refresh();
</script>
</body>
</html>
"""

# ── build ──────────────────────────────────────────────────────────────────────

def main():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    df = load_data()
    book_records  = build_records(df)
    consensus     = build_consensus_records(book_records)

    updated = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d %H:%M CT")
    consensus_json  = json.dumps(consensus,    default=str)
    book_json       = json.dumps(book_records, default=str)

    html = (HTML_TEMPLATE
            .replace("__CONSENSUS__",  consensus_json)
            .replace("__BOOK_DATA__",  book_json)
            .replace("__UPDATED__",    updated))
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"✅ Dashboard written → {OUTPUT}  ({len(consensus)} consensus / {len(book_records)} book records)")


if __name__ == "__main__":
    main()

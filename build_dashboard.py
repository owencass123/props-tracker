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

def build_records(df):
    if df.empty:
        return []

    records = []
    group_cols = ["Player", "Matchup", "Sportsbook", "Date"]

    for (player, matchup, book, date_val), grp in df.groupby(group_cols, dropna=False):
        grp = grp.sort_values("Time", na_position="last")

        for side in ("Over", "Under"):
            ev_col     = f"{side} EV%"
            odds_col   = f"{side} Odds"
            line_col   = f"{side} Line"
            result_col = f"{side} Result"

            ev_vals   = grp[ev_col].dropna()
            odds_vals = grp[odds_col].dropna()
            line_vals = grp[line_col].dropna()

            ev_cur    = float(ev_vals.iloc[-1])  if len(ev_vals)   > 0 else None
            first_odds = float(odds_vals.iloc[0]) if len(odds_vals) > 0 else None
            last_odds  = float(odds_vals.iloc[-1]) if len(odds_vals) > 0 else None
            line_val   = float(line_vals.iloc[-1]) if len(line_vals)  > 0 else None

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
  @media(max-width:600px){.cards{grid-template-columns:1fr 1fr;}.filters{flex-direction:column;}}
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
    <h2>Win Rate by Sportsbook</h2>
    <div id="book-table"></div>
  </div>

  <!-- Raw Data -->
  <div id="tab-raw" class="tab-panel section">
    <h2>Raw Data</h2>
    <div style="overflow-x:auto;"><table id="raw-table">
      <thead><tr>
        <th>Player</th><th>Date</th><th>Book</th><th>Side</th>
        <th>EV%</th><th>1st Odds</th><th>Last Odds</th><th>Movement</th>
        <th>Line</th><th>Actual Ks</th><th>Result</th>
      </tr></thead>
      <tbody id="raw-body"></tbody>
    </table></div>
  </div>

</div>

<script>
const RAW = __DATA__;

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
  const graded = base.filter(r=>r.result==='Win'||r.result==='Loss');
  const bks = [...new Set(graded.map(r=>r.book))].sort();
  let html='<table><thead><tr><th>Sportsbook</th><th>Bets</th><th>Wins</th><th>Losses</th><th>Win Rate</th><th>Avg EV%</th></tr></thead><tbody>';
  bks.forEach(b=>{
    const rows=graded.filter(r=>r.book===b);
    const {rate,wins,losses,n}=winRate(rows);
    const evVals=rows.map(r=>r.ev).filter(v=>v!==null&&!isNaN(v));
    const avgEv=evVals.length?evVals.reduce((a,c)=>a+c,0)/evVals.length:NaN;
    html+=`<tr><td>${b}</td><td>${n}</td><td class="win">${wins}</td><td class="loss">${losses}</td><td>${fmtPct(rate)}</td><td>${isNaN(avgEv)?'—':(avgEv>=0?'+':'')+avgEv.toFixed(1)+'%'}</td></tr>`;
  });
  html+='</tbody></table>';
  document.getElementById('book-table').innerHTML=html;
}

// ── raw table ─────────────────────────────────────────────────────────────────
function buildRawTable(rows){
  const MAX=300;
  const tbody=document.getElementById('raw-body');
  tbody.innerHTML='';
  rows.slice(0,MAX).forEach(r=>{
    const tr=document.createElement('tr');
    const res=r.result==='Win'?'<span class="win">Win</span>':r.result==='Loss'?'<span class="loss">Loss</span>':r.result?'<span class="pending">'+r.result+'</span>':'<span class="n">Pending</span>';
    const mov=r.movement===null?'—':(r.movement>=0?'+':'')+r.movement+(r.movFavor===true?' ↑':r.movFavor===false?' ↓':'');
    tr.innerHTML=`<td>${r.player}</td><td>${r.date||'—'}</td><td>${r.book}</td><td>${r.side}</td>`+
      `<td>${r.ev!==null?(r.ev>=0?'+':'')+r.ev.toFixed(1)+'%':'—'}</td>`+
      `<td>${fmtOdds(r.firstOdds)}</td><td>${fmtOdds(r.lastOdds)}</td><td>${mov}</td>`+
      `<td>${r.line!==null?r.line:'—'}</td><td>${r.actualKs!==null&&r.actualKs!==undefined?r.actualKs:'—'}</td><td>${res}</td>`;
    tbody.appendChild(tr);
  });
  if(rows.length>MAX){
    const tr=document.createElement('tr');
    tr.innerHTML=`<td colspan="11" style="text-align:center;color:var(--sub)">Showing ${MAX} of ${rows.length} rows — use filters to narrow</td>`;
    tbody.appendChild(tr);
  }
}

// ── main refresh ──────────────────────────────────────────────────────────────
function refresh(){
  const filtered=getFiltered();
  updateCards(filtered);
  buildEvTable(filtered);
  buildMovTable(filtered);
  buildComboTable(filtered);
  buildBookTable(filtered);
  buildRawTable(filtered);
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
    df = load_data()
    records = build_records(df)

    updated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    data_json = json.dumps(records, default=str)

    html = HTML_TEMPLATE.replace("__DATA__", data_json).replace("__UPDATED__", updated)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"✅ Dashboard written → {OUTPUT}  ({len(records)} records)")


if __name__ == "__main__":
    main()

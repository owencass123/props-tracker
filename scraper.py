"""
scraper.py — Unabated MLB strikeout props scraper
Runs headlessly, auto-logs in, auto-clicks Simulate, collects EV% + odds history
for every player × sportsbook. Appends results to data/props.csv.
"""

import os
import re
import time
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

# ── credentials (from GitHub Secrets / env vars) ──────────────────────────────
USERNAME = os.environ.get("UNABATED_USERNAME", "")
PASSWORD = os.environ.get("UNABATED_PASSWORD", "")

if not USERNAME or not PASSWORD:
    raise RuntimeError("Set UNABATED_USERNAME and UNABATED_PASSWORD environment variables.")

# ── sportsbook column IDs in the AG Grid ──────────────────────────────────────
SPORTSBOOK_COL_IDS_DEFAULT = {
    "FanDuel":    "2",
    "DraftKings": "1",
    "Sharp Book": "7",
    "Caesars":    "20",
    "BetMGM":     "4",
    "ESPN Bet":   "36",
    "Bookmaker":  "8",
    "Bovada":     "10",
    "BetRivers":  "17",
    "Fanatics":   "86",
    "Hard Rock":  "24",
}

SPORTSBOOK_COL_IDS = {}

BOOK_NAME_ALIASES = {
    "fanduel":    "FanDuel",
    "draftkings": "DraftKings",
    "sharp":      "Sharp Book",
    "caesars":    "Caesars",
    "betmgm":     "BetMGM",
    "mgm":        "BetMGM",
    "espn":       "ESPN Bet",
    "bookmaker":  "Bookmaker",
    "bovada":     "Bovada",
    "betrivers":  "BetRivers",
    "fanatics":   "Fanatics",
    "hard rock":  "Hard Rock",
    "hardrock":   "Hard Rock",
}

DATA_FILE = Path("data/props.csv")
DATA_FILE.parent.mkdir(exist_ok=True)

CSV_COLUMNS = [
    "Player", "Matchup", "Sportsbook",
    "Over EV%", "Over Odds", "Over Line",
    "Under EV%", "Under Odds", "Under Line",
    "Time", "Date", "Scrape Date",
]

_CT = timezone(timedelta(hours=-5))
TODAY = datetime.now(_CT).strftime("%m/%d/%Y")

_PW = None
_SCRAPE_DEADLINE = None
SCRAPE_LIMIT_MINS = 20  # bail out and save partial results after this many minutes


# ── column ID auto-detection ──────────────────────────────────────────────────

def detect_col_ids(page):
    global SPORTSBOOK_COL_IDS
    detected = {}
    try:
        grid_scroll_to(page, 0)
        time.sleep(0.3)
        max_scroll = get_grid_scroll_width(page)
        step = 300
        x = 0
        while x <= max_scroll + step:
            headers = page.query_selector_all(".ag-header-container .ag-header-cell[col-id]")
            for h in headers:
                col_id = h.get_attribute("col-id")
                if not col_id:
                    continue
                try:
                    label = h.query_selector(".ag-header-cell-text").inner_text().strip().lower()
                except Exception:
                    label = h.inner_text().strip().lower()
                if not label:
                    continue
                print(f"  🔎 Header col-id={col_id!r} label={label!r}")
                for alias, canonical in BOOK_NAME_ALIASES.items():
                    if alias in label and canonical not in detected:
                        detected[canonical] = col_id
                        print(f"  📌 Detected col-id for {canonical}: {col_id} (header: '{label}')")
                        break
            x += step
            grid_scroll_to(page, x)
            time.sleep(0.1)
        grid_scroll_to(page, 0)
    except Exception as e:
        print(f"  ⚠️  detect_col_ids failed: {e}")

    for book, default_id in SPORTSBOOK_COL_IDS_DEFAULT.items():
        if book not in detected:
            print(f"  ⚠️  {book}: not detected in header, using default col-id {default_id}")
            detected[book] = default_id

    SPORTSBOOK_COL_IDS = detected
    print(f"✅ Column IDs resolved: {SPORTSBOOK_COL_IDS}")


# ── browser setup ─────────────────────────────────────────────────────────────

def setup_browser():
    global _PW
    _PW = sync_playwright().start()
    browser = _PW.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
        ],
    )
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    return browser, page


# ── login ─────────────────────────────────────────────────────────────────────

_USERNAME_SELECTORS = [
    "#username",
    "input[name='username']",
    "input[type='email']",
    "input[name='email']",
    "input[placeholder*='email' i]",
    "input[placeholder*='username' i]",
]

_PASSWORD_SELECTORS = [
    "#password",
    "input[name='password']",
    "input[type='password']",
]


def _find_input(page, selectors, timeout=5000):
    for sel in selectors:
        try:
            el = page.wait_for_selector(sel, timeout=timeout, state="visible")
            if el:
                return el, sel
        except PWTimeout:
            continue
    return None, None


def login(page):
    # Try direct login URL first — skips the need to find a login button
    direct_urls = [
        "https://unabated.com/login",
        "https://app.unabated.com/login",
        "https://unabated.com/sign-in",
    ]

    username_el = None
    for url in direct_urls:
        page.goto(url)
        page.wait_for_load_state("domcontentloaded")
        time.sleep(1)
        username_el, found_sel = _find_input(page, _USERNAME_SELECTORS, timeout=4000)
        if username_el:
            print(f"✅ Login form found at {url} (field: {found_sel})")
            break

    if not username_el:
        # Fall back: homepage + find login button via JS text scan
        page.goto("https://unabated.com")
        page.wait_for_load_state("networkidle")
        time.sleep(3)

        # Find ANY visible element whose own text node is a login phrase
        handle = page.evaluate_handle("""() => {
            const phrases = ['login', 'log in', 'sign in'];
            const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT);
            while (walker.nextNode()) {
                const el = walker.currentNode;
                const st = getComputedStyle(el);
                if (st.display === 'none' || st.visibility === 'hidden') continue;
                const own = Array.from(el.childNodes)
                    .filter(n => n.nodeType === 3)
                    .map(n => n.textContent.trim().toLowerCase())
                    .join(' ').trim();
                if (phrases.some(p => own === p)) return el;
            }
            return null;
        }""")

        el = handle.as_element()
        if el:
            el.click()
            print("✅ Login button found via JS text scan")
            username_el, _ = _find_input(page, _USERNAME_SELECTORS)
        else:
            page.screenshot(path="/tmp/login_page.png", full_page=True)
            all_text = page.evaluate("""() =>
                Array.from(document.querySelectorAll('*'))
                    .filter(el => el.children.length === 0 && el.textContent.trim())
                    .map(el => el.textContent.trim())
                    .filter(t => t.length < 40)
                    .slice(0, 30)
            """)
            print(f"⚠️  Login not found. Leaf text on page: {all_text}")
            raise RuntimeError("Could not find login — screenshot at /tmp/login_page.png")

    if not username_el:
        raise RuntimeError("Found login button but no username field appeared")

    password_el, _ = _find_input(page, _PASSWORD_SELECTORS)
    if not password_el:
        raise RuntimeError("Username field found but no password field")

    username_el.fill(USERNAME)
    password_el.fill(PASSWORD)
    page.keyboard.press("Enter")
    page.wait_for_load_state("networkidle", timeout=20000)
    time.sleep(3)
    print("✅ Logged in")


# ── simulate button ───────────────────────────────────────────────────────────

def click_simulate(page):
    try:
        btn = page.locator("xpath=//a[contains(@title,'Simulate') or contains(@class,'btn-success')]")
        btn.wait_for(state="visible", timeout=15000)
        btn.evaluate("(el) => el.click()")
        time.sleep(2)
        for selector in [
            ".dropdown-menu a",
            ".dropdown-item",
            "[role='menuitem']",
            ".modal-body a",
            ".projection-set-option",
        ]:
            try:
                opt = page.wait_for_selector(selector, state="visible", timeout=2000)
                opt.evaluate("(el) => el.click()")
                print(f"✅ Selected projection set via {selector}")
                break
            except PWTimeout:
                continue
        time.sleep(4)
        print("✅ Simulate clicked")
    except Exception as e:
        print(f"⚠️  Simulate button not found or failed: {e}")


# ── parsing helpers ───────────────────────────────────────────────────────────

def parse_line_and_odds(text):
    if not text:
        return None, ""
    t = text.replace("OVER", "o").replace("UNDER", "u")
    t = re.sub(r"[^0-9ou+\-.\s]", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*([+-]\d{2,4})", t)
    if m:
        return float(m.group(1)), m.group(2)
    m = re.search(r"([+-]\d{2,4})\s*(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(2)), m.group(1)
    return None, ""


def normalize_time(s):
    if not s:
        return ""
    m = re.search(r"\b(\d{1,2}):(\d{2})(?::\d{2})?\s*([AP]M)\b", s, re.IGNORECASE)
    if m:
        return f"{m.group(1).zfill(2)}:{m.group(2)} {m.group(3).upper()}"
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", s)
    if m:
        return f"{m.group(1).zfill(2)}:{m.group(2)}"
    return s


def normalize_date(s):
    if not s:
        return ""
    m = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", s)
    if not m:
        return ""
    mm, dd = int(m.group(1)), int(m.group(2))
    yy = m.group(3)
    year = (2000 + int(yy)) if len(yy) == 2 and int(yy) <= 69 else (1900 + int(yy)) if len(yy) == 2 else int(yy)
    return f"{mm:02d}/{dd:02d}/{year:04d}"


# ── grid scroll helpers ───────────────────────────────────────────────────────

def grid_scroll_to(page, x):
    page.evaluate("""(x) => {
        const c = document.querySelector('.ag-center-cols-viewport');
        const h = document.querySelector('.ag-header-viewport');
        if (c) c.scrollLeft = Math.max(0, x);
        if (h) h.scrollLeft = Math.max(0, x);
    }""", int(x))


def grid_scroll_by(page, dx):
    page.evaluate("""(dx) => {
        const c = document.querySelector('.ag-center-cols-viewport');
        const h = document.querySelector('.ag-header-viewport');
        if (c) { c.scrollLeft += dx; if (h) h.scrollLeft = c.scrollLeft; }
    }""", int(dx))


def grid_scroll_left_edge(page):
    grid_scroll_to(page, 0)


def get_grid_scroll_width(page):
    return page.evaluate("""() => {
        const c = document.querySelector('.ag-center-cols-viewport');
        return c ? c.scrollWidth : 0;
    }""")


def jump_to_header(page, col_id, margin=120):
    try:
        hdr = page.query_selector(f".ag-header-container [col-id='{col_id}']")
        if not hdr:
            return False
        page.evaluate("""([hdr, margin]) => {
            const c = document.querySelector('.ag-center-cols-viewport');
            const h = document.querySelector('.ag-header-viewport');
            if (!c || !hdr) return;
            const t = Math.max(0, (hdr.offsetLeft || 0) - margin);
            c.scrollLeft = t; if (h) h.scrollLeft = t;
        }""", [hdr, int(margin)])
        time.sleep(0.08)
        return True
    except Exception:
        return False


def vert_scroll_row_into_view(page, row_id):
    try:
        el = page.query_selector(f".ag-pinned-left-cols-container [row-id='{row_id}']")
        if not el:
            return False
        el.evaluate("(el) => el.scrollIntoView({block:'nearest'})")
        time.sleep(0.05)
        return True
    except Exception:
        return False


def find_center_cell(page, row_id, col_id):
    return page.query_selector(
        f".ag-center-cols-container [row-id='{row_id}'] [col-id='{col_id}']"
    )


def wait_for_cell(page, row_id, col_id, timeout=0.8):
    sel = f".ag-center-cols-container [row-id='{row_id}'] [col-id='{col_id}']"
    try:
        page.wait_for_selector(sel, timeout=timeout * 1000)
        return page.query_selector(sel)
    except PWTimeout:
        return None


def ensure_cell_visible(page, row_id, col_id):
    if not vert_scroll_row_into_view(page, row_id):
        return None
    cell = find_center_cell(page, row_id, col_id)
    if cell:
        return cell
    grid_scroll_left_edge(page)
    cell = wait_for_cell(page, row_id, col_id)
    if cell:
        return cell
    if jump_to_header(page, col_id):
        cell = wait_for_cell(page, row_id, col_id)
        if cell:
            return cell
    for _ in range(18):
        grid_scroll_by(page, 200)
        cell = wait_for_cell(page, row_id, col_id)
        if cell:
            return cell
    grid_scroll_to(page, get_grid_scroll_width(page))
    time.sleep(0.05)
    jump_to_header(page, col_id)
    return wait_for_cell(page, row_id, col_id)


# ── right panel (odds history) helpers ───────────────────────────────────────

def find_right_panel(page):
    roots = page.query_selector_all("div.ag-root")
    rightmost, rightmost_x = None, -1
    for r in roots:
        try:
            x = r.evaluate("(el) => el.getBoundingClientRect().x")
            if x > rightmost_x:
                rightmost_x, rightmost = x, r
        except Exception:
            continue
    return rightmost


def get_panel_col_indices(panel):
    headers = panel.query_selector_all(".ag-header .ag-header-cell")
    labels = []
    for h in headers:
        try:
            labels.append(h.query_selector(".ag-header-cell-text").inner_text().strip().lower())
        except Exception:
            labels.append(h.inner_text().strip().lower())

    def idx_of(*keys):
        for i, t in enumerate(labels):
            for k in keys:
                if k in t:
                    return i
        return None

    return idx_of("time", "timestamp", "updated"), idx_of("over"), idx_of("under")


def wait_for_panel_rows(page, timeout=4.0):
    end = time.time() + timeout
    while time.time() < end:
        panel = find_right_panel(page)
        if panel:
            rows = panel.query_selector_all(".ag-center-cols-container .ag-row")
            if rows:
                return panel, rows
        time.sleep(0.1)
    return find_right_panel(page), []


def extract_panel_history(panel):
    time_idx, over_idx, under_idx = get_panel_col_indices(panel)
    out = []
    for r in panel.query_selector_all(".ag-center-cols-container .ag-row"):
        cells = r.query_selector_all(".ag-cell")
        if not cells:
            continue
        ts  = cells[time_idx].inner_text().strip()  if (time_idx  is not None and time_idx  < len(cells)) else ""
        ov  = cells[over_idx].inner_text().strip()  if (over_idx  is not None and over_idx  < len(cells)) else ""
        un  = cells[under_idx].inner_text().strip() if (under_idx is not None and under_idx < len(cells)) else ""
        ov_line, ov_odds = parse_line_and_odds(ov)
        un_line, un_odds = parse_line_and_odds(un)
        out.append((ts, ov_line, ov_odds, un_line, un_odds))
    return out


# ── EV% extraction ────────────────────────────────────────────────────────────

def extract_ev(cell):
    spans = cell.query_selector_all("xpath=.//span[contains(text(), '%')]")
    vals = []
    for sp in spans:
        t = sp.inner_text().strip().replace("−", "-")
        m = re.search(r"[+-]?\d+(?:\.\d+)?\s*%", t)
        if m:
            v = m.group(0).replace(" ", "")
            if v not in vals:
                vals.append(v)
    return vals[0] if len(vals) > 0 else "", vals[1] if len(vals) > 1 else ""


def open_panel(page, cell):
    for selector in [".props-hover-cells span", None]:
        try:
            if selector:
                targets = [c for c in cell.query_selector_all(selector) if c.is_visible()]
                target = targets[0] if targets else cell
            else:
                target = cell
            target.evaluate("(el) => el.scrollIntoView({block:'nearest',inline:'center'})")
            try:
                target.click()
            except Exception:
                target.evaluate("(el) => el.click()")
            time.sleep(0.25)
            return True
        except Exception:
            continue
    return False


# ── frozen column (player info) ───────────────────────────────────────────────

def extract_frozen_info(page, row_id):
    player, matchup = "N/A", "N/A"
    try:
        frozen = page.query_selector(f".ag-pinned-left-cols-container [row-id='{row_id}']")
        if not frozen:
            return player, matchup
        cells = frozen.query_selector_all(".ag-cell")
        if len(cells) > 1:
            try:
                player  = cells[1].query_selector("div[style*='font-size: 0.9rem']").inner_text().strip()
                matchup = cells[1].query_selector("div[style*='font-size: 0.65rem']").inner_text().strip().replace("\xa0", " ")
            except Exception:
                pass
    except Exception:
        pass
    return player, matchup


# ── row / cell processing ─────────────────────────────────────────────────────

def cell_has_data(cell):
    try:
        return bool(cell.inner_text().strip())
    except Exception:
        return False


def process_cell(page, cell, player, matchup, book, rows_out):
    if not cell_has_data(cell):
        print(f"  ⏭️  {book}: empty cell, skipping")
        return

    ev_over, ev_under = extract_ev(cell)

    if not open_panel(page, cell):
        print(f"  ⚠️  {book}: could not open panel")
        return

    panel, _ = wait_for_panel_rows(page, timeout=4.0)
    if not panel:
        print(f"  ⚠️  {book}: panel not found")
        return

    history = extract_panel_history(panel)
    if not history:
        print(f"  ⚠️  {book}: empty panel")
        return

    for (ts, ov_line, ov_odds, un_line, un_odds) in history:
        def fmt_line(val, prefix):
            if val is None:
                return ""
            s = str(val).rstrip("0").rstrip(".")
            return prefix + s

        panel_date = normalize_date(ts)
        if panel_date != TODAY:
            continue

        rows_out.append({
            "Player":      player,
            "Matchup":     matchup,
            "Sportsbook":  book,
            "Over EV%":    ev_over,
            "Over Odds":   ov_odds,
            "Over Line":   fmt_line(ov_line, "o"),
            "Under EV%":   ev_under,
            "Under Odds":  un_odds,
            "Under Line":  fmt_line(un_line, "u"),
            "Time":        normalize_time(ts),
            "Date":        TODAY,
            "Scrape Date": TODAY,
        })


def process_row(page, row_id, rows_out):
    grid_scroll_left_edge(page)
    vert_scroll_row_into_view(page, row_id)
    player, matchup = extract_frozen_info(page, row_id)

    for book, col_id in SPORTSBOOK_COL_IDS.items():
        try:
            cell = ensure_cell_visible(page, row_id, col_id)
            if not cell:
                grid_scroll_left_edge(page)
                for _ in range(24):
                    cell = wait_for_cell(page, row_id, col_id, timeout=0.25)
                    if cell:
                        break
                    grid_scroll_by(page, 120)
            if not cell:
                print(f"  ⛔ {book}: cell not found for row {row_id}")
                continue

            process_cell(page, cell, player, matchup, book, rows_out)
            print(f"  ✅ {player} | {book}")

        except Exception as e:
            # Playwright doesn't have StaleElementReferenceException — just retry once
            try:
                cell = ensure_cell_visible(page, row_id, col_id)
                if cell:
                    process_cell(page, cell, player, matchup, book, rows_out)
            except Exception as e2:
                print(f"  ⚠️  {book} row {row_id}: {e2}")


def scroll_and_process_all_rows(page, rows_out):
    seen = set()
    finalized = load_finalized_keys()
    print(f"ℹ️  Skipping {len(finalized)} already-finalized player+date combos")

    for attempt in range(60):
        if _SCRAPE_DEADLINE and time.time() > _SCRAPE_DEADLINE:
            print(f"⏱️  Scrape time limit reached — saving {len(rows_out)} rows collected so far")
            break

        rows = page.query_selector_all(".ag-center-cols-container .ag-row")
        new_found = False
        for row in rows:
            if _SCRAPE_DEADLINE and time.time() > _SCRAPE_DEADLINE:
                print(f"⏱️  Scrape time limit reached mid-page — saving partial results")
                return

            row_id = row.get_attribute("row-id")
            if not row_id or row_id in seen:
                continue
            seen.add(row_id)

            player, matchup = extract_frozen_info(page, row_id)
            player_lower = player.strip().lower()
            if (player_lower, TODAY) in finalized:
                print(f"  ⏭️  {player} — already finalized for {TODAY}, skipping")
                continue

            new_found = True
            process_row(page, row_id, rows_out)

        if not new_found:
            print("🛑 No new rows — done scrolling")
            break

        page.evaluate("""() => {
            const c = document.querySelector('.ag-body-viewport');
            if (c) c.scrollBy(0, 540);
        }""")
        time.sleep(0.5)


# ── CSV save ──────────────────────────────────────────────────────────────────

def load_finalized_keys():
    finalized = set()
    if not DATA_FILE.exists() or DATA_FILE.stat().st_size == 0:
        return finalized
    try:
        import pandas as pd
        df = pd.read_csv(DATA_FILE, dtype=str)
        final_results = {"win", "loss", "push"}
        for col in ("Over Result", "Under Result"):
            if col not in df.columns:
                continue
            mask = df[col].str.strip().str.lower().isin(final_results)
            for _, row in df[mask].iterrows():
                player = str(row.get("Player", "")).strip().lower()
                date   = str(row.get("Date", "")).strip()
                if player and date:
                    finalized.add((player, date))
    except Exception as e:
        print(f"⚠️  Could not load finalized keys: {e}")
    return finalized


def append_to_csv(rows):
    file_exists = DATA_FILE.exists() and DATA_FILE.stat().st_size > 0
    with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in CSV_COLUMNS})
    print(f"\n💾 Appended {len(rows)} rows to {DATA_FILE}")
    _dedup_csv()


ROLLING_DAYS = 30


def _archive_compress(df_old):
    import pandas as pd
    key = ['Player', 'Matchup', 'Sportsbook', 'Date', 'Over Line', 'Under Line']
    existing_key = [c for c in key if c in df_old.columns]
    parts = []
    for _, grp in df_old.groupby(existing_key, sort=False, dropna=False):
        parts.append(grp.iloc[[0, -1]] if len(grp) > 1 else grp)
    return pd.concat(parts, ignore_index=True) if parts else df_old.iloc[0:0]


def _normalize_odds(series):
    import pandas as pd
    def fix(v):
        if pd.isna(v) or str(v).strip() == "":
            return v
        s = str(v).strip()
        try:
            f = float(s)
            i = int(f)
            if f == i:
                return ("+" if i > 0 else "") + str(i)
        except ValueError:
            pass
        return s
    return series.map(fix)


def _dedup_csv():
    import pandas as pd
    try:
        before = len(pd.read_csv(DATA_FILE, dtype=str))
        df = pd.read_csv(DATA_FILE, dtype=str)

        normalized = False
        for col in ('Over Odds', 'Under Odds'):
            if col in df.columns:
                fixed = _normalize_odds(df[col])
                if not fixed.equals(df[col]):
                    df[col] = fixed
                    normalized = True

        dedup_cols = ['Player','Matchup','Sportsbook','Date','Time',
                      'Over Odds','Under Odds','Over Line','Under Line',
                      'Over EV%','Under EV%']
        existing = [c for c in dedup_cols if c in df.columns]
        df = df.drop_duplicates(subset=existing, keep='last')

        if 'Date' in df.columns:
            cutoff = datetime.utcnow() - timedelta(days=ROLLING_DAYS)
            dates = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
            recent = df[dates >= cutoff]
            old    = df[dates < cutoff]
            old_compressed = _archive_compress(old) if not old.empty else old
            df = pd.concat([old_compressed, recent], ignore_index=True).drop_duplicates()

        after = len(df)
        if after < before or normalized:
            df.to_csv(DATA_FILE, index=False)
            print(f"🧹 Deduped/archived CSV: {before} → {after} rows" + (" (odds normalized)" if normalized else ""))
    except Exception as e:
        print(f"⚠️  CSV dedup failed (non-fatal): {e}")


# ── main ──────────────────────────────────────────────────────────────────────

def _dismiss_modals(page):
    for sel in [
        "[class*='modal'] button[class*='close']",
        "[class*='modal'] button[aria-label*='close' i]",
        "[class*='dialog'] button[class*='close']",
        "[role='dialog'] button[class*='close']",
        "button[aria-label='Close']",
        "button.close",
        "[data-dismiss='modal']",
    ]:
        try:
            el = page.query_selector(sel)
            if el and el.is_visible():
                el.click()
                time.sleep(0.5)
                print(f"  ✅ Dismissed modal via {sel}")
        except Exception:
            pass


def main():
    global _SCRAPE_DEADLINE
    _SCRAPE_DEADLINE = time.time() + SCRAPE_LIMIT_MINS * 60
    print(f"⏱️  Scrape deadline: {SCRAPE_LIMIT_MINS} minutes from now")
    rows_out = []
    browser, page = setup_browser()
    try:
        login(page)

        page.goto("https://tools.unabated.com/mlb/props")
        page.wait_for_load_state("networkidle", timeout=30000)
        time.sleep(3)
        _dismiss_modals(page)

        try:
            page.wait_for_selector(".ag-center-cols-container", timeout=60000)
        except PWTimeout:
            page.screenshot(path="/tmp/props_page.png", full_page=True)
            print(f"  Current URL: {page.url}")
            leaf_text = page.evaluate("""() =>
                Array.from(document.querySelectorAll('*'))
                    .filter(el => el.children.length === 0 && el.textContent.trim())
                    .map(el => el.textContent.trim())
                    .filter(t => t.length < 60)
                    .slice(0, 40)
            """)
            print(f"  Page text: {leaf_text}")
            raise

        time.sleep(3)
        click_simulate(page)
        page.wait_for_selector(".ag-center-cols-container .ag-row", timeout=20000)
        detect_col_ids(page)
        scroll_and_process_all_rows(page, rows_out)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
    finally:
        if rows_out:
            append_to_csv(rows_out)
        try:
            browser.close()
        except Exception:
            pass
        try:
            _PW.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()

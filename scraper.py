"""
scraper.py — Unabated MLB strikeout props scraper
Runs headlessly, auto-logs in, auto-clicks Simulate, collects EV% + odds history
for every player × sportsbook. Appends results to data/props.csv.
"""

import os
import re
import time
import csv
from datetime import date
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import (
    NoSuchElementException, StaleElementReferenceException, TimeoutException
)
from webdriver_manager.chrome import ChromeDriverManager

# ── credentials (from GitHub Secrets / env vars) ──────────────────────────────
USERNAME = os.environ.get("UNABATED_USERNAME", "")
PASSWORD = os.environ.get("UNABATED_PASSWORD", "")

if not USERNAME or not PASSWORD:
    raise RuntimeError("Set UNABATED_USERNAME and UNABATED_PASSWORD environment variables.")

# ── sportsbook column IDs in the AG Grid ──────────────────────────────────────
SPORTSBOOK_COL_IDS = {
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

DATA_FILE = Path("data/props.csv")
DATA_FILE.parent.mkdir(exist_ok=True)

CSV_COLUMNS = [
    "Player", "Matchup", "Sportsbook",
    "Over EV%", "Over Odds", "Over Line",
    "Under EV%", "Under Odds", "Under Line",
    "Time", "Date", "Scrape Date",
]

TODAY = date.today().strftime("%m/%d/%Y")


# ── driver setup ──────────────────────────────────────────────────────────────

def setup_driver():
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)
    driver.execute_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )
    return driver


# ── login ─────────────────────────────────────────────────────────────────────

def login(driver):
    driver.get("https://unabated.com")
    WebDriverWait(driver, 15).until(
        EC.element_to_be_clickable((By.XPATH, "//button[normalize-space()='LOGIN']"))
    ).click()
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "username"))
    ).send_keys(USERNAME)
    driver.find_element(By.ID, "password").send_keys(PASSWORD + Keys.RETURN)
    WebDriverWait(driver, 20).until(EC.url_contains("unabated.com"))
    time.sleep(3)
    print("✅ Logged in")


# ── simulate button ───────────────────────────────────────────────────────────

def click_simulate(driver):
    """Auto-click the Simulate button and select the first projection set."""
    try:
        btn = WebDriverWait(driver, 15).until(EC.element_to_be_clickable((
            By.XPATH,
            "//a[contains(@title,'Simulate') or contains(@class,'btn-success')]"
        )))
        driver.execute_script("arguments[0].click();", btn)
        time.sleep(2)
        # If a dropdown/modal appeared, click the first option
        for selector in [
            ".dropdown-menu a",
            ".dropdown-item",
            "[role='menuitem']",
            ".modal-body a",
            ".projection-set-option",
        ]:
            try:
                opt = WebDriverWait(driver, 2).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                driver.execute_script("arguments[0].click();", opt)
                print(f"✅ Selected projection set via {selector}")
                break
            except TimeoutException:
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

def grid_scroll_to(driver, x):
    driver.execute_script("""
        const c = document.querySelector('.ag-center-cols-viewport');
        const h = document.querySelector('.ag-header-viewport');
        if (c) c.scrollLeft = Math.max(0, arguments[0]);
        if (h) h.scrollLeft = Math.max(0, arguments[0]);
    """, int(x))


def grid_scroll_by(driver, dx):
    driver.execute_script("""
        const c = document.querySelector('.ag-center-cols-viewport');
        const h = document.querySelector('.ag-header-viewport');
        if (c) { c.scrollLeft += arguments[0]; if (h) h.scrollLeft = c.scrollLeft; }
    """, int(dx))


def grid_scroll_left_edge(driver):
    grid_scroll_to(driver, 0)


def get_grid_scroll_width(driver):
    return driver.execute_script("""
        const c = document.querySelector('.ag-center-cols-viewport');
        return c ? c.scrollWidth : 0;
    """)


def jump_to_header(driver, col_id, margin=120):
    try:
        hdr = driver.find_element(By.CSS_SELECTOR, f".ag-header-container [col-id='{col_id}']")
        driver.execute_script("""
            const c = document.querySelector('.ag-center-cols-viewport');
            const h = document.querySelector('.ag-header-viewport');
            if (!c || !arguments[0]) return;
            const t = Math.max(0, (arguments[0].offsetLeft || 0) - arguments[1]);
            c.scrollLeft = t; if (h) h.scrollLeft = t;
        """, hdr, int(margin))
        time.sleep(0.08)
        return True
    except NoSuchElementException:
        return False


def vert_scroll_row_into_view(driver, row_id):
    try:
        el = driver.find_element(
            By.CSS_SELECTOR, f".ag-pinned-left-cols-container [row-id='{row_id}']"
        )
        driver.execute_script("arguments[0].scrollIntoView({block:'nearest'});", el)
        time.sleep(0.05)
        return True
    except NoSuchElementException:
        return False


def find_center_cell(driver, row_id, col_id):
    try:
        return driver.find_element(
            By.CSS_SELECTOR,
            f".ag-center-cols-container [row-id='{row_id}'] [col-id='{col_id}']"
        )
    except NoSuchElementException:
        return None


def wait_for_cell(driver, row_id, col_id, timeout=0.8):
    try:
        return WebDriverWait(driver, timeout, poll_frequency=0.1).until(
            EC.presence_of_element_located((
                By.CSS_SELECTOR,
                f".ag-center-cols-container [row-id='{row_id}'] [col-id='{col_id}']"
            ))
        )
    except TimeoutException:
        return None


def ensure_cell_visible(driver, row_id, col_id):
    if not vert_scroll_row_into_view(driver, row_id):
        return None
    cell = find_center_cell(driver, row_id, col_id)
    if cell:
        return cell
    grid_scroll_left_edge(driver)
    cell = wait_for_cell(driver, row_id, col_id)
    if cell:
        return cell
    if jump_to_header(driver, col_id):
        cell = wait_for_cell(driver, row_id, col_id)
        if cell:
            return cell
    for _ in range(18):
        grid_scroll_by(driver, 200)
        cell = wait_for_cell(driver, row_id, col_id)
        if cell:
            return cell
    grid_scroll_to(driver, get_grid_scroll_width(driver))
    time.sleep(0.05)
    jump_to_header(driver, col_id)
    return wait_for_cell(driver, row_id, col_id)


# ── right panel (odds history) helpers ───────────────────────────────────────

def find_right_panel(driver):
    roots = driver.find_elements(By.CSS_SELECTOR, "div.ag-root")
    rightmost, rightmost_x = None, -1
    for r in roots:
        try:
            x = driver.execute_script(
                "return arguments[0].getBoundingClientRect().x;", r
            )
            if x > rightmost_x:
                rightmost_x, rightmost = x, r
        except Exception:
            continue
    return rightmost


def get_panel_col_indices(panel):
    headers = panel.find_elements(By.CSS_SELECTOR, ".ag-header .ag-header-cell")
    labels = []
    for h in headers:
        try:
            labels.append(h.find_element(By.CSS_SELECTOR, ".ag-header-cell-text").text.strip().lower())
        except Exception:
            labels.append(h.text.strip().lower())

    def idx_of(*keys):
        for i, t in enumerate(labels):
            for k in keys:
                if k in t:
                    return i
        return None

    return idx_of("time", "timestamp", "updated"), idx_of("over"), idx_of("under")


def wait_for_panel_rows(driver, timeout=4.0):
    end = time.time() + timeout
    while time.time() < end:
        panel = find_right_panel(driver)
        if panel:
            rows = panel.find_elements(By.CSS_SELECTOR, ".ag-center-cols-container .ag-row")
            if rows:
                return panel, rows
        time.sleep(0.1)
    return find_right_panel(driver), []


def extract_panel_history(panel):
    time_idx, over_idx, under_idx = get_panel_col_indices(panel)
    out = []
    for r in panel.find_elements(By.CSS_SELECTOR, ".ag-center-cols-container .ag-row"):
        cells = r.find_elements(By.CSS_SELECTOR, ".ag-cell")
        if not cells:
            continue
        ts  = cells[time_idx].text.strip()  if (time_idx  is not None and time_idx  < len(cells)) else ""
        ov  = cells[over_idx].text.strip()  if (over_idx  is not None and over_idx  < len(cells)) else ""
        un  = cells[under_idx].text.strip() if (under_idx is not None and under_idx < len(cells)) else ""
        ov_line, ov_odds = parse_line_and_odds(ov)
        un_line, un_odds = parse_line_and_odds(un)
        out.append((ts, ov_line, ov_odds, un_line, un_odds))
    return out


# ── EV% extraction ────────────────────────────────────────────────────────────

def extract_ev(cell):
    spans = cell.find_elements(By.XPATH, ".//span[contains(text(), '%')]")
    vals = []
    for sp in spans:
        t = sp.text.strip().replace("−", "-")
        m = re.search(r"[+-]?\d+(?:\.\d+)?\s*%", t)
        if m:
            v = m.group(0).replace(" ", "")
            if v not in vals:
                vals.append(v)
    return vals[0] if len(vals) > 0 else "", vals[1] if len(vals) > 1 else ""


def open_panel(driver, cell):
    for selector in [".props-hover-cells span", None]:
        try:
            if selector:
                targets = [c for c in cell.find_elements(By.CSS_SELECTOR, selector) if c.is_displayed()]
                target = targets[0] if targets else cell
            else:
                target = cell
            driver.execute_script("arguments[0].scrollIntoView({block:'nearest',inline:'center'});", target)
            try:
                target.click()
            except Exception:
                driver.execute_script("arguments[0].click();", target)
            time.sleep(0.25)
            return True
        except Exception:
            continue
    return False


# ── frozen column (player info) ───────────────────────────────────────────────

def extract_frozen_info(driver, row_id):
    player, matchup = "N/A", "N/A"
    try:
        frozen = driver.find_element(
            By.CSS_SELECTOR, f".ag-pinned-left-cols-container [row-id='{row_id}']"
        )
        cells = frozen.find_elements(By.CLASS_NAME, "ag-cell")
        if len(cells) > 1:
            try:
                player  = cells[1].find_element(By.CSS_SELECTOR, "div[style*='font-size: 0.9rem']").text.strip()
                matchup = cells[1].find_element(By.CSS_SELECTOR, "div[style*='font-size: 0.65rem']").text.strip().replace("\xa0", " ")
            except Exception:
                pass
    except Exception:
        pass
    return player, matchup


# ── row / cell processing ─────────────────────────────────────────────────────

def process_cell(driver, cell, player, matchup, book, rows_out):
    ev_over, ev_under = extract_ev(cell)

    if not open_panel(driver, cell):
        print(f"  ⚠️  {book}: could not open panel")
        return

    panel, _ = wait_for_panel_rows(driver, timeout=4.0)
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
            "Date":        normalize_date(ts) or TODAY,
            "Scrape Date": TODAY,
        })


def process_row(driver, row_id, rows_out):
    grid_scroll_left_edge(driver)
    vert_scroll_row_into_view(driver, row_id)
    player, matchup = extract_frozen_info(driver, row_id)

    for book, col_id in SPORTSBOOK_COL_IDS.items():
        try:
            cell = ensure_cell_visible(driver, row_id, col_id)
            if not cell:
                # brute-force
                grid_scroll_left_edge(driver)
                for _ in range(24):
                    cell = wait_for_cell(driver, row_id, col_id, timeout=0.25)
                    if cell:
                        break
                    grid_scroll_by(driver, 120)
            if not cell:
                print(f"  ⛔ {book}: cell not found for row {row_id}")
                continue

            process_cell(driver, cell, player, matchup, book, rows_out)
            print(f"  ✅ {player} | {book}")

        except StaleElementReferenceException:
            try:
                cell = ensure_cell_visible(driver, row_id, col_id)
                if cell:
                    process_cell(driver, cell, player, matchup, book, rows_out)
            except Exception as e:
                print(f"  ♻️  {book}: stale retry failed — {e}")
        except Exception as e:
            print(f"  ⚠️  {book} row {row_id}: {e}")


def scroll_and_process_all_rows(driver, rows_out):
    seen = set()
    finalized = load_finalized_keys()
    print(f"ℹ️  Skipping {len(finalized)} already-finalized player+date combos")

    for attempt in range(60):
        rows = driver.find_elements(By.CSS_SELECTOR, ".ag-center-cols-container .ag-row")
        new_found = False
        for row in rows:
            row_id = row.get_attribute("row-id")
            if not row_id or row_id in seen:
                continue
            seen.add(row_id)

            # Check if this player already has a finalized result on any date
            player, matchup = extract_frozen_info(driver, row_id)
            player_lower = player.strip().lower()
            if any(k[0] == player_lower for k in finalized):
                print(f"  ⏭️  {player} — already finalized, skipping")
                continue

            new_found = True
            process_row(driver, row_id, rows_out)

        if not new_found:
            print("🛑 No new rows — done scrolling")
            break

        driver.execute_script("""
            const c = document.querySelector('.ag-body-viewport');
            if (c) c.scrollBy(0, 540);
        """)
        time.sleep(0.5)


# ── CSV save ──────────────────────────────────────────────────────────────────

def load_finalized_keys():
    """
    Returns a set of (player_lower, date) tuples that already have a
    final result (Win/Loss/Push) in the CSV. These should not be re-scraped.
    """
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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    rows_out = []
    driver = setup_driver()
    try:
        login(driver)
        driver.get("https://unabated.com/mlb/props")
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".ag-center-cols-container"))
        )
        time.sleep(3)
        click_simulate(driver)
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".ag-center-cols-container .ag-row"))
        )
        scroll_and_process_all_rows(driver, rows_out)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
    finally:
        if rows_out:
            append_to_csv(rows_out)
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()

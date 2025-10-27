#!/usr/bin/env python3
"""
nse_live_scanner.py

- Fetches full NSE symbol list live (nsetools)
- Scans entire universe in batches (yfinance)
- Computes RSI, MACD, EMA20, Bollinger Bands, volume MA
- Ranks top BUY candidates (Top N) and SELL candidates
- Maintains buys.csv (your bought list) and evaluates each holding
- Optional Telegram alerts
- Runs continuously and retries on network failures

Configure parameters in CONFIG section below.
"""

import time
import os
import csv
import math
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import yfinance as yf

# ---------------- CONFIG ----------------
POLL_INTERVAL = 180             # seconds between full scans (tune as needed)
BATCH_SIZE = 60                 # number of symbols per yfinance batch
INTERVAL = "5m"                 # yfinance interval ("1m","5m","15m","30m","60m","1d")
PERIOD = "7d"                   # data period for yfinance (enough history for indicators)
TOP_N = 10                      # number of top buys/sells to show
EXCHANGE_SUFFIX = ".NS"         # yfinance suffix for NSE tickers
RSI_PERIOD = 14
EMA_FAST = 12
EMA_SLOW = 26
EMA_SIGNAL = 9
BB_WINDOW = 20
VOLUME_MA = 10
VOLUME_SPIKE_FACTOR = 1.5
RATE_LIMIT_SLEEP = 1.5          # seconds between batch downloads

# Files
BASE_DIR = os.path.dirname(__file__)
BUYS_FILE = os.path.join(BASE_DIR, "buys.csv")
LOG_FILE = os.path.join(BASE_DIR, "scanner_log.csv")

# Telegram (optional) - set TOKEN and CHAT_ID if you want alerts
TELEGRAM_TOKEN = ""            # e.g., "123456789:ABC..."; leave blank to disable
TELEGRAM_CHAT_ID = ""          # e.g., "987654321" or "-1001234567890"

# Scoring weights (tweak to taste)
W_RSI = 0.6
W_MACD = 1.0
W_EMA = 0.4
W_VOL = 0.3

# Retry / symbol refresh
SYMBOL_FETCH_RETRIES = 6
SYMBOL_REFRESH_HOURS = 8     # refresh symbol list this often (hours)
# ----------------------------------------

# Ensure CSVs exist
if not os.path.exists(BUYS_FILE):
    with open(BUYS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_utc","symbol","entry_price","entry_rsi","note"])

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_utc","symbol","action","price","rsi","macd_hist","note"])

# ---------- Utilities / Indicators ----------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_indicators(df):
    """Input dataframe must contain 'Close' and 'Volume' (yfinance default)."""
    if df.empty:
        return df
    df = df.copy()
    # Ensure column names lower-case for easier access
    df.columns = [c.lower() for c in df.columns]
    # rename if using older formats (some yfinance outputs)
    if 'close' not in df.columns and 'adj close' in df.columns:
        df['close'] = df['adj close']

    close = df['close']
    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = ema(close, EMA_FAST)
    ema_slow = ema(close, EMA_SLOW)
    macd_line = ema_fast - ema_slow
    macd_signal = ema(macd_line, EMA_SIGNAL)
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # EMA20
    df['ema20'] = ema(close, 20)

    # Bollinger
    ma = close.rolling(BB_WINDOW, min_periods=BB_WINDOW).mean()
    std = close.rolling(BB_WINDOW, min_periods=BB_WINDOW).std()
    df['bb_upper'] = ma + 2 * std
    df['bb_lower'] = ma - 2 * std

    # volume MA
    if 'volume' in df.columns:
        df['vol_ma'] = df['volume'].rolling(VOLUME_MA, min_periods=1).mean()
    else:
        df['vol_ma'] = np.nan

    return df

def score_for_buy(meta):
    # meta: rsi, macd_hist, price_vs_ema20, vol_ratio
    rsi = meta.get('rsi', 50)
    macd_hist = meta.get('macd_hist', 0)
    price_vs_ema20 = meta.get('price_vs_ema20', 0)
    vol_ratio = meta.get('vol_ratio', 1)
    # normalize components
    rsi_score = max(0, (50 - rsi)) / 50.0          # 0..1 (lower rsi better)
    macd_score = max(0, macd_hist)                 # could be small; used as-is
    ema_score = max(0, price_vs_ema20) / 100.0     # percentage -> 0..?
    vol_score = max(0, vol_ratio - 1.0)            # extra vol above MA

    score = W_RSI * rsi_score + W_MACD * macd_score + W_EMA * ema_score + W_VOL * vol_score
    return float(score)

# ---------- NSE symbols fetch (live) ----------
def fetch_nse_symbols():
    """Fetch NSE symbols live using nsetools. Retries on failure."""
    try:
        from nsetools import Nse
    except Exception as e:
        raise RuntimeError("nsetools not installed or import failed: " + str(e))
    nse = Nse()
    retries = 0
    while retries < SYMBOL_FETCH_RETRIES:
        try:
            print(f"[{datetime.utcnow().isoformat()}] Fetching NSE symbols (attempt {retries+1})...")
            codes = nse.get_stock_codes()  # dict: symbol->name, includes header 'SYMBOL'
            symbols = [k.strip() for k in list(codes.keys()) if k and k != "SYMBOL"]
            print(f"[{datetime.utcnow().isoformat()}] Fetched {len(symbols)} symbols from NSE.")
            return symbols
        except Exception as e:
            retries += 1
            print(f"[WARN] Failed to fetch NSE symbols: {e}. Retrying in 5s...")
            time.sleep(5)
    raise RuntimeError("Unable to fetch NSE symbols after retries.")

# ---------- Batch fetch OHLCV via yfinance ----------
def fetch_batch(ticker_batch, period=PERIOD, interval=INTERVAL):
    """
    ticker_batch: list of tickers with exchange suffix e.g. ["RELIANCE.NS", ...]
    Returns dict ticker -> dataframe (most recent index is last row)
    """
    if not ticker_batch:
        return {}
    try:
        # yfinance.download supports multiple tickers via list
        raw = yf.download(tickers=ticker_batch, period=period, interval=interval, group_by='ticker', threads=True, progress=False, prepost=False)
    except Exception as e:
        print(f"[ERROR] yfinance.download failed for batch: {e}")
        return {}
    out = {}
    # Single ticker returns DataFrame directly without ticker wrapper
    if len(ticker_batch) == 1:
        t = ticker_batch[0]
        if raw is None or raw.empty:
            return {}
        out[t] = raw
        return out
    # Multi-ticker: top-level columns per ticker
    for t in ticker_batch:
        try:
            df = raw[t].dropna(how='all')
            if df is not None and not df.empty:
                out[t] = df
        except Exception:
            # sometimes data shape differs, skip
            continue
    return out

# ---------- Bought list helpers ----------
def load_buys():
    buys = []
    if not os.path.exists(BUYS_FILE):
        return buys
    with open(BUYS_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            buys.append(r)
    return buys

def add_buy(symbol, price, rsi, note="manual"):
    now = datetime.utcnow().isoformat()
    with open(BUYS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([now, symbol, f"{price:.4f}", f"{rsi:.2f}", note])
    log_event(symbol, "BUY_ADDED", price, rsi, 0, note)
    send_telegram(f"🟢 BUY added: {symbol} @ {price:.2f} (RSI={rsi:.1f})")

def log_event(symbol, action, price, rsi, macd_hist, note=""):
    now = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([now, symbol, action, price, rsi, macd_hist, note])

# ---------- Telegram ----------
def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        resp = requests.post(url, json=payload, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False

# ---------- Market hours check (NSE) ----------
def is_market_open_nse():
    # NSE regular hours: 09:15 - 15:30 IST (Mon-Fri).
    now_utc = datetime.utcnow()
    now_ist = now_utc + timedelta(hours=5, minutes=30)
    if now_ist.weekday() >= 5:
        return False
    start = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    end = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= now_ist <= end

# ---------- Core scanning ----------
def analyze_universe(symbols):
    """Main analyzer: returns top_buys, top_sells, latest_lookup"""
    yf_tickers = [s + EXCHANGE_SUFFIX for s in symbols]
    results = []
    sell_candidates = []
    latest_lookup = {}

    # process in batches to avoid big downloads
    for i in range(0, len(yf_tickers), BATCH_SIZE):
        batch = yf_tickers[i:i+BATCH_SIZE]
        fetched = fetch_batch(batch)
        time.sleep(RATE_LIMIT_SLEEP)
        for t in batch:
            df_raw = fetched.get(t)
            if df_raw is None or df_raw.empty:
                continue
            try:
                df = compute_indicators(df_raw)
            except Exception:
                continue
            # require at least a few rows
            if len(df) < max(RSI_PERIOD, BB_WINDOW, 3):
                continue
            last = df.iloc[-1]
            # extract meta
            symbol_plain = t.replace(EXCHANGE_SUFFIX, "")
            price = float(last['close'])
            rsi = float(last['rsi']) if not np.isnan(last['rsi']) else None
            macd_hist = float(last.get('macd_hist', 0.0))
            ema20 = float(last.get('ema20', price))
            vol = float(last.get('volume', 0.0))
            vol_ma = float(last.get('vol_ma', vol if vol>0 else 1.0))
            price_vs_ema20 = (price - ema20) / (ema20 + 1e-9) * 100.0
            vol_ratio = (vol / (vol_ma + 1e-9)) if vol_ma > 0 else 1.0

            meta = {
                "symbol": symbol_plain,
                "price": price,
                "rsi": rsi,
                "macd_hist": macd_hist,
                "ema20": ema20,
                "price_vs_ema20": price_vs_ema20,
                "vol": vol,
                "vol_ma": vol_ma,
                "vol_ratio": vol_ratio
            }
            meta['score'] = score_for_buy(meta)
            results.append(meta)
            latest_lookup[symbol_plain] = meta

            # simple sell candidate heuristics
            if (rsi is not None and rsi > 70) or (macd_hist < 0 and price < ema20):
                sell_candidates.append(meta)

    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    top_buys = results_sorted[:TOP_N]
    sell_sorted = sorted(sell_candidates, key=lambda x: (- (x['rsi'] or 0), -x['score']))
    top_sells = sell_sorted[:TOP_N]
    return top_buys, top_sells, latest_lookup

def evaluate_bought_list(latest_lookup):
    buys = load_buys()
    decisions = []
    for b in buys:
        sym = b.get('symbol')
        if not sym:
            continue
        entry_price = float(b.get('entry_price') or 0.0)
        meta = latest_lookup.get(sym)
        if not meta:
            decisions.append({"symbol": sym, "action":"UNKNOWN", "reason":"no data", "entry":entry_price, "price":None})
            continue
        price = meta['price']
        rsi = meta['rsi']
        macd = meta['macd_hist']
        ema20 = meta['ema20']
        action = "HOLD"
        reason = "OK"
        # sell rules for holdings
        if rsi is not None and rsi > 70:
            action = "SELL"; reason = f"RSI high ({rsi:.1f})"
        elif price < entry_price * 0.95:
            action = "SELL"; reason = f"Stop-loss (price {price:.2f} < entry {entry_price:.2f})"
        elif macd < 0 and price < ema20:
            action = "SELL"; reason = "MACD negative & below EMA20"
        decisions.append({"symbol": sym, "entry": entry_price, "price": price, "action": action, "reason": reason, "rsi": rsi})
    return decisions

# ---------- Main loop ----------
def main():
    print("NSE Live Scanner starting...")
    symbols = []
    last_symbol_fetch = None

    while True:
        try:
            now_utc = datetime.utcnow()
            # refresh symbol list periodically or if empty
            if not symbols or (last_symbol_fetch and (now_utc - last_symbol_fetch) > timedelta(hours=SYMBOL_REFRESH_HOURS)):
                try:
                    symbols = fetch_nse_symbols()
                    last_symbol_fetch = datetime.utcnow()
                except Exception as e:
                    print(f"[ERROR] Unable to fetch NSE symbols: {e}")
                    # If we have no symbols, wait and retry
                    if not symbols:
                        time.sleep(10)
                        continue

            market_open = is_market_open_nse()
            print(f"[{datetime.utcnow().isoformat()}] Market open: {market_open} — scanning {len(symbols)} symbols in batches of {BATCH_SIZE} ...")

            top_buys, top_sells, latest_lookup = analyze_universe(symbols)

            # show top buys
            print("\n=== TOP BUY CANDIDATES ===")
            for idx, b in enumerate(top_buys, start=1):
                print(f"{idx:2d}. {b['symbol']:<8} price={b['price']:.2f} rsi={b['rsi'] if b['rsi'] is not None else 'N/A':>4} score={b['score']:.3f} macd_hist={b['macd_hist']:.4f} vol_ratio={b['vol_ratio']:.2f}")
                # alert if strong score
                if b['score'] > 1.0:
                    send_telegram(f"🔔 STRONG BUY #{idx}: {b['symbol']} @ {b['price']:.2f} RSI={b['rsi']:.1f} score={b['score']:.3f}")

            # show sell candidates
            print("\n=== SELL CANDIDATES ===")
            for idx, s in enumerate(top_sells, start=1):
                print(f"{idx:2d}. {s['symbol']:<8} price={s['price']:.2f} rsi={s['rsi'] if s['rsi'] is not None else 'N/A':>4} macd_hist={s['macd_hist']:.4f} ema20={s['ema20']:.2f}")
                send_telegram(f"⚠️ SELL alert: {s['symbol']} @ {s['price']:.2f} RSI={s['rsi']:.1f}")

            # evaluate bought list
            print("\n=== YOUR BOUGHT LIST (HOLD/SELL) ===")
            bought_decisions = evaluate_bought_list(latest_lookup)
            for d in bought_decisions:
                print(f"{d['symbol']:<8} entry={d['entry']:.2f} now={d['price'] if d['price'] is not None else 'N/A':>7} action={d['action']:<4} reason={d['reason']}")
                if d['action'] == "SELL":
                    send_telegram(f"🔴 SELL recommendation for your holding {d['symbol']}: {d['reason']} now {d['price']:.2f}")

            log_event("UNIVERSE", "SCAN_COMPLETE", 0, 0, 0, f"top_buy={len(top_buys)} top_sell={len(top_sells)}")

            # sleep until next full scan
            print(f"\nSleeping {POLL_INTERVAL}s until next scan...\n")
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] Main loop exception: {e}")
            try:
                log_event("ERROR", "EXCEPTION", 0, 0, 0, str(e))
            except:
                pass
            time.sleep(10)
            continue

if __name__ == "__main__":
    main()

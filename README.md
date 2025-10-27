#!/usr/bin/env python3
"""
nse_top10_scanner.py

- Scans NSE universe (requires nsetools or a local symbol list)
- Computes RSI, MACD, EMA(20), Bollinger, volume MA
- Ranks all stocks to show Top 10 BUY candidates
- Shows SELL-now candidates (strong sell signals)
- Maintains a Bought List (buys.csv). You can add buys manually or via the app's confirm function.
- Runs forever; continues when market closed using last available candles.
- Sends Telegram alerts when buy/sell recommendations appear (optional).

Install:
    pip install yfinance pandas numpy requests nsetools

Run:
    python nse_top10_scanner.py
"""

import time
import os
import math
import csv
import json
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import yfinance as yf

# Optional - get NSE symbols
try:
    from nsetools import Nse
    NSE_AVAILABLE = True
except Exception:
    NSE_AVAILABLE = False

# --------------- CONFIG ----------------
MARKET = "NSE"
POLL_INTERVAL = 180            # seconds between full scans (tune as needed)
BATCH_SIZE = 60                # how many tickers to fetch per batch via yfinance.download
INTERVAL = "5m"                # candle interval: "1m","5m","15m","30m","60m","1d"
PERIOD = "7d"                  # how much history to fetch per ticker (5d/7d/30d)
RSI_PERIOD = 14
EMA_FAST = 12
EMA_SLOW = 26
EMA_SIGNAL = 9
BB_WINDOW = 20
VOLUME_MA = 10
VOLUME_SPIKE_FACTOR = 1.5
TOP_N = 10                     # top N buy suggestions
RATE_LIMIT_SLEEP = 2           # seconds between yfinance batch calls (avoid throttling)

# Files
BASE_DIR = os.path.dirname(__file__)
BUYS_FILE = os.path.join(BASE_DIR, "buys.csv")        # your bought list (persisted)
LOG_FILE = os.path.join(BASE_DIR, "scanner_log.csv") # scanner events
SYMBOLS_FILE = os.path.join(BASE_DIR, "nse_symbols.txt")  # fallback list (one ticker per line)

# Telegram (optional)
TELEGRAM_TOKEN = ""   # fill your bot token
TELEGRAM_CHAT_ID = "" # fill your chat id (int or string)

# Scoring weights (tweakable)
W_RSI = 0.5
W_MACD = 1.0
W_EMA = 0.6
W_VOL = 0.4

# yfinance expects NSE tickers with suffix ".NS" for many symbols
EXCHANGE_SUFFIX = ".NS"

# Ensure files exist
if not os.path.exists(BUYS_FILE):
    with open(BUYS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time","symbol","entry_price","entry_rsi","note"])

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time","symbol","action","price","rsi","macd","note"])


# --------------- UTILITIES & INDICATORS ----------------
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_indicators(df):
    """
    Accepts df with columns: Open/High/Low/Close/Volume (yfinance format)
    Returns df with added columns: rsi, macd, macd_signal, bb_upper, bb_lower, ema20, vol_ma
    """
    df = df.copy()
    # standard column names
    if 'Close' in df.columns:
        df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
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

    # EMA(20) (short term trend)
    df['ema20'] = ema(close, 20)

    # Bollinger
    ma = close.rolling(BB_WINDOW, min_periods=BB_WINDOW).mean()
    std = close.rolling(BB_WINDOW, min_periods=BB_WINDOW).std()
    df['bb_upper'] = ma + 2*std
    df['bb_lower'] = ma - 2*std

    # volume MA
    df['vol_ma'] = df['volume'].rolling(VOLUME_MA, min_periods=1).mean()

    return df

def score_for_buy(meta):
    """
    meta: dict with keys rsi, macd_hist, price_vs_ema20 (percentage), vol_ratio
    Higher score = better buy candidate
    """
    # Score components normalized roughly
    rsi_score = max(0, (50 - meta['rsi'])) / 50        # lower RSI => higher score (0..1)
    macd_score = max(0, meta['macd_hist'])             # macd hist positive boost
    ema_score = max(0, meta['price_vs_ema20']) / 100   # price above ema20 percent (if >0)
    vol_score = max(0, meta['vol_ratio'] - 1)         # extra volume above ma

    # Weighted sum
    score = W_RSI * rsi_score + W_MACD * macd_score + W_EMA * ema_score + W_VOL * vol_score
    return float(score)

# --------------- SYMBOLS LOADING ----------------
def get_nse_symbols():
    """
    Returns a list of NSE tickers (without .NS). Uses nsetools if available, else fallback to symbols file.
    """
    symbols = []
    if NSE_AVAILABLE:
        try:
            nse = Nse()
            data = nse.get_stock_codes()  # dict: symbol->name (first entry is 'SYMBOL')
            # Convert keys ignoring header
            for k in data.keys():
                if k and k != "SYMBOL":
                    symbols.append(k.strip())
            print(f"[INFO] Loaded {len(symbols)} symbols from nsetools.")
            return symbols
        except Exception as e:
            print("[WARN] nsetools failed:", e)

    # Fallback: read from nse_symbols.txt
    if os.path.exists(SYMBOLS_FILE):
        with open(SYMBOLS_FILE, "r") as f:
            for line in f:
                s = line.strip()
                if s:
                    symbols.append(s.upper())
        print(f"[INFO] Loaded {len(symbols)} symbols from {SYMBOLS_FILE}.")
        return symbols

    # If no source, raise error and instruct user
    raise RuntimeError("No NSE symbol source found. Install nsetools (pip install nsetools) or provide nse_symbols.txt file.")

# --------------- DATA FETCHING (batched) ----------------
def fetch_batch_yfinance(ticker_batch, period=PERIOD, interval=INTERVAL):
    """
    Fetches multiple tickers at once using yfinance.download
    Input tickers should include .NS suffix where appropriate.
    Returns dict: ticker -> dataframe (index ascending).
    """
    # yfinance can accept list of tickers separated by space
    tickers_joined = " ".join(ticker_batch)
    try:
        df = yf.download(tickers=tickers_joined, period=period, interval=interval, group_by='ticker', threads=True, progress=False, prepost=False)
    except Exception as e:
        print("[ERROR] yfinance.download failed:", e)
        return {}
    out = {}
    # if single ticker, structure differs
    if len(ticker_batch) == 1:
        t = ticker_batch[0]
        if df is None or df.empty:
            return {}
        out[t] = df
        return out

    # multi ticker: df has top-level columns per ticker if group_by='ticker'
    for t in ticker_batch:
        try:
            dfi = df[t].dropna(how='all')
            if not dfi.empty:
                out[t] = dfi
        except Exception:
            continue
    return out

# --------------- BUY/SELL LOGGING & Bought List ----------------
def load_buys():
    buys = []
    if os.path.exists(BUYS_FILE):
        with open(BUYS_FILE, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                buys.append(r)
    return buys

def add_buy(symbol, price, rsi, note="manual"):
    now = datetime.utcnow().isoformat()
    with open(BUYS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([now, symbol, price, rsi, note])
    log_event(symbol, "BUY_ADDED", price, rsi, 0, note)
    send_telegram(f"🟢 BUY added to bought list: {symbol} @ {price} (RSI={rsi:.1f})")

def log_event(symbol, action, price, rsi, macd, note=""):
    now = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([now, symbol, action, price, rsi, macd, note])

# --------------- TELEGRAM ----------------
import requests
def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False

# --------------- MARKET OPEN CHECK ----------------
def is_market_open_nse():
    # NSE regular hours: 09:15 - 15:30 IST (Mon-Fri). We'll use UTC conversion.
    now_utc = datetime.utcnow()
    # IST = UTC + 5:30
    now_ist = now_utc + timedelta(hours=5, minutes=30)
    if now_ist.weekday() >= 5:
        return False
    start = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    end = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= now_ist <= end

# --------------- CORE SCAN LOGIC ----------------
def analyze_universe_and_rank(symbols):
    """
    symbols: list of NSE tickers (e.g. ["RELIANCE","TCS"...])
    returns: buy_list (sorted top candidates), sell_list (strong sells)
    """
    # Prepare list with .NS suffix for yfinance
    yf_tickers = [s + EXCHANGE_SUFFIX for s in symbols]

    results = []   # list of dicts with score and meta
    sell_candidates = []

    # process in batches
    for i in range(0, len(yf_tickers), BATCH_SIZE):
        batch = yf_tickers[i:i+BATCH_SIZE]
        # fetch data
        fetched = fetch_batch_yfinance(batch)
        # small pause between batches to avoid throttling
        time.sleep(RATE_LIMIT_SLEEP)
        for t in batch:
            if t not in fetched:
                continue
            df_raw = fetched[t]
            # Standardize index ascending and columns
            if isinstance(df_raw, pd.DataFrame) and not df_raw.empty:
                df = df_raw.copy()
                # Ensure columns lower-case standardized
                df.columns = [c.lower() for c in df.columns]
                # Ensure we have close and volume
                if 'close' not in df.columns or 'volume' not in df.columns:
                    continue
                # compute indicators
                try:
                    df = compute_indicators(df)
                except Exception:
                    continue
                # take last row
                last = df.iloc[-1]
                prev = df.iloc[-2] if len(df) >= 2 else last
                # meta
                symbol_plain = t.replace(EXCHANGE_SUFFIX, "")
                rsi = float(last.get('rsi', math.nan))
                macd_hist = float(last.get('macd_hist', 0.0))
                ema20 = float(last.get('ema20', last['close']))
                price = float(last['close'])
                vol = float(last['volume'])
                vol_ma = float(last.get('vol_ma', vol))
                price_vs_ema20 = (price - ema20) / (ema20 + 1e-9) * 100.0
                vol_ratio = (vol / (vol_ma + 1e-9)) if vol_ma>0 else 1.0

                meta = {
                    "symbol": symbol_plain,
                    "price": price,
                    "rsi": rsi,
                    "macd_hist": macd_hist,
                    "price_vs_ema20": price_vs_ema20,
                    "vol_ratio": vol_ratio,
                    "ema20": ema20,
                    "vol": vol,
                    "vol_ma": vol_ma
                }

                # compute buy score
                score = score_for_buy({"rsi": rsi, "macd_hist": macd_hist, "price_vs_ema20": price_vs_ema20, "vol_ratio": vol_ratio})
                meta['score'] = score
                results.append(meta)

                # compute sell condition (strong)
                sell_cond = False
                # Example sell rules: RSI > 70 or macd_hist negative and price below ema20
                if (rsi > 70) or ((macd_hist < 0) and (price < ema20)):
                    sell_cond = True
                if sell_cond:
                    sell_candidates.append(meta)

    # now rank results by score descending
    results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)
    top_buys = results_sorted[:TOP_N]
    # top sells sorted by rsi high first or negative macd
    sell_sorted = sorted(sell_candidates, key=lambda x: ( -x['rsi'], -x['score'] ))
    top_sells = sell_sorted[:TOP_N]

    return top_buys, top_sells, results_sorted

# --------------- Evaluate Bought List ----------------
def evaluate_bought_list(latest_lookup):
    """
    latest_lookup: dict symbol->meta with current price, rsi, macd_hist, ema20
    returns: list of dicts with action 'HOLD' or 'SELL' and reason
    """
    buys = load_buys()
    decisions = []
    for b in buys:
        sym = b['symbol']
        entry_price = float(b.get('entry_price') or b.get('price') or 0)
        meta = latest_lookup.get(sym)
        if not meta:
            decisions.append({"symbol": sym, "action": "UNKNOWN", "reason": "no data"})
            continue
        # rules for holding vs sell
        rsi = meta['rsi']
        macd = meta['macd_hist']
        price = meta['price']
        ema20 = meta['ema20']
        # simple rules:
        # If rsi > 70 -> SELL
        # If price < entry_price * 0.95 -> SELL (stop loss 5%)
        # If macd_hist turns negative and price < ema20 -> SELL
        action = "HOLD"
        reason = "OK"
        if rsi is not None and rsi > 70:
            action = "SELL"
            reason = f"RSI high ({rsi:.1f})"
        elif price < entry_price * 0.95:
            action = "SELL"
            reason = f"Stop-loss triggered (price {price:.2f} < entry {entry_price:.2f})"
        elif (macd < 0) and (price < ema20):
            action = "SELL"
            reason = f"MACD neg & below EMA20"
        decisions.append({"symbol": sym, "entry": entry_price, "price": price, "action": action, "reason": reason, "rsi": rsi})
    return decisions

# --------------- MAIN LOOP ----------------
def main_loop():
    print("Starting NSE Top-10 Scanner...")
    symbols = get_nse_symbols()
    print(f"Universe symbols: {len(symbols)}")

    # Optionally reduce universe for faster dev testing
    # symbols = symbols[:500]

    while True:
        start_time = datetime.utcnow()
        try:
            market_open = is_market_open_nse()
            print(f"[{start_time.isoformat()}] Market open: {market_open}. Starting fetch and analysis...")

            # analyze universe
            top_buys, top_sells, full_sorted = analyze_universe_and_rank(symbols)

            # prepare latest_lookup for bought list evaluation
            latest_lookup = {}
            # include top results to lookup quickly
            for meta in full_sorted[:500]:  # top 500 cached (tunable)
                latest_lookup[meta['symbol']] = meta

            # Evaluate bought list
            bought_decisions = evaluate_bought_list(latest_lookup)

            # Print top buys
            print("\n=== TOP BUY CANDIDATES ===")
            for idx, b in enumerate(top_buys, start=1):
                print(f"{idx:2d}. {b['symbol']:<8} price={b['price']:.2f} rsi={b['rsi']:.1f} score={b['score']:.3f} macd_hist={b['macd_hist']:.4f} vol_ratio={b['vol_ratio']:.2f} ema20={b['ema20']:.2f}")
                # alert if very strong buy
                if b['score'] > 1.0:
                    send_telegram(f"🔔 STRONG BUY #{idx}: {b['symbol']} @ {b['price']:.2f} RSI={b['rsi']:.1f} score={b['score']:.3f}")

            # Print sell-now
            print("\n=== SELL NOW CANDIDATES ===")
            for idx, s in enumerate(top_sells[:TOP_N], start=1):
                print(f"{idx:2d}. {s['symbol']:<8} price={s['price']:.2f} rsi={s['rsi']:.1f} macd_hist={s['macd_hist']:.4f} ema20={s['ema20']:.2f}")
                send_telegram(f"⚠️ SELL alert: {s['symbol']} @ {s['price']:.2f} RSI={s['rsi']:.1f}")

            # Print bought list evaluation
            print("\n=== YOUR BOUGHT LIST (HOLD/SELL) ===")
            for d in bought_decisions:
                print(f"{d['symbol']:<8} entry={d['entry']:.2f} now={d['price']:.2f} action={d['action']} reason={d['reason']} rsi={d['rsi']:.1f if d['rsi'] else 'N/A'}")
                if d['action'] == "SELL":
                    send_telegram(f"🔴 SELL recommendation for your holding {d['symbol']}: {d['reason']} now {d['price']:.2f}")

            # log scan summary
            log_event("UNIVERSE", "SCAN_COMPLETE", 0, 0, 0, f"top_buy={len(top_buys)} top_sell={len(top_sells)}")

            # Sleep until next poll
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            sleep_for = max(10, POLL_INTERVAL - elapsed)
            print(f"\nSleeping {sleep_for:.0f}s until next scan...\n")
            time.sleep(sleep_for)

        except KeyboardInterrupt:
            print("Interrupted by user. Exiting.")
            break
        except Exception as e:
            print("Error in main loop:", e)
            try:
                log_event("ERROR", "EXCEPTION", 0, 0, 0, str(e))
            except:
                pass
            time.sleep(10)
            continue

if __name__ == "__main__":
    main_loop()

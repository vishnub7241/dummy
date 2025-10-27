import time
import json
import datetime
import pandas as pd
import yfinance as yf
from nsetools import Nse

# Initialize NSE API
nse = Nse()

def fetch_all_symbols():
    try:
        all_stocks = nse.get_stock_codes()
        tickers = [sym for sym in all_stocks.keys() if sym != 'SYMBOL']
        return tickers
    except Exception as e:
        print("⚠️ Error fetching NSE symbols:", e)
        return []

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def analyze_stock(symbol):
    try:
        data = yf.download(f"{symbol}.NS", period="3mo", interval="1h", progress=False)
        if data.empty:
            return None

        data['RSI'] = calculate_rsi(data['Close'])
        data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
        data['MACD'] = data['EMA20'] - data['EMA50']

        last = data.iloc[-1]
        decision = "HOLD"
        reason = ""

        if last['RSI'] < 30 and last['EMA20'] > last['EMA50']:
            decision = "BUY"
            reason = f"RSI={last['RSI']:.1f} (<30), Uptrend"
        elif last['RSI'] > 70 and last['EMA20'] < last['EMA50']:
            decision = "SELL"
            reason = f"RSI={last['RSI']:.1f} (>70), Downtrend"
        else:
            reason = f"RSI={last['RSI']:.1f}"

        return {
            "symbol": symbol,
            "price": last['Close'],
            "RSI": round(last['RSI'], 2),
            "decision": decision,
            "reason": reason
        }
    except Exception as e:
        print(f"❌ {symbol}: {e}")
        return None

def save_bought(stock_info):
    try:
        with open("bought_stocks.json", "r") as f:
            bought = json.load(f)
    except FileNotFoundError:
        bought = {}

    bought[stock_info['symbol']] = {
        "price": stock_info['price'],
        "time": str(datetime.datetime.now())
    }

    with open("bought_stocks.json", "w") as f:
        json.dump(bought, f, indent=4)

def check_bought_for_sell():
    try:
        with open("bought_stocks.json", "r") as f:
            bought = json.load(f)
    except FileNotFoundError:
        return

    print("\n📊 Checking Bought Stocks for Sell Signal:")
    for sym in list(bought.keys()):
        info = analyze_stock(sym)
        if info and info["decision"] == "SELL":
            print(f"💰 SELL {sym} - Profit Target Hit ({info['reason']})")
            del bought[sym]

    with open("bought_stocks.json", "w") as f:
        json.dump(bought, f, indent=4)

def main():
    print("🚀 Starting Live Trader...")
    symbols = fetch_all_symbols()
    if not symbols:
        print("⚠️ No symbols fetched from NSE.")
        return

    print(f"✅ Found {len(symbols)} stocks to monitor")

    while True:
        print(f"\n🕒 Checking market conditions @ {datetime.datetime.now()}")
        results = []

        for sym in symbols[:100]:  # limit to top 100 for performance; can increase later
            info = analyze_stock(sym)
            if info:
                results.append(info)
            time.sleep(0.2)

        df = pd.DataFrame(results)
        buy_list = df[df["decision"] == "BUY"].sort_values("RSI").head(10)
        sell_list = df[df["decision"] == "SELL"].sort_values("RSI", ascending=False).head(10)

        print("\n📈 Top 10 BUY Opportunities:")
        print(buy_list[["symbol", "price", "RSI", "reason"]])

        print("\n📉 Top 10 SELL Alerts:")
        print(sell_list[["symbol", "price", "RSI", "reason"]])

        for _, stock in buy_list.iterrows():
            save_bought(stock)

        check_bought_for_sell()

        print("\n⏳ Sleeping 15 minutes before next check...")
        time.sleep(900)

if __name__ == "__main__":
    main()

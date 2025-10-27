import time

def get_nse_symbols():
    """Fetch all NSE symbols directly from the internet using nsetools."""
    from nsetools import Nse
    nse = Nse()

    retry_count = 0
    while retry_count < 5:
        try:
            print("🔄 Fetching latest NSE stock symbols from NSE India...")
            codes = nse.get_stock_codes()  # returns dict { 'SYMBOL': 'Company Name' }
            symbols = list(codes.keys())[1:]  # skip 'SYMBOL' header
            print(f"✅ Loaded {len(symbols)} NSE symbols.")
            return symbols
        except Exception as e:
            retry_count += 1
            print(f"⚠️ Failed to fetch NSE symbols (Attempt {retry_count}/5): {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

    # If all retries fail
    raise RuntimeError("❌ Unable to fetch NSE symbols from the internet after 5 attempts.")

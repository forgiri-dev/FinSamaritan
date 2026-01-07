"""
Stock Data Generator for FinSamaritan
Generates backup stock data CSV file
"""
import yfinance as yf
import pandas as pd
import time
from datetime import datetime

# Top 50 Nifty stocks
NIFTY_50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "BHARTIARTL.NS", "SBIN.NS", "BAJFINANCE.NS", "LICI.NS",
    "ITC.NS", "HCLTECH.NS", "AXISBANK.NS", "KOTAKBANK.NS", "LT.NS",
    "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS",
    "NTPC.NS", "NESTLEIND.NS", "ONGC.NS", "POWERGRID.NS", "WIPRO.NS",
    "ADANIENT.NS", "JSWSTEEL.NS", "BAJAJFINSV.NS", "TATAMOTORS.NS", "ADANIPORTS.NS",
    "TATASTEEL.NS", "HDFCLIFE.NS", "COALINDIA.NS", "DIVISLAB.NS", "SBILIFE.NS",
    "HINDALCO.NS", "GRASIM.NS", "CIPLA.NS", "TECHM.NS", "APOLLOHOSP.NS",
    "BRITANNIA.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "DABUR.NS", "DRREDDY.NS",
    "BPCL.NS", "MARICO.NS", "INDUSINDBK.NS", "VEDL.NS", "PIDILITIND.NS"
]

def generate_stock_data():
    """Fetch and save stock data to CSV"""
    print("🔄 Generating stock data backup...")
    
    stock_data = []
    failed = []
    consecutive_failures = 0
    
    for symbol in NIFTY_50_SYMBOLS:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get current price
            hist = ticker.history(period="1d", interval="1m")
            current_price = hist['Close'].iloc[-1] if not hist.empty else info.get('currentPrice', 0)

            stock_data.append({
                "symbol": symbol,
                "name": info.get("longName", symbol),
                "current_price": round(current_price, 2),
                "previous_close": round(info.get("previousClose", 0), 2),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": round(info.get("trailingPE", 0), 2) if info.get("trailingPE") else None,
                "dividend_yield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else None,
                "52_week_high": round(info.get("fiftyTwoWeekHigh", 0), 2),
                "52_week_low": round(info.get("fiftyTwoWeekLow", 0), 2),
                "volume": info.get("volume", 0),
                "avg_volume": info.get("averageVolume", 0),
                "beta": round(info.get("beta", 0), 2) if info.get("beta") else None,
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "currency": info.get("currency", "INR"),
                "exchange": info.get("exchange", "NSE"),
                "fetched_at": datetime.now().isoformat()
            })

            print(f"✅ Fetched {symbol}")
            consecutive_failures = 0
            time.sleep(0.2)  # Rate limiting

        except Exception as e:
            print(f"❌ Failed {symbol}: {e}")
            failed.append(symbol)
            consecutive_failures += 1

            # If we keep hitting 429 / network errors, stop hammering the API
            if consecutive_failures >= 5:
                print("\n⚠️ Multiple consecutive fetch failures detected (likely rate limiting).")
                print("   Stopping live fetch attempts and falling back to synthetic backup data.\n")
                break
    
    # If we couldn't fetch anything, fall back to a synthetic/static dataset
    if not stock_data:
        print("⚠️ No live data fetched. Creating synthetic backup rows instead.")
        for symbol in NIFTY_50_SYMBOLS:
            stock_data.append({
                "symbol": symbol,
                "name": symbol,
                "current_price": 0.0,
                "previous_close": 0.0,
                "market_cap": 0,
                "pe_ratio": None,
                "dividend_yield": None,
                "52_week_high": 0.0,
                "52_week_low": 0.0,
                "volume": 0,
                "avg_volume": 0,
                "beta": None,
                "sector": "Unknown",
                "industry": "Unknown",
                "currency": "INR",
                "exchange": "NSE",
                "fetched_at": datetime.now().isoformat()
            })

    # Save to CSV
    df = pd.DataFrame(stock_data)
    df.to_csv("stock_data.csv", index=False)
    
    print(f"\n✅ Generated stock_data.csv with {len(stock_data)} stocks")
    if failed:
        print(f"⚠️ Failed to fetch: {', '.join(failed)}")
    
    return df

if __name__ == "__main__":
    generate_stock_data()



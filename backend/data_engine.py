"""
Data Engine module for FinSamaritan
Hybrid cache system for efficient stock data retrieval using yfinance
"""
import yfinance as yf
import pandas as pd
from typing import Dict, Optional, List
import time

# Top 50 Nifty stocks (Indian market)
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

class DataEngine:
    """Hybrid cache system for stock data"""
    
    def __init__(self):
        self.cache: Dict[str, Dict] = {}
        self.last_update = 0
        self.cache_ttl = 300  # 5 minutes cache TTL
        
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid"""
        return time.time() - self.last_update < self.cache_ttl
    
    def _fetch_stock_data(self, symbol: str) -> Optional[Dict]:
        """Fetch stock data from yfinance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price
            hist = ticker.history(period="1d", interval="1m")
            current_price = hist['Close'].iloc[-1] if not hist.empty else info.get('currentPrice', 0)
            
            # Get fundamental data
            data = {
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
            }
            
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def initialize_cache(self):
        """Pre-fetch Top 50 Nifty stocks on startup"""
        print("🔄 Initializing data cache with Top 50 Nifty stocks...")
        failed = []
        
        for symbol in NIFTY_50_SYMBOLS:
            data = self._fetch_stock_data(symbol)
            if data:
                self.cache[symbol] = data
            else:
                failed.append(symbol)
            time.sleep(0.1)  # Rate limiting
        
        self.last_update = time.time()
        print(f"✅ Cache initialized: {len(self.cache)} stocks loaded")
        if failed:
            print(f"⚠️ Failed to load: {', '.join(failed[:5])}")
    
    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get stock data from cache or fetch live"""
        # Normalize symbol (ensure .NS suffix for Indian stocks)
        if not symbol.endswith('.NS') and not '.' in symbol:
            symbol = f"{symbol}.NS"
        
        # Check cache first
        if symbol in self.cache and self._is_cache_valid():
            return self.cache[symbol]
        
        # Fetch live if not in cache or cache expired
        data = self._fetch_stock_data(symbol)
        if data:
            self.cache[symbol] = data
            return data
        
        return None
    
    def get_bulk_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """Get data for multiple symbols"""
        result = {}
        for symbol in symbols:
            data = self.get_stock_data(symbol)
            if data:
                result[symbol] = data
        return result
    
    def get_all_cached(self) -> pd.DataFrame:
        """Get all cached stocks as DataFrame for screening"""
        if not self.cache:
            return pd.DataFrame()
        
        df = pd.DataFrame(list(self.cache.values()))
        return df
    
    def refresh_cache(self):
        """Manually refresh the cache"""
        self.initialize_cache()

# Global instance
data_engine = DataEngine()

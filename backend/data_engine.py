"""
Data Engine module for FinSamaritan
Fetches and caches stock data
"""
import yfinance as yf
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import time

class DataEngine:
    def __init__(self):
        self.cache = {}
        self.cache_timestamp = {}
        self.cache_duration = 300  # 5 minutes cache
        
    def get_stock_data(self, symbol: str) -> Optional[Dict]:
        """Get current stock data for a symbol"""
        try:
            # Check cache
            if symbol in self.cache:
                if time.time() - self.cache_timestamp.get(symbol, 0) < self.cache_duration:
                    return self.cache[symbol]
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price
            hist = ticker.history(period="1d", interval="1m")
            if not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
                previous_close = float(info.get('previousClose', current_price))
            else:
                current_price = float(info.get('currentPrice', info.get('regularMarketPrice', 0)))
                previous_close = float(info.get('previousClose', current_price))
            
            data = {
                "symbol": symbol.upper(),
                "name": info.get('longName', info.get('shortName', symbol)),
                "current_price": round(current_price, 2),
                "previous_close": round(previous_close, 2),
                "change": round(current_price - previous_close, 2),
                "change_percent": round(((current_price - previous_close) / previous_close * 100) if previous_close > 0 else 0, 2),
                "pe_ratio": round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else None,
                "market_cap": info.get('marketCap', 0),
                "beta": round(info.get('beta', 0), 2) if info.get('beta') else None,
                "sector": info.get('sector', 'Unknown'),
                "volume": info.get('volume', 0),
                "high_52w": round(info.get('fiftyTwoWeekHigh', 0), 2) if info.get('fiftyTwoWeekHigh') else None,
                "low_52w": round(info.get('fiftyTwoWeekLow', 0), 2) if info.get('fiftyTwoWeekLow') else None,
            }
            
            # Cache it
            self.cache[symbol] = data
            self.cache_timestamp[symbol] = time.time()
            
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def get_all_cached(self) -> pd.DataFrame:
        """Get all cached stock data as DataFrame"""
        if not self.cache:
            return pd.DataFrame()
        
        # Convert cache to DataFrame
        data_list = list(self.cache.values())
        df = pd.DataFrame(data_list)
        return df
    
    def cache_multiple(self, symbols: List[str]):
        """Pre-cache multiple symbols"""
        for symbol in symbols:
            self.get_stock_data(symbol)

# Global instance
data_engine = DataEngine()


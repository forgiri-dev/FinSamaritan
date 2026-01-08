"""
Data Engine module for FinSamaritan
Fetches and caches stock data with rate limiting
"""
import yfinance as yf
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import time
import random

class DataEngine:
    def __init__(self):
        self.cache = {}
        self.cache_timestamp = {}
        self.cache_duration = 300  # 5 minutes cache
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Minimum 0.5 seconds between requests
        self.rate_limit_delay = 2.0  # Delay when rate limited
        
    def _rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def get_stock_data(self, symbol: str, retry_count: int = 0) -> Optional[Dict]:
        """Get current stock data for a symbol with rate limiting and retry logic"""
        try:
            # Check cache first
            if symbol in self.cache:
                cache_age = time.time() - self.cache_timestamp.get(symbol, 0)
                if cache_age < self.cache_duration:
                    return self.cache[symbol]
            
            # Rate limiting
            self._rate_limit()
            
            # Try to fetch data
            try:
                ticker = yf.Ticker(symbol)
                
                # Use a timeout and simpler data fetching
                try:
                    info = ticker.info
                except Exception as info_error:
                    # If info fails, try to get basic price data
                    print(f"Warning: Could not fetch full info for {symbol}: {info_error}")
                    info = {}
                
                # Get current price - try multiple methods
                current_price = None
                previous_close = None
                
                try:
                    hist = ticker.history(period="5d", interval="1d")
                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                        if len(hist) > 1:
                            previous_close = float(hist['Close'].iloc[-2])
                except Exception:
                    pass
                
                # Fallback to info if history fails
                if current_price is None:
                    current_price = float(info.get('currentPrice', info.get('regularMarketPrice', info.get('previousClose', 0))))
                
                if previous_close is None:
                    previous_close = float(info.get('previousClose', current_price))
                
                if current_price == 0:
                    print(f"Warning: Could not get price for {symbol}")
                    return None
                
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
                
            except Exception as fetch_error:
                error_str = str(fetch_error)
                
                # Handle rate limiting
                if '429' in error_str or 'Too Many Requests' in error_str:
                    if retry_count < 2:
                        # Exponential backoff
                        wait_time = self.rate_limit_delay * (2 ** retry_count) + random.uniform(0, 1)
                        print(f"Rate limited for {symbol}, waiting {wait_time:.1f}s before retry {retry_count + 1}")
                        time.sleep(wait_time)
                        return self.get_stock_data(symbol, retry_count + 1)
                    else:
                        print(f"Rate limited for {symbol} after {retry_count + 1} retries. Using cached data if available.")
                        # Return cached data even if expired
                        if symbol in self.cache:
                            return self.cache[symbol]
                        return None
                else:
                    # Other errors
                    print(f"Error fetching data for {symbol}: {fetch_error}")
                    # Return cached data if available
                    if symbol in self.cache:
                        print(f"Returning cached data for {symbol}")
                        return self.cache[symbol]
                    return None
                    
        except Exception as e:
            print(f"Unexpected error fetching data for {symbol}: {e}")
            # Return cached data if available
            if symbol in self.cache:
                return self.cache[symbol]
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
        """Pre-cache multiple symbols with delays"""
        for i, symbol in enumerate(symbols):
            self.get_stock_data(symbol)
            # Add extra delay between multiple requests
            if i < len(symbols) - 1:
                time.sleep(0.3)

# Global instance
data_engine = DataEngine()


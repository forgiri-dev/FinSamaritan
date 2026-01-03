"""
Agent Tools - Custom functions for the AI agent to interact with stock data
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Any
import os

# Load stock data into memory
STOCK_DATA_PATH = os.path.join(os.path.dirname(__file__), 'stock_data.csv')
stock_df = None

def load_stock_data():
    """Load stock data from CSV into memory"""
    global stock_df
    if stock_df is None:
        if not os.path.exists(STOCK_DATA_PATH):
            raise FileNotFoundError(
                f"stock_data.csv not found at {STOCK_DATA_PATH}. "
                "Please run stock_data_generator.py first."
            )
        stock_df = pd.read_csv(STOCK_DATA_PATH)
        # Ensure all numeric columns are properly typed
        numeric_columns = ['PE_Ratio', 'PB_Ratio', 'Sales_Growth', 'Profit_Margin', 
                          'Market_Cap', 'Current_Price', '52W_High', '52W_Low']
        for col in numeric_columns:
            if col in stock_df.columns:
                stock_df[col] = pd.to_numeric(stock_df[col], errors='coerce')
    return stock_df

def search_stocks(query: str) -> Dict[str, Any]:
    """
    Search stocks based on a pandas query string.
    
    Args:
        query: A pandas query string (e.g., "PE_Ratio < 15 and Sector == 'IT'")
    
    Returns:
        Dictionary with search results and metadata
    """
    try:
        df = load_stock_data()
        
        # Execute the query
        result_df = df.query(query)
        
        # Replace NaN values with None for JSON serialization
        result_df = result_df.replace({np.nan: None})
        
        # Convert to list of dictionaries for JSON serialization
        results = result_df.to_dict('records')
        
        return {
            "success": True,
            "count": len(results),
            "results": results,
            "query": query
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "count": 0,
            "results": []
        }

def get_live_price(symbol: str) -> Dict[str, Any]:
    """
    Fetch the current live price for a stock symbol.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE.NS", "HDFCBANK.NS")
    
    Returns:
        Dictionary with current price and metadata
    """
    try:
        # Handle Indian stocks - add .NS suffix if not present
        if not symbol.endswith('.NS') and not '.' in symbol:
            symbol_with_suffix = f"{symbol}.NS"
        else:
            symbol_with_suffix = symbol
        
        ticker = yf.Ticker(symbol_with_suffix)
        info = ticker.info
        
        # Get current price
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        # Get additional info
        data = {
            "symbol": symbol,
            "current_price": current_price,
            "currency": info.get('currency', 'INR'),
            "market_cap": info.get('marketCap'),
            "pe_ratio": info.get('trailingPE'),
            "pb_ratio": info.get('priceToBook'),
            "volume": info.get('volume'),
            "day_high": info.get('dayHigh'),
            "day_low": info.get('dayLow'),
            "previous_close": info.get('previousClose'),
            "52_week_high": info.get('fiftyTwoWeekHigh'),
            "52_week_low": info.get('fiftyTwoWeekLow'),
        }
        
        return {
            "success": True,
            "data": data
        }
    except Exception as e:
        # Fallback: try to get from our CSV if available
        try:
            df = load_stock_data()
            stock_info = df[df['Symbol'] == symbol].iloc[0].to_dict()
            return {
                "success": True,
                "data": {
                    "symbol": symbol,
                    "current_price": stock_info.get('Current_Price'),
                    "note": "Data from local database (live price unavailable)"
                }
            }
        except:
            return {
                "success": False,
                "error": str(e),
                "data": None
            }

def get_stock_info(symbol: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a stock from the database.
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Dictionary with stock information
    """
    try:
        df = load_stock_data()
        stock_info = df[df['Symbol'] == symbol]
        
        if stock_info.empty:
            return {
                "success": False,
                "error": f"Stock {symbol} not found in database",
                "data": None
            }
        
        stock_dict = stock_info.iloc[0].to_dict()
        # Replace NaN values with None for JSON serialization
        stock_dict = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in stock_dict.items()}
        
        return {
            "success": True,
            "data": stock_dict
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data": None
        }


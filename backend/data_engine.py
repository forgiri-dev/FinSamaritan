"""
Data Engine - Handles stock data retrieval from CSV database
"""
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional, List

STOCK_DATA_PATH = os.path.join(os.path.dirname(__file__), 'stock_data.csv')
_cached_df = None

def _load_dataframe() -> pd.DataFrame:
    """Load and cache the stock data CSV"""
    global _cached_df
    if _cached_df is None:
        if not os.path.exists(STOCK_DATA_PATH):
            raise FileNotFoundError(
                f"stock_data.csv not found at {STOCK_DATA_PATH}. "
                "Please run stock_data_generator.py first."
            )
        _cached_df = pd.read_csv(STOCK_DATA_PATH)
        # Ensure all numeric columns are properly typed
        numeric_columns = ['PE_Ratio', 'PB_Ratio', 'Sales_Growth', 'Profit_Margin', 
                          'Market_Cap', 'Current_Price', '52W_High', '52W_Low']
        for col in numeric_columns:
            if col in _cached_df.columns:
                _cached_df[col] = pd.to_numeric(_cached_df[col], errors='coerce')
    return _cached_df

def get_stock_data(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get stock data for a specific symbol from CSV
    
    Returns:
        Dictionary with stock data or None if not found
    """
    try:
        df = _load_dataframe()
        stock_row = df[df['Symbol'] == symbol.upper()]
        
        if stock_row.empty:
            return None
        
        stock_dict = stock_row.iloc[0].to_dict()
        
        # Normalize column names to lowercase with underscores
        normalized = {
            "symbol": stock_dict.get("Symbol", ""),
            "name": stock_dict.get("Name", ""),
            "sector": stock_dict.get("Sector", ""),
            "pe_ratio": stock_dict.get("PE_Ratio"),
            "pb_ratio": stock_dict.get("PB_Ratio"),
            "sales_growth": stock_dict.get("Sales_Growth"),
            "profit_margin": stock_dict.get("Profit_Margin"),
            "market_cap": stock_dict.get("Market_Cap"),
            "current_price": stock_dict.get("Current_Price"),
            "52w_high": stock_dict.get("52W_High"),
            "52w_low": stock_dict.get("52W_Low"),
            "previous_close": stock_dict.get("Current_Price"),  # Use current as fallback
            "beta": None  # Not in CSV, but expected by some functions
        }
        
        # Replace NaN with None for JSON serialization
        for key, value in normalized.items():
            if isinstance(value, float) and np.isnan(value):
                normalized[key] = None
        
        return normalized
    except Exception as e:
        print(f"Error getting stock data for {symbol}: {e}")
        return None

def get_all_cached() -> pd.DataFrame:
    """
    Get all cached stock data as DataFrame
    
    Returns:
        DataFrame with all stock data
    """
    return _load_dataframe().copy()

def search_stocks_by_name(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Search stocks by name or symbol (case-insensitive partial match)
    
    Args:
        query: Search query string
        limit: Maximum number of results
    
    Returns:
        List of stock dictionaries matching the query
    """
    try:
        df = _load_dataframe()
        query_upper = query.upper()
        
        # Search in both Symbol and Name columns
        mask = (
            df['Symbol'].str.upper().str.contains(query_upper, na=False) |
            df['Name'].str.upper().str.contains(query_upper, na=False)
        )
        
        results_df = df[mask].head(limit)
        
        # Convert to list of dictionaries
        results = []
        for _, row in results_df.iterrows():
            stock_dict = row.to_dict()
            # Replace NaN with None
            clean_dict = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) 
                         for k, v in stock_dict.items()}
            results.append(clean_dict)
        
        return results
    except Exception as e:
        print(f"Error searching stocks: {e}")
        return []

# Create a singleton instance for backward compatibility
data_engine = type('DataEngine', (), {
    'get_stock_data': staticmethod(get_stock_data),
    'get_all_cached': staticmethod(get_all_cached),
    'search_stocks_by_name': staticmethod(search_stocks_by_name),
})()


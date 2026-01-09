"""
Database module for portfolio and watchlist management
Uses JSON file for persistence
"""
import json
import os
from typing import List, Dict, Any

PORTFOLIO_FILE = os.path.join(os.path.dirname(__file__), 'portfolio.json')

def _load_data() -> Dict[str, Any]:
    """Load portfolio data from JSON file"""
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"portfolio": [], "watchlist": []}

def _save_data(data: Dict[str, Any]) -> bool:
    """Save portfolio data to JSON file"""
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False

def get_portfolio() -> List[Dict[str, Any]]:
    """Get all portfolio holdings"""
    data = _load_data()
    return data.get("portfolio", [])

def add_to_portfolio(symbol: str, shares: int, buy_price: float) -> bool:
    """Add or update a stock in portfolio"""
    data = _load_data()
    portfolio = data.get("portfolio", [])
    
    # Check if stock already exists
    for holding in portfolio:
        if holding.get("symbol") == symbol:
            # Update existing holding
            holding["shares"] = shares
            holding["buy_price"] = buy_price
            return _save_data(data)
    
    # Add new holding
    portfolio.append({
        "symbol": symbol,
        "shares": shares,
        "buy_price": buy_price
    })
    data["portfolio"] = portfolio
    return _save_data(data)

def remove_from_portfolio(symbol: str) -> bool:
    """Remove a stock from portfolio"""
    data = _load_data()
    portfolio = data.get("portfolio", [])
    
    # Remove the stock
    portfolio = [h for h in portfolio if h.get("symbol") != symbol]
    data["portfolio"] = portfolio
    return _save_data(data)

def get_watchlist() -> List[str]:
    """Get all watchlist symbols"""
    data = _load_data()
    return data.get("watchlist", [])

def add_to_watchlist(symbol: str) -> bool:
    """Add a symbol to watchlist"""
    data = _load_data()
    watchlist = data.get("watchlist", [])
    
    if symbol not in watchlist:
        watchlist.append(symbol)
        data["watchlist"] = watchlist
        return _save_data(data)
    return True

def remove_from_watchlist(symbol: str) -> bool:
    """Remove a symbol from watchlist"""
    data = _load_data()
    watchlist = data.get("watchlist", [])
    
    watchlist = [s for s in watchlist if s != symbol]
    data["watchlist"] = watchlist
    return _save_data(data)


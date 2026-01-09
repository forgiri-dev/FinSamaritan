"""
Database module for portfolio and watchlist management
Uses JSON file for persistence
"""
import json
import os
from typing import List, Dict, Any

DATABASE_FILE = os.path.join(os.path.dirname(__file__), 'portfolio.json')

def _load_database() -> Dict[str, Any]:
    """Load database from JSON file"""
    if os.path.exists(DATABASE_FILE):
        try:
            with open(DATABASE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {"portfolio": [], "watchlist": []}
    return {"portfolio": [], "watchlist": []}

def _save_database(data: Dict[str, Any]):
    """Save database to JSON file"""
    with open(DATABASE_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def add_to_portfolio(symbol: str, shares: int, buy_price: float) -> bool:
    """Add or update stock in portfolio"""
    try:
        data = _load_database()
        portfolio = data.get("portfolio", [])
        
        # Check if stock already exists
        for i, holding in enumerate(portfolio):
            if holding["symbol"] == symbol:
                # Update existing holding
                portfolio[i] = {
                    "symbol": symbol,
                    "shares": shares,
                    "buy_price": buy_price
                }
                _save_database(data)
                return True
        
        # Add new holding
        portfolio.append({
            "symbol": symbol,
            "shares": shares,
            "buy_price": buy_price
        })
        data["portfolio"] = portfolio
        _save_database(data)
        return True
    except Exception as e:
        print(f"Error adding to portfolio: {e}")
        return False

def get_portfolio() -> List[Dict[str, Any]]:
    """Get all portfolio holdings"""
    data = _load_database()
    return data.get("portfolio", [])

def remove_from_portfolio(symbol: str) -> bool:
    """Remove stock from portfolio"""
    try:
        data = _load_database()
        portfolio = data.get("portfolio", [])
        portfolio = [h for h in portfolio if h["symbol"] != symbol]
        data["portfolio"] = portfolio
        _save_database(data)
        return True
    except Exception as e:
        print(f"Error removing from portfolio: {e}")
        return False

def get_watchlist() -> List[str]:
    """Get watchlist symbols"""
    data = _load_database()
    return data.get("watchlist", [])

def add_to_watchlist(symbol: str) -> bool:
    """Add symbol to watchlist"""
    try:
        data = _load_database()
        watchlist = data.get("watchlist", [])
        if symbol not in watchlist:
            watchlist.append(symbol)
            data["watchlist"] = watchlist
            _save_database(data)
        return True
    except Exception as e:
        print(f"Error adding to watchlist: {e}")
        return False

def remove_from_watchlist(symbol: str) -> bool:
    """Remove symbol from watchlist"""
    try:
        data = _load_database()
        watchlist = data.get("watchlist", [])
        watchlist = [s for s in watchlist if s != symbol]
        data["watchlist"] = watchlist
        _save_database(data)
        return True
    except Exception as e:
        print(f"Error removing from watchlist: {e}")
        return False


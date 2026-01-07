"""
Database module for FinSamaritan
Manages SQLite database for portfolio and watchlist
"""
import sqlite3
import os
from typing import List, Dict, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), 'finsamaritan.db')

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database tables"""
    conn = get_db()
    cursor = conn.cursor()
    
    # Portfolio table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE,
            shares INTEGER NOT NULL,
            buy_price REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Watchlist table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def add_to_portfolio(symbol: str, shares: int, buy_price: float) -> bool:
    """Add or update stock in portfolio"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        
        # Check if symbol exists
        cursor.execute('SELECT shares, buy_price FROM portfolio WHERE symbol = ?', (symbol,))
        existing = cursor.fetchone()
        
        if existing:
            # Update existing holding (average price)
            old_shares = existing['shares']
            old_price = existing['buy_price']
            total_shares = old_shares + shares
            avg_price = ((old_shares * old_price) + (shares * buy_price)) / total_shares
            
            cursor.execute('''
                UPDATE portfolio 
                SET shares = ?, buy_price = ?
                WHERE symbol = ?
            ''', (total_shares, avg_price, symbol))
        else:
            # Insert new holding
            cursor.execute('''
                INSERT INTO portfolio (symbol, shares, buy_price)
                VALUES (?, ?, ?)
            ''', (symbol, shares, buy_price))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error adding to portfolio: {e}")
        return False

def remove_from_portfolio(symbol: str) -> bool:
    """Remove stock from portfolio"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM portfolio WHERE symbol = ?', (symbol,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted
    except Exception as e:
        print(f"Error removing from portfolio: {e}")
        return False

def get_portfolio() -> List[Dict]:
    """Get all portfolio holdings"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT symbol, shares, buy_price FROM portfolio')
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "symbol": row['symbol'],
                "shares": row['shares'],
                "buy_price": row['buy_price']
            }
            for row in rows
        ]
    except Exception as e:
        print(f"Error getting portfolio: {e}")
        return []

def add_to_watchlist(symbol: str) -> bool:
    """Add symbol to watchlist"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('INSERT OR IGNORE INTO watchlist (symbol) VALUES (?)', (symbol,))
        conn.commit()
        added = cursor.rowcount > 0
        conn.close()
        return added
    except Exception as e:
        print(f"Error adding to watchlist: {e}")
        return False

def remove_from_watchlist(symbol: str) -> bool:
    """Remove symbol from watchlist"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM watchlist WHERE symbol = ?', (symbol,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted
    except Exception as e:
        print(f"Error removing from watchlist: {e}")
        return False

def get_watchlist() -> List[str]:
    """Get all watchlist symbols"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('SELECT symbol FROM watchlist ORDER BY created_at DESC')
        rows = cursor.fetchall()
        conn.close()
        return [row['symbol'] for row in rows]
    except Exception as e:
        print(f"Error getting watchlist: {e}")
        return []

# Initialize database on import
init_db()


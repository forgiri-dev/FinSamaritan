"""
Database module for FinSamaritan
Manages SQLite database for portfolio and watchlist persistence
"""
import sqlite3
import os
from typing import List, Tuple, Optional

DB_PATH = "fin_samaritan.db"

def get_db_connection():
    """Get a connection to the SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn

def init_db():
    """Initialize the database with required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Portfolio table: stores user's stock holdings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            symbol TEXT PRIMARY KEY,
            shares INTEGER NOT NULL,
            buy_price REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Watchlist table: stores stocks user wants to track
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            symbol TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DB_PATH}")

def add_to_portfolio(symbol: str, shares: int, buy_price: float) -> bool:
    """Add or update a stock in the portfolio"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR REPLACE INTO portfolio (symbol, shares, buy_price, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """, (symbol, shares, buy_price))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error adding to portfolio: {e}")
        return False
    finally:
        conn.close()

def remove_from_portfolio(symbol: str) -> bool:
    """Remove a stock from the portfolio"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM portfolio WHERE symbol = ?", (symbol,))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error removing from portfolio: {e}")
        return False
    finally:
        conn.close()

def get_portfolio() -> List[dict]:
    """Get all portfolio holdings"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT symbol, shares, buy_price FROM portfolio")
    rows = cursor.fetchall()
    conn.close()
    
    return [{"symbol": row["symbol"], "shares": row["shares"], "buy_price": row["buy_price"]} 
            for row in rows]

def add_to_watchlist(symbol: str) -> bool:
    """Add a stock to the watchlist"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR IGNORE INTO watchlist (symbol)
            VALUES (?)
        """, (symbol,))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error adding to watchlist: {e}")
        return False
    finally:
        conn.close()

def remove_from_watchlist(symbol: str) -> bool:
    """Remove a stock from the watchlist"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        print(f"Error removing from watchlist: {e}")
        return False
    finally:
        conn.close()

def get_watchlist() -> List[str]:
    """Get all watchlist symbols"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT symbol FROM watchlist")
    rows = cursor.fetchall()
    conn.close()
    
    return [row["symbol"] for row in rows]

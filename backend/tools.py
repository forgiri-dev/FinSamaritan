"""
Agent Tools module for FinSamaritan
AI-compatible tools for the Manager Agent (Gemini) to use
"""
import database
from data_engine import data_engine
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
# Try to import TensorFlow and TensorFlow Lite
TENSORFLOW_AVAILABLE = False
TFLITE_AVAILABLE = False
tf = None
tflite_interpreter = None

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    # Check if tf.lite is available
    if hasattr(tf, 'lite') and hasattr(tf.lite, 'Interpreter'):
        TFLITE_AVAILABLE = True
        tflite_interpreter = tf.lite.Interpreter
        print("[INFO] Using tensorflow.lite.Interpreter")
    else:
        # Try alternative: tflite_runtime (lighter weight, for inference only)
        try:
            from tflite_runtime import interpreter
            TFLITE_AVAILABLE = True
            tflite_interpreter = interpreter.Interpreter
            print("[INFO] Using tflite_runtime.interpreter.Interpreter")
        except ImportError:
            TFLITE_AVAILABLE = False
            tflite_interpreter = None
except ImportError:
    TENSORFLOW_AVAILABLE = False
    # Try tflite_runtime as fallback
    try:
        from tflite_runtime import interpreter
        TFLITE_AVAILABLE = True
        tflite_interpreter = interpreter.Interpreter
        print("[INFO] Using tflite_runtime.interpreter.Interpreter (TensorFlow not installed)")
    except ImportError:
        TFLITE_AVAILABLE = False
        tflite_interpreter = None
from PIL import Image
import io
import base64

def manage_portfolio(action: str, symbol: str, shares: Optional[int] = None, buy_price: Optional[float] = None) -> Dict[str, Any]:
    """
    Tool: Manage Portfolio
    Actions: 'buy', 'sell', 'remove'
    """
    try:
        if action.lower() == 'buy':
            if shares is None or buy_price is None:
                return {"success": False, "error": "Shares and buy_price required for buy action"}
            
            success = database.add_to_portfolio(symbol, shares, buy_price)
            if success:
                holding = database.get_portfolio()
                total_invested = sum(h["shares"] * h["buy_price"] for h in holding)
                return {
                    "success": True,
                    "message": f"✅ Added {shares} shares of {symbol} at ₹{buy_price}",
                    "total_invested": total_invested,
                    "portfolio": holding
                }
            else:
                return {"success": False, "error": "Failed to add to portfolio"}
        
        elif action.lower() == 'sell':
            portfolio = database.get_portfolio()
            holding = next((h for h in portfolio if h["symbol"] == symbol), None)
            
            if not holding:
                return {"success": False, "error": f"{symbol} not found in portfolio"}
            
            if shares is None:
                shares = holding["shares"]  # Sell all
            
            if shares > holding["shares"]:
                return {"success": False, "error": f"Insufficient shares. You have {holding['shares']} shares"}
            
            remaining = holding["shares"] - shares
            if remaining == 0:
                database.remove_from_portfolio(symbol)
            else:
                database.add_to_portfolio(symbol, remaining, holding["buy_price"])
            
            return {
                "success": True,
                "message": f"✅ Sold {shares} shares of {symbol}",
                "remaining_shares": remaining
            }
        
        elif action.lower() == 'remove':
            success = database.remove_from_portfolio(symbol)
            if success:
                return {"success": True, "message": f"✅ Removed {symbol} from portfolio"}
            else:
                return {"success": False, "error": f"{symbol} not found in portfolio"}
        
        else:
            return {"success": False, "error": f"Unknown action: {action}. Use 'buy', 'sell', or 'remove'"}
    
    except Exception as e:
        return {"success": False, "error": str(e)}

def analyze_portfolio() -> Dict[str, Any]:
    """
    Tool: Analyze Portfolio
    Calculates P&L, exposure, risk ratios
    """
    try:
        portfolio = database.get_portfolio()
        if not portfolio:
            return {
                "success": True,
                "message": "Your portfolio is empty",
                "total_invested": 0,
                "current_value": 0,
                "total_pnl": 0,
                "total_pnl_percent": 0,
                "holdings": []
            }
        
        holdings_analysis = []
        total_invested = 0
        total_current = 0
        
        for holding in portfolio:
            symbol = holding["symbol"]
            shares = holding["shares"]
            buy_price = holding["buy_price"]
            invested = shares * buy_price
            
            # Get current price
            stock_data = data_engine.get_stock_data(symbol)
            if stock_data:
                current_price = stock_data["current_price"]
                current_value = shares * current_price
                pnl = current_value - invested
                pnl_percent = (pnl / invested) * 100 if invested > 0 else 0
                
                holdings_analysis.append({
                    "symbol": symbol,
                    "shares": shares,
                    "buy_price": buy_price,
                    "current_price": current_price,
                    "invested": round(invested, 2),
                    "current_value": round(current_value, 2),
                    "pnl": round(pnl, 2),
                    "pnl_percent": round(pnl_percent, 2),
                    "risk": "High" if abs(pnl_percent) > 10 else "Medium" if abs(pnl_percent) > 5 else "Low"
                })
                
                total_invested += invested
                total_current += current_value
        
        total_pnl = total_current - total_invested
        total_pnl_percent = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        return {
            "success": True,
            "total_invested": round(total_invested, 2),
            "current_value": round(total_current, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_percent": round(total_pnl_percent, 2),
            "holdings": holdings_analysis,
            "count": len(holdings_analysis)
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}

def search_stocks(query: str) -> Dict[str, Any]:
    """
    Tool: Search Stocks
    Searches stocks by name or symbol (case-insensitive partial match)
    Used by Portfolio screen for stock search functionality
    
    Args:
        query: Search query string (e.g., "RELIANCE", "TCS", "Bank")
    
    Returns:
        Dictionary with search results
    """
    try:
        results = data_engine.search_stocks_by_name(query, limit=20)
        
        return {
            "success": True,
            "count": len(results),
            "query": query,
            "stocks": results
        }
    except Exception as e:
        return {"success": False, "error": str(e), "count": 0, "stocks": []}

def run_screener(query: str) -> Dict[str, Any]:
    """
    Tool: Run Screener
    Executes pandas query on cached stock data from CSV
    Example: "PE_Ratio < 15", "Current_Price > 1000", "Sector == 'IT'"
    
    Note: Column names must match CSV column names (uppercase with underscores)
    """
    try:
        df = data_engine.get_all_cached()
        if df.empty:
            return {"success": False, "error": "No stock data available. Cache may be empty."}
        
        # Execute query
        try:
            filtered_df = df.query(query)
        except Exception as e:
            return {"success": False, "error": f"Invalid query: {str(e)}"}
        
        if filtered_df.empty:
            return {
                "success": True,
                "count": 0,
                "message": f"No stocks match the criteria: {query}",
                "stocks": []
            }
        
        # Format results - use CSV column names
        result_columns = ["Symbol", "Name", "Current_Price", "PE_Ratio", "Sector"]
        available_columns = [col for col in result_columns if col in filtered_df.columns]
        results_df = filtered_df[available_columns]
        
        # Convert to list of dictionaries and clean NaN values
        results = []
        for _, row in results_df.iterrows():
            stock_dict = row.to_dict()
            clean_dict = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) 
                         for k, v in stock_dict.items()}
            results.append(clean_dict)
        
        return {
            "success": True,
            "count": len(results),
            "query": query,
            "stocks": results[:20]  # Limit to top 20
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}

def simulate_strategy(symbol: str, strategy_type: str = "SMA_CROSSOVER", period: int = 252) -> Dict[str, Any]:
    """
    Tool: Simulate Strategy
    Backtests trading strategies
    Strategy types: SMA_CROSSOVER, RSI_OVERSOLD, MOMENTUM
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=f"{period}d")
        
        if hist.empty:
            return {"success": False, "error": f"No historical data for {symbol}"}
        
        hist['Close'] = pd.to_numeric(hist['Close'], errors='coerce')
        
        if strategy_type == "SMA_CROSSOVER":
            # Buy when price crosses above 50-day SMA
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['Signal'] = (hist['Close'] > hist['SMA_50']).astype(int)
            hist['Position'] = hist['Signal'].diff()
            
            # Calculate returns
            hist['Returns'] = hist['Close'].pct_change()
            hist['Strategy_Returns'] = hist['Position'].shift(1) * hist['Returns']
            
            total_return = hist['Strategy_Returns'].sum() * 100
            sharpe_ratio = (hist['Strategy_Returns'].mean() / hist['Strategy_Returns'].std()) * np.sqrt(252) if hist['Strategy_Returns'].std() > 0 else 0
            
            return {
                "success": True,
                "strategy": "50-Day SMA Crossover",
                "symbol": symbol,
                "period_days": period,
                "total_return_percent": round(total_return, 2),
                "sharpe_ratio": round(sharpe_ratio, 2),
                "trades": int(abs(hist['Position'].sum()))
            }
        
        elif strategy_type == "RSI_OVERSOLD":
            # Buy when RSI < 30, Sell when RSI > 70
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            hist['RSI'] = 100 - (100 / (1 + rs))
            
            hist['Signal'] = ((hist['RSI'] < 30) | (hist['RSI'] > 70)).astype(int)
            hist['Position'] = hist['Signal'].diff()
            
            hist['Returns'] = hist['Close'].pct_change()
            hist['Strategy_Returns'] = hist['Position'].shift(1) * hist['Returns']
            
            total_return = hist['Strategy_Returns'].sum() * 100
            sharpe_ratio = (hist['Strategy_Returns'].mean() / hist['Strategy_Returns'].std()) * np.sqrt(252) if hist['Strategy_Returns'].std() > 0 else 0
            
            return {
                "success": True,
                "strategy": "RSI Oversold/Overbought",
                "symbol": symbol,
                "period_days": period,
                "total_return_percent": round(total_return, 2),
                "sharpe_ratio": round(sharpe_ratio, 2),
                "trades": int(abs(hist['Position'].sum()))
            }
        
        elif strategy_type == "MOMENTUM":
            # Buy when momentum is positive
            hist['Returns'] = hist['Close'].pct_change()
            hist['Momentum'] = hist['Returns'].rolling(window=10).mean()
            hist['Signal'] = (hist['Momentum'] > 0).astype(int)
            hist['Position'] = hist['Signal'].diff()
            
            hist['Strategy_Returns'] = hist['Position'].shift(1) * hist['Returns']
            
            total_return = hist['Strategy_Returns'].sum() * 100
            sharpe_ratio = (hist['Strategy_Returns'].mean() / hist['Strategy_Returns'].std()) * np.sqrt(252) if hist['Strategy_Returns'].std() > 0 else 0
            
            return {
                "success": True,
                "strategy": "Momentum",
                "symbol": symbol,
                "period_days": period,
                "total_return_percent": round(total_return, 2),
                "sharpe_ratio": round(sharpe_ratio, 2),
                "trades": int(abs(hist['Position'].sum()))
            }
        
        else:
            return {"success": False, "error": f"Unknown strategy: {strategy_type}"}
    
    except Exception as e:
        return {"success": False, "error": str(e)}

def compare_peers(target_symbol: str, competitor_symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Tool: Compare Peers (Peer Comparison Screen Logic)
    Compares fundamental metrics of target vs competitors from CSV database
    
    Args:
        target_symbol: Stock symbol to compare
        competitor_symbols: Optional list of competitor symbols. If None, finds peers in same sector.
    
    Returns:
        Dictionary with comparison data
    """
    try:
        target_data = data_engine.get_stock_data(target_symbol)
        if not target_data:
            return {"success": False, "error": f"Could not fetch data for {target_symbol}"}
        
        # If no competitors specified, try to find peers in same sector
        if not competitor_symbols:
            df = data_engine.get_all_cached()
            sector = target_data.get("sector", "Unknown")
            # Match sector column (uppercase in CSV)
            peers_df = df[df["Sector"] == sector]
            peers = peers_df["Symbol"].tolist()
            competitor_symbols = [p for p in peers if p.upper() != target_symbol.upper()][:4]  # Top 4 peers
        
        competitors_data = []
        for symbol in competitor_symbols:
            data = data_engine.get_stock_data(symbol)
            if data:
                competitors_data.append(data)
        
        comparison = {
            "target": {
                "symbol": target_data["symbol"],
                "name": target_data["name"],
                "current_price": target_data["current_price"],
                "pe_ratio": target_data["pe_ratio"],
                "market_cap": target_data["market_cap"],
                "beta": target_data.get("beta"),
                "sector": target_data["sector"]
            },
            "competitors": [
                {
                    "symbol": c["symbol"],
                    "name": c["name"],
                    "current_price": c["current_price"],
                    "pe_ratio": c["pe_ratio"],
                    "market_cap": c["market_cap"],
                    "beta": c.get("beta")
                }
                for c in competitors_data
            ]
        }
        
        return {
            "success": True,
            "comparison": comparison
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}

def fetch_news(symbol: str, limit: int = 3) -> Dict[str, Any]:
    """
    Tool: Fetch News
    Gets latest news headlines for a stock
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        news = ticker.news[:limit] if hasattr(ticker, 'news') else []
        
        if not news:
            return {
                "success": True,
                "symbol": symbol,
                "count": 0,
                "message": "No news available",
                "headlines": []
            }
        
        headlines = []
        for item in news:
            headlines.append({
                "title": item.get("title", "No title"),
                "publisher": item.get("publisher", "Unknown"),
                "link": item.get("link", ""),
                "published": item.get("providerPublishTime", 0)
            })
        
        return {
            "success": True,
            "symbol": symbol,
            "count": len(headlines),
            "headlines": headlines
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}

def view_watchlist() -> Dict[str, Any]:
    """
    Tool: View Watchlist
    Returns all stocks in the user's watchlist
    """
    try:
        symbols = database.get_watchlist()
        
        if not symbols:
            return {
                "success": True,
                "count": 0,
                "message": "Your watchlist is empty",
                "stocks": []
            }
        
        # Get current data for watchlist stocks
        stocks_data = []
        for symbol in symbols:
            data = data_engine.get_stock_data(symbol)
            if data:
                stocks_data.append({
                    "symbol": data["symbol"],
                    "name": data["name"],
                    "current_price": data["current_price"],
                    "change_percent": round(((data["current_price"] - data["previous_close"]) / data["previous_close"]) * 100, 2) if data["previous_close"] > 0 else 0,
                    "pe_ratio": data["pe_ratio"],
                    "sector": data["sector"]
                })
        
        return {
            "success": True,
            "count": len(stocks_data),
            "stocks": stocks_data
        }
    
    except Exception as e:
        return {"success": False, "error": str(e)}

# Edge Sentinel Model Loader and Chart Analysis
_EDGE_SENTINEL_INTERPRETER = None
_EDGE_SENTINEL_LABELS = None

def load_edge_sentinel_model() -> Dict[str, Any]:
    """
    Tool: Load Edge Sentinel Model
    Loads the trained Edge Sentinel model from model_training directory
    
    Returns:
        Dictionary with model status
    """
    global _EDGE_SENTINEL_INTERPRETER, _EDGE_SENTINEL_LABELS
    
    try:
        if _EDGE_SENTINEL_INTERPRETER is not None:
            return {
                "success": True,
                "message": "Edge Sentinel model already loaded",
                "model_path": "model_training/models/model_unquant.tflite"
            }
        
        # Path to model files
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'model_training', 'models')
        model_path = os.path.join(model_dir, 'model_unquant.tflite')
        labels_path = os.path.join(model_dir, 'labels.txt')
        
        if not TENSORFLOW_AVAILABLE:
            return {
                "success": False,
                "error": "TensorFlow is not installed. Please install it to use Edge Sentinel model."
            }
        
        if not os.path.exists(model_path):
            return {
                "success": False,
                "error": f"Model file not found at {model_path}"
            }
        
        if not os.path.exists(labels_path):
            return {
                "success": False,
                "error": f"Labels file not found at {labels_path}"
            }
        
        # Load TFLite model
        if tflite_interpreter is None:
            return {
                "success": False,
                "error": "TensorFlow Lite interpreter is not available. Please install tensorflow or tflite-runtime."
            }
        
        # Use the interpreter - works for both tf.lite.Interpreter and tflite_runtime.interpreter.Interpreter
        _EDGE_SENTINEL_INTERPRETER = tflite_interpreter(model_path=model_path)
        _EDGE_SENTINEL_INTERPRETER.allocate_tensors()
        
        # Load labels
        _EDGE_SENTINEL_LABELS = []
        with open(labels_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    _EDGE_SENTINEL_LABELS.append(parts[1])
        
        return {
            "success": True,
            "message": f"Edge Sentinel model loaded successfully with {len(_EDGE_SENTINEL_LABELS)} classes",
            "model_path": model_path,
            "num_classes": len(_EDGE_SENTINEL_LABELS)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load Edge Sentinel model: {str(e)}"
        }

def analyze_chart_with_edge_sentinel(image_base64: str) -> Dict[str, Any]:
    """
    Tool: Analyze Chart with Edge Sentinel (Chart Doctor Screen Logic)
    Uses the local Edge Sentinel model to analyze a chart image
    
    Args:
        image_base64: Base64 encoded image string
    
    Returns:
        Dictionary with Edge Sentinel analysis results
    """
    global _EDGE_SENTINEL_INTERPRETER, _EDGE_SENTINEL_LABELS
    
    try:
        if not TFLITE_AVAILABLE:
            error_msg = "TensorFlow Lite is not available. "
            if not TENSORFLOW_AVAILABLE:
                error_msg += "Please install TensorFlow or tflite-runtime:\n"
                error_msg += "  pip install tensorflow\n"
                error_msg += "  OR (lighter weight):\n"
                error_msg += "  pip install tflite-runtime\n"
            else:
                error_msg += "TensorFlow is installed but TensorFlow Lite module is not available. "
                error_msg += "Try: pip install tensorflow (full version) or pip install tflite-runtime"
            return {
                "success": False,
                "error": error_msg
            }
        
        # Ensure model is loaded
        if _EDGE_SENTINEL_INTERPRETER is None:
            load_result = load_edge_sentinel_model()
            if not load_result.get("success"):
                return load_result
        
        # Decode image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess image
        img_size = 224
        image = image.resize((img_size, img_size))
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Run inference
        input_details = _EDGE_SENTINEL_INTERPRETER.get_input_details()
        output_details = _EDGE_SENTINEL_INTERPRETER.get_output_details()
        
        _EDGE_SENTINEL_INTERPRETER.set_tensor(input_details[0]['index'], img_array)
        _EDGE_SENTINEL_INTERPRETER.invoke()
        
        output = _EDGE_SENTINEL_INTERPRETER.get_tensor(output_details[0]['index'])[0]
        
        # Get top 3 predictions
        top_k = 3
        top_indices = np.argsort(output)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            class_name = _EDGE_SENTINEL_LABELS[idx]
            confidence = float(output[idx])
            # Parse pattern and trend from class name (e.g., "hammer_uptrend")
            parts = class_name.split('_', 1)
            pattern = parts[0] if len(parts) > 0 else class_name
            trend = parts[1] if len(parts) > 1 else "unknown"
            
            predictions.append({
                "pattern": pattern,
                "trend": trend,
                "confidence": confidence,
                "class_name": class_name
            })
        
        return {
            "success": True,
            "model": "Edge Sentinel",
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Edge Sentinel analysis failed: {str(e)}"
        }


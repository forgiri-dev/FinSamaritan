"""
Flask API server for FinSamaritan
Provides endpoints for tools and agentic AI chat
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import google.generativeai as genai
from typing import Dict, Any, List
import base64
import io
from PIL import Image
import tensorflow as tf
import numpy as np
import tools

app = Flask(__name__)
CORS(app)

# Configure Gemini API
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
else:
    model = None
    print("Warning: GEMINI_API_KEY not set. AI features will be disabled.")

# Load Edge Sentinel model (if available)
EDGE_SENTINEL_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model_training', 'models', 'model_unquant.tflite')
EDGE_SENTINEL_LABELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'model_training', 'models', 'labels.txt')
edge_sentinel_interpreter = None
edge_sentinel_labels = []

try:
    if os.path.exists(EDGE_SENTINEL_MODEL_PATH):
        edge_sentinel_interpreter = tf.lite.Interpreter(model_path=EDGE_SENTINEL_MODEL_PATH)
        edge_sentinel_interpreter.allocate_tensors()
        
        # Load labels
        if os.path.exists(EDGE_SENTINEL_LABELS_PATH):
            with open(EDGE_SENTINEL_LABELS_PATH, 'r') as f:
                edge_sentinel_labels = [line.strip().split(' ', 1)[1] for line in f.readlines()]
        print("Edge Sentinel model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load Edge Sentinel model: {e}")

# Tool definitions for Gemini
TOOLS_DEFINITION = [
    {
        "name": "manage_portfolio",
        "description": "Add, remove, or sell stocks from the user's portfolio. Actions: 'buy' (requires shares and buy_price), 'sell' (requires shares), 'remove'.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["buy", "sell", "remove"]},
                "symbol": {"type": "string"},
                "shares": {"type": "integer", "description": "Required for buy/sell"},
                "buy_price": {"type": "number", "description": "Required for buy"}
            },
            "required": ["action", "symbol"]
        }
    },
    {
        "name": "analyze_portfolio",
        "description": "Analyze the user's portfolio to calculate total P&L, exposure, and risk ratios for all holdings.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "run_screener",
        "description": "Search and filter stocks using pandas query syntax. Example: 'pe_ratio < 15', 'current_price > 1000'.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Pandas query string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "simulate_strategy",
        "description": "Backtest trading strategies on a stock. Strategy types: SMA_CROSSOVER, RSI_OVERSOLD, MOMENTUM.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "strategy_type": {"type": "string", "enum": ["SMA_CROSSOVER", "RSI_OVERSOLD", "MOMENTUM"]},
                "period": {"type": "integer", "description": "Days of historical data", "default": 252}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "compare_peers",
        "description": "Compare fundamental metrics of a target stock with its competitors in the same sector.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_symbol": {"type": "string"},
                "competitor_symbols": {"type": "array", "items": {"type": "string"}, "description": "Optional list of competitor symbols"}
            },
            "required": ["target_symbol"]
        }
    },
    {
        "name": "fetch_news",
        "description": "Get the latest news headlines for a specific stock ticker.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "limit": {"type": "integer", "default": 3}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "view_watchlist",
        "description": "View all stocks in the user's watchlist with current prices and metrics.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]

def predict_candlestick_pattern(image_data: bytes) -> Dict[str, Any]:
    """Use Edge Sentinel model to predict candlestick pattern"""
    if edge_sentinel_interpreter is None:
        return {"error": "Edge Sentinel model not loaded"}
    
    try:
        # Preprocess image
        image = Image.open(io.BytesIO(image_data))
        image = image.resize((224, 224))
        image = image.convert('RGB')
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Run inference
        input_details = edge_sentinel_interpreter.get_input_details()
        output_details = edge_sentinel_interpreter.get_output_details()
        
        edge_sentinel_interpreter.set_tensor(input_details[0]['index'], image_array)
        edge_sentinel_interpreter.invoke()
        
        output = edge_sentinel_interpreter.get_tensor(output_details[0]['index'])
        predictions = output[0]
        
        # Get top prediction
        top_idx = np.argmax(predictions)
        confidence = float(predictions[top_idx])
        pattern_trend = edge_sentinel_labels[top_idx] if top_idx < len(edge_sentinel_labels) else "unknown"
        
        # Parse pattern and trend
        parts = pattern_trend.split('_')
        if len(parts) >= 2:
            trend = parts[-1]
            pattern = '_'.join(parts[:-1])
        else:
            pattern = pattern_trend
            trend = "unknown"
        
        return {
            "pattern": pattern,
            "trend": trend,
            "full_classification": pattern_trend,
            "confidence": round(confidence * 100, 2)
        }
    except Exception as e:
        return {"error": str(e)}

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

def call_tool_function(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Helper function to call tools"""
    if tool_name == 'manage_portfolio':
        return tools.manage_portfolio(
            action=params.get('action'),
            symbol=params.get('symbol'),
            shares=params.get('shares'),
            buy_price=params.get('buy_price')
        )
    elif tool_name == 'analyze_portfolio':
        return tools.analyze_portfolio()
    elif tool_name == 'run_screener':
        return tools.run_screener(query=params.get('query', ''))
    elif tool_name == 'simulate_strategy':
        return tools.simulate_strategy(
            symbol=params.get('symbol'),
            strategy_type=params.get('strategy_type', 'SMA_CROSSOVER'),
            period=params.get('period', 252)
        )
    elif tool_name == 'compare_peers':
        return tools.compare_peers(
            target_symbol=params.get('target_symbol'),
            competitor_symbols=params.get('competitor_symbols')
        )
    elif tool_name == 'fetch_news':
        return tools.fetch_news(
            symbol=params.get('symbol'),
            limit=params.get('limit', 3)
        )
    elif tool_name == 'view_watchlist':
        return tools.view_watchlist()
    else:
        return {"error": f"Unknown tool: {tool_name}"}

@app.route('/api/chat', methods=['POST'])
def chat():
    """Agentic AI chat endpoint with intelligent tool calling"""
    if not model:
        return jsonify({"error": "Gemini API key not configured"}), 500
    
    try:
        data = request.json
        user_message = data.get('message', '')
        chat_history = data.get('history', [])
        
        # Build system prompt
        system_prompt = """You are a financial advisor AI assistant for FinSamaritan. You have access to 7 specialized tools:

1. manage_portfolio(action, symbol, shares?, buy_price?) - Add/remove/sell stocks from portfolio
2. analyze_portfolio() - Calculate P&L and risk metrics for all holdings
3. run_screener(query) - Search stocks with pandas query syntax (e.g., "pe_ratio < 15")
4. simulate_strategy(symbol, strategy_type?, period?) - Backtest trading strategies (SMA_CROSSOVER, RSI_OVERSOLD, MOMENTUM)
5. compare_peers(target_symbol, competitor_symbols?) - Compare stocks with competitors
6. fetch_news(symbol, limit?) - Get latest news headlines for a stock
7. view_watchlist() - View user's watchlist with current prices

When the user asks about their portfolio, watchlist, or needs analysis, use the appropriate tools. Always explain what you're doing."""
        
        tool_calls_used = []
        tool_results = []
        user_lower = user_message.lower()
        
        # Intelligent tool detection and calling
        # Portfolio analysis
        if any(keyword in user_lower for keyword in ['portfolio', 'holdings', 'investments']) and \
           any(keyword in user_lower for keyword in ['analyze', 'performance', 'profit', 'loss', 'pnl', 'how']):
            result = tools.analyze_portfolio()
            tool_calls_used.append("analyze_portfolio")
            tool_results.append({"tool": "analyze_portfolio", "result": result})
        
        # Watchlist
        if any(keyword in user_lower for keyword in ['watchlist', 'watching', 'tracking']):
            result = tools.view_watchlist()
            tool_calls_used.append("view_watchlist")
            tool_results.append({"tool": "view_watchlist", "result": result})
        
        # News
        if any(keyword in user_lower for keyword in ['news', 'headlines', 'latest']):
            # Try to extract symbol from message
            symbol_match = re.search(r'\b([A-Z]{1,5})\b', user_message.upper())
            if symbol_match:
                symbol = symbol_match.group(1)
                result = tools.fetch_news(symbol)
                tool_calls_used.append("fetch_news")
                tool_results.append({"tool": "fetch_news", "result": result})
        
        # Screener
        if any(keyword in user_lower for keyword in ['screen', 'filter', 'find stocks', 'search']):
            # Try to extract query
            if 'pe' in user_lower or 'price' in user_lower or 'ratio' in user_lower:
                # Simple query extraction - can be improved
                query = "pe_ratio < 20"  # Default
                if 'low' in user_lower or 'cheap' in user_lower:
                    query = "pe_ratio < 15"
                result = tools.run_screener(query)
                tool_calls_used.append("run_screener")
                tool_results.append({"tool": "run_screener", "result": result})
        
        # Build conversation with tool results
        conversation_parts = [system_prompt]
        
        # Add history
        for msg in chat_history[-5:]:  # Last 5 messages for context
            if msg.get('user'):
                conversation_parts.append(f"User: {msg.get('user')}")
            if msg.get('assistant'):
                conversation_parts.append(f"Assistant: {msg.get('assistant')}")
        
        conversation_parts.append(f"User: {user_message}")
        
        # Add tool results to context
        if tool_results:
            conversation_parts.append("\n[Tool Results Available]")
            for tr in tool_results:
                conversation_parts.append(f"\nTool '{tr['tool']}' returned: {str(tr['result'])}")
        
        conversation_parts.append("\nAssistant:")
        
        full_prompt = "\n".join(conversation_parts)
        
        # Generate response with Gemini
        gemini_response = model.generate_content(full_prompt)
        response_text = gemini_response.text
        
        # Format response to include tool usage info
        if tool_calls_used:
            response_text = f"[Using tools: {', '.join(tool_calls_used)}]\n\n{response_text}"
        
        return jsonify({
            "response": response_text,
            "tools_used": tool_calls_used
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/tools/<tool_name>', methods=['POST'])
def call_tool(tool_name):
    """Direct tool calling endpoint"""
    try:
        data = request.json or {}
        
        if tool_name == 'manage_portfolio':
            result = tools.manage_portfolio(
                action=data.get('action'),
                symbol=data.get('symbol'),
                shares=data.get('shares'),
                buy_price=data.get('buy_price')
            )
        elif tool_name == 'analyze_portfolio':
            result = tools.analyze_portfolio()
        elif tool_name == 'run_screener':
            result = tools.run_screener(query=data.get('query', ''))
        elif tool_name == 'simulate_strategy':
            result = tools.simulate_strategy(
                symbol=data.get('symbol'),
                strategy_type=data.get('strategy_type', 'SMA_CROSSOVER'),
                period=data.get('period', 252)
            )
        elif tool_name == 'compare_peers':
            result = tools.compare_peers(
                target_symbol=data.get('target_symbol'),
                competitor_symbols=data.get('competitor_symbols')
            )
        elif tool_name == 'fetch_news':
            result = tools.fetch_news(
                symbol=data.get('symbol'),
                limit=data.get('limit', 3)
            )
        elif tool_name == 'view_watchlist':
            result = tools.view_watchlist()
        else:
            return jsonify({"error": f"Unknown tool: {tool_name}"}), 400
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/watchlist', methods=['GET', 'POST', 'DELETE'])
def watchlist():
    """Watchlist management endpoint"""
    import database
    
    if request.method == 'GET':
        symbols = database.get_watchlist()
        return jsonify({"symbols": symbols})
    
    elif request.method == 'POST':
        data = request.json
        symbol = data.get('symbol', '').upper()
        if database.add_to_watchlist(symbol):
            return jsonify({"success": True, "message": f"Added {symbol} to watchlist"})
        else:
            return jsonify({"success": False, "error": "Failed to add to watchlist"}), 400
    
    elif request.method == 'DELETE':
        data = request.json
        symbol = data.get('symbol', '').upper()
        if database.remove_from_watchlist(symbol):
            return jsonify({"success": True, "message": f"Removed {symbol} from watchlist"})
        else:
            return jsonify({"success": False, "error": "Symbol not in watchlist"}), 400

@app.route('/api/portfolio', methods=['GET'])
def portfolio():
    """Get portfolio endpoint"""
    import database
    holdings = database.get_portfolio()
    return jsonify({"holdings": holdings})

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """Analyze candlestick chart image with Edge Sentinel and Gemini"""
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        image_data = file.read()
        
        # Use Edge Sentinel model
        edge_result = predict_candlestick_pattern(image_data)
        
        # Use Gemini for additional analysis
        gemini_analysis = ""
        if model and 'error' not in edge_result:
            # Convert image to base64 for Gemini
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            prompt = f"""Analyze this candlestick chart image. 
The Edge Sentinel model detected:
- Pattern: {edge_result.get('pattern', 'unknown')}
- Trend: {edge_result.get('trend', 'unknown')}
- Confidence: {edge_result.get('confidence', 0)}%

Provide a detailed analysis of:
1. The candlestick pattern and its significance
2. The trend direction and strength
3. Trading implications and recommendations
4. Risk factors to consider"""
            
            try:
                # For Gemini, we need to send the image properly
                # This is a simplified version
                gemini_response = model.generate_content([
                    prompt,
                    Image.open(io.BytesIO(image_data))
                ])
                gemini_analysis = gemini_response.text
            except Exception as e:
                gemini_analysis = f"Gemini analysis unavailable: {str(e)}"
        
        return jsonify({
            "edge_sentinel": edge_result,
            "gemini_analysis": gemini_analysis
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)


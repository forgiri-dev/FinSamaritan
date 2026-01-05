"""
Main FastAPI server for FinSamaritan
The unified brain that routes user intent to specialized tools via Gemini
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import base64
import database
from data_engine import data_engine
import tools

# Import Gemini - try new package first, fallback to old
try:
    import google.genai as genai
    print("✅ Using google.genai (new package)")
except ImportError:
    try:
        import google.generativeai as genai
        print("⚠️ Using deprecated google.generativeai. Please install google-genai: pip install google-genai")
    except ImportError:
        raise ImportError("Please install google-genai: pip install google-genai")

# Initialize FastAPI app
app = FastAPI(title="FinSamaritan API", version="1.0.0")

# CORS middleware for React Native frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("⚠️ Warning: GEMINI_API_KEY not set. Please set it as an environment variable.")

# Configure API key and detect which package is being used
USE_NEW_API = hasattr(genai, 'Client')
genai_client = None

if USE_NEW_API:
    # New google.genai uses Client
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Initialized google.genai Client")
    except Exception as e:
        print(f"⚠️ Error initializing new API, falling back: {e}")
        USE_NEW_API = False
else:
    # Old google.generativeai uses configure
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Configured google.generativeai (deprecated)")

# Tool definitions (shared for both APIs)
manager_tools = [
        {
            "function_declarations": [
                {
                    "name": "manage_portfolio",
                    "description": "Add, remove, or sell stocks from the user's portfolio. Use 'buy' to add stocks, 'sell' to reduce shares, 'remove' to completely remove a stock.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "Action to perform: 'buy', 'sell', or 'remove'",
                                "enum": ["buy", "sell", "remove"]
                            },
                            "symbol": {
                                "type": "string",
                                "description": "Stock symbol (e.g., 'RELIANCE.NS', 'TATAPOWER.NS')"
                            },
                            "shares": {
                                "type": "integer",
                                "description": "Number of shares (required for buy/sell)"
                            },
                            "buy_price": {
                                "type": "number",
                                "description": "Buy price per share (required for buy)"
                            }
                        },
                        "required": ["action", "symbol"]
                    }
                },
                {
                    "name": "analyze_portfolio",
                    "description": "Analyze the user's portfolio. Calculates total P&L, current value, and risk assessment for each holding.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                },
                {
                    "name": "run_screener",
                    "description": "Screen stocks based on criteria. Use pandas query syntax like 'pe_ratio < 15', 'current_price > 1000', 'sector == \"Technology\"'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Pandas query string to filter stocks"
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "simulate_strategy",
                    "description": "Backtest a trading strategy on a stock. Strategies: SMA_CROSSOVER (50-day moving average), RSI_OVERSOLD (RSI < 30 buy, > 70 sell), MOMENTUM (momentum-based).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock symbol to backtest"
                            },
                            "strategy_type": {
                                "type": "string",
                                "description": "Strategy type",
                                "enum": ["SMA_CROSSOVER", "RSI_OVERSOLD", "MOMENTUM"]
                            },
                            "period": {
                                "type": "integer",
                                "description": "Backtest period in days (default: 252)"
                            }
                        },
                        "required": ["symbol", "strategy_type"]
                    }
                },
                {
                    "name": "compare_peers",
                    "description": "Compare a stock with its competitors or peers in the same sector.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target_symbol": {
                                "type": "string",
                                "description": "Stock symbol to compare"
                            },
                            "competitor_symbols": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of competitor symbols (optional, will auto-find peers if not provided)"
                            }
                        },
                        "required": ["target_symbol"]
                    }
                },
                {
                    "name": "fetch_news",
                    "description": "Get latest news headlines for a stock.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "Stock symbol"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of headlines (default: 3)"
                            }
                        },
                        "required": ["symbol"]
                    }
                },
                {
                    "name": "view_watchlist",
                    "description": "View all stocks in the user's watchlist with current prices.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            ]
        }
    ]

manager_system_instruction = """You are a Wealth Manager AI assistant for FinSamaritan. 
You help users manage their portfolio, analyze stocks, and make informed investment decisions.

Guidelines:
- Always use the provided tools to get real data. Never hallucinate stock prices or portfolio values.
- Be critical of losses and provide actionable advice.
- Format your responses clearly with tables when showing portfolio or screener results.
- Use markdown formatting for better readability (bold, tables, lists).
- When analyzing portfolio, always check current prices and calculate real P&L.
- Be concise but informative.
- For Indian stocks, ensure symbols end with .NS (e.g., RELIANCE.NS, TATAPOWER.NS).
"""

# Initialize Manager Agent (Gemini 1.5 Flash)
if USE_NEW_API:
    # New API - may need adjustment based on actual google-genai API structure
    # For now, try to create model with tools
    try:
        manager_model = genai_client.models.get("gemini-1.5-flash")
        # Store tools and system instruction for later use
        manager_model._tools = manager_tools
        manager_model._system_instruction = manager_system_instruction
    except Exception as e:
        print(f"⚠️ Error with new API model initialization: {e}")
        print("   Falling back to old API structure")
        USE_NEW_API = False
        genai.configure(api_key=GEMINI_API_KEY)
        manager_model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            tools=manager_tools,
            system_instruction=manager_system_instruction
        )
else:
    # Old API
    manager_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        tools=manager_tools,
        system_instruction=manager_system_instruction
    )

# Initialize Vision Agent (Gemini 1.5 Pro)
if USE_NEW_API:
    vision_model = genai_client.models.get("gemini-1.5-pro")
    vision_system_instruction = """You are a Financial Chart Analysis Specialist. 
    Analyze the provided chart image and provide:
    1. Chart type identification (candlestick, line, bar, etc.)
    2. Key technical indicators visible
    3. Trend analysis (bullish, bearish, sideways)
    4. Support and resistance levels
    5. Trading recommendations based on technical analysis
    6. Risk assessment
    
    Be precise and professional. Use technical terminology correctly.
    """
else:
    vision_model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        system_instruction="""You are a Financial Chart Analysis Specialist. 
        Analyze the provided chart image and provide:
        1. Chart type identification (candlestick, line, bar, etc.)
        2. Key technical indicators visible
        3. Trend analysis (bullish, bearish, sideways)
        4. Support and resistance levels
        5. Trading recommendations based on technical analysis
        6. Risk assessment
        
        Be precise and professional. Use technical terminology correctly.
        """
    )

# Request/Response models
class AgentRequest(BaseModel):
    text: str

class ChartRequest(BaseModel):
    image: str  # Base64 encoded image

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database and data cache on startup"""
    print("🚀 Starting FinSamaritan Backend...")
    database.init_db()
    data_engine.initialize_cache()
    print("✅ Backend ready!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "FinSamaritan API",
        "version": "1.0.0"
    }

@app.post("/agent")
async def agent_endpoint(request: AgentRequest):
    """
    Main agent endpoint - routes user queries to specialized tools via Gemini
    """
    try:
        # Create chat session
        chat = manager_model.start_chat()
        
        # Send user message
        response = chat.send_message(request.text)
        
        # Handle function calls if any (loop until no more function calls)
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Check if response has function calls
            if not response.candidates or not response.candidates[0].content.parts:
                break
            
            # Check for function call in parts
            function_call = None
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_call = part.function_call
                    break
            
            if not function_call:
                break  # No function call, we're done
            
            function_name = function_call.name
            args = {}
            
            # Extract arguments safely
            if hasattr(function_call, 'args'):
                args = dict(function_call.args)
            
            # Map function names to tool functions
            tool_functions = {
                "manage_portfolio": tools.manage_portfolio,
                "analyze_portfolio": tools.analyze_portfolio,
                "run_screener": tools.run_screener,
                "simulate_strategy": tools.simulate_strategy,
                "compare_peers": tools.compare_peers,
                "fetch_news": tools.fetch_news,
                "view_watchlist": tools.view_watchlist
            }
            
            if function_name in tool_functions:
                # Call the tool
                try:
                    result = tool_functions[function_name](**args)
                except Exception as e:
                    result = {"success": False, "error": str(e)}
                
                # Send result back to Gemini
                # Format the function response as a Part
                function_response_part = {
                    "function_response": {
                        "name": function_name,
                        "response": result
                    }
                }
                response = chat.send_message(function_response_part)
            else:
                print(f"Unknown function: {function_name}")
                break
        
        # Extract final response text
        if response.text:
            final_response = response.text
        else:
            final_response = "I processed your request, but couldn't generate a text response."
        
        return {
            "success": True,
            "response": final_response
        }
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in agent endpoint: {e}")
        print(error_details)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-chart")
async def analyze_chart_endpoint(request: ChartRequest):
    """
    Chart analysis endpoint - uses Gemini Vision to analyze financial charts
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        
        # Analyze with Vision Agent
        if USE_NEW_API:
            # New API - adjust based on actual google-genai API
            response = vision_model.generate_content([
                "Analyze this financial chart image. Provide detailed technical analysis including trend, support/resistance, and trading recommendations.",
                {"mime_type": "image/jpeg", "data": image_data}
            ])
        else:
            # Old API
            response = vision_model.generate_content([
                "Analyze this financial chart image. Provide detailed technical analysis including trend, support/resistance, and trading recommendations.",
                {"mime_type": "image/jpeg", "data": image_data}
            ])
        
        return {
            "success": True,
            "analysis": response.text
        }
    
    except Exception as e:
        print(f"Error in chart analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "cache_size": len(data_engine.cache),
        "gemini_configured": bool(GEMINI_API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

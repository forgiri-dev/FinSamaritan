"""
Main FastAPI server for FinSamaritan
The unified brain that routes user intent to specialized tools via Gemini
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import base64
import io
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# Use TensorFlow's built-in TFLite interpreter (works on Windows via tensorflow package)
try:
    from tensorflow.lite import Interpreter
except Exception:  # pragma: no cover
    # Fallback if module path differs
    from tensorflow import lite as _lite  # type: ignore
    Interpreter = _lite.Interpreter  # type: ignore
import database
from data_engine import data_engine
import tools

# Load environment variables from .env file
load_dotenv()

# Import Gemini - try new package first, fallback to old
USE_NEW_API = False
genai = None
genai_client = None

try:
    # Try new google-genai package (correct import)
    from google import genai
    USE_NEW_API = True
    print("✅ Using google-genai (new package)")
except ImportError:
    try:
        # Fallback to deprecated google-generativeai
        import google.generativeai as genai
        USE_NEW_API = False
        print("⚠️ Using deprecated google.generativeai. Please install google-genai: pip install google-genai")
    except ImportError:
        raise ImportError("Please install google-genai: pip install google-genai")

# Initialize FastAPI app
app = FastAPI(title="FinSamaritan API", version="1.0.0")

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("❌ ERROR: GEMINI_API_KEY not set!")
    print("   Please set it using one of these methods:")
    print("   1. Create a .env file in the backend directory with: GEMINI_API_KEY=your-api-key")
    print("   2. Set environment variable:")
    print("      Windows PowerShell: $env:GEMINI_API_KEY=\"your-api-key\"")
    print("      Windows CMD: set GEMINI_API_KEY=your-api-key")
    print("      Linux/Mac: export GEMINI_API_KEY=\"your-api-key\"")
    print("   See backend/README_ENV_SETUP.md for detailed instructions")
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Configure API key and initialize client
if USE_NEW_API:
    # New google-genai uses Client
    try:
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✅ Initialized google-genai Client")
    except Exception as e:
        print(f"❌ Error initializing new API: {e}")
        print("   Falling back to deprecated google.generativeai")
        USE_NEW_API = False
        # Re-import old API if we were using new one
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        print("✅ Configured google.generativeai (deprecated)")
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

# Initialize Manager Agent
# Use generally available Gemini models (newer first, then legacy)
manager_model = None
model_names_to_try = [
    "models/gemini-1.5-flash-001",
    "models/gemini-1.5-pro-001",
    "models/gemini-1.0-pro-001",
    "models/gemini-pro",
]

# Force use of old API (google.generativeai) which is more stable
if USE_NEW_API:
    print("⚠️ New API detected, but using old API (google.generativeai) for better compatibility")
    USE_NEW_API = False
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

# Use old API with standard gemini-pro model
for model_name in model_names_to_try:
    try:
        manager_model = genai.GenerativeModel(
            model_name=model_name,
            tools=manager_tools,
            system_instruction=manager_system_instruction
        )
        print(f"✅ Initialized Manager Agent ({model_name})")
        break
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Failed to initialize {model_name}: {error_msg[:200]}")
        if "404" in error_msg or "not found" in error_msg.lower():
            print(f"   Model {model_name} is not available. Please check your API key permissions at https://makersuite.google.com/app/apikey")
        continue

if manager_model is None:
    print("❌ ERROR: Failed to initialize any Gemini model!")
    print("Tried model names:", model_names_to_try)
    print("Please check your API key and ensure you have access to at least one of these models.")
    raise RuntimeError("Failed to initialize any Gemini model. Please check your API key and model availability.")

# Initialize Vision Agent (Gemini 1.5 Pro)
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

# Initialize Vision Agent
# Use standard model names - gemini-pro can handle both text and vision
vision_model = None
vision_model_names_to_try = [
    "models/gemini-1.5-flash-001",
    "models/gemini-1.5-pro-vision-001",
    "models/gemini-pro-vision",
    "models/gemini-pro",
]

# Initialize vision model using old API (more stable)
for model_name in vision_model_names_to_try:
    try:
        vision_model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=vision_system_instruction
        )
        print(f"✅ Initialized Vision Agent ({model_name})")
        break
    except Exception as e:
        print(f"⚠️ Failed to initialize vision model {model_name}: {str(e)[:100]}")
        continue

if vision_model is None:
    print("❌ ERROR: Failed to initialize any Gemini vision model!")
    print("Tried model names:", vision_model_names_to_try)
    print("Please check your API key and ensure you have access to at least one of these models.")
    raise RuntimeError("Failed to initialize any Gemini vision model. Please check your API key and model availability.")

# --- Edge Sentinel (server-side inference) ---
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BACKEND_DIR, "..", "model_training", "models"))
TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "model_unquant.tflite")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.txt")

if not os.path.exists(TFLITE_MODEL_PATH):
    print(f"⚠️ TFLite model not found at {TFLITE_MODEL_PATH}. Server-side inference will be disabled.")

_tflite_interpreter: Optional[Interpreter] = None
_labels: List[str] = []


def load_labels() -> List[str]:
    global _labels
    if _labels:
        return _labels
    try:
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        parsed = []
        for line in lines:
            if " " in line:
                _, label = line.split(" ", 1)
                parsed.append(label.strip())
            else:
                parsed.append(line)
        _labels = parsed
        print(f"✅ Loaded {len(_labels)} labels for Edge Sentinel")
    except Exception as e:
        print(f"⚠️ Could not load labels: {e}")
        _labels = []
    return _labels


def get_interpreter() -> Optional[Interpreter]:
    global _tflite_interpreter
    if _tflite_interpreter is not None:
        return _tflite_interpreter
    if not os.path.exists(TFLITE_MODEL_PATH):
        return None
    try:
        _tflite_interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
        _tflite_interpreter.allocate_tensors()
        print("✅ TFLite interpreter initialized")
    except Exception as e:
        print(f"❌ Failed to initialize TFLite interpreter: {e}")
        _tflite_interpreter = None
    return _tflite_interpreter


def preprocess_image(base64_image: str) -> Optional[np.ndarray]:
    try:
        if "," in base64_image:
            base64_image = base64_image.split(",", 1)[1]
        image_data = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img = img.resize((224, 224))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)  # [1, 224, 224, 3]
        return arr
    except Exception as e:
        print(f"❌ Preprocess error: {e}")
        return None


def run_inference(image_array: np.ndarray) -> Dict:
    interpreter = get_interpreter()
    if interpreter is None:
        raise RuntimeError("TFLite interpreter not available")

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(input_details["index"], image_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]  # [num_classes]

    labels = load_labels()
    if not labels:
        labels = [f"class_{i}" for i in range(len(output))]

    exp = np.exp(output - np.max(output))
    probs = exp / np.sum(exp)

    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx])

    top3_indices = np.argsort(probs)[::-1][:3]
    top3 = [
        {"label": labels[i] if i < len(labels) else f"class_{i}", "confidence": float(probs[i])}
        for i in top3_indices
    ]

    return {
        "label": labels[top_idx] if top_idx < len(labels) else f"class_{top_idx}",
        "confidence": top_prob,
        "top3": top3,
    }

# Request/Response models
class AgentRequest(BaseModel):
    text: str

class ChartRequest(BaseModel):
    image: str  # Base64 encoded image


class InferenceRequest(BaseModel):
    image: str  # Base64 encoded image

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database and data cache on startup"""
    print("🚀 Starting FinSamaritan Backend...")
    database.init_db()
    # Skip prefetching Nifty-50 to avoid rate limits (yfinance 429)
    # Cache will populate lazily on-demand.
    # To re-enable prefetch, set PREFETCH_TOP50=1 in the environment.
    prefetch = os.getenv("PREFETCH_TOP50", "0")
    if prefetch == "1":
        data_engine.initialize_cache()
    else:
        print("ℹ️ Prefetch disabled (PREFETCH_TOP50!=1). Cache will fill lazily.")
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
        # Both APIs use the same generate_content method
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


@app.post("/inference/chart")
async def inference_chart(request: InferenceRequest):
    """
    Server-side Edge Sentinel inference using TFLite.
    """
    try:
        img_array = preprocess_image(request.image)
        if img_array is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        result = run_inference(img_array)
        return {"success": True, **result}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in inference: {e}")
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

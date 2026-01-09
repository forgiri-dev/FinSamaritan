"""
FinSamaritan Backend - FastAPI Server with Agentic AI
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas as pd
import json
import base64
import numpy as np
from PIL import Image
import io

from agent_tools import search_stocks, get_live_price, get_stock_info, load_stock_data
from tools import (
    search_stocks as tools_search_stocks,
    run_screener,
    compare_peers,
    analyze_chart_with_edge_sentinel,
    manage_portfolio,
    analyze_portfolio,
    view_watchlist
)

# Load environment variables
load_dotenv()

app = FastAPI(title="FinSamaritan API", version="1.0.0")

# CORS middleware for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini models
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Auto-detect available models
def get_available_model(preferred_names=None, fallback_names=None):
    """Get the first available model from preferred list, or fallback"""
    if preferred_names is None:
        preferred_names = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-flash-latest']
    if fallback_names is None:
        fallback_names = ['gemini-2.5-pro', 'gemini-pro-latest', 'gemini-pro']
    
    # First, try to list available models
    try:
        models = list(genai.list_models())
        available_model_names = []
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                model_name = m.name.replace('models/', '')
                available_model_names.append(model_name)
        
        # Try preferred names first
        for name in preferred_names:
            if name in available_model_names:
                try:
                    return genai.GenerativeModel(name), name
                except Exception:
                    continue
        
        # Try fallback names
        for name in fallback_names:
            if name in available_model_names:
                try:
                    return genai.GenerativeModel(name), name
                except Exception:
                    continue
        
        # If we have any available models, use the first one
        if available_model_names:
            try:
                return genai.GenerativeModel(available_model_names[0]), available_model_names[0]
            except Exception:
                pass
    except Exception:
        # If listing fails, try direct model creation
        pass
    
    # Fallback: try direct model creation
    all_names = preferred_names + fallback_names
    for name in all_names:
        try:
            return genai.GenerativeModel(name), name
        except Exception:
            continue
    
    raise ValueError("No available Gemini models found. Check your API key and model availability.")

# Model for agent (fast and cheap)
try:
    agent_model, agent_model_name = get_available_model(
        preferred_names=['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-flash-latest'],
        fallback_names=['gemini-2.5-pro', 'gemini-pro-latest', 'gemini-pro']
    )
    print(f"[OK] Using agent model: {agent_model_name}")
except Exception as e:
    print(f"[WARNING] Could not initialize agent model: {e}")
    raise

# Model for vision (better image handling)
vision_model_name = None
try:
    vision_model, vision_model_name = get_available_model(
        preferred_names=['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-pro-latest'],
        fallback_names=['gemini-2.0-flash', 'gemini-flash-latest', 'gemini-pro']
    )
    print(f"[OK] Using vision model: {vision_model_name}")
except Exception as e:
    print(f"[WARNING] Could not initialize vision model: {e}")
    # Use agent model as fallback
    vision_model = agent_model
    vision_model_name = agent_model_name
    print(f"  Using agent model for vision as fallback")

# Check if stock data exists
try:
    load_stock_data()
    print("[OK] Stock data loaded successfully")
except FileNotFoundError as e:
    print(f"[WARNING] {e}")
except Exception as e:
    print(f"[WARNING] Error loading stock data: {e}")


# Request/Response Models
class AgentRequest(BaseModel):
    query: str
    show_reasoning: bool = True

class ChartAnalysisRequest(BaseModel):
    image_base64: str
    additional_context: Optional[str] = None

class CompareRequest(BaseModel):
    symbol: str

class SearchStocksRequest(BaseModel):
    query: str

class AddToPortfolioRequest(BaseModel):
    symbol: str
    shares: int
    buy_price: float

class ChatRequest(BaseModel):
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []


@app.get("/")
def root():
    return {
        "message": "FinSamaritan API - Portfolio Manager with AI Agent",
        "endpoints": {
            "/agent": "AI Agent chat with tool calling",
            "/search-stocks": "Search stocks by name/symbol",
            "/portfolio": "Get portfolio analysis",
            "/portfolio/add": "Add stock to portfolio",
            "/analyze-chart": "Dual-processing chart analysis (Edge Sentinel + Gemini)",
            "/compare": "Peer comparison"
        }
    }


@app.post("/agent")
async def agent_screener(request: AgentRequest):
    """
    Endpoint 1: The Smart Screener
    Uses Gemini 1.5 Flash with function calling to search stocks based on natural language.
    """
    try:
        user_query = request.query
        
        # Initial reasoning prompt
        reasoning_text = ""
        if request.show_reasoning:
            reasoning_prompt = f"User query: {user_query}\n\nThink step by step about what the user wants and which tool to use. Show your reasoning."
            reasoning_response = agent_model.generate_content(reasoning_prompt)
            reasoning_text = reasoning_response.text
        
        # Use simpler approach: extract query directly without complex function calling
        # This is more reliable and works consistently
        pandas_query = None
        search_results = None
        summary = ""
        
        # Extract query from user's natural language
        query_extraction_prompt = f"""Based on this user query: "{user_query}"

Translate it to a pandas query string. Only return the query string, nothing else.
Example: If user says "cheap IT stocks", return: PE_Ratio < 15 and Sector == 'IT'
Example: If user says "high growth banks", return: Sales_Growth > 15 and Sector == 'Banking'

Query string:"""

        query_response = agent_model.generate_content(query_extraction_prompt)
        pandas_query = query_response.text.strip()
        
        # Clean up the query
        pandas_query = pandas_query.replace('```', '').replace('python', '').replace('"', "'").strip()
        if pandas_query.startswith("'") and pandas_query.endswith("'"):
            pandas_query = pandas_query[1:-1]
        
        # Execute search using tools
        search_results = run_screener(pandas_query)
        
        # Generate summary if we have results
        if search_results["success"] and search_results["count"] > 0:
            # Ensure no NaN values in results for JSON serialization
            clean_results = []
            for result in search_results["results"][:10]:
                clean_result = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in result.items()}
                clean_results.append(clean_result)
            results_summary = json.dumps(clean_results, indent=2, default=str)
            summary_prompt = f"""Based on these search results:

{results_summary}

Provide a clear, concise summary. Include:
- Number of stocks found
- Key highlights (top 3-5 stocks with metrics)
- Any insights

Format in markdown."""
            summary_response = agent_model.generate_content(summary_prompt)
            summary = summary_response.text
        else:
            summary = f"No stocks found. Query: {pandas_query}"
        
        # Prepare results - clean NaN values for JSON serialization
        raw_results = search_results["results"][:20] if search_results and search_results.get("success") else []
        results_list = []
        for result in raw_results:
            clean_result = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in result.items()}
            results_list.append(clean_result)
        count = search_results["count"] if search_results and search_results.get("success") else 0
        
        return {
            "success": True,
            "query": user_query,
            "pandas_query": pandas_query or "N/A",
            "reasoning": reasoning_text if request.show_reasoning else None,
            "summary": summary,
            "results": results_list,
            "count": count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in agent_screener: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/analyze-chart")
async def analyze_chart(request: ChartAnalysisRequest):
    """
    Endpoint: Dual-Processing Chart Analysis
    Uses both Edge Sentinel (local model) and Gemini 2.5 Vision for comprehensive analysis.
    """
    try:
        import asyncio
        
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        # Dual processing: Run both analyses in parallel
        async def run_edge_sentinel():
            """Run Edge Sentinel analysis"""
            try:
                return analyze_chart_with_edge_sentinel(request.image_base64)
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        async def run_gemini_vision():
            """Run Gemini Vision analysis"""
            try:
                vision_prompt = """You are an expert technical analyst. Analyze this trading chart and provide:

1. **Trend Analysis**: Identify the current trend (uptrend, downtrend, or sideways)
2. **Support & Resistance Levels**: Mark key support and resistance levels with approximate price points
3. **Trade Setup**:
   - Entry Price
   - Stop Loss
   - Target Price
   - Risk/Reward Ratio
4. **Key Patterns**: Identify any chart patterns (head and shoulders, triangles, etc.)
5. **Recommendation**: Buy, Sell, or Hold with reasoning

Format your response in clear markdown with sections. Be specific with price levels if visible."""

                if request.additional_context:
                    vision_prompt += f"\n\nAdditional Context: {request.additional_context}"

                response = vision_model.generate_content([vision_prompt, image])
                return {
                    "success": True,
                    "analysis": response.text,
                    "model": vision_model_name or "gemini-2.5-pro"
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        # Run both analyses concurrently
        edge_result, gemini_result = await asyncio.gather(
            run_edge_sentinel(),
            run_gemini_vision()
        )
        
        # Aggregate results
        combined_analysis = {
            "success": True,
            "edge_sentinel": edge_result,
            "gemini_vision": gemini_result,
            "combined_summary": ""
        }
        
        # Generate combined summary if both succeeded
        if edge_result.get("success") and gemini_result.get("success"):
            summary_prompt = f"""Combine these two chart analyses:

**Edge Sentinel (Local Model) Analysis:**
{json.dumps(edge_result.get("predictions", []), indent=2)}

**Gemini Vision Analysis:**
{gemini_result.get("analysis", "")}

Provide a unified analysis that combines insights from both models. Highlight:
- Agreement between models
- Unique insights from each
- Final recommendation

Format in markdown."""
            
            try:
                summary_response = agent_model.generate_content(summary_prompt)
                combined_analysis["combined_summary"] = summary_response.text
            except Exception as e:
                combined_analysis["combined_summary"] = "Could not generate combined summary"
        
        return combined_analysis
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing chart: {str(e)}")


@app.post("/compare")
async def compare_stock(request: CompareRequest):
    """
    Endpoint: Peer Comparison
    Uses tools.py compare_peers function for peer analysis.
    """
    try:
        result = compare_peers(request.symbol)
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", "Comparison failed"))
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing stock: {str(e)}")

@app.post("/search-stocks")
async def search_stocks_endpoint(request: SearchStocksRequest):
    """
    Endpoint: Search Stocks
    Search stocks by name or symbol for Portfolio screen.
    """
    try:
        result = tools_search_stocks(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching stocks: {str(e)}")

@app.get("/portfolio")
async def get_portfolio():
    """
    Endpoint: Get Portfolio Analysis
    Returns portfolio analysis with P&L calculations.
    """
    try:
        result = analyze_portfolio()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting portfolio: {str(e)}")

@app.post("/portfolio/add")
async def add_to_portfolio_endpoint(request: AddToPortfolioRequest):
    """
    Endpoint: Add Stock to Portfolio
    Adds a stock to the user's portfolio.
    """
    try:
        result = manage_portfolio("buy", request.symbol, request.shares, request.buy_price)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to add to portfolio"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding to portfolio: {str(e)}")

@app.post("/agent/chat")
async def agent_chat(request: ChatRequest):
    """
    Endpoint: AI Agent Chat with Tool Calling
    Main chat interface that uses Gemini 2.5 with function calling to tools.py functions.
    """
    try:
        # Define available tools for the agent
        tools_schema = [
            {
                "name": "search_stocks",
                "description": "Search stocks by name or symbol (for Portfolio screen search)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query string"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "run_screener",
                "description": "Run a stock screener with pandas query (e.g., 'PE_Ratio < 15 and Sector == \"IT\"')",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Pandas query string"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "compare_peers",
                "description": "Compare a stock with its peers in the same sector",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_symbol": {"type": "string", "description": "Stock symbol to compare"},
                        "competitor_symbols": {"type": "array", "items": {"type": "string"}, "description": "Optional list of competitor symbols"}
                    },
                    "required": ["target_symbol"]
                }
            },
            {
                "name": "manage_portfolio",
                "description": "Manage portfolio: buy, sell, or remove stocks",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["buy", "sell", "remove"], "description": "Action to perform"},
                        "symbol": {"type": "string", "description": "Stock symbol"},
                        "shares": {"type": "integer", "description": "Number of shares (required for buy/sell)"},
                        "buy_price": {"type": "number", "description": "Buy price per share (required for buy)"}
                    },
                    "required": ["action", "symbol"]
                }
            },
            {
                "name": "analyze_portfolio",
                "description": "Get comprehensive portfolio analysis with P&L",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "view_watchlist",
                "description": "View all stocks in watchlist",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]
        
        # Map tool names to functions
        tool_functions = {
            "search_stocks": tools_search_stocks,
            "run_screener": run_screener,
            "compare_peers": compare_peers,
            "manage_portfolio": manage_portfolio,
            "analyze_portfolio": analyze_portfolio,
            "view_watchlist": view_watchlist
        }
        
        # Build conversation context
        conversation_text = ""
        if request.conversation_history:
            for msg in request.conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                conversation_text += f"{role.capitalize()}: {content}\n"
        
        conversation_text += f"User: {request.message}\nAssistant:"
        
        # Use Gemini with function calling
        # Note: This is a simplified version. For production, use proper function calling API
        system_prompt = """You are FinSamaritan, an AI financial assistant. You can help users:
- Search and analyze stocks
- Manage their portfolio
- Compare stocks with peers
- Run stock screeners
- Analyze charts

When users ask questions, use the available tools to get data, then provide helpful analysis."""
        
        full_prompt = f"{system_prompt}\n\n{conversation_text}"
        
        # For now, use a simple approach - in production, implement proper function calling
        response = agent_model.generate_content(full_prompt)
        
        return {
            "success": True,
            "response": response.text,
            "tools_used": []  # Would be populated with actual tool calls in production
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in agent chat: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


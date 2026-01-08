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
    print(f"✓ Using agent model: {agent_model_name}")
except Exception as e:
    print(f"⚠ Warning: Could not initialize agent model: {e}")
    raise

# Model for vision (better image handling)
vision_model_name = None
try:
    vision_model, vision_model_name = get_available_model(
        preferred_names=['gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-pro-latest'],
        fallback_names=['gemini-2.0-flash', 'gemini-flash-latest', 'gemini-pro']
    )
    print(f"✓ Using vision model: {vision_model_name}")
except Exception as e:
    print(f"⚠ Warning: Could not initialize vision model: {e}")
    # Use agent model as fallback
    vision_model = agent_model
    vision_model_name = agent_model_name
    print(f"  Using agent model for vision as fallback")

# Check if stock data exists
try:
    load_stock_data()
    print("✓ Stock data loaded successfully")
except FileNotFoundError as e:
    print(f"⚠ Warning: {e}")
except Exception as e:
    print(f"⚠ Warning: Error loading stock data: {e}")


# Request/Response Models
class AgentRequest(BaseModel):
    query: str
    show_reasoning: bool = True

class ChartAnalysisRequest(BaseModel):
    image_base64: str
    additional_context: Optional[str] = None

class CompareRequest(BaseModel):
    symbol: str


@app.get("/")
def root():
    return {
        "message": "FinSamaritan API - Agentic AI Financial Assistant",
        "endpoints": {
            "/agent": "Natural language stock screener",
            "/analyze-chart": "Visual technical analysis",
            "/compare": "Competitive landscape with grounding"
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
        
        # Execute search
        search_results = search_stocks(pandas_query)
        
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
    Endpoint 2: Visual Technical Analysis
    Uses Gemini 1.5 Pro Vision to analyze trading charts.
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        
        # Prepare the vision prompt
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

        # Create image part
        image = Image.open(io.BytesIO(image_data))
        
        # Generate analysis
        response = vision_model.generate_content([vision_prompt, image])
        
        return {
            "success": True,
            "analysis": response.text,
            "model": vision_model_name or "gemini-2.5-pro"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing chart: {str(e)}")


@app.post("/compare")
async def compare_stock(request: CompareRequest):
    """
    Endpoint 3: Competitive Landscape with Grounding
    Combines database fundamentals with Google Search for news/sentiment.
    """
    try:
        symbol = request.symbol
        
        # Get stock info from database
        stock_info = get_stock_info(symbol)
        
        if not stock_info["success"]:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found in database")
        
        # Get live price
        live_price = get_live_price(symbol)
        
        # Prepare data for Gemini
        fundamentals = stock_info["data"]
        live_data = live_price.get("data", {}) if live_price.get("success") else {}
        
        # Use Gemini with Google Search grounding
        # Note: Gemini's search grounding is built-in when using certain models
        comparison_prompt = f"""Analyze the stock {symbol} ({fundamentals.get('Name', 'N/A')}) comprehensively.

**Fundamental Data from Database:**
- Sector: {fundamentals.get('Sector', 'N/A')}
- PE Ratio: {fundamentals.get('PE_Ratio', 'N/A')}
- PB Ratio: {fundamentals.get('PB_Ratio', 'N/A')}
- Sales Growth: {fundamentals.get('Sales_Growth', 'N/A')}%
- Profit Margin: {fundamentals.get('Profit_Margin', 'N/A')}%
- Market Cap: {fundamentals.get('Market_Cap', 'N/A')}
- Current Price: {live_data.get('current_price', fundamentals.get('Current_Price', 'N/A'))}

**Your Task:**
1. Search for recent news and developments about {symbol} (use web search if needed)
2. Compare its fundamentals against industry peers
3. Analyze the competitive landscape
4. Provide a comprehensive report with:
   - Recent news/sentiment
   - Fundamental analysis
   - Peer comparison
   - Investment recommendation
   - Key risks and opportunities

Format your response in markdown with clear sections. Include citations for any news sources."""

        # Use Gemini with search capability
        # For grounding, we'll use a model that supports web search
        # Note: You may need to enable search grounding in your Gemini API settings
        response = agent_model.generate_content(
            comparison_prompt,
            generation_config={
                "temperature": 0.7,
            }
        )
        
        # If you have access to Google Search API, you can enhance this:
        # For now, Gemini's training data includes recent information
        
        return {
            "success": True,
            "symbol": symbol,
            "fundamentals": fundamentals,
            "live_data": live_data,
            "analysis": response.text,
            "citations": []  # Add if using Google Search API
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing stock: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# FinSamaritan Backend

The backend server for FinSamaritan - A Hybrid Agentic Financial Platform.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variable**
   ```bash
   # Windows PowerShell
   $env:GEMINI_API_KEY="your-api-key-here"
   
   # Linux/Mac
   export GEMINI_API_KEY="your-api-key-here"
   ```

3. **Generate Backup Stock Data (Optional)**
   ```bash
   python stock_generator.py
   ```
   This will create `stock_data.csv` with Top 50 Nifty stocks.

4. **Start the Server**
   ```bash
   uvicorn main:app --reload
   ```
   
   The server will start on `http://localhost:8000`

## API Endpoints

### `GET /`
Health check endpoint.

### `POST /agent`
Main agent endpoint that routes user queries to specialized tools via Gemini.

**Request:**
```json
{
  "text": "How is my portfolio?"
}
```

**Response:**
```json
{
  "success": true,
  "response": "Your portfolio analysis..."
}
```

### `POST /analyze-chart`
Analyzes financial chart images using Gemini Vision.

**Request:**
```json
{
  "image": "base64_encoded_image_string"
}
```

**Response:**
```json
{
  "success": true,
  "analysis": "Technical analysis of the chart..."
}
```

### `GET /health`
Detailed health check with cache status.

## Architecture

- **Database**: SQLite (`fin_samaritan.db`)
  - `portfolio` table: User stock holdings
  - `watchlist` table: Stocks user wants to track

- **Data Engine**: Hybrid cache system
  - Pre-loads Top 50 Nifty stocks on startup
  - Fetches live data for uncached stocks via yfinance

- **Agent Tools**: 7 specialized tools
  1. `manage_portfolio` - CRUD operations on portfolio
  2. `analyze_portfolio` - Calculate P&L and risk
  3. `run_screener` - Filter stocks by criteria
  4. `simulate_strategy` - Backtest trading strategies
  5. `compare_peers` - Compare stocks with competitors
  6. `fetch_news` - Get latest news headlines
  7. `view_watchlist` - View tracked stocks

- **AI Models**:
  - Manager Agent: Gemini 1.5 Flash (tool selection & orchestration)
  - Vision Agent: Gemini 1.5 Pro (chart analysis)

## Testing

Test the agent endpoint:
```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"text": "Show me stocks with PE ratio less than 15"}'
```


# FinSamaritan

A web-based financial analysis application with agentic AI features, powered by Gemini AI and Edge Sentinel model for candlestick pattern detection.

## Features

### 🤖 Agentic AI Chat
- Intelligent financial advisor powered by Google Gemini
- Autonomously uses 7 specialized tools based on user queries
- Displays which tools are being used for transparency
- Sidebar overlay that doesn't cover the entire screen

### 📊 7 Specialized Tools
1. **manage_portfolio** - CRUD operations for portfolio holdings
2. **analyze_portfolio** - Calculate total P&L, exposure, and risk ratios
3. **run_screener** - Search and filter stocks using pandas queries
4. **simulate_strategy** - Backtest trading strategies (SMA, RSI, Momentum)
5. **compare_peers** - Compare fundamental metrics with competitors
6. **fetch_news** - Get latest news headlines for stocks
7. **view_watchlist** - View watchlist with current prices

### 📈 Edge Sentinel Model
- Upload candlestick chart images
- Automatic pattern detection (24 pattern-trend combinations)
- Combined analysis with Gemini AI for detailed insights

### 💼 Frontend Features
- **Dark Theme** - Professional dark UI
- **Watchlist** - Add/remove stocks to track
- **Portfolio Management** - Manage holdings with P&L tracking
- **Search Bar** - Quick symbol search at the top
- **AI Chat Sidebar** - Overlay chat interface
- **Image Upload** - Analyze candlestick charts

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+
- Google Gemini API key (free tier available)

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set your Gemini API key:
```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Linux/Mac
export GEMINI_API_KEY=your_api_key_here
```

4. Initialize the database (automatic on first run):
```bash
python -c "import database; database.init_db()"
```

5. Start the Flask server:
```bash
python app.py
```

The backend will run on `http://localhost:5000`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file (optional, defaults to localhost:5000):
```bash
REACT_APP_API_URL=http://localhost:5000/api
```

4. Start the development server:
```bash
npm start
```

The frontend will run on `http://localhost:3000`

### Edge Sentinel Model

The Edge Sentinel model should be located at:
```
model_training/models/model_unquant.tflite
model_training/models/labels.txt
```

If the model files are not found, the image analysis will still work with Gemini, but without the Edge Sentinel pattern detection.

## Project Structure

```
FinSamaritan/
├── backend/
│   ├── app.py              # Flask API server
│   ├── tools.py            # 7 specialized tools
│   ├── database.py         # SQLite database management
│   ├── data_engine.py      # Stock data fetching
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API service
│   │   └── App.tsx         # Main app component
│   └── package.json
├── model_training/
│   ├── models/             # Trained Edge Sentinel model
│   └── ...                 # Training scripts
└── README.md
```

## Usage

### Adding Stocks to Watchlist
1. Use the search bar at the top
2. Enter a stock symbol (e.g., AAPL, TSLA)
3. Click "Add to Watchlist"

### Managing Portfolio
1. Navigate to the Portfolio tab
2. Click "+ Add Holding"
3. Enter symbol, shares, and buy price
4. View P&L analysis automatically

### Using AI Chat
1. Click the chat button (💬) in the bottom right
2. Ask questions like:
   - "Analyze my portfolio"
   - "Show my watchlist"
   - "Find stocks with PE ratio less than 15"
   - "Get news for AAPL"
   - "Compare TSLA with competitors"

### Analyzing Candlestick Charts
1. Click the chart button (📊) in the bottom right
2. Upload a candlestick chart image
3. Click "Analyze with Edge Sentinel & Gemini"
4. View pattern detection and AI analysis

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/chat` - AI chat with tool calling
- `GET /api/watchlist` - Get watchlist
- `POST /api/watchlist` - Add to watchlist
- `DELETE /api/watchlist` - Remove from watchlist
- `GET /api/portfolio` - Get portfolio holdings
- `POST /api/tools/<tool_name>` - Direct tool calling
- `POST /api/analyze-image` - Analyze candlestick chart image

## Technologies Used

- **Backend**: Flask, Python, SQLite
- **Frontend**: React, TypeScript
- **AI**: Google Gemini API
- **ML**: TensorFlow Lite (Edge Sentinel)
- **Data**: yfinance

## License

MIT

## Notes

- The Gemini API key is required for AI features
- Stock data is fetched from yfinance (free, but rate-limited)
- Edge Sentinel model is optional but recommended for image analysis
- Database is SQLite (local file: `backend/finsamaritan.db`)


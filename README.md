# FinSamaritan: The Hybrid Agentic Financial Platform

> **Not a chatbot** - An autonomous Multi-Agent System designed to democratize institutional-grade financial intelligence.

## 🏗️ Architecture

FinSamaritan employs a **Hybrid Architecture**:

- **The Cloud Hive (Backend)**: A centralized "Manager Agent" (Gemini) that autonomously routes user intent to 7 specialized Python tools (Quant, Auditor, Portfolio Manager, etc.)

- **The Edge Sentinel (Frontend)**: An offline Neural Network (TensorFlow.js) running in the browser that filters visual data in real-time (0.1s latency) before it reaches the cloud.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+
- Google Gemini API Key ([Get it here](https://makersuite.google.com/app/apikey))

### Backend Setup

**Windows:**
```powershell
cd backend
.\start_backend.bat
```

**Linux/Mac:**
```bash
cd backend
chmod +x start_backend.sh
./start_backend.sh
```

**Manual:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY="your-api-key"  # Windows: $env:GEMINI_API_KEY="your-api-key"
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Then open your browser to `http://localhost:3000`

## 📚 Documentation

- **[Complete Setup & Testing Guide](SETUP_AND_TESTING_GUIDE.md)** - Comprehensive step-by-step setup and testing instructions
- **[Quick Start Guide](QUICK_START.md)** - Get running in 5 minutes
- **[Backend README](backend/README.md)** - Backend API documentation

## 🧪 Testing

### Backend Tests
```bash
cd backend
python test_backend.py
```

### Manual Testing
```bash
# Health check
curl http://localhost:8000/health

# Agent query
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"text": "What is the price of RELIANCE.NS?"}'
```

## 🛠️ Tech Stack

| Layer | Component | Technology |
|-------|-----------|------------|
| Frontend | Web App | React (TypeScript) + Vite |
| Edge AI | Offline Model | TensorFlow.js |
| Backend | API Server | Python (FastAPI) |
| Cloud AI | Manager Agent | Gemini 1.5 Flash |
| Cloud AI | Vision Agent | Gemini 1.5 Pro |
| Data | Live Feed | yfinance |
| Storage | Database | SQLite |

## 🎯 Features

### 7 Specialized Agent Tools

1. **manage_portfolio** - Add/remove stocks from portfolio
2. **analyze_portfolio** - Calculate P&L, exposure, risk ratios
3. **run_screener** - Filter stocks by criteria (PE, price, sector, etc.)
4. **simulate_strategy** - Backtest trading strategies (SMA, RSI, Momentum)
5. **compare_peers** - Compare stocks with competitors
6. **fetch_news** - Get latest news headlines
7. **view_watchlist** - View tracked stocks

### Key Capabilities

- ✅ **Autonomous Portfolio Management** - Natural language commands
- ✅ **Real-time Stock Data** - Live prices from yfinance
- ✅ **Technical Analysis** - Chart pattern recognition
- ✅ **Strategy Backtesting** - Test trading strategies
- ✅ **Data Persistence** - SQLite database
- ✅ **Edge AI Filtering** - Pre-filter images locally in browser

## 📁 Project Structure

```
FinSamaritan/
├── backend/
│   ├── database.py          # SQLite database management
│   ├── data_engine.py       # Hybrid cache system
│   ├── tools.py             # 7 agent tools
│   ├── main.py              # FastAPI server
│   ├── stock_generator.py   # Backup data generator
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── api/             # Backend API client
│   │   ├── services/        # Edge Sentinel service
│   │   ├── screens/         # App screens
│   │   └── components/      # UI components
│   ├── index.html           # HTML entry point
│   └── vite.config.ts       # Vite configuration
├── SETUP_AND_TESTING_GUIDE.md   # Complete setup guide
└── QUICK_START.md                # Quick start guide
```

## 🎬 Demo Scenarios

### Scenario 1: Portfolio Management
```
User: "I bought 100 shares of Tata Power at 250"
Agent: "✅ Added. Your total invested amount is ₹25,000."
```

### Scenario 2: Portfolio Analysis
```
User: "Is my portfolio safe?"
Agent: "You are down 2% on Tata Power. However, news suggests a renewable energy boom, so hold."
```

### Scenario 3: Strategy Backtest
```
User: "What if I bought Reliance whenever it crossed its 50-day moving average?"
Agent: "That strategy would have yielded a 14% return over the last year."
```

## 🔒 Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive data
- In production, restrict CORS origins
- Use HTTPS for API endpoints

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

This is a demonstration project. For production use, consider:
- Adding authentication
- Implementing rate limiting
- Adding error monitoring
- Optimizing database queries
- Enhancing Edge Sentinel model

## 📞 Support

For setup issues, refer to:
1. [SETUP_AND_TESTING_GUIDE.md](SETUP_AND_TESTING_GUIDE.md) - Detailed troubleshooting
2. Backend logs (terminal output)
3. Browser console logs

---

**Built with ❤️ for the FinSights Hackathon**

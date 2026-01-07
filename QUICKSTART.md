# Quick Start Guide

## 1. Get Your Gemini API Key

1. Go to https://makersuite.google.com/app/apikey
2. Create a free API key
3. Copy the key

## 2. Start Backend

```bash
# Windows
cd backend
set GEMINI_API_KEY=your_key_here
pip install -r requirements.txt
python app.py

# Linux/Mac
cd backend
export GEMINI_API_KEY=your_key_here
pip install -r requirements.txt
python app.py
```

Backend runs on http://localhost:5000

## 3. Start Frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs on http://localhost:3000

## 4. Test the App

1. Open http://localhost:3000
2. Add a stock to watchlist (e.g., AAPL)
3. Click the chat button (💬) and ask: "Show my watchlist"
4. Try uploading a candlestick chart image

## Troubleshooting

- **Backend won't start**: Make sure GEMINI_API_KEY is set
- **Frontend can't connect**: Check that backend is running on port 5000
- **Image analysis fails**: Edge Sentinel model is optional, Gemini will still work
- **Stock data not loading**: Check internet connection, yfinance may be rate-limited


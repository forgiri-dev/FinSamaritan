# FinSamaritan - Agentic AI Financial Assistant

**FinSamaritan** is an "Agentic AI" Financial Assistant that goes beyond simple chatbots. It has tools: it can query databases, analyze images of charts, and cross-reference real-time news.

## 🎯 Project Overview

**Goal**: To democratize financial literacy by translating complex data into simple insights.

**The "Winning" Hook**: Uses a Hybrid Architecture—combining a static local database (for speed) with Google Search Grounding (for accuracy) and Gemini Vision (for technical analysis).

## 🏗️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Frontend | Flutter (Dart) | Single codebase for Android/Web. Fast UI prototyping. |
| Backend | Python (FastAPI) | Native support for AI libraries. Fast and lightweight. |
| AI Brain | Gemini 1.5 Flash | For the "Agent" (Screener) because it's fast and cheap. |
| AI Vision | Gemini 1.5 Pro | For "Chart Analysis" because it handles complex images better. |
| Database | Pandas (In-Memory) | For the Hackathon, a CSV loaded into RAM is 100x faster than SQL. |
| Data Source | yfinance | To fetch real-time prices for the "Top 50" stocks. |

## 📁 Project Structure

```
FinSamaritan/
├── backend/
│   ├── main.py            (FastAPI Server)
│   ├── agent_tools.py     (Custom Python functions for the AI)
│   ├── stock_data.csv     (Your dataset - generated)
│   ├── stock_data_generator.py  (Your existing generator)
│   ├── requirements.txt
│   └── .env               (API Keys - create this)
├── frontend/
│   ├── lib/
│   │   ├── main.dart
│   │   ├── services/      (API logic)
│   │   └── screens/       (UI Pages)
│   └── pubspec.yaml
└── .gitignore
```

## 📚 Setup Guides

**New to FinSamaritan? Start here:**

- **[QUICK_START.md](QUICK_START.md)** - Get running in 5 minutes! ⚡
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Complete step-by-step guide for Chrome, Windows, and Android 📖

## 🚀 Step-by-Step Setup Instructions

### Step 1: Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get your Gemini API Key:**
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Sign in with your Google account
   - Click "Create API Key"
   - Copy the API key

5. **Create `.env` file:**
   ```bash
   # Copy the example file
   copy .env.example .env   # Windows
   cp .env.example .env    # Mac/Linux
   ```
   
   Then edit `.env` and add your API key:
   ```
   GEMINI_API_KEY=your_actual_api_key_here
   ```

6. **Generate stock data:**
   ```bash
   # Run your stock data generator
   python stock_data_generator.py
   ```
   
   This should create `stock_data.csv` in the backend directory.

7. **Start the backend server:**
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   
   You should see:
   ```
   ✓ Stock data loaded successfully
   INFO:     Uvicorn running on http://0.0.0.0:8000
   ```

8. **Test the backend:**
   - Open your browser and go to: `http://localhost:8000`
   - You should see the API welcome message
   - Test endpoint: `http://localhost:8000/docs` (FastAPI Swagger UI)

### Step 2: Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install Flutter dependencies:**
   ```bash
   flutter pub get
   ```

3. **Check Flutter setup:**
   ```bash
   flutter doctor
   ```
   
   Make sure you have at least one device/emulator available.

4. **Run the Flutter app:**
   ```bash
   # For Android Emulator
   flutter run

   # For iOS Simulator (Mac only)
   flutter run -d ios

   # For Web
   flutter run -d chrome
   ```

### Step 3: Connect Frontend to Backend

**Important**: The frontend is configured to connect to:
- **Android Emulator**: `http://10.0.2.2:8000` (automatically mapped to localhost)
- **iOS Simulator**: `http://localhost:8000`
- **Web**: `http://localhost:8000`

The API service (`frontend/lib/services/api_service.dart`) handles this automatically.

**If you're running on a physical device:**
- Find your computer's IP address:
  ```bash
  # Windows
  ipconfig

  # Mac/Linux
  ifconfig
  ```
- Update `api_service.dart` to use your IP: `http://YOUR_IP:8000`

### Step 4: Testing the Features

#### Feature 1: Smart Screener
1. Open the app
2. Go to the "Screener" tab
3. Try queries like:
   - "Show me undervalued IT stocks"
   - "Find high growth banks"
   - "Show me cheap pharma stocks"
4. Watch the "Reasoning Trace" to see how the AI thinks

#### Feature 2: Chart Doctor
1. Go to the "Chart Doctor" tab
2. Click "Gallery" or "Camera"
3. Upload a trading chart image
4. Click "Analyze Chart"
5. See the AI's technical analysis with support/resistance levels

#### Feature 3: Peer Comparison
1. Go to the "Compare" tab
2. Enter a stock symbol (e.g., "RELIANCE", "TCS")
3. Click search
4. See comprehensive analysis combining fundamentals + news

## 🎬 Demo Mode

The app includes demo query buttons for quick testing during presentations:
- **Screener**: Pre-filled queries like "Show me undervalued IT stocks"
- **Compare**: Quick buttons for popular stocks (RELIANCE, TCS, HDFCBANK, etc.)

## 🐛 Troubleshooting

### Backend Issues

**Error: "GEMINI_API_KEY not found"**
- Make sure you created `.env` file in the `backend/` directory
- Check that the file contains: `GEMINI_API_KEY=your_key_here`
- Restart the server after creating `.env`

**Error: "stock_data.csv not found"**
- Run `python stock_data_generator.py` in the backend directory
- Make sure `stock_data.csv` is created in `backend/`

**Port 8000 already in use:**
```bash
# Find and kill the process (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Mac/Linux
lsof -ti:8000 | xargs kill
```

### Frontend Issues

**"Error connecting to API"**
- Make sure the backend is running on port 8000
- Check the API URL in `api_service.dart`
- For Android emulator, use `10.0.2.2:8000`
- For physical device, use your computer's IP address

**"flutter pub get" fails**
- Make sure you have Flutter SDK installed
- Run `flutter doctor` to check setup
- Try `flutter clean` then `flutter pub get`

**App crashes on image picker**
- Make sure you've granted camera/gallery permissions
- For Android: Check `AndroidManifest.xml` permissions
- For iOS: Check `Info.plist` permissions

## 📝 API Endpoints

### POST `/agent`
Natural language stock screener
```json
{
  "query": "Show me cheap IT stocks",
  "show_reasoning": true
}
```

### POST `/analyze-chart`
Visual technical analysis
```json
{
  "image_base64": "base64_encoded_image_string",
  "additional_context": "Optional context"
}
```

### POST `/compare`
Competitive landscape analysis
```json
{
  "symbol": "RELIANCE"
}
```

## 🎯 Key Features

1. **Natural Language Screener**: Ask in plain English, get filtered results
2. **Visual Chart Analysis**: Upload charts, get technical analysis
3. **Competitive Landscape**: Combine fundamentals with real-time news
4. **Reasoning Trace**: See how the AI thinks (for judges!)

## 📦 Dependencies

### Backend (`requirements.txt`)
- fastapi
- uvicorn
- pandas
- google-generativeai
- python-dotenv
- yfinance
- pillow
- pydantic

### Frontend (`pubspec.yaml`)
- flutter
- http
- flutter_markdown
- image_picker



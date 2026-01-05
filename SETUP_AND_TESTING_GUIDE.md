# FinSamaritan: Complete Setup & Testing Guide

This comprehensive guide will walk you through setting up and testing the entire FinSamaritan application, including the backend, frontend, and Edge Sentinel model.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Backend Setup](#backend-setup)
3. [Frontend Setup](#frontend-setup)
4. [Model Verification](#model-verification)
5. [Testing Guide](#testing-guide)
6. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### Required Software

1. **Python 3.8+**
   ```bash
   python --version  # Should be 3.8 or higher
   ```

2. **Node.js 18+**
   ```bash
   node --version  # Should be 18 or higher
   ```

3. **npm or yarn**
   ```bash
   npm --version
   ```

4. **Android Studio** (for Android development)
   - Android SDK
   - Android Emulator or Physical Device
   - JDK 11+

5. **Xcode** (for iOS development - Mac only)
   - Xcode 14+
   - CocoaPods

### Required API Keys

1. **Google Gemini API Key**
   - Get it from: https://makersuite.google.com/app/apikey
   - Save it securely - you'll need it for the backend

---

## 🚀 Backend Setup

### Step 1: Navigate to Backend Directory

```bash
cd backend
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed fastapi-0.104.1 uvicorn-0.24.0 ...
```

### Step 4: Set Environment Variable

**Windows PowerShell:**
```powershell
$env:GEMINI_API_KEY="your-gemini-api-key-here"
```

**Windows CMD:**
```cmd
set GEMINI_API_KEY=your-gemini-api-key-here
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```

**For permanent setup (Linux/Mac):**
```bash
echo 'export GEMINI_API_KEY="your-gemini-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Step 5: Generate Backup Stock Data (Optional)

```bash
python stock_generator.py
```

This will:
- Fetch Top 50 Nifty stocks
- Save to `stock_data.csv`
- Take ~2-3 minutes

**Expected output:**
```
🔄 Generating stock data backup...
✅ Fetched RELIANCE.NS
✅ Fetched TCS.NS
...
✅ Generated stock_data.csv with 50 stocks
```

### Step 6: Start the Backend Server

```bash
uvicorn main:app --reload
```

**Expected output:**
```
🚀 Starting FinSamaritan Backend...
✅ Database initialized at fin_samaritan.db
🔄 Initializing data cache with Top 50 Nifty stocks...
✅ Cache initialized: 50 stocks loaded
✅ Backend ready!
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Server is now running on:** `http://localhost:8000`

### Step 7: Verify Backend is Running

Open a new terminal and test:

```bash
curl http://localhost:8000/
```

**Expected response:**
```json
{
  "status": "online",
  "service": "FinSamaritan API",
  "version": "1.0.0"
}
```

Or visit in browser: `http://localhost:8000/docs` (FastAPI Swagger UI)

---

## 📱 Frontend Setup

### Step 1: Navigate to Frontend Directory

```bash
cd frontend
```

### Step 2: Install Dependencies

```bash
npm install
```

**Expected output:**
```
added 500+ packages in 30s
```

### Step 3: Configure API Endpoint (if needed)

Edit `frontend/src/api/agent.ts` if your backend is not on `localhost:8000`:

- **Android Emulator:** Use `http://10.0.2.2:8000`
- **Physical Device:** Use your computer's IP address (e.g., `http://192.168.1.100:8000`)
- **iOS Simulator:** Use `http://localhost:8000`

### Step 4: Start Metro Bundler

**Terminal 1:**
```bash
npm start
```

**Expected output:**
```
Welcome to Metro!
...
Metro waiting on exp://192.168.1.100:8081
```

### Step 5: Run on Android

**Terminal 2 (new terminal):**
```bash
npm run android
```

**Or for iOS (Mac only):**
```bash
npm run ios
```

**Expected output:**
```
info Running "FinSights" on "Pixel_5_API_33"
...
BUILD SUCCESSFUL
```

The app should launch on your emulator/device.

---

## 🤖 Model Verification

### Step 1: Verify Model Files Exist

Check that these files exist:
```bash
ls frontend/assets/
```

**Should show:**
- `model_unquant.tflite` (TensorFlow Lite model)
- `labels.txt` (Model labels)

### Step 2: Verify Model Integration

The Edge Sentinel service is implemented in `frontend/src/services/EdgeSentinel.ts`.

**Current Status:**
- ✅ Service structure is ready
- ✅ Placeholder implementation for testing
- ⚠️ Full TFLite integration requires `react-native-fast-tflite` (optional for basic testing)

**For Production TFLite Integration:**
1. Install: `npm install react-native-fast-tflite`
2. Uncomment the production code in `EdgeSentinel.ts` (lines 191-242)
3. Rebuild the app

**For Testing:**
- The placeholder implementation will work for basic functionality testing
- It simulates chart detection with random patterns

---

## 🧪 Testing Guide

### Backend Testing

#### Test 1: Health Check

```bash
curl http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "cache_size": 50,
  "gemini_configured": true
}
```

#### Test 2: Agent Endpoint (Simple Query)

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"What is the current price of RELIANCE.NS?\"}"
```

**Expected response:**
```json
{
  "success": true,
  "response": "The current price of RELIANCE.NS is ₹2,450.50..."
}
```

#### Test 3: Portfolio Management

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"I bought 100 shares of RELIANCE.NS at 2400\"}"
```

**Expected response:**
```json
{
  "success": true,
  "response": "✅ Added 100 shares of RELIANCE.NS at ₹2400. Your total invested amount is ₹240,000."
}
```

#### Test 4: Portfolio Analysis

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"How is my portfolio performing?\"}"
```

**Expected response:**
```json
{
  "success": true,
  "response": "Your portfolio analysis:\n\n**Total Invested:** ₹240,000\n**Current Value:** ₹245,050\n**Total P&L:** +₹5,050 (+2.10%)\n\n..."
}
```

#### Test 5: Stock Screener

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Show me stocks with PE ratio less than 15\"}"
```

**Expected response:**
```json
{
  "success": true,
  "response": "Found 12 stocks matching your criteria:\n\n| Symbol | Name | Price | PE Ratio |\n|--------|------|-------|----------|\n| ..."
}
```

#### Test 6: Strategy Backtest

```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"What if I bought RELIANCE.NS whenever it crossed its 50-day moving average?\"}"
```

**Expected response:**
```json
{
  "success": true,
  "response": "Backtesting 50-Day SMA Crossover strategy on RELIANCE.NS:\n\n**Total Return:** +14.2%\n**Sharpe Ratio:** 1.35\n**Number of Trades:** 8\n\n..."
}
```

#### Test 7: Chart Analysis (with base64 image)

```bash
# First, convert an image to base64
# Windows PowerShell:
$base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("chart.jpg"))

# Then send:
curl -X POST http://localhost:8000/analyze-chart \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$base64\"}"
```

**Expected response:**
```json
{
  "success": true,
  "analysis": "Chart Analysis:\n\n**Chart Type:** Candlestick\n**Trend:** Uptrend\n**Support Level:** ₹2,400\n**Resistance Level:** ₹2,500\n\n..."
}
```

### Frontend Testing

#### Test 1: App Launch

1. Launch the app on emulator/device
2. You should see the welcome message from FinSights AI
3. Chat interface should be visible

#### Test 2: Text Query

1. Type: "What is the price of TCS.NS?"
2. Press Send
3. **Expected:** AI responds with current stock price

#### Test 3: Portfolio Management

1. Type: "I bought 50 shares of INFY.NS at 1500"
2. Press Send
3. **Expected:** Confirmation message with total invested

#### Test 4: Portfolio Analysis

1. Type: "Show me my portfolio"
2. Press Send
3. **Expected:** Portfolio table with P&L calculations

#### Test 5: Image Upload (Edge Sentinel)

1. Tap the image icon in chat
2. Select a financial chart image
3. **Expected:**
   - Edge Sentinel processes image (0.1s)
   - If valid chart: Shows pattern detection
   - Vision Agent analyzes (2-3s)
   - Returns detailed technical analysis

#### Test 6: Invalid Image Test

1. Upload a selfie or random image
2. **Expected:** Alert "Not a Chart - Edge Sentinel detected this is not a financial chart"

#### Test 7: Persistence Test

1. Add stocks to portfolio
2. Close the app completely
3. Reopen the app
4. Ask: "Show my portfolio"
5. **Expected:** Portfolio data persists (SQLite)

### Integration Testing

#### Test 1: End-to-End Flow

1. **Backend:** Start server
2. **Frontend:** Launch app
3. **Query:** "I bought 100 shares of RELIANCE.NS at 2400"
4. **Query:** "How is my portfolio?"
5. **Query:** "Show me stocks with PE < 20"
6. **Upload:** Chart image
7. **Verify:** All responses are accurate and formatted

#### Test 2: Real-time Data Verification

1. Ask: "What is the current price of RELIANCE.NS?"
2. Check the price on Google Finance or NSE website
3. **Expected:** Prices should match (within 1-2 minutes)

#### Test 3: Multi-Tool Interaction

1. Ask: "Compare RELIANCE.NS with its competitors"
2. **Expected:** AI uses `compare_peers` tool, then `fetch_news`, synthesizes response

---

## 🔍 Troubleshooting

### Backend Issues

#### Issue: "GEMINI_API_KEY not set"

**Solution:**
```bash
# Verify environment variable is set
echo $GEMINI_API_KEY  # Linux/Mac
echo %GEMINI_API_KEY%  # Windows CMD
$env:GEMINI_API_KEY   # Windows PowerShell
```

#### Issue: "Module not found" errors

**Solution:**
```bash
# Ensure virtual environment is activated
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### Issue: Port 8000 already in use

**Solution:**
```bash
# Use a different port
uvicorn main:app --reload --port 8001

# Update frontend API URL accordingly
```

#### Issue: Database errors

**Solution:**
```bash
# Delete and recreate database
rm fin_samaritan.db  # Linux/Mac
del fin_samaritan.db  # Windows
# Restart server (will auto-create)
```

#### Issue: yfinance rate limiting

**Solution:**
- Wait a few minutes between requests
- The cache system should prevent excessive API calls
- Check your internet connection

### Frontend Issues

#### Issue: "Network error: Could not reach server"

**Solution:**
1. Verify backend is running: `curl http://localhost:8000/`
2. Check API URL in `frontend/src/api/agent.ts`
3. For Android emulator, use `http://10.0.2.2:8000`
4. For physical device, use your computer's IP address

#### Issue: Metro bundler won't start

**Solution:**
```bash
# Clear cache
npm start -- --reset-cache

# Or
watchman watch-del-all  # If using watchman
```

#### Issue: Build fails on Android

**Solution:**
```bash
cd android
./gradlew clean
cd ..
npm run android
```

#### Issue: App crashes on image upload

**Solution:**
1. Check permissions in `AndroidManifest.xml`:
   ```xml
   <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
   <uses-permission android:name="android.permission.CAMERA" />
   ```
2. For Android 13+, request runtime permissions

### Model Issues

#### Issue: Edge Sentinel not detecting charts

**Solution:**
- Current implementation uses placeholder logic
- For production, integrate actual TFLite model
- Check `frontend/src/services/EdgeSentinel.ts` for implementation

#### Issue: Model file not found

**Solution:**
```bash
# Verify files exist
ls frontend/assets/model_unquant.tflite
ls frontend/assets/labels.txt

# If missing, re-download or regenerate from Teachable Machine
```

---

## ✅ Verification Checklist

Before considering the setup complete, verify:

### Backend
- [ ] Server starts without errors
- [ ] Health check returns "healthy"
- [ ] Database initialized
- [ ] Cache loaded with 50 stocks
- [ ] Gemini API key configured
- [ ] Agent endpoint responds to queries
- [ ] Chart analysis endpoint works

### Frontend
- [ ] App builds successfully
- [ ] App launches on emulator/device
- [ ] Welcome message appears
- [ ] Can send text messages
- [ ] Can upload images
- [ ] Responses are formatted correctly
- [ ] Markdown rendering works

### Integration
- [ ] Backend and frontend communicate
- [ ] Portfolio data persists
- [ ] Real-time stock prices are accurate
- [ ] Edge Sentinel filters images
- [ ] Vision Agent analyzes charts
- [ ] All 7 tools work correctly

---

## 📊 Performance Benchmarks

Expected performance metrics:

- **Backend Startup:** < 5 seconds
- **Cache Initialization:** 2-3 minutes (one-time)
- **Agent Response:** 2-5 seconds
- **Chart Analysis:** 3-8 seconds
- **Edge Sentinel:** < 0.1 seconds (local)
- **Database Queries:** < 50ms

---

## 🎯 Next Steps

After successful setup:

1. **Customize:** Add more stocks to cache
2. **Enhance:** Integrate full TFLite model
3. **Deploy:** Set up production backend (Heroku, AWS, etc.)
4. **Optimize:** Fine-tune cache TTL and strategies
5. **Extend:** Add more agent tools as needed

---

## 📞 Support

If you encounter issues not covered here:

1. Check the logs (backend terminal and Metro bundler)
2. Verify all prerequisites are met
3. Ensure API keys are valid
4. Review the code comments in each module
5. Check FastAPI docs at `http://localhost:8000/docs`

---

**Happy Testing! 🚀**


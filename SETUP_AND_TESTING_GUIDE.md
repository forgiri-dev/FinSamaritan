## FinSamaritan: Complete Setup & Testing Guide

This comprehensive guide will walk you through setting up and testing the entire FinSamaritan application, including:
- **Backend API** (FastAPI + Gemini integration)
- **Frontend web app** (Vite/React)
- **Model training pipeline** (Edge Sentinel: data generation + TensorFlow training)

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

1. **Python 3.11 (64‑bit, recommended)**

   The project is tested with Python 3.11, and some ML libraries do **not** yet have stable support for 3.13.

   ```bash
   python --version
   ```

   On Windows, you can list all installed versions:

   ```powershell
   py -0p
   ```

   Ensure you have a 64‑bit Python 3.11 install (path typically under `Python311`).

2. **Node.js 18+**
   ```bash
   node --version  # Should be 18 or higher
   ```

3. **npm or yarn**
   ```bash
   npm --version
   ```

4. **Modern Web Browser** (Chrome, Firefox, Edge, or Safari)
   - Chrome recommended for best compatibility

### Required API Keys

1. **Google Gemini API Key**
   - Get it from: https://makersuite.google.com/app/apikey
   - Save it securely – you'll need it for the backend

---

## 🧱 Project Structure Overview

At the repo root:

- `backend/` – FastAPI server, Gemini tools, SQLite DB
- `frontend/` – Vite/React web client
- `model_training/` – Edge Sentinel data generation + TensorFlow training pipeline
- `SETUP_AND_TESTING_GUIDE.md` – this guide

---

## 🚀 Backend Setup (FastAPI + Gemini)

The backend is designed to run cleanly on **Python 3.11+** with pinned, compatible dependencies.

### Step 1: Create and Activate a Virtual Environment

Always create the venv in the **project root**, then work inside `backend/`.

**Windows (PowerShell, Python 3.11):**

```powershell
cd "C:\Users\Zaid Iqbal\FinSamaritan"

# Create venv with Python 3.11 explicitly (adjust path if different)
& "C:\Users\Zaid Iqbal\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv

# Allow script execution for this session only
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate venv
.\.venv\Scripts\Activate.ps1

python --version  # Should now show Python 3.11.x
```

**Windows (CMD alternative):**

```cmd
cd C:\Users\Zaid Iqbal\FinSamaritan
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**Linux/Mac:**

```bash
cd /path/to/FinSamaritan
python3 -m venv .venv
source .venv/bin/activate
python --version  # Should be 3.11+
```

### Step 2: Install Backend Dependencies

From the **project root** with the venv active:

```bash
cd backend
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Backend requirements are pinned to versions compatible with Python 3.11+ and Gemini (`fastapi`, `pydantic`, `google-generativeai`, `numpy`, `pandas`, etc.).

### Step 3: Set GEMINI_API_KEY Environment Variable

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

### Step 4: (Optional) Generate Backup Stock Data

```bash
cd backend
python stock_generator.py
```

This will:
- Fetch Top 50 Nifty stocks
- Save to `stock_data.csv`
- Take ~2–3 minutes

### Step 5: Start the Backend Server

From `backend/` with the venv active:

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

### Step 6: Verify Backend is Running

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

## 🌐 Frontend Setup (Vite/React)

### Step 1: Navigate to Frontend Directory

```bash
cd frontend
```

You can run the frontend either from the same venv session (Python is irrelevant here) or from a normal shell.

### Step 2: Install Dependencies

```bash
npm install
```

### Step 3: Configure API Endpoint (if needed)

The frontend calls the backend via an API URL:

- Default backend URL: `http://localhost:8000`
- If you change backend port/host, update either:
  - `frontend/src/api/agent.ts`, or
  - Set `VITE_API_URL` in a `.env` file for Vite.

### Step 4: Start Development Server

```bash
npm run dev
```

**Expected output:**
```
  VITE v5.0.8  ready in 500 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

### Step 5: Open in Browser

Open your web browser and navigate to:
```
http://localhost:3000
```

The app should load in your browser.

---

## 🤖 Model Training & Verification (Edge Sentinel)

The `model_training/` folder contains everything needed to **generate training data** and **train the Edge Sentinel CNN** that ultimately produces the `model_unquant.tflite` and `labels.txt` used on the frontend.

### 1. Model Training Environment Setup

Model training requires TensorFlow and scientific Python packages. These are pinned in `model_training/requirements.txt` to versions that work well together on Python 3.11.

You can either:
- **Reuse the same `.venv`** as the backend (recommended), or
- Create a **separate venv** if you want to isolate heavy ML deps.

**Reuse same venv (from project root):**

```bash
cd model_training
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

This installs:
- `tensorflow`
- `numpy`, `pandas`, `matplotlib`, `pillow`
- `scikit-learn`
- `yfinance`, `mplfinance`, `opencv-python`

### 2. Generate Training Data (optional if you already have `training_data/`)

The data generator script creates candlestick chart images for different pattern + trend combinations.

From `model_training/`:

```bash
python data_generator.py
```

This will:
- Create a `training_data/` directory
- For each pattern in `PATTERNS` and trend in `TRENDS`, generate images into class folders like `hammer_uptrend`, `doji_sideways`, etc.
- Create a `labels.txt` mapping class indices to names.

You can control sample size by editing `samples_per_class` in the `__main__` block of `data_generator.py`.

### 3. Train the Edge Sentinel Model

From `model_training/`:

```bash
python train_model.py --data-dir training_data --output-dir models
```

What this does:
- Loads and splits the dataset into train/val/test
- Builds a CNN defined in `train_model.py`
- Trains with data augmentation and callbacks (checkpointing, early stopping, LR reduction)
- Saves:
  - `models/best_model.keras`
  - `models/edge_sentinel_model.keras`
  - `models/model_unquant.tflite`  ✅ (used by the mobile/web client)
  - `models/labels.txt`
  - `models/training_history.json`
  - `models/model_info.json`

Training can take several minutes depending on your hardware and dataset size.

### 4. Test the Trained Model

From `model_training/`:

```bash
python test_model.py --model models/model_unquant.tflite --labels models/labels.txt --test-dir training_data
```

You can also test a single image:

```bash
python test_model.py --model models/model_unquant.tflite --labels models/labels.txt --image path/to/chart.jpg
```

The script prints:
- Top‑k predictions for single images
- Approximate accuracy over a sample of images in the test directory.

### 5. Connecting Model Outputs to the App

- The **React Native / web client** expects:
  - `model_unquant.tflite`
  - `labels.txt`
- In this repo, the frontend’s `assets/` folder already contains:
  - `frontend/assets/model_unquant.tflite`
  - `frontend/assets/labels.txt`
- To update the model used by the app:
  1. Train a new model as above.
  2. Copy the new `model_unquant.tflite` and `labels.txt` from `model_training/models/` into `frontend/assets/`.

The current web frontend uses a **placeholder Edge Sentinel service** (`frontend/src/services/EdgeSentinel.ts`) to simulate chart detection; for full on-device inference you would integrate TensorFlow.js and load the converted model.

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

1. Open `http://localhost:3000` in your browser
2. You should see the welcome message from FinSights AI
3. Chat interface should be visible

#### Test 2: Text Query

1. Type: "What is the price of TCS.NS?"
2. Press Send or Enter
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

1. Click the image icon (📷) in chat
2. Select a financial chart image from your computer
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
2. Refresh the browser
3. Ask: "Show my portfolio"
4. **Expected:** Portfolio data persists (SQLite on backend)

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
3. Check browser console for CORS errors
4. Verify backend CORS is configured correctly

#### Issue: Vite dev server won't start

**Solution:**
```bash
# Clear cache and node_modules
rm -rf node_modules package-lock.json
npm install

# Or check if port 3000 is in use
# Use a different port: npm run dev -- --port 3001
```

#### Issue: Build fails

**Solution:**
```bash
# Check TypeScript errors
npm run build

# Fix any type errors shown
```

#### Issue: Image upload not working

**Solution:**
1. Check browser console for errors
2. Verify file input is working (check browser permissions)
3. Ensure image file is valid (jpg, png, etc.)

### Model Issues

#### Issue: Edge Sentinel not detecting charts

**Solution:**
- Current implementation uses placeholder logic
- For production, integrate actual TensorFlow.js model
- Check `frontend/src/services/EdgeSentinel.ts` for implementation

#### Issue: Model integration

**Solution:**
- Current implementation uses simulated detection
- For production, convert TensorFlow Lite model to TensorFlow.js format
- Load model in browser using TensorFlow.js

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
- [ ] App loads in browser at `http://localhost:3000`
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
- **Edge Sentinel:** < 0.1 seconds (browser-based)
- **Database Queries:** < 50ms
- **Frontend Load Time:** < 2 seconds

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


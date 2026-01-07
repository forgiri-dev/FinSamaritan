## FinSamaritan: Complete Setup & Testing Guide

This comprehensive guide will walk you through setting up and testing the entire FinSamaritan application, including:
- **Backend API** (FastAPI + Gemini integration)
- **Frontend web app** (Vite/React)
- **Model training pipeline** (Edge Sentinel: data generation + TensorFlow training)

---

## 📋 Table of Contents

0. [End-to-End Setup (Windows, Recommended)](#-end-to-end-setup-windows-recommended)
1. [Prerequisites](#-prerequisites)
2. [Backend Setup](#-backend-setup-fastapi--gemini)
3. [Frontend Setup](#-frontend-setup-vitereact)
4. [Model Training & Verification](#-model-training--verification-edge-sentinel)
   - [Quick Start: Simple Training (Recommended)](#-quick-start-simple-training-recommended)
   - [Advanced: Full TensorFlow Training (Alternative)](#-advanced-full-tensorflow-training-alternative)
5. [Testing Guide](#-testing-guide)
6. [Troubleshooting](#-troubleshooting)
7. [Verification Checklist](#-verification-checklist)
8. [Performance Benchmarks](#-performance-benchmarks)
9. [Next Steps](#-next-steps)
10. [Support](#-support)

## 🧭 End-to-End Setup (Windows, Recommended)

This section gives you a **“do these in order”** flow for a fresh Windows machine using PowerShell.
After completing these steps, your **backend, frontend, and model training** environments will all be ready.

### Step 0: Open PowerShell in the Project Root

```powershell
cd "C:\Users\<YourUser>\FinSamaritan"
```

Replace `<YourUser>` with your Windows username.

### Step 1: Create and Activate the Python Virtual Environment

```powershell
# Create venv with Python 3.11
py -3.11 -m venv .venv

# Allow script execution for this session only
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate venv
.\.venv\Scripts\Activate.ps1

python --version  # should show 3.11.x
```

Keep this terminal open and activated whenever you work on the backend or model training.

### Step 2: Install Backend Dependencies

From the same activated venv:

```powershell
cd backend

# Recommended: use the pinned requirements file
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

> You can also use the helper script on Windows:
> ```powershell
> .\install_requirements.bat
> ```

### Step 3: Configure GEMINI_API_KEY

Still in the same PowerShell session:

```powershell
$env:GEMINI_API_KEY = "your-gemini-api-key-here"
```

To confirm:

```powershell
$env:GEMINI_API_KEY
```

### Step 4: (Optional) Pre-generate Stock Cache Data

```powershell
cd backend
python stock_generator.py
```

This creates or refreshes `stock_data.csv` with Top 50 Nifty stocks used by the backend.

### Step 5: Start the Backend Server

From `backend/` in the activated venv:

```powershell
# Option A: use uvicorn directly
uvicorn main:app --reload

# Option B: use helper script
.\start_backend.bat
```

Leave this terminal running. The API will be available at `http://localhost:8000`.

### Step 6: Start the Frontend (New PowerShell Window)

Open a **new** PowerShell window (frontend does not need the Python venv):

```powershell
cd "C:\Users\<YourUser>\FinSamaritan\frontend"

# Install Node dependencies (first time only)


# Start Vite dev server
npm run dev
```

Once it starts, open `http://localhost:3000` in your browser.

### Step 7: (Optional but Recommended) Train / Re-train the Model

Back in your **first** PowerShell window where the Python venv is activated:

```powershell
cd "C:\Users\<YourUser>\FinSamaritan\model_training"

# Install ML dependencies (one-time)
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# If you don't already have training data:
python data_generator.py

# Simple Keras-based training (recommended)
python train_simple.py --data-dir training_data --output-dir models

# Convert to TFLite
python convert_to_tflite.py --model-path models/model.keras --output-path models/model_unquant.tflite
```

> On Windows you can also use the convenience script:
> ```powershell
> .\train.bat
> ```

### Step 8: Run the Tests

With backend and frontend both running:

- **Backend tests:** follow the commands in the [Testing Guide](#-testing-guide) (health check, agent queries, chart analysis).
- **Frontend tests:** follow the manual tests in the [Frontend Testing](#frontend-testing) section.
- **Model tests:** optionally run `test_model.py` from `model_training/` or use the sample inference snippets in the Model Training section.

When all these pass, your FinSamaritan environment is fully set up end-to-end.

## ⚡ Quick Model Training Summary

**Recommended Approach (Simple & Error-Free):**
1. Install dependencies: `pip install -r requirements.txt`
2. Train model: `python train_simple.py --data-dir training_data --output-dir models`
3. Convert to TFLite: `python convert_to_tflite.py --model-path models/model.keras --output-path models/model_unquant.tflite`

**Why Simple Training?**
- ✅ Uses Keras Sequential API (much easier than raw TensorFlow)
- ✅ Error-free and straightforward
- ✅ Standard formats (Keras → TFLite)
- ✅ No complex configuration needed
- ✅ Faster training (10 epochs default vs 50)

See the [Model Training section](#-model-training--verification-edge-sentinel) for detailed instructions.

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

The `model_training/` folder contains everything needed to **generate training data** and **train the Edge Sentinel CNN** that produces model weights and `labels.txt` for use in the application.

### ⚡ Quick Start: Simple Training (Recommended)

**For a simple, error-free training experience**, use the new `train_simple.py` script which uses Keras (much simpler than raw TensorFlow):

**Step 1: Install Dependencies**

```bash
cd model_training
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

**Step 2: Train the Model**

```bash
python train_simple.py --data-dir training_data --output-dir models
```

This will:
- Load images from `training_data/` directory
- Train a simple Keras CNN model
- Save `models/model.keras` (Keras format)
- Save `models/labels.txt` (Class labels)
- Save `models/model_metadata.json` and `models/training_history.json`

**Step 3: Convert to TFLite Format**

After training, convert the model to TensorFlow Lite format:

```bash
python convert_to_tflite.py --model-path models/model.keras --output-path models/model_unquant.tflite
```

This creates `models/model_unquant.tflite` ready for your frontend!

**Why use this approach?**
- ✅ Uses Keras Sequential API (much simpler than raw TensorFlow)
- ✅ Error-free and straightforward
- ✅ Standard formats (Keras → TFLite)
- ✅ No complex configuration needed

---

### 🔧 Advanced: Full TensorFlow Training (Alternative)

If you prefer the full TensorFlow implementation, you can use `train_model.py`:

### 1. Model Training Environment Setup

**Important:** The advanced training script (`train_model.py`) uses **TensorFlow** (low-level API, no Keras). TensorFlow is required for training.

You can either:
- **Reuse the same `.venv`** as the backend (recommended), or
- Create a **separate venv** if you want to isolate ML dependencies.

**Reuse same venv (from project root):**

```bash
cd model_training
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

This installs:
- `tensorflow` (required for model training)
- `numpy`, `pandas`, `matplotlib`, `pillow`, `scipy` (core scientific computing)
- `scikit-learn` (for data splitting)
- `yfinance`, `mplfinance` (for data generation)

> **Note:** TensorFlow is required for training. The advanced script uses TensorFlow's low-level API (not Keras) for a lightweight implementation. For simpler training, use `train_simple.py` instead.

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

> **Note:** If you encounter Yahoo Finance rate limiting (429 errors), the script will automatically fall back to synthetic data generation, so training data will still be created successfully.

### 3. Train the Edge Sentinel Model

**Option A: Simple Training (Recommended)**

From `model_training/`:

```bash
python train_simple.py --data-dir training_data --output-dir models
```

**What this does:**
- Loads and splits the dataset into train/test (80/20)
- Builds and trains a simple CNN using **Keras Sequential API**
- Uses Adam optimizer with sparse categorical crossentropy loss
- Saves:
  - `models/model.keras` ✅ (Keras model format)
  - `models/labels.txt` ✅ (Class labels mapping)
  - `models/model_metadata.json` ✅ (Model information)
  - `models/training_history.json` ✅ (Training metrics per epoch)

**Then convert to TFLite:**
```bash
python convert_to_tflite.py --model-path models/model.keras --output-path models/model_unquant.tflite
```

This creates:
- `models/model_unquant.tflite` ✅ (TensorFlow Lite format for mobile/web)

**Training output:**
- Progress printed each epoch: loss, accuracy, validation loss, validation accuracy
- Final test set evaluation
- Model saved in Keras format, then converted to TFLite

**Option B: Advanced TensorFlow Training**

From `model_training/`:

```bash
python train_model.py --data-dir training_data --output-dir models
```

**What this does:**
- Loads and splits the dataset into train/val/test
- Builds and trains a CNN using **TensorFlow** (no Keras, uses low-level TensorFlow API)
- Uses TensorFlow's automatic differentiation and Adam optimizer
- Saves:
  - `models/edge_sentinel_model/` ✅ (TensorFlow SavedModel format)
  - `models/model_unquant.tflite` ✅ (TensorFlow Lite format for mobile/web)
  - `models/edge_sentinel_model_weights.npz` ✅ (NumPy weights for compatibility)
  - `models/labels.txt` ✅ (Class labels mapping)
  - `models/training_history.json` ✅ (Training metrics per epoch)
  - `models/model_info.json` ✅ (Model metadata and test accuracy)

**Training output:**
- Progress printed each epoch: loss, accuracy, validation loss, validation accuracy
- Final test set evaluation with top-3 accuracy
- Model saved in TensorFlow SavedModel and TFLite formats

Training can take several minutes to hours depending on your hardware and dataset size. The simple Keras approach is faster and easier to use.

### 4. Test the Trained Model

You can test the trained model using TensorFlow or TFLite:

**Option A: Test with TensorFlow SavedModel:**

```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.saved_model.load('models/edge_sentinel_model')

# Load labels
labels = []
with open('models/labels.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            labels.append(parts[1])

# Test image
img = Image.open('path/to/test_image.jpg').convert('RGB')
img = img.resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0
img_batch = np.expand_dims(img_array, axis=0)
img_tensor = tf.constant(img_batch, dtype=tf.float32)

# Predict
logits = model(img_tensor)
probs = tf.nn.softmax(logits).numpy()
pred_idx = np.argmax(probs[0])
print(f"Predicted: {labels[pred_idx]} ({probs[0][pred_idx]:.2%})")
```

**Option B: Test with TFLite (for mobile/web):**

The `test_model.py` script should work with the generated `.tflite` file.

### 5. Using the Trained Model

**For Python inference (TensorFlow):**
```python
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.saved_model.load('models/edge_sentinel_model')

# Load labels
labels = []
with open('models/labels.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(' ', 1)
        if len(parts) == 2:
            labels.append(parts[1])

# Preprocess image
img = Image.open('chart.jpg').convert('RGB')
img = img.resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0
img_batch = np.expand_dims(img_array, axis=0)
img_tensor = tf.constant(img_batch, dtype=tf.float32)

# Predict
logits = model(img_tensor)
probs = tf.nn.softmax(logits).numpy()
pred_idx = np.argmax(probs[0])
print(f"Class: {labels[pred_idx]}, Confidence: {probs[0][pred_idx]:.2%}")
```

**For frontend integration:**
- The current web frontend uses a **placeholder Edge Sentinel service** (`frontend/src/services/EdgeSentinel.ts`)
- To use the actual model, you can:
  1. Use the `.tflite` file with TensorFlow.js Lite runtime, OR
  2. Create a backend API endpoint that loads the TensorFlow SavedModel and performs inference server-side
  3. Update the frontend to call this endpoint

**Model files location:**
- All model files are saved in `model_training/models/`
- Copy `labels.txt` and `model_unquant.tflite` to `frontend/assets/` if needed
- The TensorFlow SavedModel can be used directly in Python applications
- The `.tflite` file can be used in mobile/web applications

### 6. Quick Start: What to Do Now

**Step-by-step workflow (Simple Approach - Recommended):**

1. **Activate your virtual environment:**
   ```powershell
   cd "C:\Users\Zaid Iqbal\FinSamaritan"
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install model training dependencies:**
   ```powershell
   cd model_training
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -r requirements.txt
   ```

3. **Generate training data (if you don't have it):**
   ```powershell
   python data_generator.py
   ```
   - This creates `training_data/` with labeled candlestick chart images
   - If Yahoo Finance rate limits you, it will automatically use synthetic data

4. **Train the model (Simple Keras approach):**
   ```powershell
   python train_simple.py --data-dir training_data --output-dir models
   ```
   - Watch for epoch-by-epoch progress
   - Training completes when you see "🎉 Training complete!"
   - Model files are saved in `models/` directory

5. **Convert to TFLite format:**
   ```powershell
   python convert_to_tflite.py --model-path models/model.keras --output-path models/model_unquant.tflite
   ```
   - This converts the Keras model to TFLite format
   - Creates `models/model_unquant.tflite` ready for frontend use

6. **Verify training succeeded:**
   ```powershell
   # Check that these files exist:
   ls models/model.keras
   ls models/model_unquant.tflite
   ls models/labels.txt
   ls models/training_history.json
   ls models/model_metadata.json
   ```

7. **Test the model (optional):**
   - Create a simple test script as shown in section 4 above
   - Or load the model in Python and test with sample images

**Alternative: Advanced TensorFlow Approach**

If you prefer the full TensorFlow implementation:

```powershell
python train_model.py --data-dir training_data --output-dir models
```

This uses raw TensorFlow (more complex but more control).

**Expected training output (Simple Approach):**
```
🚀 Starting Simple Model Training (Keras)
============================================================

📂 Loading dataset...
📁 Found 24 classes
  📊 hammer_uptrend: 200 images
  ...

✂️ Splitting dataset...
  Training: 3840 samples
  Testing: 960 samples

🏗️ Creating model (Keras CNN)...
📊 Total parameters: 1,234,567

🎓 Training model...
============================================================
Epoch 1/10 - loss: 2.3456 - accuracy: 0.1234 - val_loss: 2.1234 - val_accuracy: 0.2345
Epoch 2/10 - loss: 1.9876 - accuracy: 0.3456 - val_loss: 1.8765 - val_accuracy: 0.4567
...

📊 Evaluating model...
  Training Accuracy: 0.8234
  Test Accuracy: 0.7890

💾 Saving model...
✅ Saved Keras model: models/model.keras
✅ Saved metadata: models/model_metadata.json
✅ Saved labels: models/labels.txt
✅ Saved training history: models/training_history.json

============================================================
🎉 Training complete!
📁 Model saved to: models
📦 Model format: Keras (.keras)
💡 Run convert_to_tflite.py to convert to .tflite format
============================================================
```

**Then convert to TFLite:**
```
🔄 Converting Keras model to TensorFlow Lite...
============================================================
📂 Loading model from: models/model.keras
✅ Model loaded successfully

🔄 Converting to TFLite format...
✅ TFLite model saved: models/model_unquant.tflite
📦 Model size: 4.56 MB

============================================================
🎉 Conversion complete!
📁 TFLite model: models/model_unquant.tflite
💡 You can now use this model in your frontend
============================================================
```

**Expected training output (Advanced TensorFlow Approach):**
```
🚀 Starting Edge Sentinel Model Training (TensorFlow)
============================================================

📂 Loading dataset...
📁 Found 24 classes
  📊 hammer_uptrend: 200 images
  ...

✂️ Splitting dataset...
  Training: 4320 samples
  Validation: 1080 samples
  Testing: 600 samples

🏗️ Creating model architecture (TensorFlow)...
📊 Total parameters: 2,123,456

🎓 Starting training...
============================================================
Epoch 1/50 - loss: 2.3456 - acc: 0.1234 - val_loss: 2.1234 - val_acc: 0.2345
Epoch 2/50 - loss: 1.9876 - acc: 0.3456 - val_loss: 1.8765 - val_acc: 0.4567
...

📊 Evaluating on test set...
  Test Accuracy: 0.8234
  Test Top-3 Accuracy: 0.9456

💾 Saving model...
✅ Saved TensorFlow model: models/edge_sentinel_model
✅ Saved model weights: models/edge_sentinel_model_weights.npz
🔄 Converting to TFLite...
✅ Saved TFLite model: models/model_unquant.tflite
✅ Saved labels: models/labels.txt
✅ Saved training history: models/training_history.json
✅ Saved model info: models/model_info.json

============================================================
🎉 Training complete!
📁 Models saved to: models
📦 Model format: TensorFlow SavedModel + TFLite
============================================================
```

**Troubleshooting:**
- If training fails, check the error message
- Common issues are covered in the "Model Issues" section below
- For simple training, ensure TensorFlow/Keras is installed: `python -c "import tensorflow as tf; print(tf.__version__)"`
- For advanced training, ensure TensorFlow is properly installed: `python -c "import tensorflow as tf; print(tf.__version__)"`
- If you get import errors, make sure scikit-learn is installed: `pip install scikit-learn`

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

#### Issue: TensorFlow/Keras import errors during training

**Solution:**
- Ensure TensorFlow is installed: `python -m pip install tensorflow`
- Check TensorFlow version: `python -c "import tensorflow as tf; print(tf.__version__)"`
- For simple training (`train_simple.py`), TensorFlow 2.x includes Keras automatically
- For advanced training (`train_model.py`), ensure you're using TensorFlow 2.x (the script uses low-level TensorFlow API)
- For GPU support, install `tensorflow-gpu` instead of `tensorflow`
- Ensure scikit-learn is installed: `pip install scikit-learn`

#### Issue: Training is slow

**Solution:**
- TensorFlow will use GPU automatically if available
- For faster training, consider:
  - Using `train_simple.py` (simpler, faster) instead of `train_model.py`
  - Reducing `samples_per_class` in `data_generator.py`
  - Reducing `EPOCHS` in training script (default is 10 for simple, 50 for advanced)
  - Using a smaller `BATCH_SIZE` if memory is limited
  - Enabling GPU support if you have a compatible GPU

#### Issue: TFLite conversion fails

**Solution:**
- Make sure you've trained the model first using `train_simple.py`
- Check that `models/model.keras` exists
- Verify TensorFlow Lite converter is available: `python -c "import tensorflow as tf; print(tf.lite)"`
- If conversion fails, the Keras model can still be used directly in Python

#### Issue: Model files not found after training

**Solution:**
- Check that training completed successfully (look for "🎉 Training complete!" message)
- For simple training, verify files exist in `model_training/models/`:
  - `model.keras` (Keras model format)
  - `model_unquant.tflite` (after running convert_to_tflite.py)
  - `labels.txt` (Class labels)
  - `training_history.json` (Training metrics)
  - `model_metadata.json` (Model metadata)
- For advanced training, verify files exist:
  - `edge_sentinel_model/` (TensorFlow SavedModel directory)
  - `model_unquant.tflite` (TFLite format)
  - `edge_sentinel_model_weights.npz` (NumPy weights for compatibility)
  - `labels.txt` (Class labels)
  - `training_history.json` (Training metrics)
  - `model_info.json` (Model metadata)
- If TFLite file is missing, run `convert_to_tflite.py` after simple training

#### Issue: Edge Sentinel not detecting charts (frontend)

**Solution:**
- Current implementation uses placeholder logic in `frontend/src/services/EdgeSentinel.ts`
- To use the actual trained model:
  1. Load the TensorFlow SavedModel or TFLite model server-side (in backend)
  2. Create an API endpoint for image classification
  3. Update frontend to call this endpoint
  4. OR use the `.tflite` file with TensorFlow.js Lite runtime for client-side inference

#### Issue: Cannot load model for inference

**Solution:**
```python
# Make sure you're loading the correct file format
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Option 1: Load Keras model (from train_simple.py)
model = keras.models.load_model('models/model.keras')

# Option 2: Load TensorFlow SavedModel (from train_model.py)
model = tf.saved_model.load('models/edge_sentinel_model')

# Option 3: Load TFLite model (for mobile/web)
interpreter = tf.lite.Interpreter(model_path='models/model_unquant.tflite')
interpreter.allocate_tensors()

# Option 4: Load NumPy weights (for compatibility - requires model reconstruction)
data = np.load('models/edge_sentinel_model_weights.npz')
# Note: You'll need to reconstruct the model architecture and load weights manually
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


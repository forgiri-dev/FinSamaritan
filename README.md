# 🚀 FinSamaritan: Smart Portfolio Manager with AI Agent Overlay

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-2.5-orange.svg)](https://ai.google.dev/)
[![AI](https://img.shields.io/badge/AI-Agentic%20AI-green.svg)](https://ai.google.dev/)
[![Flutter](https://img.shields.io/badge/Flutter-Mobile%20%26%20Web-blue.svg)](https://flutter.dev/)

**Revolutionize Your Investing: AI-Powered Portfolio Management in One Unified Dashboard!** 💡📈

## 🏆 Team FinSamaritan

- **Harsh Giri** (Team Leader) 
- **Malika Parveen** 
- **Zaid Iqbal** 
- **Lakshay Garg** 

## 🔥 The Problem

Investors today are overwhelmed—juggling **10+ different tabs** for news, charts, screeners, and analysis tools. Switching between platforms wastes time, increases errors, and misses critical insights. Traditional portfolio managers lack intelligence, while AI tools are fragmented and expensive. 📉😩

## ✨ The Solution

**FinSamaritan** is your AI-powered financial companion: a unified dashboard with an intelligent agent that handles the heavy lifting. Search stocks, manage portfolios, and get expert-level advice—all in one place. Our AI "Brain" uses custom tools for personalized financial guidance, while dual-engine analysis delivers unparalleled accuracy. 🎯🤖

## 🛠️ Tech Stack

| Component | Technology | Why It Rocks |
|-----------|------------|--------------|
| 🤖 **AI Brain** | Google Gemini 2.5 | Lightning-fast agentic AI with custom tools for financial advice |
| 🧠 **Custom ML** | Edge Sentinel (TensorFlow Lite) | Local model for instant technical analysis |
| 👁️ **Vision AI** | Google Gemini Vision | Advanced chart interpretation for support/resistance |
| 🐍 **Backend** | Python + FastAPI | Scalable API with native AI/ML support |
| 📱 **Frontend** | Flutter (Dart) | Cross-platform magic: Android, iOS, Web from one codebase |
| 💾 **Database** | CSV-based (Pandas) | Lightning-fast in-memory stock data for hackathon speed |

## 🔄 How It Works

1. **📊 Portfolio Management**: Search and manage holdings from our local CSV database of top stocks
2. **🧠 AI Agent Activation**: Query in natural language—"Show me undervalued tech stocks under $50"
3. **📈 Dual-Engine Analysis**: Upload a chart → Edge Sentinel processes patterns + Gemini Vision analyzes visuals → Combined insights delivered instantly
4. **💡 Smart Advice**: Get personalized recommendations with reasoning traces

**Flow**: Upload Chart → Dual Processing (Local ML + Cloud Vision) → AI-Powered Result 📊➡️🤖➡️💡

## 📊 Edge Sentinel
**Edge Sentinel** is our proprietary machine learning model that brings AI-powered technical analysis directly to your device. Trained on thousands of candlestick patterns, it provides instant, privacy-preserving insights without relying on cloud services.

### 🚀 Capabilities
- **Pattern Recognition**: Detects 12+ candlestick patterns including:
  - **Reversal Patterns**: Hammer, Shooting Star, Morning Star, Evening Star
  - **Continuation Patterns**: Doji, Engulfing (Bullish/Bearish)
  - **Complex Patterns**: Multiple candlestick formations
- **Trend Context**: Analyzes patterns in different market contexts:
  - 📈 **Uptrend**: Bullish reversals, continuations
  - 📉 **Downtrend**: Bearish reversals, continuations  
  - ➡️ **Sideways**: Range-bound market signals
- **Real-time Analysis**: Processes charts instantly on-device
- **Accuracy Boost**: Complements Gemini Vision for dual-engine precision

### 🏗️ Technical Details
- **Framework**: TensorFlow Lite for edge deployment
- **Training Data**: 10,000+ labeled candlestick images across 15 pattern categories
- **Model Size**: Lightweight (< 5MB) for mobile optimization
- **Inference Speed**: < 100ms per analysis
- **Privacy**: All processing happens locally - no data sent to servers

### 🎯 Why Edge Sentinel?
- **⚡ Speed**: Instant results without network latency
- **🔒 Privacy**: Your charts never leave your device
- **💰 Cost-Effective**: No API calls for basic pattern recognition
- **🔄 Offline**: Works without internet connection
- **🤝 Synergy**: Pairs perfectly with Gemini Vision for comprehensive analysis

**Training Pipeline**: Raw chart images → Data augmentation → CNN feature extraction → Pattern classification → TFLite conversion → Edge deployment


##  Monorepo Structure

```
FinSamaritan/
├── 📄 generate_stock_data.py          # Stock data generation script
├── 📄 new__version                    # Version notes
├── 📄 QUICK_START.md                  # Quick start guide
├── 📄 README.md                       # This file
├── 📄 SETUP_GUIDE.md                  # Detailed setup guide
├── 🔧 backend/                        # Python FastAPI backend
│   ├── 📄 agent_tools.py              # AI agent custom tools
│   ├── 📄 check_gemini_api.py         # API key validation
│   ├── 📄 data_engine.py              # Data processing engine
│   ├── 📄 database.py                 # Database utilities
│   ├── 📄 DEBUG_ERRORS.md             # Debug documentation
│   ├── 📄 main.py                     # FastAPI server entry point
│   ├── 📄 portfolio.json              # Portfolio data
│   ├── 📄 requirements.txt            # Python dependencies
│   ├── 📄 restart_server.ps1         # Windows server restart script
│   ├── 📄 stock_data_generator.py     # Generate stock CSV data
│   ├── 📄 stock_data.csv              # Generated stock database
│   ├── 📄 test_backend.py             # Backend tests
│   ├── 📄 tools.py                    # Utility functions
│   ├── 📄 TROUBLESHOOTING.md          # Backend troubleshooting
│   └── 📄 .env.example                # Environment variables template
├── 📱 frontend/                       # Flutter cross-platform app
│   ├── 📄 analysis_options.yaml       # Dart analysis config
│   ├── 📄 pubspec.yaml                # Flutter dependencies
│   ├── 📱 android/                    # Android platform files
│   ├── 🍎 ios/                        # iOS platform files
│   ├── 🐧 linux/                      # Linux platform files
│   ├── 🍏 macos/                      # macOS platform files
│   ├── 🌐 web/                        # Web platform files
│   ├── 🧪 test/                       # Flutter tests
│   └── 📱 lib/                        # Flutter source code
│       ├── 📄 main.dart               # App entry point
│       ├── 📱 screens/                # UI screens
│       ├── 🔧 services/               # API services
│       └── 🧩 widgets/                # Reusable UI components
├── 🤖 model_training/                 # ML model training & Edge Sentinel
│   ├── 📄 convert_to_tflite.py        # Convert to TensorFlow Lite
│   ├── 📄 data_generator.py           # Training data generation
│   ├── 📄 IMPLEMENTATION_SUMMARY.md   # Implementation details
│   ├── 📄 QUICK_START.md              # Training quick start
│   ├── 📄 README.md                   # Training documentation
│   ├── 📄 requirements.txt            # Training dependencies
│   ├── 📄 SIMPLE_TRAINING_GUIDE.md    # Simple training guide
│   ├── 📄 test_model.py               # Model testing
│   ├── 📄 train_model.ipynb           # Jupyter training notebook
│   ├── 📄 train_model.py              # Training script
│   ├── 📄 train_simple.py             # Simplified training
│   ├── 📄 train.bat                   # Windows training batch
│   ├── 📄 train.sh                    # Linux/Mac training script
│   ├── 📄 TRAINING_GUIDE.md           # Comprehensive training guide
│   ├── 🤖 models/                     # Trained models
│   │   ├── 📄 labels.txt              # Model labels
│   │   ├── 📄 model_metadata.json     # Model metadata
│   │   ├── 📄 model_unquant.tflite    # Edge Sentinel model
│   │   └── 📄 training_history.json   # Training metrics
│   └── 📊 training_data/              # Training datasets
│       ├── 📄 labels.txt              # Data labels
│       ├── 📊 doji_downtrend/         # Doji pattern data
│       ├── 📊 doji_sideways/
│       ├── 📊 doji_uptrend/
│       ├── 📊 engulfing_bearish_downtrend/
│       ├── 📊 engulfing_bearish_sideways/
│       ├── 📊 engulfing_bearish_uptrend/
│       ├── 📊 engulfing_bullish_downtrend/
│       ├── 📊 engulfing_bullish_sideways/
│       ├── 📊 engulfing_bullish_uptrend/
│       ├── 📊 evening_star_downtrend/
│       ├── 📊 evening_star_sideways/
│       ├── 📊 evening_star_uptrend/
│       ├── 📊 hammer_downtrend/
│       ├── 📊 hammer_sideways/
│       ├── 📊 hammer_uptrend/
│       ├── 📊 morning_star_downtrend/
│       ├── 📊 morning_star_sideways/
│       ├── 📊 morning_star_uptrend/
│       ├── 📊 shooting_star_downtrend/
│       ├── 📊 shooting_star_sideways/
│       ├── 📊 shooting_star_uptrend/
│       └── 📊 normal_downtrend/       # Normal patterns
└── 📂 flutter/                        # Flutter SDK (if extracted here)
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11+
- Flutter SDK
- Google Gemini API Key

### Quick Start (5 Minutes!)

1. **Clone & Navigate**:
   ```bash
   git clone <your-repo>
   cd FinSamaritan
   ```

2. **Backend Setup**:
   ```bash
   cd backend
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   # Edit .env with your GEMINI_API_KEY
   python main.py
   ```

3. **Frontend Setup**:
   ```bash
   cd ../frontend
   flutter pub get
   flutter run  # Choose your platform
   ```

4. **🎉 Done!** Open the app and start managing your portfolio with AI!

### Environment Variables
Create `backend/.env`:
```
GEMINI_API_KEY=your_api_key_here
```


## 🎯 Key Features

- **📱 Unified Dashboard**: No more tab-switching—everything in one app
- **🧠 AI Agent**: Conversational financial advice with custom tools
- **📊 Portfolio Manager**: Search, track, and manage stock holdings
- **🔍 Dual-Engine Analysis**: Local ML + Cloud Vision for unbeatable accuracy
- **⚡ Fast & Local**: CSV-based backend for instant responses
- **🌐 Cross-Platform**: Flutter powers Android, iOS, and Web

## 🐛 Troubleshooting

- **API Key Issues**: Ensure `.env` is in `backend/` with correct key
- **Connection Errors**: Check backend is running on port 8000
- **Flutter Issues**: Run `flutter doctor` and ensure devices are connected

## 📝 API Endpoints

- `POST /agent` - AI-powered stock screening
- `POST /analyze-chart` - Dual-engine chart analysis
- `POST /portfolio` - Manage holdings
- 
---

<p align="center">
  Built with ❤️ by CACHE CAT
</p>






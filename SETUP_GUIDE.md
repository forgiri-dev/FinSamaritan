# FinSamaritan - Complete Setup & Usage Guide

**Portfolio Manager with AI Agent Overlay**

This guide will walk you through setting up and running FinSamaritan on **Google Chrome (Web)**, **Windows Desktop**, and **Android**.

## 🆕 What's New

FinSamaritan has been refactored into a **Portfolio Manager** with an **AI Agent overlay**:

- **Portfolio Screen**: Main dashboard for managing your stock portfolio
- **AI Chat Assistant**: Global chat overlay accessible from anywhere (floating button)
- **Chart Analysis**: Dual-processing with Edge Sentinel (local model) + Gemini Vision
- **Stock Search**: Search stocks from CSV database and add to portfolio
- **Portfolio Analytics**: Real-time P&L calculations and performance tracking

---

## 📋 Prerequisites

Before starting, make sure you have:

1. **Python 3.8+** installed
   - Check: `python --version` or `python3 --version`
   - Download: https://www.python.org/downloads/

2. **Flutter SDK** installed
   - Check: `flutter --version`
   - Download: https://docs.flutter.dev/get-started/install

3. **Google Gemini API Key**
   - Get it from: https://makersuite.google.com/app/apikey
   - Sign in with Google account and create a new API key

4. **Node.js** (optional, for some Flutter tools)

---

## 🔧 Part 1: Backend Setup (Required for All Platforms)

The backend must be running for the app to work on any platform.

### Step 1: Navigate to Backend Directory
```powershell
cd path\to\FinSamaritan\backend
# Replace with your actual path, e.g.:
# cd C:\Users\YourName\FinSamaritan\backend
```

### Step 2: Set Up Python Virtual Environment

**Option A: Using Python directly (Recommended for Windows)**
```powershell
# Create virtual environment
python -m venv ..\.venv

# Activate it
..\.venv\Scripts\activate
```

**Option B: If you get execution policy errors, use Python directly:**
```powershell
# Skip activation, use Python directly from venv
# We'll use the full path instead
```

### Step 3: Install Dependencies
```powershell
# If virtual environment is activated:
pip install -r requirements.txt

# OR if using Python directly:
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 4: Create .env File
Create a file named `.env` in the `backend` directory:

**File: `backend\.env`**
```
GEMINI_API_KEY=your_actual_api_key_here
```

Replace `your_actual_api_key_here` with your actual Gemini API key.

### Step 5: Install TensorFlow (Optional - For Edge Sentinel Model)
If you want to use the local Edge Sentinel model for chart analysis:

```powershell
# If virtual environment is activated:
pip install tensorflow

# OR using Python directly:
..\.venv\Scripts\python.exe -m pip install tensorflow
```

**Note:** TensorFlow is optional. Chart analysis will still work with Gemini Vision even without it.

### Step 6: Generate Stock Data (If Not Already Done)
```powershell
# If virtual environment is activated:
python stock_data_generator.py

# OR using Python directly:
..\.venv\Scripts\python.exe stock_data_generator.py
```

This will create `stock_data.csv` (may take a few minutes to fetch all stock data).

**Important:** All stock data must come from this CSV file. The app does not fetch live data from external APIs for stock fundamentals.

### Step 7: Start the Backend Server
```powershell
# If virtual environment is activated:
python main.py

# OR using Python directly:
..\.venv\Scripts\python.exe main.py
```

You should see:
```
✓ Using agent model: gemini-2.5-flash
✓ Using vision model: gemini-2.5-pro
✓ Stock data loaded successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**⚠️ Keep this terminal window open!** The backend must stay running.

**Test the backend:**
- Open browser: http://localhost:8000
- You should see: `{"message":"FinSamaritan API - Portfolio Manager with AI Agent",...}`
- API docs: http://localhost:8000/docs

**New API Endpoints:**
- `/search-stocks` - Search stocks by name/symbol
- `/portfolio` - Get portfolio analysis
- `/portfolio/add` - Add stock to portfolio
- `/agent/chat` - AI agent chat with tool calling
- `/analyze-chart` - Dual-processing chart analysis (Edge Sentinel + Gemini)

---

## 🌐 Part 2: Running on Google Chrome (Web)

### Step 1: Navigate to Frontend Directory
Open a **NEW** PowerShell/Command Prompt window (keep backend running):

```powershell
cd path\to\FinSamaritan\frontend
# Replace with your actual path, e.g.:
# cd C:\Users\YourName\FinSamaritan\frontend
```

### Step 2: Verify Flutter is Working

**First, make sure Flutter is recognized:**

```powershell
flutter --version
```

If you see an error, go back to **Flutter Installation** section above and install Flutter.

### Step 3: Install Flutter Dependencies
```powershell
flutter pub get
```

**If you still get "flutter is not recognized":**
- Make sure you opened a NEW PowerShell window after adding Flutter to PATH
- Or restart your computer
- Or use the full path: `C:\src\flutter\bin\flutter.bat pub get`

### Step 4: Check Flutter Setup
```powershell
flutter doctor
```

Make sure you have at least:
- ✅ Flutter (Channel stable)
- ✅ Chrome browser available

### Step 4: Run on Chrome
```powershell
flutter run -d chrome
```

**Or select Chrome when prompted:**
```powershell
flutter run
# Then select Chrome from the device list
```

### Step 5: Using the App in Chrome

The app will open in Chrome automatically. You'll see:

1. **Portfolio Tab** (Portfolio icon) - **Main Dashboard**
   - **Search Bar**: Search stocks by name or symbol from CSV database
   - **Add to Portfolio**: Click "Add" on any search result to add stocks
   - **Portfolio Summary**: View total invested, current value, and P&L
   - **Holdings List**: See all your stocks with real-time performance

2. **Chart Analysis Tab** (Chart icon)
   - Click "Gallery" to upload a chart image
   - Click "Camera" to take a photo (if available)
   - Click "Analyze Chart" for **dual-processing analysis**:
     - **Edge Sentinel** (local model): Pattern and trend detection
     - **Gemini Vision**: Detailed technical analysis
     - **Combined Summary**: Unified insights from both models

3. **AI Chat Assistant** (Floating button - Chat icon)
   - Click the floating chat button (bottom-right) to open AI assistant
   - Ask questions like:
     - "Compare RELIANCE and TCS"
     - "Show me undervalued IT stocks"
     - "Analyze my portfolio"
     - "What's the PE ratio of HDFCBANK?"
   - The AI agent uses tools to fetch data and provide insights

### Troubleshooting Chrome

**Error: "Error connecting to API"**
- Make sure backend is running on http://localhost:8000
- Check backend terminal for errors
- Try refreshing the page (F5)

**App doesn't load**
- Check console: Press F12 → Console tab
- Make sure backend is running
- Verify `api_service.dart` uses `http://localhost:8000`

---

## 🪟 Part 3: Running on Windows Desktop

### Step 1: Enable Windows Desktop Support (If Needed)
```powershell
flutter config --enable-windows-desktop
```

### Step 2: Navigate to Frontend Directory
```powershell
cd path\to\FinSamaritan\frontend
# Replace with your actual path
```

### Step 3: Install Dependencies
```powershell
flutter pub get
```

### Step 4: Run on Windows
```powershell
flutter run -d windows
```

**Or select Windows when prompted:**
```powershell
flutter run
# Then select Windows from the device list
```

### Step 5: Using the Windows App

The app will open as a desktop window. Usage is the same as Chrome:
- Two tabs at the bottom: Portfolio and Chart Analysis
- Floating chat button for AI assistant
- Same features: Portfolio management, Chart analysis, AI chat
- Backend must be running on localhost:8000

### Troubleshooting Windows

**Build errors:**
```powershell
flutter clean
flutter pub get
flutter run -d windows
```

**Missing Visual Studio Build Tools:**
- Install Visual Studio 2022 with "Desktop development with C++" workload
- Or use: `flutter doctor` to see what's missing

---

## 📱 Part 4: Running on Android

### Step 1: Set Up Android Development

**A. Install Android Studio**
- Download: https://developer.android.com/studio
- Install with default settings
- Open Android Studio → Tools → SDK Manager
- Install: Android SDK, Android SDK Platform-Tools, Android SDK Build-Tools

**B. Set Up Android Emulator (Optional but Recommended)**
- Open Android Studio → Tools → Device Manager
- Click "Create Device"
- Select a device (e.g., Pixel 5)
- Download a system image (e.g., Android 13)
- Click "Finish" and start the emulator

**C. Enable USB Debugging (For Physical Device)**
- On your Android phone: Settings → About Phone
- Tap "Build Number" 7 times to enable Developer Options
- Settings → Developer Options → Enable "USB Debugging"
- Connect phone via USB

### Step 2: Check Flutter Setup
```powershell
flutter doctor
```

Make sure you see:
- ✅ Android toolchain
- ✅ Android Studio
- ✅ Android license status (accept licenses if needed)

**Accept Android licenses:**
```powershell
flutter doctor --android-licenses
# Type 'y' for each license
```

### Step 3: Verify Flutter is Installed
```powershell
flutter --version
```

**If you see "flutter is not recognized":**
- Go back to **Flutter Installation** section at the top of this guide
- Or see: [FLUTTER_INSTALL_WINDOWS.md](FLUTTER_INSTALL_WINDOWS.md)

### Step 4: Navigate to Frontend Directory
```powershell
cd path\to\FinSamaritan\frontend
# Replace with your actual path
```

### Step 5: Install Dependencies
```powershell
flutter pub get
```

### Step 6: Check Connected Devices
```powershell
flutter devices
```

You should see your Android device or emulator listed.

### Step 7: Run on Android

**For Android Emulator:**
```powershell
flutter run -d emulator-5554
# Replace emulator-5554 with your device ID from 'flutter devices'
```

**For Physical Device:**
```powershell
flutter run -d <device-id>
# Replace <device-id> with your device ID from 'flutter devices'
```

**Or let Flutter auto-select:**
```powershell
flutter run
# Select Android device from the list
```

### Step 6: Important - Update API URL for Android

**For Android Emulator:**
- The app should automatically use `http://10.0.2.2:8000` (maps to localhost)
- This is already configured in `api_service.dart`

**For Physical Android Device:**
You need to update the API URL to use your computer's IP address:

1. **Find your computer's IP address:**
   ```powershell
   ipconfig
   ```
   Look for "IPv4 Address" under your network adapter (usually 192.168.x.x or 10.x.x.x)

2. **Update `frontend/lib/services/api_service.dart`:**
   
   Open: `frontend/lib/services/api_service.dart`
   
   Find this section (around line 18-22):
   ```dart
   if (platform == 'android') {
     return 'http://10.0.2.2:8000';
   }
   ```
   
   **Option A: Temporary change for physical device testing:**
   Change to (replace with your IP):
   ```dart
   if (platform == 'android') {
     return 'http://192.168.1.100:8000';  // Replace 192.168.1.100 with your IP from ipconfig
   }
   ```
   
   **Option B: Better solution - Add device detection:**
   You can modify the code to detect if it's an emulator vs physical device, but for quick testing, Option A works.

3. **Make sure your phone and computer are on the same WiFi network**

4. **Allow Windows Firewall (if prompted):**
   - Windows may ask to allow Python/FastAPI through firewall
   - Click "Allow access"

5. **Restart the Flutter app:**
   ```powershell
   flutter run -d <device-id>
   ```
   
   **Note:** After testing on physical device, change back to `http://10.0.2.2:8000` for emulator use.

### Step 7: Using the Android App

Once the app installs and opens on your Android device:
- Two tabs at the bottom: Portfolio and Chart Analysis
- Floating chat button for AI assistant
- Same features as web/desktop
- Camera access works for Chart Analysis
- Gallery access works for uploading charts
- Search stocks and manage portfolio on the go

### Troubleshooting Android

**Error: "Error connecting to API" on Physical Device**
- Make sure computer and phone are on same WiFi
- Check Windows Firewall isn't blocking port 8000
- Verify backend is running
- Use your computer's IP address in `api_service.dart`

**"flutter is not recognized":**
- Flutter is not installed or not in PATH
- See **Flutter Installation** section at the top of this guide
- Or see: [FLUTTER_INSTALL_WINDOWS.md](FLUTTER_INSTALL_WINDOWS.md)

**Build fails:**
```powershell
flutter clean
flutter pub get
flutter run -d <device-id>
```

**App crashes on image picker:**
- Check AndroidManifest.xml has permissions (should be auto-added by image_picker package)
- Grant camera/gallery permissions when prompted

**Device not detected:**
- Make sure USB debugging is enabled
- Try different USB cable/port
- Run `adb devices` to check connection
- For emulator: Make sure it's started in Android Studio

---

## 🎯 Quick Test Guide

Test all features on any platform:

### 1. Portfolio Management Test
- Go to Portfolio tab (home screen)
- Search for a stock (e.g., "RELIANCE" or "TCS")
- Click "Add" on a search result
- Enter shares and buy price
- View your portfolio with P&L calculations

### 2. Chart Analysis Test (Dual-Processing)
- Go to Chart Analysis tab
- Click "Gallery" → Select a stock chart image
- Click "Analyze Chart"
- Should show:
  - **Edge Sentinel Analysis**: Pattern and trend with confidence scores
  - **Gemini Vision Analysis**: Detailed technical analysis
  - **Combined Summary**: Unified insights from both models

### 3. AI Chat Assistant Test
- Click the floating chat button (bottom-right)
- Try these queries:
  - "Search for IT stocks"
  - "Compare RELIANCE with TCS"
  - "What's in my portfolio?"
  - "Show me stocks with PE ratio less than 15"
- The AI should respond with relevant data and insights

---

## 🔥 Common Issues & Solutions

### Backend Issues

**Port 8000 already in use:**
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual number)
taskkill /PID <PID> /F
```

**"GEMINI_API_KEY not found"**
- Make sure `.env` file exists in `backend` folder
- Check the file contains: `GEMINI_API_KEY=your_key`
- Restart backend server

**"stock_data.csv not found"**
- Run: `python stock_data_generator.py` in backend folder
- Wait for it to complete (downloads 500+ stocks)
- **Important**: All stock data comes from this CSV file

**"TensorFlow is not installed" (for Edge Sentinel)**
- This is optional - chart analysis still works with Gemini Vision
- To enable Edge Sentinel: `pip install tensorflow`
- The app will gracefully handle missing TensorFlow

### Frontend Issues

**"flutter pub get" fails**
- Run `flutter doctor` to check setup
- Make sure you're in `frontend` directory
- Try: `flutter clean` then `flutter pub get`

**App won't build**
- Check `flutter doctor` for missing components
- Clean build: `flutter clean`
- Re-get dependencies: `flutter pub get`

**Hot reload not working**
- Press `r` in terminal to manually reload
- Press `R` to restart
- Press `q` to quit

---

## 📝 Summary Checklist

### Backend (Do this first, keep running)
- [ ] Python installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] TensorFlow installed (optional, for Edge Sentinel model)
- [ ] `.env` file created with Gemini API key
- [ ] Stock data generated (`stock_data_generator.py`)
- [ ] Backend running on http://localhost:8000
- [ ] Tested in browser: http://localhost:8000/docs
- [ ] Verified new endpoints are working

### Chrome (Web)
- [ ] Flutter SDK installed
- [ ] In `frontend` directory
- [ ] `flutter pub get` completed
- [ ] `flutter run -d chrome` working
- [ ] App opens in Chrome
- [ ] Portfolio search and add tested
- [ ] Chart analysis (dual-processing) tested
- [ ] AI chat assistant tested

### Windows Desktop
- [ ] Windows desktop support enabled
- [ ] Visual Studio Build Tools installed (if needed)
- [ ] `flutter run -d windows` working
- [ ] Desktop app opens
- [ ] Portfolio management tested
- [ ] Chart analysis tested
- [ ] AI chat assistant tested

### Android
- [ ] Android Studio installed
- [ ] Android SDK installed
- [ ] Android licenses accepted
- [ ] Device/emulator connected
- [ ] API URL updated for physical device (if needed)
- [ ] `flutter run -d <device-id>` working
- [ ] App installed and opens
- [ ] Portfolio management tested
- [ ] Chart analysis tested
- [ ] AI chat assistant tested

---

## 🎉 You're All Set!

Once you have the backend running and the frontend app open on your chosen platform, you can start using FinSamaritan!

## 📚 Key Features

### Portfolio Manager
- **Search Stocks**: Query the CSV database by name or symbol
- **Add Holdings**: Add stocks with shares and buy price
- **Real-time P&L**: Automatic calculation of profit/loss
- **Portfolio Analytics**: Track total invested, current value, and performance

### AI Agent Chat
- **Natural Language Queries**: Ask questions in plain English
- **Tool Integration**: AI uses backend tools to fetch data
- **Contextual Responses**: Understands portfolio and stock context
- **Always Accessible**: Floating button available from any screen

### Chart Analysis
- **Dual-Processing**: 
  - Edge Sentinel (local model) for pattern recognition
  - Gemini Vision for detailed technical analysis
- **Combined Insights**: Unified analysis from both models
- **Image Upload**: Gallery or camera support

**Remember:**
- Backend must stay running in a terminal window
- Backend runs on http://localhost:8000
- Each platform (Chrome/Windows/Android) connects to the same backend
- All stock data comes from `stock_data.csv` (no external API calls for fundamentals)
- TensorFlow is optional (chart analysis works with Gemini Vision alone)

Happy portfolio management! 📈💰


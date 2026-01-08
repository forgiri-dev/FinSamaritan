# FinSamaritan - Quick Start Guide

**Fastest way to get the app running!**

---

## 🚀 Quick Start (5 minutes)

### Step 1: Start Backend (Terminal 1)

```powershell
cd C:\Users\124sa\FinSamaritan\backend

# Install dependencies (first time only)
..\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Start server
..\.venv\Scripts\python.exe main.py
```

✅ Wait for: `INFO: Uvicorn running on http://0.0.0.0:8000`

**Keep this terminal open!**

### Step 2: Run Flutter App (Terminal 2)

```powershell
cd C:\Users\124sa\FinSamaritan\frontend

# Install dependencies (first time only)
flutter pub get

# Run on Chrome
flutter run -d chrome
```

✅ App opens in Chrome automatically!

---

## 📱 Platform-Specific Quick Starts

### Chrome (Web) - Easiest!
```powershell
cd frontend
flutter run -d chrome
```

### Windows Desktop
```powershell
cd frontend
flutter run -d windows
```

### Android Emulator
1. Start emulator in Android Studio
2. Then:
```powershell
cd frontend
flutter run
# Select your emulator
```

### Android Physical Device
1. Connect phone via USB (enable USB debugging)
2. Update `frontend/lib/services/api_service.dart` line 20:
   ```dart
   return 'http://YOUR_IP:8000';  // Get IP from: ipconfig
   ```
3. Then:
```powershell
cd frontend
flutter run
# Select your device
```

---

## ✅ Pre-Flight Checklist

Before running, make sure:

- [ ] **Flutter is installed** (check: `flutter --version`)
  - If you see "flutter is not recognized", install Flutter first!
  - See: [FLUTTER_INSTALL_WINDOWS.md](FLUTTER_INSTALL_WINDOWS.md)
- [ ] Backend has `.env` file with `GEMINI_API_KEY=your_key`
- [ ] Backend has `stock_data.csv` (run `stock_data_generator.py` if missing)
- [ ] Backend is running on port 8000
- [ ] Test backend: Open http://localhost:8000 in browser

---

## 🎯 Test It Works

1. **Screener Tab**: Click "Show me undervalued IT stocks"
2. **Chart Doctor Tab**: Upload any image → Analyze
3. **Compare Tab**: Click "RELIANCE" button

All three should work! 🎉

---

## 📖 Need More Details?

See `SETUP_GUIDE.md` for complete step-by-step instructions.


# FinSamaritan Quick Start Guide

Get up and running in 5 minutes!

## 🚀 Quick Start (Windows)

### Backend (Terminal 1)

```powershell
cd backend
.\start_backend.bat
```

Or manually:
```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
$env:GEMINI_API_KEY="your-api-key-here"
uvicorn main:app --reload
```

### Frontend (Terminal 2)

```powershell
cd frontend
npm install
npm start
```

### Frontend (Terminal 3)

```powershell
cd frontend
npm run android
```

## 🧪 Quick Test

Once backend is running, test it:

```powershell
cd backend
python test_backend.py
```

Or manually:
```powershell
curl http://localhost:8000/health
```

## 📱 Test in App

1. Open the app on your emulator/device
2. Type: "What is the price of RELIANCE.NS?"
3. You should get a response!

## 🔑 Get Gemini API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Create a new API key
3. Copy and set it as environment variable

## ⚠️ Common Issues

**Backend won't start:**
- Check if port 8000 is free
- Verify GEMINI_API_KEY is set
- Ensure all dependencies are installed

**Frontend can't connect:**
- Verify backend is running
- Check API URL in `frontend/src/api/agent.ts`
- For Android emulator, use `http://10.0.2.2:8000`

**For detailed setup, see:** `SETUP_AND_TESTING_GUIDE.md`


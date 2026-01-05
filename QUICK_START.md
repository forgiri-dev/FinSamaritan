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
npm run dev
```

Then open your browser to `http://localhost:3000`

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

## 🌐 Test in Browser

1. Open `http://localhost:3000` in Chrome
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
- Verify backend is running on `http://localhost:8000`
- Check browser console for errors
- Ensure CORS is properly configured

**For detailed setup, see:** `SETUP_AND_TESTING_GUIDE.md`

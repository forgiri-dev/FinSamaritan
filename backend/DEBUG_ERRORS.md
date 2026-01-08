# Debugging 500 Internal Server Error

## How to Get the Full Error Message

The backend prints detailed error information to the terminal. When you see "500 Internal Server Error", look for:

1. **In the Python terminal** (where you ran `python main.py`), you should see:
   ```
   Error in agent_screener: <full traceback>
   ```
   
2. **Copy the entire error message** from the terminal - it shows:
   - The exact line where the error occurred
   - The error type and message
   - The full stack trace

## Common 500 Error Causes

### 1. Gemini API Key Issues
**Error might say:** "API key not valid" or "Invalid API key"

**Solution:**
- Check `.env` file exists in `backend/` directory
- Verify `GEMINI_API_KEY=your_key_here` (no quotes, no spaces)
- Restart backend server after changing `.env`

### 2. Stock Data Not Loaded
**Error might say:** "stock_data.csv not found" or pandas errors

**Solution:**
```powershell
cd backend
python stock_data_generator.py
```

### 3. Gemini API Rate Limits
**Error might say:** "429" or "quota exceeded"

**Solution:**
- Wait a few minutes
- Check your Gemini API quota
- Make sure API key is valid

### 4. JSON Serialization Errors
**Error might say:** "Object of type float is not JSON serializable" or NaN errors

**Status:** Should be fixed, but if you still see this:
- Check that `numpy` is imported: `import numpy as np`
- Check that NaN values are being cleaned

### 5. Missing Dependencies
**Error might say:** "No module named X"

**Solution:**
```powershell
pip install -r requirements.txt
```

## Steps to Debug

1. **Run diagnostic test first:**
   ```powershell
   cd backend
   python test_backend.py
   ```
   This will check common issues and tell you what's wrong.

2. **Look at the backend terminal** - the full error is printed there
3. **Copy the error message** - especially the last few lines showing the actual error
4. **Check what endpoint was called** - Was it `/agent`, `/analyze-chart`, or `/compare`?
5. **Check the request** - What data was sent to the endpoint?

## Getting Error Details

The backend code prints errors like this:
```python
print(f"Error in agent_screener: {error_details}")
```

So in your terminal, you should see something like:
```
Error in agent_screener: Traceback (most recent call last):
  File "...", line X, in agent_screener
    ...
SomeError: error message here
```

**Please share this full error message** so we can identify the exact issue!


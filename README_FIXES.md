# Fixes Applied for 500 Error

## Problem
The backend was returning a 500 Internal Server Error when calling the `/agent` endpoint. This was caused by **NaN (Not a Number) values** in pandas DataFrames that cannot be serialized to JSON.

## Solution
Fixed NaN value handling in two files:

### 1. `backend/agent_tools.py`
- Added `numpy` import
- Replaced NaN values with `None` before converting DataFrames to dictionaries
- Applied to both `search_stocks()` and `get_stock_info()` functions

### 2. `backend/main.py`
- Added `numpy` import
- Added additional cleanup when creating JSON summary for Gemini

## Next Steps

1. **Restart your backend server** to apply the fixes:
   ```powershell
   # In the backend directory
   python main.py
   ```

2. **Test the API** in your browser:
   - Go to: http://localhost:8000/docs
   - Try the `/agent` endpoint with a test query

3. **If you still get errors**, check the PowerShell terminal running the backend for the detailed error message (it should now show more details).

## PowerShell Execution Policy Issue (Separate)

The PowerShell execution policy error is a Windows security feature. You have two options:

### Option 1: Use Python directly (Recommended)
Instead of activating the virtual environment with PowerShell, use Python directly:
```powershell
# Navigate to backend folder
cd backend

# Run directly (if virtual environment is activated elsewhere)
python main.py

# OR use the virtual environment Python directly
..\.venv\Scripts\python.exe main.py
```

### Option 2: Change Execution Policy (One-time setup)
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Testing

After restarting the server, try these in your Flutter app:
1. Screener: "Show me undervalued IT stocks"
2. Chart Doctor: Upload a chart image
3. Compare: Enter "RELIANCE"

All should now work without 500 errors!


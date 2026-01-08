# Backend Troubleshooting Guide

## Common Errors and Solutions

### Error: "GenerativeModel.__init__() got an unexpected keyword argument 'system_instruction'"

**Status: ✅ FIXED**

This was caused by using an unsupported parameter in the Google Generative AI library version 0.3.2.

**Solution Applied:**
- Removed the unused `system_instruction` parameter from model creation
- The code now uses the simpler `agent_model.generate_content()` approach which works with all versions

---

### Error: JSON Serialization Error (NaN values)

**Status: ✅ FIXED**

NaN values from pandas DataFrames cannot be serialized to JSON.

**Solution Applied:**
- Added NaN value cleaning in `agent_tools.py` using `np.nan` replacement
- Added additional cleaning in `main.py` for results list
- All NaN values are now converted to `None` before JSON serialization

---

### Error: "Error connecting to API" (Frontend)

**Possible Causes:**
1. Backend server is not running
2. Backend is running on wrong port
3. CORS issues

**Solutions:**
1. Make sure backend is running: `python main.py` (should see "Uvicorn running on http://0.0.0.0:8000")
2. Test backend directly: Open http://localhost:8000 in browser
3. Check backend terminal for errors
4. Verify CORS middleware is enabled (already configured in main.py)

---

### Error: "GEMINI_API_KEY not found"

**Solution:**
1. Create `.env` file in `backend/` directory
2. Add: `GEMINI_API_KEY=your_actual_api_key_here`
3. Restart backend server

---

### Error: "stock_data.csv not found"

**Solution:**
```bash
cd backend
python stock_data_generator.py
```

Wait for it to complete (downloads 500+ stocks, may take a few minutes).

---

### Error: Port 8000 already in use

**Windows Solution:**
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace <PID> with actual number from above)
taskkill /PID <PID> /F
```

---

## Testing the Backend

### 1. Check if backend starts:
```bash
cd backend
python main.py
```

Should see:
```
✓ Stock data loaded successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### 2. Test in browser:
- Open: http://localhost:8000
- Should see: API welcome message
- API docs: http://localhost:8000/docs

### 3. Test /agent endpoint:
Use the Swagger UI at http://localhost:8000/docs:
1. Click on `/agent` endpoint
2. Click "Try it out"
3. Enter test query: `"Show me IT stocks"`
4. Click "Execute"
5. Should return results (not 500 error)

---

## If Errors Persist

1. **Check the backend terminal** - it shows detailed error messages
2. **Check browser console** (F12) - for frontend errors
3. **Verify all dependencies installed:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Restart backend server** after any code changes
5. **Check Python version:** Should be 3.8+
   ```bash
   python --version
   ```

---

## Current Status

✅ `system_instruction` error - FIXED
✅ NaN JSON serialization - FIXED
✅ Error handling improved
✅ All endpoints should work

If you see a new error, check the backend terminal output and share the full error message.


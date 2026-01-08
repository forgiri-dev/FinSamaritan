# Backend Restart Instructions

## Important: Restart Required

After updating the Gemini model configuration, you **MUST restart the backend server** for changes to take effect.

## Steps to Restart

1. **Stop the current backend server:**
   - Press `Ctrl+C` in the terminal where the backend is running

2. **Restart the backend:**
   ```bash
   cd backend
   python app.py
   ```

3. **Check the console output:**
   You should see one of these messages:
   - `✓ Gemini AI model 'gemini-pro' loaded successfully`
   - `✓ Gemini AI model 'models/gemini-pro' loaded successfully`

4. **If you see warnings:**
   - Make sure `GEMINI_API_KEY` is set
   - Check that your API key is valid
   - Verify the API key has access to Gemini models

## Why Restart is Needed

Flask's debug mode caches Python modules. When you change the code, you need to restart the server for the new code to run. The model initialization happens when the module is first imported, so a restart is required.

## Troubleshooting

If you still see `gemini-1.5-flash` errors after restarting:
1. Check that you saved the `app.py` file
2. Verify the server actually restarted (check the console output)
3. Clear any Python cache files: `__pycache__` directories
4. Try stopping and starting again


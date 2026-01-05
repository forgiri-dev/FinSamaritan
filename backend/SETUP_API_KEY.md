# Quick Setup: GEMINI_API_KEY

## Easiest Method: Use .env File

1. **Create a `.env` file** in the `backend` directory (same folder as `main.py`)

2. **Add this line** to the file:
   ```
   GEMINI_API_KEY=your-actual-api-key-here
   ```
   Replace `your-actual-api-key-here` with your actual Gemini API key.

3. **Save the file**

4. **Run the startup script:**
   ```powershell
   .\start_backend.bat
   ```

That's it! The script will automatically load the API key from the `.env` file.

## Example .env File

```
GEMINI_API_KEY=AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz1234567
```

## Get Your API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and paste it in your `.env` file

## Important Notes

- The file must be named exactly `.env` (with the dot at the beginning)
- No spaces around the `=` sign
- No quotes needed around the API key value
- The `.env` file is automatically ignored by git (won't be committed)

## Troubleshooting

**Still getting "GEMINI_API_KEY is not set" error?**
- Make sure the `.env` file is in the `backend` directory (same folder as `main.py`)
- Make sure the file is named `.env` (not `env.txt` or `.env.txt`)
- Make sure there are no spaces: `GEMINI_API_KEY=your-key` (not `GEMINI_API_KEY = your-key`)
- Make sure `python-dotenv` is installed: `pip install python-dotenv`


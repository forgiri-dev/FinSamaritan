# Setting Up GEMINI_API_KEY

There are three ways to set your Gemini API key. **Method 1 (using .env file) is the easiest and recommended.**

## Method 1: Using .env File (Recommended - Easiest)

1. In the `backend` directory, create a file named `.env` (note the dot at the beginning)
2. Add this line to the file:
   ```
   GEMINI_API_KEY=your-actual-api-key-here
   ```
3. Save the file
4. Run the startup script - it will automatically load the key from the .env file

**Example:**
```bash
cd backend
# Create .env file with: GEMINI_API_KEY=AIzaSy...
# Then run:
.\start_backend.bat
```

## Method 2: Set in Current Session (Temporary)

### Windows PowerShell:
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

### Windows CMD:
```cmd
set GEMINI_API_KEY=your-api-key-here
```

### Linux/Mac:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**Note:** This only works for the current terminal session. If you close the terminal, you'll need to set it again.

## Method 3: Set as System Environment Variable (Permanent)

### Windows:
1. Press `Win + R`, type `sysdm.cpl`, press Enter
2. Click "Environment Variables"
3. Under "User variables" (or "System variables"), click "New"
4. Variable name: `GEMINI_API_KEY`
5. Variable value: `your-api-key-here`
6. Click OK on all dialogs
7. **IMPORTANT:** Close and reopen your terminal/PowerShell window
8. Run the startup script

### Linux/Mac:
Add to your `~/.bashrc` or `~/.zshrc`:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Then reload:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

## Getting Your API Key

1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key
5. Use it in one of the methods above

## Troubleshooting

**"GEMINI_API_KEY is not set" even after setting it:**
- If you used Method 3 (System Variables), make sure you **closed and reopened** your terminal
- Try Method 1 (.env file) instead - it's the most reliable
- Verify the key is correct (no extra spaces, quotes, etc.)

**".env file not working":**
- Make sure the file is named exactly `.env` (with the dot)
- Make sure it's in the `backend` directory (same folder as `main.py`)
- Make sure the format is: `GEMINI_API_KEY=your-key` (no spaces around the `=`)


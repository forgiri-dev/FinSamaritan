@echo off
REM Change to the script's directory (backend folder)
cd /d "%~dp0"

echo ========================================
echo FinSamaritan Backend Startup Script
echo ========================================
echo.
echo Current directory: %CD%
echo.

REM Refresh environment variables from registry (for system/user variables)
call refreshenv >nul 2>&1
if errorlevel 1 (
    REM If refreshenv doesn't exist, try to refresh manually
    for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v GEMINI_API_KEY 2^>nul') do set GEMINI_API_KEY=%%b
    for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v GEMINI_API_KEY 2^>nul') do set GEMINI_API_KEY=%%b
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies are installed
echo Checking dependencies...
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    echo.
    echo If installation fails, try running: install_requirements.bat
    echo.
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Installation failed. Trying alternative method...
        echo Installing packages in order...
        python -m pip install "numpy>=1.24.0,<2.0.0" "pydantic>=2.5.0,<3.0.0"
        python -m pip install "fastapi>=0.104.0" "uvicorn[standard]>=0.24.0" "python-multipart>=0.0.6"
        python -m pip install "pandas>=2.0.0,<3.0.0"
        python -m pip install "google-genai>=0.2.0" "yfinance>=0.2.28" "requests>=2.31.0"
    )
)

REM Check for GEMINI_API_KEY in environment or .env file
REM If .env file exists, Python will load it automatically via dotenv
if "%GEMINI_API_KEY%"=="" (
    if exist ".env" (
        echo Found .env file - Python will load GEMINI_API_KEY from it
        echo Continuing startup...
        goto :continue_start
    )
    
    echo.
    echo ========================================
    echo ERROR: GEMINI_API_KEY is not set!
    echo ========================================
    echo.
    echo Please set it using one of these methods:
    echo.
    echo Method 1: Create .env file (EASIEST - Recommended)
    echo   1. Create a file named .env in the backend directory
    echo   2. Add this line: GEMINI_API_KEY=your-api-key-here
    echo   3. Save the file and run this script again
    echo   4. Make sure the file is named exactly .env (not .env.txt)
    echo.
    echo Method 2: Set in this session (temporary)
    echo   set GEMINI_API_KEY=your-api-key-here
    echo   Then run this script again
    echo.
    echo Method 3: Set permanently in System Variables
    echo   1. Open System Properties ^> Environment Variables
    echo   2. Add new User variable: GEMINI_API_KEY
    echo   3. Set value to your API key
    echo   4. CLOSE AND REOPEN this terminal, then run this script
    echo   (Note: Existing terminals don't see new system variables)
    echo.
    echo Get your API key from: https://makersuite.google.com/app/apikey
    echo.
    echo Current directory: %CD%
    echo Checking for .env file in: %CD%\.env
    if exist ".env" (
        echo .env file EXISTS in %CD%
        echo File contents:
        type .env
        echo.
        echo If the file looks correct, the script will continue and Python will load it.
        echo Press any key to continue anyway, or CTRL+C to exit...
        pause >nul
        goto :continue_start
    ) else (
        echo .env file NOT FOUND in current directory: %CD%
    )
    echo.
    pause
    exit /b 1
)

:continue_start

REM Initialize database (will be done automatically on startup)
echo.
echo Starting FinSamaritan Backend...
echo Server will be available at http://localhost:8000
echo Press CTRL+C to stop the server
echo.

REM Start the server
uvicorn main:app --reload

pause


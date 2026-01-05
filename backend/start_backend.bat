@echo off
echo ========================================
echo FinSamaritan Backend Startup Script
echo ========================================
echo.

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
        python -m pip install "google-generativeai>=0.3.0" "yfinance>=0.2.28" "requests>=2.31.0"
    )
)

REM Check for GEMINI_API_KEY
if "%GEMINI_API_KEY%"=="" (
    echo.
    echo WARNING: GEMINI_API_KEY environment variable is not set!
    echo Please set it using:
    echo   set GEMINI_API_KEY=your-api-key-here
    echo.
    pause
)

REM Initialize database (will be done automatically on startup)
echo.
echo Starting FinSamaritan Backend...
echo Server will be available at http://localhost:8000
echo Press CTRL+C to stop the server
echo.

REM Start the server
uvicorn main:app --reload

pause


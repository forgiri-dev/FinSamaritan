@echo off
echo ========================================
echo Installing FinSamaritan Backend Dependencies
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

echo Step 1: Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

echo.
echo Step 2: Installing core dependencies (numpy, pydantic)...
python -m pip install "numpy>=1.24.0,<2.0.0" "pydantic>=2.5.0,<3.0.0"

echo.
echo Step 3: Installing web framework...
python -m pip install "fastapi>=0.104.0" "uvicorn[standard]>=0.24.0" "python-multipart>=0.0.6"

echo.
echo Step 4: Installing data processing...
python -m pip install "pandas>=2.0.0,<3.0.0"

echo.
echo Step 5: Installing external APIs...
python -m pip install "google-genai>=0.2.0" "yfinance>=0.2.28" "requests>=2.31.0"

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Verifying installation...
python -c "import fastapi; import uvicorn; import pandas; import numpy; import google.genai; import yfinance; import requests; print('All packages installed successfully!')"

if errorlevel 1 (
    echo.
    echo WARNING: Some packages may not have installed correctly
    echo Try installing manually: pip install -r requirements.txt
) else (
    echo.
    echo All packages verified successfully!
)

pause


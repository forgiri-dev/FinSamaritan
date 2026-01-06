#!/bin/bash

echo "========================================"
echo "Installing FinSamaritan Backend Dependencies"
echo "========================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    exit 1
fi

echo "Step 1: Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

echo ""
echo "Step 2: Installing core dependencies (numpy, pydantic)..."
python3 -m pip install "numpy>=2.1.0,<3.0.0" "pydantic==2.7.4"

echo ""
echo "Step 3: Installing web framework..."
python3 -m pip install "fastapi==0.110.2" "uvicorn[standard]==0.29.0" "python-multipart==0.0.9"

echo ""
echo "Step 4: Installing data processing..."
python3 -m pip install "pandas>=2.2.0,<3.0.0"

echo ""
echo "Step 5: Installing external APIs..."
python3 -m pip install "google-generativeai==0.8.3" "google-genai==0.3.0" "yfinance==0.2.40" "requests==2.32.3" "python-dotenv==1.0.1"

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Verifying installation..."
python3 -c "import fastapi; import uvicorn; import pandas; import numpy; import google.generativeai; import google.genai; import yfinance; import requests; import dotenv; print('All packages installed successfully!')"

if [ $? -eq 0 ]; then
    echo ""
    echo "All packages verified successfully!"
else
    echo ""
    echo "WARNING: Some packages may not have installed correctly"
    echo "Try installing manually: pip install -r requirements.txt"
fi


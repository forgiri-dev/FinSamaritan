#!/bin/bash

echo "========================================"
echo "FinSamaritan Backend Startup Script"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
echo "Checking dependencies..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    echo ""
    echo "If installation fails, try running: ./install_requirements.sh"
    echo ""
    python3 -m pip install --upgrade pip setuptools wheel
    python3 -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo ""
        echo "Installation failed. Trying alternative method..."
        echo "Installing packages in order..."
        python3 -m pip install "numpy>=1.24.0,<2.0.0" "pydantic>=2.5.0,<3.0.0"
        python3 -m pip install "fastapi>=0.104.0" "uvicorn[standard]>=0.24.0" "python-multipart>=0.0.6"
        python3 -m pip install "pandas>=2.0.0,<3.0.0"
        python3 -m pip install "google-generativeai>=0.3.0" "yfinance>=0.2.28" "requests>=2.31.0"
    fi
fi

# Check for GEMINI_API_KEY
if [ -z "$GEMINI_API_KEY" ]; then
    echo ""
    echo "WARNING: GEMINI_API_KEY environment variable is not set!"
    echo "Please set it using:"
    echo "  export GEMINI_API_KEY='your-api-key-here'"
    echo ""
    read -p "Press enter to continue anyway, or CTRL+C to exit..."
fi

# Initialize database (will be done automatically on startup)
echo ""
echo "Starting FinSamaritan Backend..."
echo "Server will be available at http://localhost:8000"
echo "Press CTRL+C to stop the server"
echo ""

# Start the server
uvicorn main:app --reload


# Installation Guide for Backend Dependencies

## Quick Installation

### Option 1: Automated Script (Recommended)

**Windows:**
```powershell
.\install_requirements.bat
```

**Linux/Mac:**
```bash
chmod +x install_requirements.sh
./install_requirements.sh
```

### Option 2: Standard pip install

```bash
pip install -r requirements.txt
```

### Option 3: If Standard Install Fails

Try the fixed version:
```bash
pip install -r requirements-fixed.txt
```

### Option 4: Minimal Installation (No Version Constraints)

If all else fails:
```bash
pip install -r requirements-minimal.txt
```

## Installation Order Matters

Some packages have dependencies that must be installed first:

1. **NumPy** (required by Pandas)
2. **Pydantic** (required by FastAPI)
3. **FastAPI & Uvicorn** (web framework)
4. **Pandas** (data processing)
5. **External APIs** (google-generativeai, yfinance, requests)

The installation scripts handle this automatically.

## Common Issues

### "ERROR: Could not find a version"

**Solution:** Upgrade pip first:
```bash
python -m pip install --upgrade pip setuptools wheel
```

### "Microsoft Visual C++ 14.0 is required" (Windows)

**Solution:** Install Visual C++ Build Tools or use pre-built wheels:
```bash
pip install --only-binary :all: -r requirements.txt
```

### Version Conflicts

**Solution:** Use the installation script which installs packages in the correct order, or install manually:

```bash
# Step by step
pip install "numpy>=1.24.0,<2.0.0"
pip install "pydantic>=2.5.0,<3.0.0"
pip install "fastapi>=0.104.0"
pip install "uvicorn[standard]>=0.24.0"
pip install "python-multipart>=0.0.6"
pip install "pandas>=2.0.0,<3.0.0"
pip install "google-generativeai>=0.3.0"
pip install "yfinance>=0.2.28"
pip install "requests>=2.31.0"
```

## Verify Installation

After installation, verify all packages:

```bash
python -c "import fastapi; import uvicorn; import pandas; import numpy; import google.generativeai; import yfinance; import requests; print('✅ All packages installed!')"
```

## Using Virtual Environment (Strongly Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Need More Help?

See [TROUBLESHOOTING_INSTALL.md](TROUBLESHOOTING_INSTALL.md) for detailed troubleshooting steps.


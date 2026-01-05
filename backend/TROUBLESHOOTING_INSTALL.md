# Troubleshooting Requirements Installation

If `pip install -r requirements.txt` fails, follow these steps:

## Common Issues and Solutions

### Issue 1: Version Conflicts

**Error:** `ERROR: Cannot install package X because it requires Y version Z`

**Solution:** Use the installation script that installs packages in the correct order:

**Windows:**
```powershell
.\install_requirements.bat
```

**Linux/Mac:**
```bash
chmod +x install_requirements.sh
./install_requirements.sh
```

### Issue 2: Pydantic/FastAPI Compatibility

**Error:** `ImportError: cannot import name 'X' from 'pydantic'`

**Solution:** FastAPI 0.104+ requires Pydantic 2.x. Install in this order:
```bash
pip install "pydantic>=2.5.0,<3.0.0"
pip install "fastapi>=0.104.0"
```

### Issue 3: NumPy/Pandas Compatibility

**Error:** `numpy.core.multiarray failed to import`

**Solution:** Install NumPy before Pandas:
```bash
pip install "numpy>=1.24.0,<2.0.0"
pip install "pandas>=2.0.0,<3.0.0"
```

### Issue 4: Outdated pip

**Error:** `ERROR: Could not find a version that satisfies the requirement`

**Solution:** Upgrade pip first:
```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Issue 5: Platform-Specific Build Failures

**Error:** `Microsoft Visual C++ 14.0 is required` (Windows)

**Solution:** 
- Install Visual C++ Build Tools: https://visualstudio.microsoft.com/downloads/
- Or use pre-built wheels:
```bash
pip install --only-binary :all: -r requirements.txt
```

### Issue 6: Network/Proxy Issues

**Error:** `Could not fetch URL` or `Connection timeout`

**Solution:**
```bash
# Use a different index
pip install -r requirements.txt -i https://pypi.org/simple

# Or with timeout
pip install --default-timeout=100 -r requirements.txt
```

## Manual Installation (Step by Step)

If automated installation fails, install packages manually in this order:

```bash
# 1. Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# 2. Core dependencies
pip install "numpy>=1.24.0,<2.0.0"
pip install "pydantic>=2.5.0,<3.0.0"

# 3. Web framework
pip install "fastapi>=0.104.0"
pip install "uvicorn[standard]>=0.24.0"
pip install "python-multipart>=0.0.6"

# 4. Data processing
pip install "pandas>=2.0.0,<3.0.0"

# 5. External APIs
pip install "google-generativeai>=0.3.0"
pip install "yfinance>=0.2.28"
pip install "requests>=2.31.0"
```

## Using Virtual Environment (Recommended)

Always use a virtual environment to avoid conflicts:

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Alternative: Use requirements-fixed.txt

If the main requirements.txt still fails, try the fixed version:

```bash
pip install -r requirements-fixed.txt
```

## Verify Installation

After installation, verify all packages:

```bash
python -c "import fastapi; import uvicorn; import pandas; import numpy; import google.generativeai; import yfinance; import requests; print('✅ All packages installed!')"
```

## Still Having Issues?

1. Check Python version: `python --version` (should be 3.8+)
2. Check pip version: `pip --version` (should be 23.0+)
3. Try installing without version constraints:
   ```bash
   pip install fastapi uvicorn pandas numpy google-generativeai yfinance requests python-multipart
   ```
4. Check for conflicting packages: `pip check`


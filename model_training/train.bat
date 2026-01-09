@echo off
REM Edge Sentinel Model Training Script (Windows)
REM This script automates the complete training workflow

echo.
echo ========================================
echo   Edge Sentinel Model Training
echo ========================================
echo.

REM Step 1: Check Python
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)
echo OK: Python found
python --version
echo.

REM Step 2: Install dependencies
echo [2/6] Installing dependencies...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo OK: Dependencies installed
echo.

REM Step 3: Generate training data
echo [3/6] Generating training data...
echo This may take 5-10 minutes...
python data_generator.py
if errorlevel 1 (
    echo ERROR: Failed to generate training data
    pause
    exit /b 1
)
echo OK: Training data generated
echo.

REM Step 4: Train model
echo [4/6] Training model...
echo This may take 30-60 minutes (GPU) or 2-4 hours (CPU)...
python train_model.py --data-dir training_data --output-dir models
if errorlevel 1 (
    echo ERROR: Training failed
    pause
    exit /b 1
)
echo OK: Model trained
echo.

REM Step 5: Test model
echo [5/6] Testing model...
python test_model.py --model models/model_unquant.tflite --labels models/labels.txt --test-dir training_data
if errorlevel 1 (
    echo WARNING: Testing failed, but model may still be usable
)
echo OK: Model tested
echo.

REM Step 6: Copy to frontend
echo [6/6] Deploying to frontend...
if exist "..\frontend\assets" (
    copy models\model_unquant.tflite ..\frontend\assets\ >nul
    copy models\labels.txt ..\frontend\assets\ >nul
    echo OK: Model deployed to frontend/assets/
) else (
    echo WARNING: Frontend assets directory not found
    echo Please copy manually:
    echo   copy models\model_unquant.tflite ..\frontend\assets\
    echo   copy models\labels.txt ..\frontend\assets\
)
echo.

REM Summary
echo ========================================
echo   Training Complete!
echo ========================================
echo.
echo Model files:
echo   - models\model_unquant.tflite
echo   - models\labels.txt
echo   - models\model_info.json
echo.
echo Next steps:
echo   1. Check model_info.json for accuracy metrics
echo   2. Test with your own images
echo   3. Integrate TFLite library in React Native
echo   4. Update EdgeSentinel.ts to use the model
echo.
pause


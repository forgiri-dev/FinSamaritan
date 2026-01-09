# Edge Sentinel Model Training

This directory contains everything needed to train the Edge Sentinel model for detecting candlestick patterns and trends in financial charts.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training data
python data_generator.py

# 3. Train the model
python train_model.py --data-dir training_data --output-dir models

# 4. Test the model
python test_model.py --model models/model_unquant.tflite --labels models/labels.txt --test-dir training_data
```

## Files

- `data_generator.py` - Generates synthetic candlestick charts with labeled patterns
- `train_model.py` - Trains the CNN model and exports to TFLite
- `test_model.py` - Tests the trained model
- `requirements.txt` - Python dependencies
- `TRAINING_GUIDE.md` - Comprehensive training guide

## Model Classes

The model detects combinations of:

**Patterns (8):**
- hammer
- doji
- engulfing_bullish
- engulfing_bearish
- shooting_star
- morning_star
- evening_star
- normal

**Trends (3):**
- uptrend
- downtrend
- sideways

**Total Classes: 8 × 3 = 24**

## Output

After training, you'll get:
- `model_unquant.tflite` - TensorFlow Lite model for React Native
- `labels.txt` - Class labels
- `model_info.json` - Model metrics and information

Copy these to `frontend/assets/` to use in the app.

## See Also

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Detailed training instructions
- Frontend Edge Sentinel service: `frontend/src/services/EdgeSentinel.ts`


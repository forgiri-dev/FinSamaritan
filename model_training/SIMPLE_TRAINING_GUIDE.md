# Simple Model Training Guide

This guide shows you how to train a model using the simple, error-free training scripts.

## Quick Start

### Step 1: Install Dependencies

Make sure you have the required packages installed:

```bash
pip install tensorflow numpy pillow scikit-learn
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

Run the simple training script:

```bash
python train_simple.py --data-dir training_data --output-dir models
```

This will:
- Load all images from `training_data/` directory
- Train a simple Keras CNN model
- Save the model as `models/model.keras`
- Save labels as `models/labels.txt`
- Save training metadata and history

**Expected output:**
- `models/model.keras` - The trained Keras model
- `models/labels.txt` - Class labels
- `models/model_metadata.json` - Model information
- `models/training_history.json` - Training metrics

### Step 3: Convert to TFLite Format

After training, convert the model to TensorFlow Lite format:

```bash
python convert_to_tflite.py --model-path models/model.keras --output-path models/model_unquant.tflite
```

This will:
- Load the Keras model
- Convert it to TensorFlow Lite format
- Save as `models/model_unquant.tflite`

**Expected output:**
- `models/model_unquant.tflite` - Ready to use in your frontend!

## Usage

### Basic Training

```bash
python train_simple.py
```

Uses default directories:
- Data: `training_data/`
- Output: `models/`

### Custom Directories

```bash
python train_simple.py --data-dir /path/to/data --output-dir /path/to/output
```

### Convert to TFLite

```bash
python convert_to_tflite.py
```

Uses default paths:
- Model: `models/model.keras`
- Output: `models/model_unquant.tflite`

### Custom Paths

```bash
python convert_to_tflite.py --model-path models/model.keras --output-path models/model_unquant.tflite
```

## Model Architecture

The simple model uses:
- **Input**: 224x224 RGB images
- **Architecture**: Simple CNN with 3 convolutional blocks
- **Output**: Softmax probabilities for each class

## Troubleshooting

### "ModuleNotFoundError: No module named 'tensorflow'"

Install TensorFlow:
```bash
pip install tensorflow
```

### "ModuleNotFoundError: No module named 'sklearn'"

Install scikit-learn:
```bash
pip install scikit-learn
```

### Training is slow

The default is 10 epochs. You can modify `EPOCHS` in `train_simple.py` if needed.

### Model file is large

The TFLite converter uses default optimizations. The model size should be reasonable for mobile use.

## What Makes This Simple?

1. **Uses Keras Sequential API** - Much simpler than raw TensorFlow
2. **No complex configuration** - Just run and it works
3. **Clear error messages** - If something fails, you'll know why
4. **Standard formats** - Uses standard Keras and TFLite formats

## Next Steps

After converting to TFLite:
1. Copy `models/model_unquant.tflite` to `frontend/assets/`
2. Copy `models/labels.txt` to `frontend/assets/`
3. Use the model in your React Native app!

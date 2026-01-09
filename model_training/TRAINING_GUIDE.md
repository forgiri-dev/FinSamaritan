# Edge Sentinel Model Training Guide

## Overview

This guide will help you train the Edge Sentinel model to detect:
- **Candlestick Patterns**: Hammer, Doji, Engulfing, Shooting Star, Morning/Evening Star, etc.
- **Trend Analysis**: Uptrend, Downtrend, Sideways

The model will be exported as TensorFlow Lite for use in the React Native frontend.

## Prerequisites

1. **Python 3.8+** installed
2. **GPU (Optional but recommended)** - Training will be much faster with GPU
3. **At least 10GB free disk space** for training data and models

## Step 1: Install Dependencies

```bash
cd model_training
pip install -r requirements.txt
```

**Note**: If you have a GPU, install TensorFlow with GPU support:
```bash
pip install tensorflow[and-cuda]
```

## Step 2: Generate Training Data

The data generator creates synthetic candlestick charts with labeled patterns and trends.

### Basic Usage

```bash
python data_generator.py
```

This will:
- Generate 200 samples per class (24 classes = 4,800 total images)
- Mix synthetic and real stock data
- Save images to `training_data/` directory
- Create `labels.txt` file

### Customize Data Generation

Edit `data_generator.py` and modify the `generate_training_dataset()` call:

```python
generate_training_dataset(
    output_dir='training_data',
    samples_per_class=500,  # Increase for better accuracy
    use_real_data=True      # Mix real stock data
)
```

### Data Structure

The generated data will be organized as:
```
training_data/
├── hammer_uptrend/
│   ├── hammer_uptrend_0000.jpg
│   ├── hammer_uptrend_0001.jpg
│   └── ...
├── hammer_downtrend/
├── doji_uptrend/
├── ...
└── labels.txt
```

### Tips for Better Data

1. **Increase samples**: More data = better model (aim for 500-1000 per class)
2. **Use real data**: The generator mixes real stock data for more realistic patterns
3. **Vary parameters**: The generator uses random volatility and time periods
4. **Add your own data**: You can manually add images to class directories

## Step 3: Train the Model

### Basic Training

```bash
python train_model.py --data-dir training_data --output-dir models
```

### Advanced Options

```bash
python train_model.py \
    --data-dir training_data \
    --output-dir models \
    --epochs 100
```

### Training Process

The training script will:
1. Load all images from the training directory
2. Split into train/validation/test sets (70%/20%/10%)
3. Apply data augmentation (rotation, shifts, brightness)
4. Train a CNN model with:
   - 4 convolutional blocks
   - Batch normalization
   - Dropout for regularization
   - Dense layers for classification
5. Save the best model based on validation accuracy
6. Convert to TensorFlow Lite format
7. Generate evaluation metrics

### Expected Training Time

- **CPU**: ~2-4 hours for 4,800 images, 50 epochs
- **GPU**: ~30-60 minutes for 4,800 images, 50 epochs

### Monitor Training

The script will show:
- Training progress per epoch
- Training and validation accuracy
- Loss values
- Early stopping if validation doesn't improve

## Step 4: Evaluate the Model

After training, check the output:

```
models/
├── best_model.keras          # Best model during training
├── edge_sentinel_model.keras  # Final model
├── model_unquant.tflite       # TFLite model for React Native ⭐
├── labels.txt                 # Class labels
├── training_history.json      # Training metrics
└── model_info.json            # Model information
```

### Check Model Performance

```python
import json

with open('models/model_info.json', 'r') as f:
    info = json.load(f)
    print(f"Test Accuracy: {info['test_accuracy']:.2%}")
    print(f"Top-3 Accuracy: {info['test_top3_accuracy']:.2%}")
```

**Target Metrics:**
- Test Accuracy: >85% (good), >90% (excellent)
- Top-3 Accuracy: >95% (good), >98% (excellent)

## Step 5: Deploy to React Native

1. **Copy TFLite model**:
   ```bash
   cp models/model_unquant.tflite ../frontend/assets/
   cp models/labels.txt ../frontend/assets/
   ```

2. **Update Edge Sentinel service** (see next section)

## Troubleshooting

### Out of Memory

If you get OOM errors:
- Reduce `BATCH_SIZE` in `train_model.py` (default: 32, try 16 or 8)
- Reduce `samples_per_class` in data generation
- Use smaller image size (change `IMG_SIZE` from 224 to 128)

### Low Accuracy

If accuracy is low (<80%):
- Increase training data (more samples per class)
- Train for more epochs
- Check data quality (some images might be corrupted)
- Try different model architecture

### Training is Slow

- Use GPU if available
- Reduce image size
- Reduce batch size
- Use fewer samples initially to test

### Model Size Too Large

The TFLite model should be <10MB for mobile. If larger:
- Use quantization (already enabled)
- Reduce model complexity
- Use smaller image size

## Advanced: Custom Patterns

To add custom patterns:

1. **Add pattern to `PATTERNS` list** in `data_generator.py`:
   ```python
   PATTERNS = [
       'hammer',
       'doji',
       'your_custom_pattern',  # Add here
       ...
   ]
   ```

2. **Implement pattern injection** in `inject_pattern()` function:
   ```python
   elif pattern == 'your_custom_pattern':
       # Your pattern logic here
   ```

3. **Regenerate training data** and retrain

## Advanced: Transfer Learning

For better accuracy, you can use transfer learning:

```python
# In train_model.py, replace create_model() with:
base_model = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
```

## Next Steps

1. Test the model with real chart images
2. Fine-tune based on performance
3. Deploy to React Native frontend
4. Monitor model performance in production

## Model Architecture

```
Input (224x224x3)
  ↓
Conv2D(32) + BatchNorm + ReLU
Conv2D(32) + BatchNorm + ReLU
MaxPool(2x2) + Dropout(0.25)
  ↓
Conv2D(64) + BatchNorm + ReLU
Conv2D(64) + BatchNorm + ReLU
MaxPool(2x2) + Dropout(0.25)
  ↓
Conv2D(128) + BatchNorm + ReLU
Conv2D(128) + BatchNorm + ReLU
MaxPool(2x2) + Dropout(0.25)
  ↓
Conv2D(256) + BatchNorm + ReLU
Conv2D(256) + BatchNorm + ReLU
MaxPool(2x2) + Dropout(0.25)
  ↓
Flatten
  ↓
Dense(512) + BatchNorm + ReLU + Dropout(0.5)
Dense(256) + BatchNorm + ReLU + Dropout(0.5)
  ↓
Dense(num_classes) + Softmax
```

Total parameters: ~2-3 million (depending on num_classes)

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review training logs
3. Verify data quality
4. Check TensorFlow version compatibility


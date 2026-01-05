# Quick Start Guide - Edge Sentinel Training

## 🚀 5-Minute Setup

### Step 1: Install Dependencies

```bash
cd model_training
pip install -r requirements.txt
```

### Step 2: Generate Training Data (5-10 minutes)

```bash
python data_generator.py
```

This creates ~4,800 labeled chart images in `training_data/` directory.

### Step 3: Train the Model (30-60 minutes on GPU, 2-4 hours on CPU)

```bash
python train_model.py --data-dir training_data --output-dir models
```

### Step 4: Deploy to Frontend

```bash
# Copy trained model to frontend
cp models/model_unquant.tflite ../frontend/assets/
cp models/labels.txt ../frontend/assets/
```

### Step 5: Test the Model

```bash
python test_model.py --model models/model_unquant.tflite --labels models/labels.txt
```

## 📊 What Gets Detected

The model detects **24 combinations** of:

**Patterns (8):**
- 🔨 Hammer
- ⚖️ Doji
- 📈 Bullish Engulfing
- 📉 Bearish Engulfing
- ⭐ Shooting Star
- 🌅 Morning Star
- 🌆 Evening Star
- 📊 Normal

**Trends (3):**
- 📈 Uptrend
- 📉 Downtrend
- ↔️ Sideways

## 🎯 Expected Results

- **Test Accuracy**: 85-90%+
- **Model Size**: <10MB (TFLite)
- **Inference Time**: <100ms on mobile

## ⚙️ Customization

### More Training Data

Edit `data_generator.py`:
```python
samples_per_class=500  # Increase from 200
```

### More Epochs

```bash
python train_model.py --epochs 100
```

### Test Your Own Images

```bash
python test_model.py --image path/to/your/chart.jpg
```

## 🐛 Troubleshooting

**Out of Memory?**
- Reduce batch size in `train_model.py` (line 15): `BATCH_SIZE = 16`

**Low Accuracy?**
- Increase training data: `samples_per_class=500`
- Train longer: `--epochs 100`

**Model Too Large?**
- Already optimized with quantization
- Should be <10MB

## 📚 Next Steps

- Read [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed instructions
- Integrate model in React Native (see `frontend/src/services/EdgeSentinel.ts`)
- Fine-tune based on your specific use case

## ✅ Checklist

- [ ] Dependencies installed
- [ ] Training data generated
- [ ] Model trained
- [ ] Model tested
- [ ] Model copied to frontend
- [ ] Edge Sentinel service updated

---

**Need Help?** Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed documentation.


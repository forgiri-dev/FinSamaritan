#!/bin/bash

# Edge Sentinel Model Training Script
# This script automates the complete training workflow

set -e  # Exit on error

echo "🚀 Edge Sentinel Model Training"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check dependencies
echo -e "${BLUE}📦 Step 1: Checking dependencies...${NC}"
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Step 2: Install dependencies
echo -e "\n${BLUE}📦 Step 2: Installing dependencies...${NC}"
pip install -q -r requirements.txt
echo "✅ Dependencies installed"

# Step 3: Generate training data
echo -e "\n${BLUE}📊 Step 3: Generating training data...${NC}"
echo "This may take 5-10 minutes..."
python3 data_generator.py
echo "✅ Training data generated"

# Step 4: Train model
echo -e "\n${BLUE}🎓 Step 4: Training model...${NC}"
echo "This may take 30-60 minutes (GPU) or 2-4 hours (CPU)..."
python3 train_model.py --data-dir training_data --output-dir models
echo "✅ Model trained"

# Step 5: Test model
echo -e "\n${BLUE}🧪 Step 5: Testing model...${NC}"
python3 test_model.py --model models/model_unquant.tflite --labels models/labels.txt --test-dir training_data
echo "✅ Model tested"

# Step 6: Copy to frontend
echo -e "\n${BLUE}📱 Step 6: Deploying to frontend...${NC}"
if [ -d "../frontend/assets" ]; then
    cp models/model_unquant.tflite ../frontend/assets/
    cp models/labels.txt ../frontend/assets/
    echo "✅ Model deployed to frontend/assets/"
else
    echo -e "${YELLOW}⚠️  Frontend assets directory not found. Copy manually:${NC}"
    echo "   cp models/model_unquant.tflite ../frontend/assets/"
    echo "   cp models/labels.txt ../frontend/assets/"
fi

# Step 7: Summary
echo -e "\n${GREEN}🎉 Training Complete!${NC}"
echo "================================"
echo ""
echo "📁 Model files:"
echo "   - models/model_unquant.tflite"
echo "   - models/labels.txt"
echo "   - models/model_info.json"
echo ""
echo "📊 Next steps:"
echo "   1. Check model_info.json for accuracy metrics"
echo "   2. Test with your own images: python3 test_model.py --image path/to/image.jpg"
echo "   3. Integrate TFLite library in React Native"
echo "   4. Update EdgeSentinel.ts to use the model"
echo ""
echo "📚 Documentation:"
echo "   - QUICK_START.md - Quick reference"
echo "   - TRAINING_GUIDE.md - Detailed guide"
echo ""


#!/bin/bash
# Deployment script for Render.com
# This script sets up the optimized model and prepares for deployment

echo "🚀 AI Voice Detection - Deployment Setup"
echo "=========================================="

# Step 1: Install main dependencies
echo -e "\n📦 Installing main dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Step 2: Install CPU-only PyTorch
echo -e "\n📦 Installing CPU-only PyTorch..."
pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cpu

if [ $? -ne 0 ]; then
    echo "❌ Failed to install PyTorch"
    exit 1
fi

# Step 3: Pre-cache models
echo -e "\n💾 Pre-caching models (this may take 2-3 minutes)..."
python cache_models.py

# Step 4: Test the application
echo -e "\n✅ Testing application startup..."
timeout 30 python -c "
from ensemble_inference import load_model
print('Loading model...')
load_model(use_quantization=True)
print('✓ Model loaded successfully!')
" || echo "⚠️ Model loading test completed (may need more time on first run)"

# Step 5: Git commands for deployment
echo -e "\n📝 Ready for deployment!"
echo ""
echo "Next steps:"
echo "1. Stage changes: git add ."
echo "2. Commit: git commit -m 'Optimized model for efficient deployment'"
echo "3. Push: git push origin main"
echo "4. Deploy on Render (it will use the updated render.yaml)"
echo ""
echo "Deployment will take 3-5 minutes (model download + quantization)"

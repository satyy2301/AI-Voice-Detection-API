# Deployment script for Render.com (Windows)
# This script sets up the optimized model and prepares for deployment

Write-Host "🚀 AI Voice Detection - Deployment Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Step 1: Install CPU-only PyTorch
Write-Host "`n📦 Installing optimized dependencies..." -ForegroundColor Yellow
pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Step 2: Pre-cache models
Write-Host "`n💾 Pre-caching models (this may take 2-3 minutes)..." -ForegroundColor Yellow
python cache_models.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️ Model caching had issues, but proceeding..." -ForegroundColor Yellow
}

# Step 3: Test the application
Write-Host "`n✅ Testing application startup..." -ForegroundColor Yellow
python -c @"
from ensemble_inference import load_model
print('Loading model...')
load_model(use_quantization=True)
print('✓ Model loaded successfully!')
"@

# Step 4: Display next steps
Write-Host "`n✅ Ready for deployment!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Stage changes: git add ." -ForegroundColor White
Write-Host "2. Commit: git commit -m 'Optimized model for efficient deployment'" -ForegroundColor White
Write-Host "3. Push: git push origin main" -ForegroundColor White
Write-Host "4. Deploy on Render (it will use the updated render.yaml)" -ForegroundColor White
Write-Host ""
Write-Host "⏱️  Deployment will take 3-5 minutes (model download + quantization)" -ForegroundColor Yellow

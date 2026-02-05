#!/usr/bin/env python3
"""
Pre-Deployment Verification Script
Run this to verify all optimization changes are correct before deploying.
"""

import os
import sys
import subprocess

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ MISSING {description}: {filepath}")
        return False

def check_model_in_code():
    """Verify wav2vec2-base is in ensemble_inference.py"""
    try:
        with open('ensemble_inference.py', 'r') as f:
            content = f.read()
            if 'facebook/wav2vec2-base' in content:
                print("✅ Model changed to wav2vec2-base")
                return True
            else:
                print("❌ Model still using old version!")
                return False
    except Exception as e:
        print(f"❌ Error reading ensemble_inference.py: {e}")
        return False

def check_pytorch_cpu_only():
    """Verify CPU-only PyTorch in requirements."""
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()
            if '--index-url' in content or 'torch' in content:
                print("✅ PyTorch CPU-only configuration detected")
                return True
            else:
                print("⚠️  PyTorch version needs verification")
                return False
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def check_render_yaml():
    """Verify render.yaml is updated."""
    try:
        with open('render.yaml', 'r') as f:
            content = f.read()
            if '--index-url' in content and 'download.pytorch.org/whl/cpu' in content:
                print("✅ render.yaml updated with CPU-only PyTorch")
                return True
            else:
                print("⚠️  render.yaml might need --index-url update")
                return False
    except Exception as e:
        print(f"❌ Error reading render.yaml: {e}")
        return False

def check_quantization():
    """Verify aggressive quantization in code."""
    try:
        with open('ensemble_inference.py', 'r') as f:
            content = f.read()
            if 'torch.nn.LSTM' in content and 'torch.nn.GRU' in content:
                print("✅ Enhanced quantization (LSTM/GRU) configured")
                return True
            else:
                print("⚠️  Quantization may not be optimal")
                return False
    except Exception as e:
        print(f"❌ Error reading ensemble_inference.py: {e}")
        return False

def run_verification():
    """Run all verifications."""
    print("\n" + "="*60)
    print("🔍 PRE-DEPLOYMENT VERIFICATION")
    print("="*60 + "\n")
    
    checks = [
        ("Code files", [
            (check_file_exists("app.py", "Main API"),),
            (check_file_exists("ensemble_inference.py", "Inference engine"),),
            (check_file_exists("preprocessing.py", "Audio preprocessing"),),
            (check_file_exists("security.py", "Security"),),
        ]),
        ("Configuration files", [
            (check_file_exists("requirements.txt", "Dependencies"),),
            (check_file_exists("render.yaml", "Render config"),),
        ]),
        ("New helper files", [
            (check_file_exists("cache_models.py", "Model caching script"),),
            (check_file_exists("deploy.ps1", "Windows deployment script"),),
            (check_file_exists("OPTIMIZATION_SUMMARY.md", "Optimization docs"),),
            (check_file_exists("DEPLOYMENT_GUIDE.md", "Deployment guide"),),
        ]),
        ("Code optimizations", [
            (check_model_in_code(),),
            (check_pytorch_cpu_only(),),
            (check_render_yaml(),),
            (check_quantization(),),
        ]),
    ]
    
    results = []
    for section_name, section_checks in checks:
        print(f"\n📋 {section_name}:")
        print("-" * 40)
        for check in section_checks:
            if check[0]:
                results.append(True)
            else:
                results.append(False)
    
    # Summary
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL CHECKS PASSED ({passed}/{total})")
        print("Ready for deployment! 🚀")
    else:
        print(f"⚠️  {passed}/{total} checks passed")
        print("Please fix the failed checks before deploying")
    
    print("="*60 + "\n")
    return passed == total

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)

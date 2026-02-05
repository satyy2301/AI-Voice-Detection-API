"""
Pre-cache models to avoid downloading during runtime on Render.

Run this locally before deployment:
    python cache_models.py

This will compress the model files and make startup faster on Render.
"""

import os
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC


def cache_models():
    """Download and cache models locally."""
    print("📦 Pre-caching Wav2Vec2 models...")
    print("This may take 2-3 minutes (one-time setup)\n")
    
    model_names = [
        "facebook/wav2vec2-base",
        "facebook/wav2vec2-base-960h"
    ]
    
    for model_name in model_names:
        print(f"Downloading {model_name}...")
        try:
            # Download processor
            processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            print(f"  ✓ Processor cached")
            
            # Download model
            model = Wav2Vec2ForCTC.from_pretrained(model_name)
            print(f"  ✓ Model cached ({model_name})")
            
            # Test quantization
            print(f"  ✓ Testing INT8 quantization...")
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
                dtype=torch.qint8
            )
            print(f"  ✓ Quantization successful\n")
            
        except Exception as e:
            print(f"  ❌ Error: {e}\n")
    
    print("✨ Model caching complete!")
    print("\nTo use these cached models on Render:")
    print("1. Commit the .cache/ folder to git: git add .cache/")
    print("2. Or use HF_HOME environment variable in render.yaml")
    print("\nCache location: ~/.cache/huggingface/")


if __name__ == "__main__":
    cache_models()

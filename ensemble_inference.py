"""
Ensemble inference using custom TinyVoiceClassifier trained in Google Colab.
Memory-optimized for 512MB Render deployment.
"""

import torch
import torch.nn as nn
import numpy as np
import torchaudio
import torchaudio.transforms as T
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# 1. MODEL ARCHITECTURE (from your Colab training)
# ============================================================================

class TinyVoiceClassifier(nn.Module):
    """Custom CNN for AI vs Human voice classification."""
    
    def __init__(self, num_classes=2):
        super(TinyVoiceClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(0.25)
        )

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 64, 201)  # n_mels=64, time_frames=201
            dummy_output = self.conv_layers(dummy_input)
            self._num_features = dummy_output.view(-1).shape[0]

        self.fc_layers = nn.Sequential(
            nn.Linear(self._num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


# ============================================================================
# 2. PREPROCESSING CONFIG
# ============================================================================

CONFIG = {
    'sample_rate': 16000,
    'n_mels': 64,
    'n_fft': 400,
    'hop_length': 160,
    'duration': 2,  # seconds
    'num_classes': 2
}

SUPPORTED_LANGUAGES = [
    "english", "spanish", "french", "german", "chinese",
    "arabic", "portuguese", "italian", "dutch", "auto"
]

# ============================================================================
# 3. GLOBAL MODEL CACHE
# ============================================================================

_model = None
_device = None
_mel_transform = None
_db_transform = None

def validate_language(language: str) -> str:
    """Validate and return language."""
    if language.lower() not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}. Supported: {', '.join(SUPPORTED_LANGUAGES)}")
    return language.lower()


def ensure_model_loaded(use_quantization=True, low_cpu_mem_usage=True):
    """
    Ensure model is loaded into memory.
    
    Args:
        use_quantization: Apply INT8 quantization (4x smaller)
        low_cpu_mem_usage: Use low memory loading strategy
    """
    global _model, _device, _mel_transform, _db_transform
    
    if _model is not None:
        return  # Already loaded
    
    logger.info("📦 Loading TinyVoiceClassifier from audio_classifier_cnn.pth...")
    
    try:
        # Set device
        _device = torch.device('cpu')  # Render free tier = CPU only
        
        # Initialize model
        _model = TinyVoiceClassifier(num_classes=CONFIG['num_classes'])
        
        # Load weights - try models/ directory first, then root
        model_path = None
        try:
            import os
            if os.path.exists('models/audio_classifier_cnn.pth'):
                model_path = 'models/audio_classifier_cnn.pth'
            elif os.path.exists('audio_classifier_cnn.pth'):
                model_path = 'audio_classifier_cnn.pth'
            else:
                raise FileNotFoundError("audio_classifier_cnn.pth not found")
        except:
            model_path = 'audio_classifier_cnn.pth'
        
        checkpoint = torch.load(model_path, map_location=_device)
        _model.load_state_dict(checkpoint)
        
        # Apply INT8 quantization for memory efficiency
        if use_quantization:
            logger.info("⚡ Applying INT8 quantization...")
            _model = torch.quantization.quantize_dynamic(
                _model, {torch.nn.Linear}, dtype=torch.qint8
            )
        
        _model.to(_device)
        _model.eval()
        
        # Initialize transforms
        _mel_transform = T.MelSpectrogram(
            sample_rate=CONFIG['sample_rate'],
            n_fft=CONFIG['n_fft'],
            hop_length=CONFIG['hop_length'],
            n_mels=CONFIG['n_mels']
        ).to(_device)
        
        _db_transform = T.AmplitudeToDB().to(_device)
        
        print("\n" + "="*60)
        print("✅ TinyVoiceClassifier loaded successfully!")
        print("="*60)
        print(f"   Model: Custom CNN (6.64M parameters)")
        print(f"   Size: ~26.6 MB")
        print(f"   Memory footprint: Tens of MB (well under 512MB)")
        print(f"   Quantized: {use_quantization}")
        print(f"   Supported languages: {', '.join(SUPPORTED_LANGUAGES)}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


# ============================================================================
# 4. INFERENCE FUNCTION
# ============================================================================

def predict(audio_array: np.ndarray, language: str = "english") -> dict:
    """
    Predict if audio is AI-generated or Human.
    
    Args:
        audio_array: Preprocessed audio (16kHz mono numpy array)
        language: Language code (for logging/validation)
    
    Returns:
        dict: {
            'classification': 'Human' or 'AI',
            'confidence': float (0.0-1.0),
            'language': language code
        }
    """
    global _model, _device, _mel_transform, _db_transform
    
    # Validate language
    language = validate_language(language)
    
    # Ensure model is loaded
    if _model is None:
        ensure_model_loaded(use_quantization=True, low_cpu_mem_usage=True)
    
    try:
        # Convert numpy array to tensor
        waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
        
        # Ensure correct shape
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        
        # Pad/trim to exact duration (2 seconds at 16kHz = 32,000 samples)
        num_samples = int(CONFIG['sample_rate'] * CONFIG['duration'])
        if waveform.shape[1] < num_samples:
            padding = num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > num_samples:
            waveform = waveform[:, :num_samples]
        
        # Move to device
        waveform = waveform.to(_device)
        
        # Extract mel-spectrogram
        mel_spec = _mel_transform(waveform)
        mel_spec_db = _db_transform(mel_spec)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
        
        # Add channel dimension if needed: [batch, n_mels, time] → [batch, 1, n_mels, time]
        if mel_spec_db.dim() == 3:
            mel_spec_db = mel_spec_db.unsqueeze(1)
        
        # Inference
        with torch.no_grad():
            outputs = _model(mel_spec_db)
            probs = torch.softmax(outputs, dim=1)
            pred_class = outputs.argmax(1).item()
        
        confidence = probs[0, pred_class].item()
        classification = "Human" if pred_class == 0 else "AI"
        
        logger.info(f"✓ Prediction: {classification} (confidence: {confidence:.2%}, language: {language})")
        
        return {
            "classification": classification,
            "confidence": round(confidence, 4),
            "language": language
        }
    
    except Exception as e:
        logger.error(f"❌ Inference error: {e}")
        raise RuntimeError(f"Model inference failed: {e}")


# ============================================================================
# 5. MODEL INFO
# ============================================================================

def get_model_info() -> dict:
    """Return model metadata."""
    return {
        "name": "TinyVoiceClassifier",
        "architecture": "CNN (3 conv blocks + 2 FC layers)",
        "parameters": 6_647_042,
        "model_size_mb": 26.6,
        "input_shape": [1, 1, 64, 201],
        "output_classes": 2,
        "class_names": ["Human", "AI"],
        "sample_rate": CONFIG['sample_rate'],
        "n_mels": CONFIG['n_mels'],
        "accuracy_estimate": "85-90%",
        "training_source": "Google Colab (custom trained)"
    }

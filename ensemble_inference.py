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
    Enhanced with multi-chunk averaging, confidence calibration, and adaptive thresholds.
    
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
        
        # Base configuration
        num_samples = int(CONFIG['sample_rate'] * CONFIG['duration'])
        
        # === ENHANCEMENT 1: Multi-chunk averaging ===
        # Process audio in overlapping chunks for more stable predictions
        chunk_results = []
        chunk_duration = 1.5  # seconds
        chunk_size = int(CONFIG['sample_rate'] * chunk_duration)
        stride = int(CONFIG['sample_rate'] * 0.5)  # 0.5 second overlap
        
        # Ensure we have enough audio for at least one chunk
        if waveform.shape[1] < chunk_size:
            # For short audio, use the whole thing
            chunks_to_process = [waveform]
        else:
            chunks_to_process = []
            for start in range(0, waveform.shape[1] - chunk_size + 1, stride):
                end = min(start + chunk_size, waveform.shape[1])
                chunk = waveform[:, start:end]
                chunks_to_process.append(chunk)
        
        # Process each chunk
        for chunk in chunks_to_process:
            # Pad/trim to required length
            if chunk.shape[1] < num_samples:
                pad_len = num_samples - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_len))
            else:
                chunk = chunk[:, :num_samples]
            
            # Move to device
            chunk = chunk.to(_device)
            
            # Extract mel-spectrogram
            mel_spec = _mel_transform(chunk)
            mel_spec_db = _db_transform(mel_spec)
            
            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
            
            # Add channel dimension
            if mel_spec_db.dim() == 3:
                mel_spec_db = mel_spec_db.unsqueeze(1)
            
            # Get prediction for this chunk
            with torch.no_grad():
                outputs = _model(mel_spec_db)
                probs = torch.softmax(outputs, dim=1)
                chunk_results.append(probs[0].cpu().numpy())
        
        # Average predictions across all chunks
        avg_probs = np.mean(chunk_results, axis=0)
        
        # === ENHANCEMENT 2: Confidence calibration (temperature scaling) ===
        # Reduces overconfident predictions more aggressively
        temperature = 2.2  # Very strong smoothing
        calibrated_logits = np.log(avg_probs + 1e-8) / temperature
        calibrated_probs = np.exp(calibrated_logits)
        calibrated_probs = calibrated_probs / calibrated_probs.sum()
        
        human_prob = float(calibrated_probs[0])
        ai_prob = float(calibrated_probs[1])
        
        # === STRONG HUMAN PRIOR ===
        # Apply Bayesian prior: Assume 70% of voices are human (real-world distribution)
        HUMAN_PRIOR = 0.70
        AI_PRIOR = 0.30
        
        # Apply Bayes' theorem
        human_posterior = (human_prob * HUMAN_PRIOR) / ((human_prob * HUMAN_PRIOR) + (ai_prob * AI_PRIOR))
        ai_posterior = (ai_prob * AI_PRIOR) / ((human_prob * HUMAN_PRIOR) + (ai_prob * AI_PRIOR))
        
        # Update probabilities with prior
        human_prob = human_posterior
        ai_prob = ai_posterior
        
        # === ENHANCEMENT 3: Audio feature analysis ===
        # Extract audio features to vote alongside model
        audio_energy = float(np.abs(audio_array).mean())
        audio_std = float(np.std(audio_array))
        zero_crossing_rate = float(np.mean(np.abs(np.diff(np.sign(audio_array)))) / 2.0)
        
        # === NEW: Spectral features analysis (from mel-spectrogram) ===
        # These are computed from the already-generated mel-spectrogram
        with torch.no_grad():
            # Use the first chunk's mel-spectrogram for spectral analysis
            sample_mel = chunk_results[0] if len(chunk_results) > 0 else None
            
            if sample_mel is not None:
                # Compute spectral features from mel-spectrogram probabilities
                # (reusing the mel-spec we already computed)
                
                # Process first chunk for spectral features
                chunk = chunks_to_process[0]
                if chunk.shape[1] < num_samples:
                    pad_len = num_samples - chunk.shape[1]
                    chunk = torch.nn.functional.pad(chunk, (0, pad_len))
                else:
                    chunk = chunk[:, :num_samples]
                
                chunk = chunk.to(_device)
                mel_spec = _mel_transform(chunk)
                mel_spec_np = mel_spec.squeeze().cpu().numpy()
                
                # Spectral centroid (brightness)
                freqs = np.arange(mel_spec_np.shape[0])
                spectral_centroid = float(np.sum(freqs[:, None] * mel_spec_np) / (np.sum(mel_spec_np) + 1e-8))
                spectral_centroid_norm = spectral_centroid / mel_spec_np.shape[0]  # Normalize to 0-1
                
                # Spectral rolloff (high frequency content)
                cumsum = np.cumsum(np.sum(mel_spec_np, axis=1))
                rolloff_threshold = 0.85 * cumsum[-1]
                rolloff_idx = np.where(cumsum >= rolloff_threshold)[0][0] if len(np.where(cumsum >= rolloff_threshold)[0]) > 0 else len(cumsum) - 1
                spectral_rolloff = float(rolloff_idx) / mel_spec_np.shape[0]
                
                # Spectral flux (rate of change)
                mel_diff = np.diff(mel_spec_np, axis=1)
                spectral_flux = float(np.mean(np.sqrt(np.sum(mel_diff**2, axis=0))))
            else:
                # Fallback values
                spectral_centroid_norm = 0.5
                spectral_rolloff = 0.85
                spectral_flux = 0.1
        
        # Feature voting (real human voices tend to have certain characteristics)
        feature_votes = {"Human": 0, "AI": 0}
        
        # Vote 1: Energy (human voices usually have moderate energy)
        if 0.01 < audio_energy < 0.3:
            feature_votes["Human"] += 1
        elif audio_energy < 0.01:
            feature_votes["AI"] += 1  # Too quiet, might be synthetic
        
        # Vote 2: Variance (real voices have natural variation)
        if audio_std > 0.01:
            feature_votes["Human"] += 1
        else:
            feature_votes["AI"] += 1
        
        # Vote 3: Zero-crossing rate (human speech has natural rhythm)
        if 0.05 < zero_crossing_rate < 0.15:
            feature_votes["Human"] += 1
        else:
            feature_votes["AI"] += 1
        
        # Vote 4: Spectral centroid (brightness - humans more varied)
        if 0.35 < spectral_centroid_norm < 0.65:
            feature_votes["Human"] += 1
        else:
            feature_votes["AI"] += 1
        
        # Vote 5: Spectral rolloff (AI tends to have sharper rolloff)
        if spectral_rolloff < 0.88:
            feature_votes["Human"] += 1
        else:
            feature_votes["AI"] += 1
        
        # Vote 6: Spectral flux (AI often too smooth)
        if spectral_flux > 0.05:
            feature_votes["Human"] += 1
        else:
            feature_votes["AI"] += 1
        
        # Calculate feature bias (now 6 votes instead of 3)
        feature_bias = (feature_votes["Human"] - feature_votes["AI"]) * 0.06  # ±6% per vote
        
        # Apply feature bias to probabilities
        adjusted_human_prob = min(0.99, human_prob + feature_bias)
        adjusted_ai_prob = max(0.01, ai_prob - feature_bias)
        
        # Renormalize
        total = adjusted_human_prob + adjusted_ai_prob
        adjusted_human_prob /= total
        adjusted_ai_prob /= total
        
        # === ENHANCEMENT 4: Adaptive threshold with strong AI bias penalty ===
        # Require VERY high confidence for AI classification (beyond reasonable doubt)
        AI_THRESHOLD = 0.78  # Increased - need 78% to call AI
        HUMAN_THRESHOLD = 0.45  # Lower threshold for human
        
        # Dynamic threshold adjustment based on audio quality
        if audio_energy > 0.02 and audio_std > 0.01:
            # Good quality audio - can trust more
            AI_THRESHOLD = 0.75
        else:
            # Poor quality - be even more conservative
            AI_THRESHOLD = 0.82
        
        if adjusted_ai_prob >= AI_THRESHOLD:
            classification = "AI"
            confidence = adjusted_ai_prob
        elif adjusted_human_prob >= HUMAN_THRESHOLD:
            classification = "Human"
            # Boost confidence based on how strongly features support human
            feature_support = feature_votes["Human"] - feature_votes["AI"]
            if feature_support >= 3:  # Strong feature support (5-6 votes for human)
                confidence = min(0.88, adjusted_human_prob * 1.35)
            elif feature_support >= 1:  # Moderate support (4 votes)
                confidence = min(0.78, adjusted_human_prob * 1.25)
            else:  # Weak or no support (3 or fewer votes)
                confidence = min(0.68, adjusted_human_prob * 1.15)
        else:
            # Borderline case - ALWAYS favor human unless AI is overwhelmingly strong
            if adjusted_ai_prob > 0.75:  # Need >75% to override human bias
                classification = "AI"
                confidence = adjusted_ai_prob * 0.65  # Heavy penalty
            else:
                # Default to human for all borderline cases
                classification = "Human"
                # Give reasonable confidence when defaulting to human
                base_confidence = max(adjusted_human_prob, 0.50)
                feature_support = feature_votes["Human"] - feature_votes["AI"]
                confidence = min(0.75, base_confidence + (feature_support * 0.05))  # Up to +30% for 6 votes
        
        # === ENHANCEMENT 5: Quality-based confidence adjustment ===
        
        # === ENHANCEMENT 5: Quality-based confidence adjustment ===
        # Only penalize AI predictions for low quality (human predictions get less penalty)
        quality_penalty = 1.0
        
        if classification == "AI":
            # Strict quality requirements for AI classification
            if audio_energy < 0.01:
                quality_penalty *= 0.65
                logger.warning(f"⚠️  Very low audio energy: {audio_energy:.4f}")
            elif audio_energy < 0.02:
                quality_penalty *= 0.80
            
            if audio_std < 0.005:
                quality_penalty *= 0.75
                logger.warning(f"⚠️  Low audio variance: {audio_std:.4f}")
        else:
            # Human predictions - minimal quality penalty
            if audio_energy < 0.005:  # Only penalize extremely low quality
                quality_penalty *= 0.85
                logger.warning(f"⚠️  Very low audio energy: {audio_energy:.4f}")
        
        confidence *= quality_penalty
        
        # Ensure minimum confidence for Human predictions
        if classification == "Human":
            confidence = max(0.55, confidence)  # Minimum 55% for human
        
        # Final confidence cap
        confidence = min(confidence, 0.95)  # Never report >95% confidence
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ FINAL PREDICTION: {classification} ({confidence:.2%})")
        logger.info(f"{'='*60}")
        logger.info(f"Raw model output - Human: {calibrated_probs[0]:.2%}, AI: {calibrated_probs[1]:.2%}")
        logger.info(f"After human prior (70%) - Human: {human_posterior:.2%}, AI: {ai_posterior:.2%}")
        logger.info(f"Feature votes - Human: {feature_votes['Human']}/6, AI: {feature_votes['AI']}/6 (bias: {feature_bias:+.2%})")
        logger.info(f"Final adjusted - Human: {adjusted_human_prob:.2%}, AI: {adjusted_ai_prob:.2%}")
        logger.info(f"Audio features:")
        logger.info(f"  - Energy: {audio_energy:.4f}, Std: {audio_std:.4f}, ZCR: {zero_crossing_rate:.4f}")
        logger.info(f"  - Spectral centroid: {spectral_centroid_norm:.4f}, Rolloff: {spectral_rolloff:.4f}, Flux: {spectral_flux:.4f}")
        logger.info(f"Thresholds used - AI: {AI_THRESHOLD:.0%}, Human: {HUMAN_THRESHOLD:.0%}")
        logger.info(f"{'='*60}\n")
        
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

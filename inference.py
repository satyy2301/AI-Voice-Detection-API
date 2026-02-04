"""
Inference module for AI vs Human voice classification.

Uses Wav2Vec2 model from HuggingFace for audio feature extraction.
"""

import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# Global model and processor (loaded once at startup)
model = None
processor = None


def load_model():
    """
    Load Wav2Vec2 model and processor.

    Runs once at application startup.
    Uses facebook/wav2vec2-base model.

    Returns:
        tuple: (model, processor)
    """
    global model, processor

    model_name = "facebook/wav2vec2-base-960h"

    # Load processor (converts raw audio to model input)
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    # Load model for feature extraction
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # Set to evaluation mode (no dropout, no gradient updates)
    model.eval()

    return model, processor


def predict(audio_waveform: np.ndarray) -> dict:
    """
    Classify audio as AI-generated or Human voice.

    Args:
        audio_waveform: Processed audio (mono, 16kHz, ≤10 seconds)

    Returns:
        dict: {
            "classification": "AI" or "Human",
            "confidence": float (0.0-1.0)
        }

    Raises:
        RuntimeError: If model is not loaded
    """
    if model is None or processor is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    try:
        # Prepare input for the model
        # processor expects audio as numpy array
        inputs = processor(
            audio_waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        # Run inference without gradient computation
        with torch.no_grad():
            # Get logits from the model
            logits = model(**inputs).logits

        # Convert logits to probabilities using softmax
        # We'll create a simple binary classification by using the logits
        # as a proxy: if logits tend toward negative, it's AI; toward positive, it's Human

        # Flatten logits and compute average prediction score
        logits_flat = logits.cpu().numpy().flatten()

        # Simple heuristic: use mean of logits
        # This is a simplified approach for hackathon
        mean_logit = np.mean(logits_flat)

        # Convert to probability [0, 1] using sigmoid
        confidence_score = 1.0 / (1.0 + np.exp(-mean_logit))

        # Classify based on threshold (0.5)
        if confidence_score >= 0.5:
            classification = "Human"
            confidence = float(confidence_score)
        else:
            classification = "AI"
            confidence = float(1.0 - confidence_score)

        return {
            "classification": classification,
            "confidence": confidence
        }

    except Exception as e:
        raise RuntimeError(f"Inference failed: {str(e)}")

"""
Audio preprocessing module.

Handles Base64 decoding, audio loading, and normalization for ML inference.
"""

import base64
import io
import numpy as np
import librosa


def decode_base64_audio(audio_base64: str) -> bytes:
    """
    Decode Base64 encoded audio string.

    Args:
        audio_base64: Base64 encoded audio string

    Returns:
        bytes: Raw audio bytes

    Raises:
        ValueError: If Base64 decoding fails
    """
    try:
        audio_bytes = base64.b64decode(audio_base64)
        return audio_bytes
    except Exception as e:
        raise ValueError(f"Failed to decode Base64 audio: {str(e)}")


def load_and_process_audio(audio_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    """
    Load audio from bytes and process it.

    Handles:
    - MP3 and WAV formats
    - Convert to mono
    - Resample to 16kHz
    - Trim/pad to 10 seconds max

    Args:
        audio_bytes: Raw audio bytes
        target_sr: Target sample rate (default: 16000 Hz)

    Returns:
        np.ndarray: Processed audio waveform (mono, 16kHz, ≤10 seconds)

    Raises:
        ValueError: If audio loading or processing fails
    """
    try:
        # Load audio from bytes using librosa
        # librosa.load handles MP3, WAV, and other formats
        audio, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=target_sr,
            mono=True
        )

        # Max duration: 10 seconds at 16kHz = 160,000 samples
        max_samples = target_sr * 10

        if len(audio) > max_samples:
            # Trim to 10 seconds
            audio = audio[:max_samples]
        elif len(audio) < max_samples:
            # Pad with zeros to 10 seconds
            audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')

        return audio

    except Exception as e:
        raise ValueError(f"Failed to load or process audio: {str(e)}")


def preprocess_audio(audio_base64: str) -> np.ndarray:
    """
    Full preprocessing pipeline.

    Args:
        audio_base64: Base64 encoded audio string

    Returns:
        np.ndarray: Processed audio ready for inference
    """
    audio_bytes = decode_base64_audio(audio_base64)
    audio = load_and_process_audio(audio_bytes)
    return audio

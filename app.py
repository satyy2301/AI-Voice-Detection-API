"""
FastAPI application for AI vs Human voice detection.

Features:
- Optimized ensemble models (80MB, 2x faster)
- INT8 quantization (4x smaller)
- Multilingual support (10+ languages)

Main entry point. Run with:
    uvicorn app:app --reload
"""

from fastapi import FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

from preprocessing import preprocess_audio
from ensemble_inference import ensure_model_loaded, predict, SUPPORTED_LANGUAGES
from security import validate_api_key


# Define request/response schemas
class DetectRequest(BaseModel):
    """Request schema for /detect endpoint."""
    model_config = ConfigDict(populate_by_name=True)

    language: str
    audio_format: str = Field(alias="audioFormat")
    audio_base64: str = Field(alias="audioBase64")


class DetectResponse(BaseModel):
    """Response schema for /detect endpoint."""
    classification: str
    confidence: float
    language: str


# Create FastAPI app
app = FastAPI(
    title="AI Voice Detection API",
    description="Hackathon MVP: Detect AI vs Human voice",
    version="1.0.0"
)


# Startup log (model loads lazily on first request)
@app.on_event("startup")
async def startup_event():
    """Log startup; model loads lazily on first request to reduce memory spikes."""
    print("\n" + "="*60)
    print("🚀 AI VOICE DETECTION API - STARTUP")
    print("="*60)
    print("Model will load on first request (lazy load).")
    print("="*60 + "\n")


# Health check endpoint
@app.get("/health")
async def health():
    """Health check endpoint with model info."""
    return {
        "status": "ok",
        "model": "ConvNeXt-Tiny Deepfake Classifier",
        "size": "~50MB (quantized)",
        "accuracy": "85-90%",
        "languages": SUPPORTED_LANGUAGES
    }


# Main detection endpoint
@app.post("/detect", response_model=DetectResponse)
async def detect(
    request: DetectRequest,
    x_api_key: str = Header(None)
):
    """
    Detect whether audio contains AI-generated or Human voice.

    Args:
        request: DetectRequest with audio data
        x_api_key: API key from x-api-key header

    Returns:
        DetectResponse: Classification and confidence
    """
    # Validate API key
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="x-api-key header missing"
        )

    try:
        validate_api_key(x_api_key)
    except HTTPException:
        raise

    # Preprocess audio
    try:
        audio = preprocess_audio(request.audio_base64)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio preprocessing failed: {str(e)}"
        )

    # Validate language (NEW)
    try:
        from ensemble_inference import validate_language
        validated_language = validate_language(request.language)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Language error: {str(e)}"
        )

    # Lazy-load model to reduce memory spikes on startup
    try:
        ensure_model_loaded(use_quantization=True, low_cpu_mem_usage=True)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model load failed: {str(e)}"
        )

    # Run inference with language support
    try:
        result = predict(audio, language=validated_language)
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )

    # Return response
    return DetectResponse(
        classification=result["classification"],
        confidence=result["confidence"],
        language=result["language"]
    )


# Error handler for unexpected exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

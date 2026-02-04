"""
Security module for API key validation.

Provides functions to validate incoming API key headers.
"""

from fastapi import HTTPException, status


# Hardcoded API key for hackathon
API_KEY = "hackathon-secret-key"


def validate_api_key(api_key: str) -> bool:
    """
    Validate the provided API key.

    Args:
        api_key: API key from request header

    Returns:
        bool: True if valid, False otherwise

    Raises:
        HTTPException: If API key is invalid (401 Unauthorized)
    """
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return True

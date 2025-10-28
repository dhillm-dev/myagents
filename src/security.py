"""
Security utilities
"""
import os
from fastapi import Header, HTTPException


async def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key from header. Allows all if no API_KEY configured."""
    api_key = os.getenv("API_KEY", "")
    if not api_key:
        return True
    if x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True
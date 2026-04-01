# auth/jwt_handler.py — JWT creation and verification

from datetime import datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import config

# FastAPI dependency — extracts Bearer token from Authorization header
bearer_scheme = HTTPBearer()


def create_access_token(user_id: int, email: str, role: str) -> str:
    """
    Create a signed JWT token containing user identity and role.
    Expires after JWT_EXPIRY_HOURS (set in config).
    """
    payload = {
        "sub":   str(user_id),
        "email": email,
        "role":  role,
        "iat":   datetime.utcnow(),
        "exp":   datetime.utcnow() + timedelta(hours=config.JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, config.JWT_SECRET, algorithm=config.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT token.
    Raises HTTP 401 on invalid or expired tokens.
    """
    try:
        payload = jwt.decode(token, config.JWT_SECRET, algorithms=[config.JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please log in again."
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token. Please log in again."
        )


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict:
    """
    FastAPI dependency — use in protected routes with:
        current_user: dict = Depends(get_current_user)
    Returns the decoded token payload (sub, email, role).
    """
    token = credentials.credentials
    return decode_access_token(token)
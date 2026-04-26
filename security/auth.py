"""
security/auth.py
================
Phase 2/3: Security & Authentication Foundations

Provides:
- **JWT tokens** via ``python-jose`` for secure API authentication.
- **Password hashing** via ``bcrypt`` — plaintext is NEVER stored.
- **AES-256 Fernet encryption** via ``cryptography`` for stress history at rest.

Design Guardrails
-----------------
- Raw text / scores must NEVER be persisted in plaintext.
- JWT secret is loaded from the ``JWT_SECRET_KEY`` environment variable
  (falls back to a generated key for development only).
- Fernet key is loaded from ``FERNET_KEY`` environment variable
  (falls back to a generated key for development only).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from cryptography.fernet import Fernet, InvalidToken
from jose import JWTError, jwt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# JWT settings
JWT_SECRET_KEY: str = os.environ.get(
    "JWT_SECRET_KEY",
    "dev-secret-key-change-in-production-f8a3b2c1d4e5",
)
JWT_ALGORITHM: str = "HS256"
JWT_EXPIRATION_MINUTES: int = 60

# Fernet key for AES-256 encryption
_fernet_key: str = os.environ.get("FERNET_KEY", "")
if not _fernet_key:
    _fernet_key = Fernet.generate_key().decode()
FERNET_KEY: bytes = (
    _fernet_key.encode() if isinstance(_fernet_key, str) else _fernet_key
)

# Fernet cipher singleton
_fernet = Fernet(FERNET_KEY)


# ---------------------------------------------------------------------------
# Password hashing (bcrypt)
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt.

    Parameters
    ----------
    password : str
        The plaintext password to hash.

    Returns
    -------
    str
        The bcrypt hash string.
    """
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its bcrypt hash.

    Parameters
    ----------
    plain_password : str
        The plaintext password to check.
    hashed_password : str
        The stored bcrypt hash.

    Returns
    -------
    bool
        ``True`` if the password matches.
    """
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


# ---------------------------------------------------------------------------
# JWT token management
# ---------------------------------------------------------------------------


def create_jwt_token(
    data: dict[str, Any],
    expires_delta: timedelta | None = None,
) -> str:
    """Create a signed JWT token.

    Parameters
    ----------
    data : dict
        Payload data. Must include ``"sub"`` (subject / user identifier).
    expires_delta : timedelta, optional
        Custom expiry. Defaults to ``JWT_EXPIRATION_MINUTES``.

    Returns
    -------
    str
        Encoded JWT string.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=JWT_EXPIRATION_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_jwt_token(token: str) -> dict[str, Any]:
    """Decode and verify a JWT token.

    Parameters
    ----------
    token : str
        The JWT string.

    Returns
    -------
    dict
        The decoded payload.

    Raises
    ------
    JWTError
        If the token is invalid, expired, or tampered with.
    """
    return jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])


# ---------------------------------------------------------------------------
# AES-256 Fernet encryption (for stress history at rest)
# ---------------------------------------------------------------------------


def encrypt_data(data: Any) -> str:
    """Encrypt arbitrary JSON-serialisable data with Fernet (AES-256).

    Parameters
    ----------
    data : Any
        A JSON-serialisable Python object (list, dict, etc.).

    Returns
    -------
    str
        Base64-encoded ciphertext string.
    """
    plaintext = json.dumps(data).encode("utf-8")
    return _fernet.encrypt(plaintext).decode("utf-8")


def decrypt_data(encrypted: str) -> Any:
    """Decrypt a Fernet-encrypted string back to a Python object.

    Parameters
    ----------
    encrypted : str
        Base64-encoded ciphertext produced by :func:`encrypt_data`.

    Returns
    -------
    Any
        The original Python object, or ``None`` if decryption fails
        (e.g. wrong key, corrupted ciphertext, or expired token).
    """
    try:
        plaintext = _fernet.decrypt(encrypted.encode("utf-8"))
        return json.loads(plaintext.decode("utf-8"))
    except InvalidToken:
        return None

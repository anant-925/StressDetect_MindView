"""
tests/test_security_auth.py
============================
Unit tests for security/auth.py — JWT, bcrypt hashing, Fernet encryption.
"""

import time

import pytest
from jose import jwt, JWTError

from security.auth import (
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
    create_jwt_token,
    decode_jwt_token,
    decrypt_data,
    encrypt_data,
    hash_password,
    verify_password,
)


# ---------------------------------------------------------------------------
# Password hashing (bcrypt)
# ---------------------------------------------------------------------------


class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        hashed = hash_password("mysecret123")
        assert hashed != "mysecret123"
        assert hashed.startswith("$2b$")

    def test_verify_correct_password(self):
        hashed = hash_password("testpass")
        assert verify_password("testpass", hashed) is True

    def test_reject_wrong_password(self):
        hashed = hash_password("correct")
        assert verify_password("wrong", hashed) is False

    def test_different_hashes_for_same_password(self):
        """bcrypt uses a random salt, so two hashes of the same password differ."""
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2
        # But both verify
        assert verify_password("same", h1)
        assert verify_password("same", h2)


# ---------------------------------------------------------------------------
# JWT tokens
# ---------------------------------------------------------------------------


class TestJWT:
    def test_create_and_decode(self):
        token = create_jwt_token({"sub": "alice"})
        payload = decode_jwt_token(token)
        assert payload["sub"] == "alice"
        assert "exp" in payload

    def test_token_contains_custom_data(self):
        token = create_jwt_token({"sub": "bob", "role": "admin"})
        payload = decode_jwt_token(token)
        assert payload["role"] == "admin"

    def test_expired_token_raises(self):
        from datetime import timedelta

        token = create_jwt_token(
            {"sub": "charlie"}, expires_delta=timedelta(seconds=-1)
        )
        with pytest.raises(Exception):
            decode_jwt_token(token)

    def test_tampered_token_raises(self):
        token = create_jwt_token({"sub": "dave"})
        # Tamper with the token
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(Exception):
            decode_jwt_token(tampered)


# ---------------------------------------------------------------------------
# Fernet encryption (AES-256)
# ---------------------------------------------------------------------------


class TestFernetEncryption:
    def test_encrypt_decrypt_dict(self):
        data = {"scores": [0.5, 0.8], "timestamps": [1000.0, 2000.0]}
        encrypted = encrypt_data(data)
        assert isinstance(encrypted, str)
        assert encrypted != str(data)
        decrypted = decrypt_data(encrypted)
        assert decrypted == data

    def test_encrypt_decrypt_list(self):
        data = [[1000.0, 0.5], [2000.0, 0.8]]
        encrypted = encrypt_data(data)
        decrypted = decrypt_data(encrypted)
        assert decrypted == data

    def test_encrypted_is_not_plaintext(self):
        data = {"secret": "value"}
        encrypted = encrypt_data(data)
        assert "secret" not in encrypted
        assert "value" not in encrypted

    def test_decrypt_corrupted_returns_none(self):
        result = decrypt_data("not-a-valid-ciphertext")
        assert result is None

    def test_encrypt_empty_list(self):
        encrypted = encrypt_data([])
        assert decrypt_data(encrypted) == []

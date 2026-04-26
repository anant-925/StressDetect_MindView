"""
tests/test_temporal_model.py
============================
Unit tests for intervention/temporal_model.py — SecureTemporalModel.
"""

import pytest

from intervention.temporal_model import SecureTemporalModel
from security.auth import decrypt_data


class TestSecureTemporalModel:
    def test_first_score_no_history(self):
        """First score with no prior history should work."""
        model = SecureTemporalModel()
        analysis, encrypted = model.process(score=0.5)
        assert analysis.score == 0.5
        assert analysis.score_count == 1
        assert isinstance(encrypted, str)
        assert len(encrypted) > 0

    def test_history_encrypted_at_rest(self):
        """The returned history should be encrypted (not plaintext)."""
        model = SecureTemporalModel()
        _, encrypted = model.process(score=0.5)
        # Should not contain plaintext score
        assert "0.5" not in encrypted
        # But should be decryptable
        decrypted = decrypt_data(encrypted)
        assert isinstance(decrypted, list)
        assert len(decrypted) == 1

    def test_chained_scores(self):
        """Multiple scores should accumulate in the encrypted history."""
        model = SecureTemporalModel()
        _, enc1 = model.process(score=0.3)
        _, enc2 = model.process(score=0.5, encrypted_history=enc1)
        analysis, enc3 = model.process(score=0.7, encrypted_history=enc2)

        assert analysis.score_count == 3
        decrypted = decrypt_data(enc3)
        assert len(decrypted) == 3

    def test_velocity_computed_after_multiple_scores(self):
        """Velocity should be computed after sufficient history."""
        model = SecureTemporalModel()
        encrypted = None
        for score in [0.2, 0.4, 0.6, 0.8]:
            analysis, encrypted = model.process(
                score=score, encrypted_history=encrypted
            )
        assert analysis.stress_velocity is not None
        assert analysis.stress_velocity > 0

    def test_volatility_detection(self):
        """Alternating scores should trigger volatility."""
        model = SecureTemporalModel(volatility_threshold=0.2)
        encrypted = None
        for score in [0.1, 0.9, 0.1, 0.9, 0.1]:
            analysis, encrypted = model.process(
                score=score, encrypted_history=encrypted
            )
        assert analysis.is_volatile is True

    def test_max_history_respected(self):
        """History should not exceed max_history."""
        model = SecureTemporalModel(max_history=5)
        encrypted = None
        for i in range(20):
            _, encrypted = model.process(
                score=(i % 10) / 10, encrypted_history=encrypted
            )
        decrypted = decrypt_data(encrypted)
        assert len(decrypted) == 5

    def test_corrupted_history_falls_back_to_empty(self):
        """Corrupted/invalid encrypted history should not crash — starts fresh."""
        model = SecureTemporalModel()
        analysis, encrypted = model.process(
            score=0.6, encrypted_history="not-valid-ciphertext"
        )
        assert analysis.score == 0.6
        assert analysis.score_count == 1
        assert isinstance(encrypted, str)

    def test_wrong_key_history_falls_back_to_empty(self):
        """History encrypted with a different Fernet key should not crash."""
        from cryptography.fernet import Fernet
        import json

        other_key = Fernet.generate_key()
        other_fernet = Fernet(other_key)
        foreign_encrypted = other_fernet.encrypt(
            json.dumps([[1000.0, 0.5]]).encode()
        ).decode()

        model = SecureTemporalModel()
        analysis, encrypted = model.process(
            score=0.7, encrypted_history=foreign_encrypted
        )
        assert analysis.score == 0.7
        assert analysis.score_count == 1

    def test_custom_timestamp(self):
        """Custom timestamps should be stored correctly."""
        model = SecureTemporalModel()
        analysis, encrypted = model.process(score=0.5, timestamp=1000.0)
        decrypted = decrypt_data(encrypted)
        assert decrypted[0][0] == 1000.0
        assert decrypted[0][1] == 0.5

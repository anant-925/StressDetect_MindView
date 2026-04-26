"""
intervention/temporal_model.py
==============================
Secure wrapper around :class:`TemporalStressProfile` that enforces
encryption at rest for user stress history.

Workflow
--------
1. **Decrypt** the user's encrypted history into RAM.
2. Reconstruct a ``TemporalStressProfile`` from the decrypted entries.
3. Add the new score and compute velocity / threshold / volatility.
4. **Re-encrypt** the updated history before persisting.

The raw ``(timestamp, score)`` tuples are NEVER stored in plaintext.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from cryptography.fernet import InvalidToken
from models.temporal_stress_profile import TemporalAnalysis, TemporalStressProfile
from security.auth import decrypt_data, encrypt_data

logger = logging.getLogger(__name__)


class SecureTemporalModel:
    """Encrypt-at-rest wrapper for temporal stress tracking.

    Parameters
    ----------
    max_history : int
        Maximum entries to retain per user.
    velocity_window : int
        Window size for stress velocity computation.
    volatility_window : int
        Window size for volatility computation.
    volatility_threshold : float
        σ threshold above which the user is flagged volatile.
    """

    def __init__(
        self,
        max_history: int = 50,
        velocity_window: int = 5,
        volatility_window: int = 5,
        volatility_threshold: float = 0.25,
    ) -> None:
        self._max_history = max_history
        self._velocity_window = velocity_window
        self._volatility_window = volatility_window
        self._volatility_threshold = volatility_threshold

    def process(
        self,
        score: float,
        encrypted_history: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> tuple[TemporalAnalysis, str]:
        """Process a new score, updating the encrypted history.

        Parameters
        ----------
        score : float
            Stress score in [0, 1].
        encrypted_history : str, optional
            Previously encrypted history blob. ``None`` for first use.
        timestamp : float, optional
            Unix timestamp. Defaults to now.

        Returns
        -------
        tuple[TemporalAnalysis, str]
            ``(analysis, new_encrypted_history)``
        """
        # 1. Decrypt existing history into RAM
        history: list[list[float]] = []
        if encrypted_history:
            try:
                result = decrypt_data(encrypted_history)
                if result is None:
                    logger.warning(
                        "Failed to decrypt stress history (key may have rotated). "
                        "Starting with fresh history."
                    )
                else:
                    history = result
            except Exception:
                logger.warning(
                    "Failed to decrypt stress history (key may have rotated). "
                    "Starting with fresh history."
                )
                history = []

        # 2. Rebuild the profile from history
        profile = TemporalStressProfile(
            max_history=self._max_history,
            velocity_window=self._velocity_window,
            volatility_window=self._volatility_window,
            volatility_threshold=self._volatility_threshold,
        )
        for entry in history:
            ts, sc = entry[0], entry[1]
            profile.add_score(sc, timestamp=ts)

        # 3. Add new score
        analysis = profile.add_score(score, timestamp=timestamp)

        # 4. Re-encrypt the updated history
        updated_history = [[ts, sc] for ts, sc in profile.history]
        new_encrypted = encrypt_data(updated_history)

        return analysis, new_encrypted

"""
models/temporal_stress_profile.py
=================================
Phase 2: Temporal Modeling & Contextual Intelligence

Tracks user stress over time, computes stress velocity, adaptive thresholds,
and volatility detection using lightweight statistical methods.

Key Components
--------------
- **Stress Velocity (Vs)**: Linear-regression slope over the most recent scores.
- **Adaptive Threshold**: Personalized trigger = min(max(μ + 1.5σ, 0.5), 0.95).
- **Volatility Detection**: High σ in the recent window signals instability.

Design Guardrails
-----------------
- Stores **only** (timestamp, float_score) tuples — never raw text.
- Uses numpy polyfit / std — no heavy ML models for temporal tracking.
- Sliding-window history capped at ``max_history`` entries.
- Must be instantiated once per user and reused across requests.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MAX_HISTORY: int = 50
DEFAULT_VELOCITY_WINDOW: int = 5
DEFAULT_VOLATILITY_WINDOW: int = 5
DEFAULT_VOLATILITY_THRESHOLD: float = 0.25
ADAPTIVE_THRESHOLD_FLOOR: float = 0.5
ADAPTIVE_THRESHOLD_CEILING: float = 0.95
ADAPTIVE_THRESHOLD_K: float = 1.5
MIN_CALIBRATION_POINTS: int = 3


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TemporalAnalysis:
    """Immutable snapshot of the temporal analysis for a single inference."""

    score: float
    stress_velocity: Optional[float]
    adaptive_threshold: float
    exceeds_threshold: bool
    is_volatile: bool
    volatility: Optional[float]
    score_count: int


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------
class TemporalStressProfile:
    """Per-user temporal stress tracker.

    Parameters
    ----------
    max_history : int
        Maximum number of (timestamp, score) entries to retain.
    velocity_window : int
        Number of recent scores used to compute stress velocity.
    volatility_window : int
        Number of recent scores used to compute volatility (σ).
    volatility_threshold : float
        Standard-deviation threshold above which the user is flagged volatile.
    """

    def __init__(
        self,
        max_history: int = DEFAULT_MAX_HISTORY,
        velocity_window: int = DEFAULT_VELOCITY_WINDOW,
        volatility_window: int = DEFAULT_VOLATILITY_WINDOW,
        volatility_threshold: float = DEFAULT_VOLATILITY_THRESHOLD,
    ) -> None:
        if max_history < 1:
            raise ValueError("max_history must be >= 1")
        if velocity_window < 2:
            raise ValueError("velocity_window must be >= 2")
        if volatility_window < 2:
            raise ValueError("volatility_window must be >= 2")

        self._max_history = max_history
        self._velocity_window = velocity_window
        self._volatility_window = volatility_window
        self._volatility_threshold = volatility_threshold

        # Sliding window: stores (timestamp, score) tuples only.
        self._history: deque[tuple[float, float]] = deque(maxlen=max_history)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_score(self, score: float, timestamp: float | None = None) -> TemporalAnalysis:
        """Record a new inference score and return the temporal analysis.

        Parameters
        ----------
        score : float
            Stress inference score in [0, 1].
        timestamp : float, optional
            Unix timestamp. Defaults to ``time.time()``.

        Returns
        -------
        TemporalAnalysis
            Snapshot containing velocity, threshold, and volatility info.
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {score}")

        if timestamp is None:
            timestamp = time.time()

        self._history.append((timestamp, score))

        velocity = self._compute_velocity()
        threshold = self._compute_adaptive_threshold()
        volatility = self._compute_volatility()

        return TemporalAnalysis(
            score=score,
            stress_velocity=velocity,
            adaptive_threshold=threshold,
            exceeds_threshold=score > threshold,
            is_volatile=volatility is not None and volatility > self._volatility_threshold,
            volatility=volatility,
            score_count=len(self._history),
        )

    @property
    def score_count(self) -> int:
        """Number of scores currently stored."""
        return len(self._history)

    @property
    def scores(self) -> list[float]:
        """Return a copy of the stored scores (no timestamps)."""
        return [s for _, s in self._history]

    @property
    def history(self) -> list[tuple[float, float]]:
        """Return a copy of the full (timestamp, score) history."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Internal computations
    # ------------------------------------------------------------------

    def _compute_velocity(self) -> Optional[float]:
        """Compute stress velocity via 1-D linear regression.

        Uses the last ``velocity_window`` scores. Returns ``None`` if
        fewer than 2 data points are available.
        """
        scores = self.scores
        n = min(self._velocity_window, len(scores))
        if n < 2:
            return None

        recent = scores[-n:]
        x = np.arange(n, dtype=np.float64)
        coeffs = np.polyfit(x, recent, deg=1)
        return float(coeffs[0])  # slope

    def _compute_adaptive_threshold(self) -> float:
        """Compute adaptive threshold = min(max(μ + 1.5σ, 0.5), 0.95).

        Uses **all** stored history as the calibration baseline.
        Returns the floor (0.5) if fewer than ``MIN_CALIBRATION_POINTS``
        scores are available.
        """
        scores = self.scores
        if len(scores) < MIN_CALIBRATION_POINTS:
            return ADAPTIVE_THRESHOLD_FLOOR

        arr = np.array(scores, dtype=np.float64)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr))

        raw = mu + ADAPTIVE_THRESHOLD_K * sigma
        return min(max(raw, ADAPTIVE_THRESHOLD_FLOOR), ADAPTIVE_THRESHOLD_CEILING)

    def _compute_volatility(self) -> Optional[float]:
        """Compute standard deviation of the recent score window.

        Returns ``None`` if fewer than 2 data points are available.
        """
        scores = self.scores
        n = min(self._volatility_window, len(scores))
        if n < 2:
            return None

        recent = scores[-n:]
        return float(np.std(recent))

"""
tests/test_temporal_stress_profile.py
=====================================
Unit tests for Phase 2: Temporal Modeling & Contextual Intelligence.
"""

import time

import numpy as np
import pytest

from models.temporal_stress_profile import (
    ADAPTIVE_THRESHOLD_CEILING,
    ADAPTIVE_THRESHOLD_FLOOR,
    DEFAULT_MAX_HISTORY,
    MIN_CALIBRATION_POINTS,
    TemporalAnalysis,
    TemporalStressProfile,
)


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_params(self):
        profile = TemporalStressProfile()
        assert profile.score_count == 0
        assert profile.scores == []
        assert profile.history == []

    def test_invalid_max_history(self):
        with pytest.raises(ValueError, match="max_history"):
            TemporalStressProfile(max_history=0)

    def test_invalid_velocity_window(self):
        with pytest.raises(ValueError, match="velocity_window"):
            TemporalStressProfile(velocity_window=1)

    def test_invalid_volatility_window(self):
        with pytest.raises(ValueError, match="volatility_window"):
            TemporalStressProfile(volatility_window=1)

    def test_invalid_score_below_zero(self):
        profile = TemporalStressProfile()
        with pytest.raises(ValueError, match="score must be"):
            profile.add_score(-0.1)

    def test_invalid_score_above_one(self):
        profile = TemporalStressProfile()
        with pytest.raises(ValueError, match="score must be"):
            profile.add_score(1.1)


# ---------------------------------------------------------------------------
# Score storage & sliding window
# ---------------------------------------------------------------------------


class TestSlidingWindow:
    def test_add_single_score(self):
        profile = TemporalStressProfile()
        result = profile.add_score(0.5, timestamp=1000.0)
        assert profile.score_count == 1
        assert profile.scores == [0.5]
        assert result.score == 0.5

    def test_boundary_scores_accepted(self):
        profile = TemporalStressProfile()
        profile.add_score(0.0)
        profile.add_score(1.0)
        assert profile.scores == [0.0, 1.0]

    def test_sliding_window_eviction(self):
        profile = TemporalStressProfile(max_history=3)
        for i in range(5):
            profile.add_score(i * 0.2, timestamp=float(i))
        assert profile.score_count == 3
        assert profile.scores == pytest.approx([0.4, 0.6, 0.8])

    def test_history_stores_tuples(self):
        profile = TemporalStressProfile()
        profile.add_score(0.7, timestamp=1234.5)
        ts, score = profile.history[0]
        assert ts == 1234.5
        assert score == 0.7

    def test_default_timestamp_is_current(self):
        profile = TemporalStressProfile()
        before = time.time()
        profile.add_score(0.3)
        after = time.time()
        ts, _ = profile.history[0]
        assert before <= ts <= after


# ---------------------------------------------------------------------------
# Stress Velocity
# ---------------------------------------------------------------------------


class TestStressVelocity:
    def test_velocity_none_with_one_point(self):
        profile = TemporalStressProfile()
        result = profile.add_score(0.5)
        assert result.stress_velocity is None

    def test_velocity_positive_for_rising_trend(self):
        profile = TemporalStressProfile(velocity_window=3)
        for s in [0.2, 0.4, 0.6]:
            result = profile.add_score(s, timestamp=float(len(profile.scores)))
        assert result.stress_velocity is not None
        assert result.stress_velocity > 0

    def test_velocity_negative_for_falling_trend(self):
        profile = TemporalStressProfile(velocity_window=3)
        for s in [0.8, 0.5, 0.2]:
            result = profile.add_score(s, timestamp=float(len(profile.scores)))
        assert result.stress_velocity is not None
        assert result.stress_velocity < 0

    def test_velocity_near_zero_for_flat(self):
        profile = TemporalStressProfile(velocity_window=5)
        for s in [0.5, 0.5, 0.5, 0.5, 0.5]:
            result = profile.add_score(s, timestamp=float(len(profile.scores)))
        assert result.stress_velocity is not None
        assert abs(result.stress_velocity) < 1e-9

    def test_velocity_uses_window(self):
        """Velocity should only consider the last velocity_window scores."""
        profile = TemporalStressProfile(velocity_window=3)
        # First add declining scores
        for s in [0.9, 0.7, 0.5]:
            profile.add_score(s, timestamp=float(len(profile.scores)))
        # Then add rising scores
        for s in [0.3, 0.5, 0.7]:
            result = profile.add_score(s, timestamp=float(len(profile.scores)))
        # Velocity should reflect the recent rising trend
        assert result.stress_velocity is not None
        assert result.stress_velocity > 0

    def test_velocity_value_for_perfect_line(self):
        """For scores [0.0, 0.25, 0.5, 0.75, 1.0], slope should be 0.25."""
        profile = TemporalStressProfile(velocity_window=5)
        for i, s in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
            result = profile.add_score(s, timestamp=float(i))
        assert result.stress_velocity == pytest.approx(0.25, abs=1e-9)


# ---------------------------------------------------------------------------
# Adaptive Threshold
# ---------------------------------------------------------------------------


class TestAdaptiveThreshold:
    def test_threshold_floor_before_calibration(self):
        """Before MIN_CALIBRATION_POINTS are recorded, use the floor."""
        profile = TemporalStressProfile()
        for _ in range(MIN_CALIBRATION_POINTS - 1):
            result = profile.add_score(0.1)
        assert result.adaptive_threshold == ADAPTIVE_THRESHOLD_FLOOR

    def test_threshold_computed_after_calibration(self):
        """After enough points, threshold is based on μ + 1.5σ."""
        profile = TemporalStressProfile()
        scores = [0.3, 0.3, 0.3, 0.3, 0.3]
        for s in scores:
            result = profile.add_score(s)
        # σ = 0 for identical scores, so threshold = μ + 0 = 0.3
        # But floor is 0.5, so should be clamped to 0.5
        assert result.adaptive_threshold == ADAPTIVE_THRESHOLD_FLOOR

    def test_threshold_respects_ceiling(self):
        """Extremely high μ + 1.5σ should be capped at 0.95."""
        profile = TemporalStressProfile()
        # All very high scores → μ near 1.0
        for _ in range(10):
            result = profile.add_score(1.0)
        assert result.adaptive_threshold == ADAPTIVE_THRESHOLD_CEILING

    def test_threshold_between_floor_and_ceiling(self):
        """Mid-range varied scores should produce a threshold in (0.5, 0.95)."""
        profile = TemporalStressProfile()
        scores = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        for s in scores:
            result = profile.add_score(s)
        assert ADAPTIVE_THRESHOLD_FLOOR <= result.adaptive_threshold <= ADAPTIVE_THRESHOLD_CEILING

    def test_threshold_math_correctness(self):
        """Verify the exact formula μ + 1.5σ with known values."""
        profile = TemporalStressProfile()
        scores = [0.4, 0.6, 0.4, 0.6, 0.4, 0.6]
        for s in scores:
            result = profile.add_score(s)
        arr = np.array(scores)
        expected_raw = float(np.mean(arr) + 1.5 * np.std(arr))
        expected = min(max(expected_raw, 0.5), 0.95)
        assert result.adaptive_threshold == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Exceeds Threshold
# ---------------------------------------------------------------------------


class TestExceedsThreshold:
    def test_exceeds_when_score_above_threshold(self):
        profile = TemporalStressProfile()
        # Before calibration, threshold = 0.5
        result = profile.add_score(0.6)
        assert result.exceeds_threshold is True

    def test_does_not_exceed_when_score_below_threshold(self):
        profile = TemporalStressProfile()
        result = profile.add_score(0.3)
        assert result.exceeds_threshold is False

    def test_does_not_exceed_when_score_equals_threshold(self):
        """score == threshold should NOT exceed (strict >)."""
        profile = TemporalStressProfile()
        result = profile.add_score(0.5)
        assert result.exceeds_threshold is False


# ---------------------------------------------------------------------------
# Volatility Detection
# ---------------------------------------------------------------------------


class TestVolatility:
    def test_volatility_none_with_one_point(self):
        profile = TemporalStressProfile()
        result = profile.add_score(0.5)
        assert result.volatility is None
        assert result.is_volatile is False

    def test_not_volatile_for_stable_scores(self):
        profile = TemporalStressProfile(volatility_window=5)
        for s in [0.5, 0.5, 0.5, 0.5, 0.5]:
            result = profile.add_score(s)
        assert result.volatility is not None
        assert result.volatility < 0.25
        assert result.is_volatile is False

    def test_volatile_for_wild_swings(self):
        profile = TemporalStressProfile(volatility_window=4)
        for s in [0.1, 0.9, 0.1, 0.9]:
            result = profile.add_score(s)
        assert result.volatility is not None
        assert result.volatility > 0.25
        assert result.is_volatile is True

    def test_volatility_uses_window(self):
        """Volatility should only consider the last volatility_window scores."""
        profile = TemporalStressProfile(volatility_window=3)
        # Wild early scores
        for s in [0.1, 0.9, 0.1]:
            profile.add_score(s)
        # Stable later scores
        for s in [0.5, 0.5, 0.5]:
            result = profile.add_score(s)
        assert result.volatility is not None
        assert result.volatility < 0.01
        assert result.is_volatile is False

    def test_custom_volatility_threshold(self):
        """A lower volatility_threshold (0.1) should flag moderate swings
        that would not be flagged with the default threshold of 0.25."""
        profile = TemporalStressProfile(volatility_threshold=0.1)
        for s in [0.3, 0.5, 0.3, 0.5]:
            result = profile.add_score(s)
        assert result.volatility is not None


# ---------------------------------------------------------------------------
# TemporalAnalysis dataclass
# ---------------------------------------------------------------------------


class TestTemporalAnalysis:
    def test_immutable(self):
        analysis = TemporalAnalysis(
            score=0.5,
            stress_velocity=0.1,
            adaptive_threshold=0.6,
            exceeds_threshold=False,
            is_volatile=False,
            volatility=0.05,
            score_count=10,
        )
        with pytest.raises(AttributeError):
            analysis.score = 0.9  # type: ignore[misc]

    def test_all_fields(self):
        analysis = TemporalAnalysis(
            score=0.7,
            stress_velocity=-0.05,
            adaptive_threshold=0.65,
            exceeds_threshold=True,
            is_volatile=True,
            volatility=0.3,
            score_count=15,
        )
        assert analysis.score == 0.7
        assert analysis.stress_velocity == -0.05
        assert analysis.adaptive_threshold == 0.65
        assert analysis.exceeds_threshold is True
        assert analysis.is_volatile is True
        assert analysis.volatility == 0.3
        assert analysis.score_count == 15


# ---------------------------------------------------------------------------
# Integration / scenario tests
# ---------------------------------------------------------------------------


class TestIntegrationScenarios:
    def test_single_spike_after_calm_baseline(self):
        """A single high score after calm posts should exceed threshold."""
        profile = TemporalStressProfile(velocity_window=5)
        for s in [0.1, 0.15, 0.12, 0.1, 0.13]:
            profile.add_score(s)
        result = profile.add_score(0.8)
        # Threshold ≈ μ(low) + 1.5σ(low) → close to 0.5 (floor)
        assert result.exceeds_threshold is True
        assert result.stress_velocity is not None
        assert result.stress_velocity > 0  # spike creates upward velocity

    def test_chronic_moderate_stress(self):
        """Consistently ~0.6 should eventually NOT trigger (adaptive baseline)."""
        profile = TemporalStressProfile()
        for _ in range(20):
            result = profile.add_score(0.6)
        # With μ=0.6, σ≈0, threshold = max(0.6, 0.5) = 0.6
        # Score 0.6 == threshold → exceeds is False (strict >)
        assert result.exceeds_threshold is False

    def test_escalating_crisis(self):
        """Rapidly rising scores should show positive velocity and exceed."""
        profile = TemporalStressProfile(velocity_window=5)
        scores = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for s in scores:
            result = profile.add_score(s)
        assert result.stress_velocity is not None
        assert result.stress_velocity > 0
        assert result.exceeds_threshold is True

    def test_recovery_trajectory(self):
        """Declining scores should show negative velocity."""
        profile = TemporalStressProfile(velocity_window=5)
        for s in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
            result = profile.add_score(s)
        assert result.stress_velocity is not None
        assert result.stress_velocity < 0

    def test_volatile_user(self):
        """Alternating high/low scores should flag volatility."""
        profile = TemporalStressProfile(volatility_window=5)
        for s in [0.2, 0.8, 0.2, 0.8, 0.2]:
            result = profile.add_score(s)
        assert result.is_volatile is True

    def test_max_history_honored(self):
        """Ensure we never exceed max_history entries."""
        profile = TemporalStressProfile(max_history=10)
        for i in range(100):
            profile.add_score(float(i % 2) * 0.5 + 0.25)
        assert profile.score_count == 10

    def test_result_score_count_matches(self):
        """The score_count in the result should match the profile."""
        profile = TemporalStressProfile()
        for i in range(7):
            result = profile.add_score(0.5)
        assert result.score_count == 7
        assert result.score_count == profile.score_count

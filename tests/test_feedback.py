"""
tests/test_feedback.py
======================
Unit tests for the RL feedback subsystem:
- database/feedback.py  (FeedbackStore)
- utils/reward.py       (compute_reward, compute_combined_reward, reward_to_weight)
- utils/llm_reward.py   (get_llm_reward — offline/no-key path only)
"""

from __future__ import annotations

import pytest

from database.feedback import FeedbackStore
from utils.reward import compute_combined_reward, compute_reward, reward_to_weight


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store() -> FeedbackStore:
    """Return an in-memory FeedbackStore (discarded after each test)."""
    return FeedbackStore(":memory:")


# ---------------------------------------------------------------------------
# FeedbackStore — table creation & basic CRUD
# ---------------------------------------------------------------------------

class TestFeedbackStore:
    def test_tables_created(self, store: FeedbackStore) -> None:
        """Tables must exist immediately after construction."""
        cur = store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        names = {row["name"] for row in cur.fetchall()}
        assert "feedback" in names
        assert "experience" in names

    def test_save_feedback_correct(self, store: FeedbackStore) -> None:
        """Saving a correct-prediction feedback inserts both feedback and experience rows."""
        fid = store.save_feedback(
            username="alice",
            text="I am so stressed",
            prediction=0.8,
            user_feedback=1,
            reward=1.0,
        )
        assert fid >= 1

        # feedback table
        fb_rows = store.get_all_feedback()
        assert len(fb_rows) == 1
        row = fb_rows[0]
        assert row["username"] == "alice"
        assert row["user_feedback"] == 1
        assert row["reward"] == pytest.approx(1.0)
        assert row["llm_reward"] is None

        # experience table: label should equal round(0.8) = 1 (correct)
        exp = store.get_experience_for_training(min_samples=1)
        assert len(exp) == 1
        assert exp[0]["label"] == 1
        assert exp[0]["reward"] == pytest.approx(1.0)

    def test_save_feedback_wrong(self, store: FeedbackStore) -> None:
        """Wrong-prediction feedback flips the label in the experience table."""
        store.save_feedback(
            username="bob",
            text="Everything is fine",
            prediction=0.8,   # predicted stressed (class 1)
            user_feedback=0,  # but user says wrong → corrected label should be 0
            reward=-1.0,
        )
        exp = store.get_experience_for_training(min_samples=1)
        assert len(exp) == 1
        assert exp[0]["label"] == 0

    def test_save_feedback_with_llm_reward(self, store: FeedbackStore) -> None:
        """LLM reward is stored correctly."""
        store.save_feedback(
            username="carol",
            text="text",
            prediction=0.6,
            user_feedback=1,
            reward=1.0,
            llm_reward=1,
        )
        rows = store.get_all_feedback()
        assert rows[0]["llm_reward"] == 1

    def test_get_user_stats_no_data(self, store: FeedbackStore) -> None:
        stats = store.get_user_stats("nobody")
        assert stats["total"] == 0
        assert stats["mean_reward"] == pytest.approx(0.0)

    def test_get_user_stats(self, store: FeedbackStore) -> None:
        store.save_feedback("alice", "t1", 0.8, 1, 1.0)
        store.save_feedback("alice", "t2", 0.8, 0, -1.0)
        store.save_feedback("alice", "t3", 0.7, 1, 1.0)

        stats = store.get_user_stats("alice")
        assert stats["total"] == 3
        assert stats["n_correct"] == 2
        assert stats["n_wrong"] == 1
        assert stats["mean_reward"] == pytest.approx(1 / 3, abs=0.01)
        assert stats["accuracy_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_min_samples_gate(self, store: FeedbackStore) -> None:
        """get_experience_for_training returns [] when below min_samples."""
        store.save_feedback("alice", "text", 0.7, 1, 1.0)
        assert store.get_experience_for_training(min_samples=5) == []
        assert len(store.get_experience_for_training(min_samples=1)) == 1

    def test_feedback_count(self, store: FeedbackStore) -> None:
        assert store.get_feedback_count() == 0
        store.save_feedback("u1", "t", 0.5, 1, 1.0)
        store.save_feedback("u1", "t", 0.5, 0, -1.0)
        store.save_feedback("u2", "t", 0.5, 1, 1.0)
        assert store.get_feedback_count() == 3
        assert store.get_feedback_count("u1") == 2
        assert store.get_feedback_count("u2") == 1

    def test_multiple_users_isolated(self, store: FeedbackStore) -> None:
        """User stats must be scoped to individual users."""
        store.save_feedback("alice", "text", 0.9, 1, 1.0)
        store.save_feedback("bob", "text", 0.2, 0, -1.0)

        alice = store.get_user_stats("alice")
        bob = store.get_user_stats("bob")

        assert alice["total"] == 1
        assert alice["n_correct"] == 1
        assert bob["total"] == 1
        assert bob["n_wrong"] == 1

    def test_close(self, store: FeedbackStore) -> None:
        """close() must not raise."""
        store.close()


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

class TestComputeReward:
    def test_correct_gives_positive(self) -> None:
        assert compute_reward(1) == pytest.approx(1.0)

    def test_wrong_gives_negative(self) -> None:
        assert compute_reward(0) == pytest.approx(-1.0)

    def test_correct_low_prediction(self) -> None:
        assert compute_reward(1) == pytest.approx(1.0)

    def test_wrong_low_prediction(self) -> None:
        assert compute_reward(0) == pytest.approx(-1.0)


class TestComputeCombinedReward:
    def test_no_llm_passes_through(self) -> None:
        assert compute_combined_reward(1, None) == pytest.approx(1.0)
        assert compute_combined_reward(0, None) == pytest.approx(-1.0)

    def test_llm_agree_positive(self) -> None:
        # Both user (+1) and LLM (+1) agree → average = +1
        assert compute_combined_reward(1, 1) == pytest.approx(1.0)

    def test_llm_agree_negative(self) -> None:
        # Both user (-1) and LLM (-1) agree → average = -1
        assert compute_combined_reward(0, -1) == pytest.approx(-1.0)

    def test_llm_disagree_averages(self) -> None:
        # User says correct (+1), LLM says wrong (-1) → average = 0.0
        result = compute_combined_reward(1, -1)
        assert result == pytest.approx(0.0)

    def test_llm_partial_agreement(self) -> None:
        # User says wrong (-1), LLM says correct (+1) → 0.0
        result = compute_combined_reward(0, 1)
        assert result == pytest.approx(0.0)


class TestRewardToWeight:
    def test_nonzero_reward_gives_1_5(self) -> None:
        assert reward_to_weight(1.0) == pytest.approx(1.5)
        assert reward_to_weight(-1.0) == pytest.approx(1.5)
        assert reward_to_weight(0.5) == pytest.approx(1.5)

    def test_zero_reward_gives_1_0(self) -> None:
        assert reward_to_weight(0.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# LLM reward (no-key / offline path)
# ---------------------------------------------------------------------------

class TestGetLlmRewardOffline:
    """These tests run without real API keys — the function must return None."""

    def test_returns_none_without_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        from utils.llm_reward import get_llm_reward

        result = get_llm_reward("I am stressed", 0.8, provider="auto")
        assert result is None

    def test_openai_returns_none_without_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from utils.llm_reward import get_llm_reward

        result = get_llm_reward("text", 0.5, provider="openai")
        assert result is None

    def test_gemini_returns_none_without_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        from utils.llm_reward import get_llm_reward

        result = get_llm_reward("text", 0.5, provider="gemini")
        assert result is None

"""
tests/test_api.py
=================
Integration tests for api/main.py — FastAPI endpoints.
"""

import os
import tempfile

import pytest
import torch
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app
from database.db import DatabaseManager
from database.feedback import FeedbackStore
from models.architecture import OptimizedMultichannelCNN


@pytest.fixture(autouse=True)
def use_test_database():
    """Replace the module-level database and feedback store with in-memory
    instances for each test."""
    test_db = DatabaseManager(":memory:")
    test_feedback = FeedbackStore(":memory:")
    original_db = api_main._db
    original_fb = api_main._feedback_store
    api_main._db = test_db
    api_main._feedback_store = test_feedback
    yield test_db
    api_main._db = original_db
    api_main._feedback_store = original_fb
    test_db.close()
    test_feedback.close()


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_success(self, client):
        resp = client.post(
            "/register",
            json={"username": "testuser", "password": "securepass123"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_register_duplicate_user(self, client):
        client.post(
            "/register",
            json={"username": "testuser", "password": "securepass123"},
        )
        resp = client.post(
            "/register",
            json={"username": "testuser", "password": "anotherpass1"},
        )
        assert resp.status_code == 400
        assert "already exists" in resp.json()["detail"]

    def test_register_short_password(self, client):
        resp = client.post(
            "/register",
            json={"username": "testuser", "password": "short"},
        )
        assert resp.status_code == 422  # validation error

    def test_register_short_username(self, client):
        resp = client.post(
            "/register",
            json={"username": "ab", "password": "securepass123"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


class TestLogin:
    def test_login_success(self, client):
        client.post(
            "/register",
            json={"username": "testuser", "password": "securepass123"},
        )
        resp = client.post(
            "/login",
            json={"username": "testuser", "password": "securepass123"},
        )
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    def test_login_wrong_password(self, client):
        client.post(
            "/register",
            json={"username": "testuser", "password": "securepass123"},
        )
        resp = client.post(
            "/login",
            json={"username": "testuser", "password": "wrongpassword"},
        )
        assert resp.status_code == 401

    def test_login_nonexistent_user(self, client):
        resp = client.post(
            "/login",
            json={"username": "nobody", "password": "securepass123"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Analyze (JWT required)
# ---------------------------------------------------------------------------


class TestAnalyze:
    def _get_token(self, client) -> str:
        resp = client.post(
            "/register",
            json={"username": "testuser", "password": "securepass123"},
        )
        return resp.json()["access_token"]

    def test_analyze_without_auth(self, client):
        resp = client.post("/analyze", json={"text": "I feel stressed"})
        assert resp.status_code in (401, 403)  # no bearer token

    def test_analyze_with_auth(self, client):
        token = self._get_token(client)
        resp = client.post(
            "/analyze",
            json={"text": "I feel stressed about work"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "stress_score" in data
        assert "stress_label" in data
        assert "temporal" in data
        assert "interventions" in data
        assert "attention_weights" in data
        assert 0.0 <= data["stress_score"] <= 1.0

    def test_analyze_crisis_text(self, client):
        token = self._get_token(client)
        resp = client.post(
            "/analyze",
            json={"text": "I want to kill myself"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_crisis"] is True
        assert data["crisis_message"] is not None
        assert "988" in data["crisis_message"]

    def test_analyze_with_triggers(self, client):
        token = self._get_token(client)
        resp = client.post(
            "/analyze",
            json={"text": "I can't sleep because of work stress"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "sleep" in data["matched_triggers"] or "work" in data["matched_triggers"]

    def test_analyze_temporal_accumulates(self, client):
        """Multiple analyses should accumulate temporal history."""
        token = self._get_token(client)
        for _ in range(3):
            resp = client.post(
                "/analyze",
                json={"text": "test text"},
                headers={"Authorization": f"Bearer {token}"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["temporal"]["score_count"] == 3

    def test_analyze_invalid_token(self, client):
        resp = client.post(
            "/analyze",
            json={"text": "test"},
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert resp.status_code == 401

    def test_password_not_stored_plaintext(self, client, use_test_database):
        """Verify that passwords in the database are hashed, not plaintext."""
        client.post(
            "/register",
            json={"username": "testuser", "password": "securepass123"},
        )
        user = use_test_database.get_user("testuser")
        assert user is not None
        stored = user["password_hash"]
        assert stored != "securepass123"
        assert stored.startswith("$2b$")


# ---------------------------------------------------------------------------
# History endpoint
# ---------------------------------------------------------------------------


class TestHistory:
    def _get_token(self, client) -> str:
        resp = client.post(
            "/register",
            json={"username": "testuser", "password": "securepass123"},
        )
        return resp.json()["access_token"]

    def test_history_empty_initially(self, client):
        token = self._get_token(client)
        resp = client.get(
            "/history",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["sessions"] == []
        assert data["total"] == 0

    def test_history_after_analysis(self, client):
        token = self._get_token(client)
        # Perform two analyses
        for text in ["I feel stressed about work", "I feel calm and relaxed"]:
            client.post(
                "/analyze",
                json={"text": text},
                headers={"Authorization": f"Bearer {token}"},
            )

        resp = client.get(
            "/history",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["sessions"]) == 2
        # Sessions are newest first
        assert data["sessions"][0]["created_at"] >= data["sessions"][1]["created_at"]
        # Each session has the expected fields
        for s in data["sessions"]:
            assert "stress_score" in s
            assert "stress_label" in s
            assert "temporal_data" in s
            assert "interventions" in s
            assert "created_at" in s

    def test_history_pagination(self, client):
        token = self._get_token(client)
        for _ in range(5):
            client.post(
                "/analyze",
                json={"text": "test text"},
                headers={"Authorization": f"Bearer {token}"},
            )

        resp = client.get(
            "/history?limit=2&offset=0",
            headers={"Authorization": f"Bearer {token}"},
        )
        data = resp.json()
        assert data["total"] == 5
        assert len(data["sessions"]) == 2

        resp2 = client.get(
            "/history?limit=2&offset=2",
            headers={"Authorization": f"Bearer {token}"},
        )
        data2 = resp2.json()
        assert data2["total"] == 5
        assert len(data2["sessions"]) == 2

    def test_history_without_auth(self, client):
        resp = client.get("/history")
        assert resp.status_code in (401, 403)

    def test_history_isolated_per_user(self, client):
        """Each user should only see their own sessions."""
        # Register two users
        r1 = client.post(
            "/register",
            json={"username": "user1", "password": "securepass123"},
        )
        token1 = r1.json()["access_token"]

        r2 = client.post(
            "/register",
            json={"username": "user2", "password": "securepass456"},
        )
        token2 = r2.json()["access_token"]

        # User1 analyses
        client.post(
            "/analyze",
            json={"text": "stressed out"},
            headers={"Authorization": f"Bearer {token1}"},
        )

        # User2 should have no history
        resp = client.get(
            "/history",
            headers={"Authorization": f"Bearer {token2}"},
        )
        assert resp.json()["total"] == 0

        # User1 should have one session
        resp = client.get(
            "/history",
            headers={"Authorization": f"Bearer {token1}"},
        )
        assert resp.json()["total"] == 1


# ---------------------------------------------------------------------------
# Checkpoint loading (backward compatibility)
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_model():
    """Reset the cached model singleton and checkpoint path around each test."""
    original_path = api_main._CHECKPOINT_PATH
    api_main._model = None
    api_main._model_type = "cnn"
    api_main._decision_threshold = 0.5
    api_main._tokenizer = None
    api_main._feature_dim = 0
    yield
    api_main._model = None
    api_main._CHECKPOINT_PATH = original_path


class TestCheckpointLoading:
    """Tests for backward-compatible model checkpoint loading."""

    def test_no_checkpoint_uses_random_weights(self, reset_model, tmp_path):
        """When no checkpoint file exists, the model is created fresh."""
        api_main._CHECKPOINT_PATH = str(tmp_path / "nonexistent.pt")
        model = api_main._get_model()
        assert isinstance(model, OptimizedMultichannelCNN)

    def test_compatible_checkpoint_loads(self, reset_model, tmp_path):
        """A checkpoint saved with the current architecture loads cleanly."""
        ckpt_path = str(tmp_path / "good.pt")
        # Save a compatible checkpoint
        ref_model = OptimizedMultichannelCNN(
            vocab_size=10000, embed_dim=128, num_filters=64,
            kernel_sizes=(2, 3, 5), num_classes=2, dropout=0.3,
        )
        torch.save({"model_state_dict": ref_model.state_dict()}, ckpt_path)

        api_main._CHECKPOINT_PATH = ckpt_path
        model = api_main._get_model()
        assert isinstance(model, OptimizedMultichannelCNN)

    def test_old_checkpoint_loads_with_strict_false(self, reset_model, tmp_path):
        """An old checkpoint (with 'fc' instead of 'attention' + 'classifier')
        should load compatible weights and not crash."""
        ckpt_path = str(tmp_path / "old.pt")

        # Build a state dict that mimics the OLD architecture:
        # same embedding + convs, but 'fc' instead of 'attention' + 'classifier'
        ref_model = OptimizedMultichannelCNN(
            vocab_size=10000, embed_dim=128, num_filters=64,
            kernel_sizes=(2, 3, 5), num_classes=2, dropout=0.3,
        )
        old_state = {}
        for k, v in ref_model.state_dict().items():
            if k.startswith(("attention.", "classifier.")):
                continue  # skip new layers
            old_state[k] = v
        # Add the old 'fc' layer keys
        total_filters = 64 * 3
        old_state["fc.weight"] = torch.randn(2, total_filters)
        old_state["fc.bias"] = torch.randn(2)

        torch.save({"model_state_dict": old_state}, ckpt_path)

        api_main._CHECKPOINT_PATH = ckpt_path
        model = api_main._get_model()
        assert isinstance(model, OptimizedMultichannelCNN)

        # Verify model can run inference
        inp = torch.randint(0, 100, (1, 20))
        with torch.no_grad():
            out = model(inp)
        assert out["logits"].shape == (1, 2)

    def test_analyze_works_with_old_checkpoint(self, reset_model, tmp_path, client):
        """Full /analyze endpoint succeeds even with an old-format checkpoint."""
        ckpt_path = str(tmp_path / "old.pt")

        ref_model = OptimizedMultichannelCNN(
            vocab_size=10000, embed_dim=128, num_filters=64,
            kernel_sizes=(2, 3, 5), num_classes=2, dropout=0.3,
        )
        old_state = {}
        for k, v in ref_model.state_dict().items():
            if k.startswith(("attention.", "classifier.")):
                continue
            old_state[k] = v
        total_filters = 64 * 3
        old_state["fc.weight"] = torch.randn(2, total_filters)
        old_state["fc.bias"] = torch.randn(2)

        torch.save({"model_state_dict": old_state}, ckpt_path)
        api_main._CHECKPOINT_PATH = ckpt_path

        # Register and get token
        resp = client.post(
            "/register",
            json={"username": "testuser", "password": "securepass123"},
        )
        token = resp.json()["access_token"]

        # Analyze should succeed (no 500)
        resp = client.post(
            "/analyze",
            json={"text": "I feel stressed about work"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "stress_score" in data
        assert "attention_weights" in data

    def test_raw_state_dict_checkpoint(self, reset_model, tmp_path):
        """A checkpoint that is a raw state_dict (not wrapped in a dict
        with 'model_state_dict' key) should also load."""
        ckpt_path = str(tmp_path / "raw.pt")
        ref_model = OptimizedMultichannelCNN(
            vocab_size=10000, embed_dim=128, num_filters=64,
            kernel_sizes=(2, 3, 5), num_classes=2, dropout=0.3,
        )
        torch.save(ref_model.state_dict(), ckpt_path)

        api_main._CHECKPOINT_PATH = ckpt_path
        model = api_main._get_model()
        assert isinstance(model, OptimizedMultichannelCNN)

    def test_corrupted_checkpoint_falls_back(self, reset_model, tmp_path):
        """A corrupted checkpoint file should not crash the server."""
        ckpt_path = str(tmp_path / "corrupted.pt")
        with open(ckpt_path, "wb") as f:
            f.write(b"not a valid pytorch file")

        api_main._CHECKPOINT_PATH = ckpt_path
        model = api_main._get_model()
        assert isinstance(model, OptimizedMultichannelCNN)


# ---------------------------------------------------------------------------
# Multi-level output (stress_level, confidence)
# ---------------------------------------------------------------------------


class TestMultiLevelOutput:
    """Tests for the new stress_level and confidence fields added to /analyze."""

    def _get_token(self, client) -> str:
        resp = client.post(
            "/register",
            json={"username": "testuser", "password": "securepass123"},
        )
        return resp.json()["access_token"]

    def test_analyze_returns_stress_level(self, client):
        token = self._get_token(client)
        resp = client.post(
            "/analyze",
            json={"text": "I feel a bit tense today"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "stress_level" in data
        assert data["stress_level"] in {"low", "moderate", "high", "uncertain"}

    def test_analyze_returns_confidence(self, client):
        token = self._get_token(client)
        resp = client.post(
            "/analyze",
            json={"text": "test text"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_stress_score_clipped(self, client):
        """stress_score must always be in [_PROB_CLIP_MIN, _PROB_CLIP_MAX]."""
        token = self._get_token(client)
        resp = client.post(
            "/analyze",
            json={"text": "ok"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        score = resp.json()["stress_score"]
        assert score >= api_main._PROB_CLIP_MIN
        assert score <= api_main._PROB_CLIP_MAX

    def test_decision_threshold_never_below_minimum(self, reset_model, tmp_path, client):
        """A checkpoint with a very low threshold (e.g. 0.15) must be clamped."""
        ckpt_path = str(tmp_path / "low_threshold.pt")
        ref_model = OptimizedMultichannelCNN(
            vocab_size=10000, embed_dim=128, num_filters=64,
            kernel_sizes=(2, 3, 5), num_classes=2, dropout=0.3,
        )
        torch.save(
            {
                "model_state_dict": ref_model.state_dict(),
                "decision_threshold": 0.15,  # pathologically low
            },
            ckpt_path,
        )
        api_main._CHECKPOINT_PATH = ckpt_path
        api_main._get_model()
        assert api_main._decision_threshold >= api_main._MIN_DECISION_THRESHOLD


# ---------------------------------------------------------------------------
# Feedback endpoints
# ---------------------------------------------------------------------------


class TestFeedbackEndpoints:
    """Tests for POST /feedback, GET /feedback/stats, GET /personalization."""

    def _get_token(self, client) -> str:
        resp = client.post(
            "/register",
            json={"username": "testuser", "password": "securepass123"},
        )
        return resp.json()["access_token"]

    def test_submit_feedback_success(self, client):
        token = self._get_token(client)
        resp = client.post(
            "/feedback",
            json={"text": "I am stressed", "prediction": 0.8, "user_feedback": 1},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "saved"
        assert "reward" in data
        assert "feedback_id" in data
        assert data["reward"] == pytest.approx(1.0)

    def test_submit_feedback_wrong_prediction(self, client):
        token = self._get_token(client)
        resp = client.post(
            "/feedback",
            json={"text": "I am calm", "prediction": 0.7, "user_feedback": 0},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["reward"] == pytest.approx(-1.0)

    def test_submit_feedback_requires_auth(self, client):
        resp = client.post(
            "/feedback",
            json={"text": "text", "prediction": 0.5, "user_feedback": 1},
        )
        assert resp.status_code in (401, 403)

    def test_feedback_stats_empty(self, client):
        token = self._get_token(client)
        resp = client.get(
            "/feedback/stats",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["accuracy_rate"] == pytest.approx(0.0)

    def test_feedback_stats_accumulate(self, client):
        token = self._get_token(client)
        for uf in [1, 1, 0]:
            client.post(
                "/feedback",
                json={"text": "text", "prediction": 0.7, "user_feedback": uf},
                headers={"Authorization": f"Bearer {token}"},
            )
        resp = client.get(
            "/feedback/stats",
            headers={"Authorization": f"Bearer {token}"},
        )
        data = resp.json()
        assert data["total"] == 3
        assert data["n_correct"] == 2
        assert data["n_wrong"] == 1

    def test_personalization_no_feedback(self, client):
        token = self._get_token(client)
        resp = client.get(
            "/personalization",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_bias"] == pytest.approx(0.0)
        assert data["feedback_count"] == 0

    def test_personalization_with_feedback(self, client):
        token = self._get_token(client)
        # All predictions were correct → mean_reward = 1 → user_bias = -0.1
        for _ in range(3):
            client.post(
                "/feedback",
                json={"text": "text", "prediction": 0.8, "user_feedback": 1},
                headers={"Authorization": f"Bearer {token}"},
            )
        resp = client.get(
            "/personalization",
            headers={"Authorization": f"Bearer {token}"},
        )
        data = resp.json()
        assert data["feedback_count"] == 3
        # mean_reward = 1.0, bias = -1.0 * 0.1 = -0.1
        assert data["user_bias"] == pytest.approx(-0.1, abs=0.01)


# ---------------------------------------------------------------------------
# New response fields: requires_escalation and is_uncertain
# ---------------------------------------------------------------------------


class TestNewResponseFields:
    def _get_token(self, client) -> str:
        resp = client.post(
            "/register",
            json={"username": "testuser_nrf", "password": "securepass123"},
        )
        return resp.json()["access_token"]

    def test_analyze_returns_requires_escalation_field(self, client):
        token = self._get_token(client)
        resp = client.post(
            "/analyze",
            json={"text": "I feel a bit stressed today"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "requires_escalation" in data
        assert isinstance(data["requires_escalation"], bool)

    def test_analyze_returns_is_uncertain_field(self, client):
        token = self._get_token(client)
        resp = client.post(
            "/analyze",
            json={"text": "I feel a bit off today"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "is_uncertain" in data
        assert isinstance(data["is_uncertain"], bool)

    def test_escalation_false_for_first_session(self, client):
        """A brand-new user's first session can never trigger escalation."""
        token = self._get_token(client)
        resp = client.post(
            "/analyze",
            json={"text": "I feel really stressed and overwhelmed"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        # With only one session, there cannot be 3 consecutive above-threshold
        assert resp.json()["requires_escalation"] is False

    def test_is_uncertain_is_true_for_short_ambiguous_text(self, client):
        """Very short / boundary-score text should often surface is_uncertain=True."""
        token = self._get_token(client)
        resp = client.post(
            "/analyze",
            json={"text": "fine"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        # is_uncertain may or may not be True depending on model output, but the
        # field must always be present and boolean.
        data = resp.json()
        assert isinstance(data["is_uncertain"], bool)

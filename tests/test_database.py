"""
tests/test_database.py
======================
Unit tests for database/db.py — SQLite persistence layer.
"""

import sqlite3

import pytest

from database.db import DatabaseManager


@pytest.fixture
def db():
    """Provide a fresh in-memory database for each test."""
    manager = DatabaseManager(":memory:")
    yield manager
    manager.close()


# ---------------------------------------------------------------------------
# User operations
# ---------------------------------------------------------------------------


class TestUserOperations:
    def test_create_and_get_user(self, db):
        uid = db.create_user("alice", "$2b$12$fakehash")
        assert uid is not None
        user = db.get_user("alice")
        assert user is not None
        assert user["username"] == "alice"
        assert user["password_hash"] == "$2b$12$fakehash"
        assert user["encrypted_history"] is None
        assert user["created_at"] > 0

    def test_user_exists(self, db):
        assert db.user_exists("alice") is False
        db.create_user("alice", "$2b$12$hash")
        assert db.user_exists("alice") is True

    def test_duplicate_username_raises(self, db):
        db.create_user("alice", "$2b$12$hash1")
        with pytest.raises(sqlite3.IntegrityError):
            db.create_user("alice", "$2b$12$hash2")

    def test_get_nonexistent_user_returns_none(self, db):
        assert db.get_user("nobody") is None

    def test_update_encrypted_history(self, db):
        db.create_user("alice", "$2b$12$hash")
        assert db.get_user("alice")["encrypted_history"] is None

        db.update_encrypted_history("alice", "encrypted-blob-123")
        user = db.get_user("alice")
        assert user["encrypted_history"] == "encrypted-blob-123"

    def test_multiple_users_independent(self, db):
        db.create_user("alice", "$2b$hash1")
        db.create_user("bob", "$2b$hash2")

        alice = db.get_user("alice")
        bob = db.get_user("bob")
        assert alice["id"] != bob["id"]
        assert alice["username"] == "alice"
        assert bob["username"] == "bob"


# ---------------------------------------------------------------------------
# Session operations
# ---------------------------------------------------------------------------


class TestSessionOperations:
    def _create_user(self, db, username="testuser"):
        db.create_user(username, "$2b$12$fakehash")

    def test_save_and_get_session(self, db):
        self._create_user(db)
        sid = db.save_session(
            username="testuser",
            stress_score=0.75,
            stress_label="stress",
            temporal_data={"velocity": 0.1, "threshold": 0.6},
            interventions=[{"title": "Deep breathing", "category": "breathing"}],
            is_crisis=False,
            crisis_message=None,
            matched_triggers=["work", "stress"],
            attention_weights=[0.1, 0.5, 0.3],
        )
        assert sid is not None

        sessions = db.get_sessions("testuser")
        assert len(sessions) == 1
        s = sessions[0]
        assert s["stress_score"] == 0.75
        assert s["stress_label"] == "stress"
        assert s["temporal_data"] == {"velocity": 0.1, "threshold": 0.6}
        assert s["interventions"] == [
            {"title": "Deep breathing", "category": "breathing"}
        ]
        assert s["is_crisis"] is False
        assert s["crisis_message"] is None
        assert s["matched_triggers"] == ["work", "stress"]
        assert s["attention_weights"] == [0.1, 0.5, 0.3]
        assert s["created_at"] > 0

    def test_sessions_ordered_newest_first(self, db):
        self._create_user(db)
        for i in range(3):
            db.save_session(
                username="testuser",
                stress_score=i * 0.1,
                stress_label="no_stress",
                temporal_data={},
                interventions=[],
                is_crisis=False,
                crisis_message=None,
                matched_triggers=[],
                attention_weights=[],
            )

        sessions = db.get_sessions("testuser")
        assert len(sessions) == 3
        # newest first (higher id = newer insertion)
        assert sessions[0]["id"] > sessions[1]["id"]
        assert sessions[1]["id"] > sessions[2]["id"]

    def test_get_sessions_pagination(self, db):
        self._create_user(db)
        for i in range(5):
            db.save_session(
                username="testuser",
                stress_score=i * 0.1,
                stress_label="no_stress",
                temporal_data={},
                interventions=[],
                is_crisis=False,
                crisis_message=None,
                matched_triggers=[],
                attention_weights=[],
            )

        page1 = db.get_sessions("testuser", limit=2, offset=0)
        assert len(page1) == 2

        page2 = db.get_sessions("testuser", limit=2, offset=2)
        assert len(page2) == 2

        page3 = db.get_sessions("testuser", limit=2, offset=4)
        assert len(page3) == 1

    def test_get_session_count(self, db):
        self._create_user(db)
        assert db.get_session_count("testuser") == 0

        db.save_session(
            username="testuser",
            stress_score=0.5,
            stress_label="stress",
            temporal_data={},
            interventions=[],
            is_crisis=False,
            crisis_message=None,
            matched_triggers=[],
            attention_weights=[],
        )
        assert db.get_session_count("testuser") == 1

    def test_sessions_isolated_per_user(self, db):
        self._create_user(db, "alice")
        self._create_user(db, "bob")

        db.save_session(
            username="alice",
            stress_score=0.9,
            stress_label="stress",
            temporal_data={},
            interventions=[],
            is_crisis=False,
            crisis_message=None,
            matched_triggers=[],
            attention_weights=[],
        )

        assert db.get_session_count("alice") == 1
        assert db.get_session_count("bob") == 0
        assert db.get_sessions("bob") == []

    def test_save_session_nonexistent_user_raises(self, db):
        with pytest.raises(ValueError, match="not found"):
            db.save_session(
                username="nobody",
                stress_score=0.5,
                stress_label="stress",
                temporal_data={},
                interventions=[],
                is_crisis=False,
                crisis_message=None,
                matched_triggers=[],
                attention_weights=[],
            )

    def test_get_sessions_nonexistent_user_returns_empty(self, db):
        assert db.get_sessions("nobody") == []

    def test_get_session_count_nonexistent_user_returns_zero(self, db):
        assert db.get_session_count("nobody") == 0

    def test_session_crisis_data(self, db):
        self._create_user(db)
        db.save_session(
            username="testuser",
            stress_score=0.99,
            stress_label="stress",
            temporal_data={},
            interventions=[],
            is_crisis=True,
            crisis_message="Please call 988",
            matched_triggers=["suicide"],
            attention_weights=[0.9],
        )

        sessions = db.get_sessions("testuser")
        assert len(sessions) == 1
        assert sessions[0]["is_crisis"] is True
        assert sessions[0]["crisis_message"] == "Please call 988"


# ---------------------------------------------------------------------------
# Database lifecycle
# ---------------------------------------------------------------------------


class TestDatabaseLifecycle:
    def test_file_based_database(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        db1 = DatabaseManager(db_path)
        db1.create_user("alice", "$2b$hash")
        db1.save_session(
            username="alice",
            stress_score=0.5,
            stress_label="stress",
            temporal_data={"key": "value"},
            interventions=[],
            is_crisis=False,
            crisis_message=None,
            matched_triggers=[],
            attention_weights=[],
        )
        db1.close()

        # Re-open the same database — data should persist
        db2 = DatabaseManager(db_path)
        user = db2.get_user("alice")
        assert user is not None
        sessions = db2.get_sessions("alice")
        assert len(sessions) == 1
        assert sessions[0]["stress_score"] == 0.5
        db2.close()

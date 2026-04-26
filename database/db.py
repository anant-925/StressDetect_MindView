"""
database/db.py
==============
SQLite persistence layer for user accounts and stress analysis sessions.

Tables
------
- ``users``    — username, password_hash, encrypted_history, created_at
- ``sessions`` — per-analysis snapshots linked to users

Thread-safety is handled by using ``check_same_thread=False`` and
relying on SQLite's internal serialisation (WAL mode).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default database path (overridable via env var or constructor arg)
_DEFAULT_DB_PATH = os.environ.get("STRESS_DB_PATH", "stress_detection.db")


class DatabaseManager:
    """Thin wrapper around a SQLite database for user + session storage.

    Parameters
    ----------
    db_path : str
        File path for the SQLite database.  Use ``":memory:"`` for an
        ephemeral in-memory database (useful in tests).
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(
            db_path, check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        """Create tables if they do not exist."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT    NOT NULL UNIQUE,
                password_hash   TEXT    NOT NULL,
                encrypted_history TEXT,
                created_at      REAL    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id           INTEGER NOT NULL,
                stress_score      REAL    NOT NULL,
                stress_label      TEXT    NOT NULL,
                temporal_data     TEXT    NOT NULL,
                interventions     TEXT    NOT NULL,
                is_crisis         INTEGER NOT NULL DEFAULT 0,
                crisis_message    TEXT,
                matched_triggers  TEXT    NOT NULL,
                attention_weights TEXT    NOT NULL,
                created_at        REAL    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_user_id
                ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_created_at
                ON sessions(created_at);
            """
        )

    # ------------------------------------------------------------------
    # User CRUD
    # ------------------------------------------------------------------

    def create_user(
        self,
        username: str,
        password_hash: str,
    ) -> int:
        """Insert a new user and return their ``id``.

        Raises
        ------
        sqlite3.IntegrityError
            If the username already exists.
        """
        cur = self._conn.execute(
            "INSERT INTO users (username, password_hash, encrypted_history, created_at) "
            "VALUES (?, ?, NULL, ?)",
            (username, password_hash, time.time()),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_user(self, username: str) -> Optional[dict[str, Any]]:
        """Return a user dict or ``None`` if not found."""
        row = self._conn.execute(
            "SELECT id, username, password_hash, encrypted_history, created_at "
            "FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def user_exists(self, username: str) -> bool:
        """Check whether a username is already taken."""
        row = self._conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone()
        return row is not None

    def update_encrypted_history(
        self, username: str, encrypted_history: str
    ) -> None:
        """Persist the user's updated encrypted temporal history."""
        self._conn.execute(
            "UPDATE users SET encrypted_history = ? WHERE username = ?",
            (encrypted_history, username),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def save_session(
        self,
        username: str,
        stress_score: float,
        stress_label: str,
        temporal_data: dict,
        interventions: list[dict],
        is_crisis: bool,
        crisis_message: Optional[str],
        matched_triggers: list[str],
        attention_weights: list[float],
    ) -> int:
        """Persist a single analysis session and return its ``id``."""
        user = self.get_user(username)
        if user is None:
            raise ValueError(f"User '{username}' not found")

        cur = self._conn.execute(
            "INSERT INTO sessions "
            "(user_id, stress_score, stress_label, temporal_data, "
            " interventions, is_crisis, crisis_message, matched_triggers, "
            " attention_weights, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user["id"],
                stress_score,
                stress_label,
                json.dumps(temporal_data),
                json.dumps(interventions),
                int(is_crisis),
                crisis_message,
                json.dumps(matched_triggers),
                json.dumps(attention_weights),
                time.time(),
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def get_sessions(
        self,
        username: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return past sessions for a user, newest first.

        Parameters
        ----------
        username : str
            The user whose sessions to retrieve.
        limit : int
            Maximum number of sessions to return.
        offset : int
            Number of sessions to skip (for pagination).
        """
        user = self.get_user(username)
        if user is None:
            return []

        rows = self._conn.execute(
            "SELECT id, stress_score, stress_label, temporal_data, "
            "interventions, is_crisis, crisis_message, matched_triggers, "
            "attention_weights, created_at "
            "FROM sessions WHERE user_id = ? "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (user["id"], limit, offset),
        ).fetchall()

        sessions = []
        for row in rows:
            session = dict(row)
            session["temporal_data"] = json.loads(session["temporal_data"])
            session["interventions"] = json.loads(session["interventions"])
            session["is_crisis"] = bool(session["is_crisis"])
            session["matched_triggers"] = json.loads(session["matched_triggers"])
            session["attention_weights"] = json.loads(session["attention_weights"])
            sessions.append(session)
        return sessions

    def get_session_count(self, username: str) -> int:
        """Return the total number of sessions for a user."""
        user = self.get_user(username)
        if user is None:
            return 0
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM sessions WHERE user_id = ?",
            (user["id"],),
        ).fetchone()
        return row["cnt"]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

"""
database/feedback.py
====================
Persistence layer for RL-style feedback and experience replay.

Tables
------
- ``feedback``   — one row per user-submitted feedback event (text, predicted
  score, user rating, optional LLM reward, computed reward scalar).
- ``experience`` — the same data shaped as an experience-replay buffer that
  ``training/retrain.py`` can query to build a fine-tuning dataset.

Both tables are stored in the same SQLite file as the main user/session DB
(``stress_detection.db``) so that a single file contains the whole
application's state.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = os.environ.get("STRESS_DB_PATH", "stress_detection.db")


class FeedbackStore:
    """Thin SQLite wrapper for feedback storage and experience replay.

    Parameters
    ----------
    db_path : str
        File path for the SQLite database.  Pass ``":memory:"`` for
        ephemeral in-memory storage (useful in tests).
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        """Create feedback tables if they do not already exist."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT    NOT NULL,
                text            TEXT    NOT NULL,
                prediction      REAL    NOT NULL,
                user_feedback   INTEGER NOT NULL,  -- 1 = correct, 0 = wrong
                llm_reward      INTEGER,            -- +1 / -1 / NULL
                reward          REAL    NOT NULL,   -- final combined reward
                created_at      REAL    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_feedback_username
                ON feedback(username);
            CREATE INDEX IF NOT EXISTS idx_feedback_created_at
                ON feedback(created_at);

            CREATE TABLE IF NOT EXISTS experience (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                text            TEXT    NOT NULL,
                label           INTEGER NOT NULL,  -- corrected label
                reward          REAL    NOT NULL,  -- sample weight for training
                source          TEXT    NOT NULL DEFAULT 'feedback',
                created_at      REAL    NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_experience_created_at
                ON experience(created_at);
            """
        )

    # ------------------------------------------------------------------
    # Feedback CRUD
    # ------------------------------------------------------------------

    def save_feedback(
        self,
        username: str,
        text: str,
        prediction: float,
        user_feedback: int,
        reward: float,
        llm_reward: Optional[int] = None,
    ) -> int:
        """Persist one feedback event and derive a corrected training sample.

        The corrected label is:
        - ``round(prediction)`` when ``user_feedback == 1`` (prediction was right).
        - ``1 - round(prediction)`` when ``user_feedback == 0`` (prediction was wrong).

        The corrected sample is also inserted into ``experience`` so that
        ``training/retrain.py`` can build a dataset without joining tables.

        Parameters
        ----------
        username : str
            User who submitted the feedback.
        text : str
            Original input text that was analysed.
        prediction : float
            Raw stress probability returned by the model (0–1).
        user_feedback : int
            1 if the prediction was correct, 0 if it was wrong.
        reward : float
            Computed reward scalar (e.g. from ``utils.reward``).
        llm_reward : int | None
            Optional reward from an LLM judge (+1 / -1 / None).

        Returns
        -------
        int
            Row id of the newly inserted feedback row.
        """
        now = time.time()

        cur = self._conn.execute(
            "INSERT INTO feedback "
            "(username, text, prediction, user_feedback, llm_reward, reward, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (username, text, prediction, user_feedback, llm_reward, reward, now),
        )
        feedback_id = cur.lastrowid

        # Derive corrected label for experience replay
        predicted_class = int(round(prediction))
        corrected_label = predicted_class if user_feedback == 1 else 1 - predicted_class

        self._conn.execute(
            "INSERT INTO experience (text, label, reward, source, created_at) "
            "VALUES (?, ?, ?, 'feedback', ?)",
            (text, corrected_label, abs(reward), now),
        )
        self._conn.commit()
        return feedback_id  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_all_feedback(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Return feedback rows ordered newest-first."""
        rows = self._conn.execute(
            "SELECT id, username, text, prediction, user_feedback, "
            "llm_reward, reward, created_at "
            "FROM feedback ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_user_stats(self, username: str) -> dict[str, Any]:
        """Return aggregated feedback statistics for one user."""
        row = self._conn.execute(
            "SELECT COUNT(*) as total, "
            "AVG(reward) as mean_reward, "
            "SUM(CASE WHEN user_feedback=1 THEN 1 ELSE 0 END) as n_correct, "
            "SUM(CASE WHEN user_feedback=0 THEN 1 ELSE 0 END) as n_wrong "
            "FROM feedback WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None or row["total"] == 0:
            return {
                "total": 0,
                "mean_reward": 0.0,
                "n_correct": 0,
                "n_wrong": 0,
                "accuracy_rate": 0.0,
            }

        total = row["total"]
        n_correct = row["n_correct"] or 0
        return {
            "total": total,
            "mean_reward": float(row["mean_reward"] or 0.0),
            "n_correct": n_correct,
            "n_wrong": row["n_wrong"] or 0,
            "accuracy_rate": n_correct / total if total > 0 else 0.0,
        }

    def get_experience_for_training(
        self,
        min_samples: int = 1,
        limit: int = 10_000,
    ) -> list[dict[str, Any]]:
        """Return experience rows suitable for building a training dataset.

        Parameters
        ----------
        min_samples : int
            Return an empty list when fewer than this many rows exist
            (avoids retraining on negligible data).
        limit : int
            Maximum rows to return.

        Returns
        -------
        list of dict with keys: ``text``, ``label``, ``reward``.
        """
        count_row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM experience"
        ).fetchone()
        if (count_row["cnt"] or 0) < min_samples:
            return []

        rows = self._conn.execute(
            "SELECT text, label, reward FROM experience "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_feedback_count(self, username: Optional[str] = None) -> int:
        """Return the total number of feedback rows (optionally per user)."""
        if username is not None:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM feedback WHERE username = ?",
                (username,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM feedback"
            ).fetchone()
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

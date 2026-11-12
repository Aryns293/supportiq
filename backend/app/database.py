from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from app.config import DATABASE_PATH


@contextmanager
def get_connection():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
    finally:
        connection.close()


def init_db() -> None:
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                category_confidence REAL NOT NULL,
                priority TEXT NOT NULL,
                priority_confidence REAL NOT NULL,
                sentiment TEXT NOT NULL,
                sentiment_confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def create_ticket(ticket: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc).isoformat()
    values = (
        ticket["subject"],
        ticket["description"],
        prediction["category"],
        prediction["category_confidence"],
        prediction["priority"],
        prediction["priority_confidence"],
        prediction["sentiment"],
        prediction["sentiment_confidence"],
        created_at,
    )

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO tickets (
                subject, description, category, category_confidence,
                priority, priority_confidence, sentiment, sentiment_confidence,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
        connection.commit()
        row = connection.execute(
            "SELECT * FROM tickets WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
        return dict(row)


def list_tickets(limit: int = 20) -> list[dict[str, Any]]:
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM tickets ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


def get_analytics() -> dict[str, Any]:
    with get_connection() as connection:
        total = connection.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]

        def grouped_counts(column: str) -> list[dict[str, Any]]:
            # Column names are fixed internally, never supplied by the request.
            rows = connection.execute(
                f"SELECT {column} AS name, COUNT(*) AS count "
                f"FROM tickets GROUP BY {column} ORDER BY count DESC"
            ).fetchall()
            return [dict(row) for row in rows]

        return {
            "total_tickets": total,
            "category_counts": grouped_counts("category"),
            "priority_counts": grouped_counts("priority"),
            "sentiment_counts": grouped_counts("sentiment"),
            "recent_tickets": list_tickets(limit=8),
        }

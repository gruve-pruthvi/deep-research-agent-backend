"""Postgres database layer for research runs and memory tables.

Provides schema initialisation, run upsert, and history / single-run
query helpers.  All connections use synchronous ``psycopg`` because the
callers run in FastAPI background tasks or at startup.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import psycopg

from utils import compact_state_for_storage, to_jsonable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------


def _get_conn() -> psycopg.Connection:
    """Return a new synchronous Postgres connection using env vars."""
    return psycopg.connect(
        dbname=os.getenv("POSTGRES_DB", "postgres"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
    )


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------


def init_db() -> None:
    """Create ``research_runs`` and ``research_memory`` tables if absent."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS research_runs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    query TEXT,
                    depth TEXT,
                    status TEXT,
                    updated_at TIMESTAMPTZ,
                    data_json JSONB
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS research_memory (
                    id TEXT PRIMARY KEY,
                    query TEXT,
                    summary TEXT,
                    created_at TIMESTAMPTZ
                )
                """
            )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Run persistence
# ---------------------------------------------------------------------------


def save_run(
    run_id: str,
    session_id: str,
    query: str,
    depth: str,
    status: str,
    data: Dict[str, Any],
) -> None:
    """Upsert a research run snapshot into Postgres.

    Args:
        run_id: Unique identifier for this run.
        session_id: Browser / client session identifier.
        query: The original research query.
        depth: Depth preset (``"shallow"`` / ``"standard"`` / ``"deep"``).
        status: Lifecycle status string (e.g. ``"running"``, ``"completed"``).
        data: Dict containing ``{"state": ResearchState}`` or a raw state dict.
    """
    conn = _get_conn()
    try:
        safe_data = {"state": compact_state_for_storage(data.get("state", data))}
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO research_runs
                    (id, session_id, query, depth, status, updated_at, data_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(id) DO UPDATE SET
                    session_id  = excluded.session_id,
                    query       = excluded.query,
                    depth       = excluded.depth,
                    status      = excluded.status,
                    updated_at  = excluded.updated_at,
                    data_json   = excluded.data_json
                """,
                (
                    run_id,
                    session_id,
                    query,
                    depth,
                    status,
                    datetime.now(timezone.utc),
                    json.dumps(safe_data, default=to_jsonable),
                ),
            )
        conn.commit()
    finally:
        conn.close()

    # Ensure the persistent memory collection exists (non-fatal).
    try:
        from memory import ensure_memory_collection  # local import avoids circularity

        ensure_memory_collection()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# History queries
# ---------------------------------------------------------------------------


def fetch_history(session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the 50 most recent research runs, optionally filtered by session.

    Args:
        session_id: If provided, only runs for this session are returned.

    Returns:
        List of run summary dicts (id, session_id, query, depth, status,
        updated_at).
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            if session_id:
                cur.execute(
                    """
                    SELECT id, session_id, query, depth, status, updated_at
                    FROM research_runs
                    WHERE session_id = %s
                    ORDER BY updated_at DESC
                    LIMIT 50
                    """,
                    (session_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, session_id, query, depth, status, updated_at
                    FROM research_runs
                    ORDER BY updated_at DESC
                    LIMIT 50
                    """
                )
            rows = cur.fetchall()
        return [
            {
                "id": row[0],
                "session_id": row[1],
                "query": row[2],
                "depth": row[3],
                "status": row[4],
                "updated_at": row[5].isoformat() if row[5] else None,
            }
            for row in rows
        ]
    finally:
        conn.close()


def fetch_run(run_id: str) -> Optional[Dict[str, Any]]:
    """Return a single research run by ID including the stored report.

    Args:
        run_id: The run's unique identifier.

    Returns:
        A dict with run metadata and report fields, or ``None`` if not found.
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, session_id, query, depth, status, updated_at, data_json
                FROM research_runs
                WHERE id = %s
                """,
                (run_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        state: Dict[str, Any] = row[6] if row[6] else {}
        if isinstance(state, dict) and "state" in state:
            state = state["state"]
        return {
            "id": row[0],
            "session_id": row[1],
            "query": row[2],
            "depth": row[3],
            "status": row[4],
            "updated_at": row[5].isoformat() if row[5] else None,
            "report": state.get("report", ""),
            "sources": state.get("sources", []),
            "evaluation": state.get("evaluation", {}),
            "uncertainty_score": state.get("uncertainty_score"),
            "confidence_score": state.get("confidence_score"),
        }
    finally:
        conn.close()

"""Semantic session-memory layer backed by Qdrant + Postgres.

``save_memory`` persists a (query, summary) pair to both Postgres (for
durability) and a persistent Qdrant collection (for semantic search).
``load_memory`` performs a similarity search and falls back to a Postgres
ILIKE search when Qdrant is unavailable.
"""

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List

import psycopg
from qdrant_client.http import models as qdrant_models

from config import build_embeddings
from vectorstore import get_qdrant_client

logger = logging.getLogger(__name__)

MEMORY_COLLECTION = "research_memory_embeddings"


# ---------------------------------------------------------------------------
# Connection helper (local copy avoids circular import with database.py)
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
# Collection bootstrap
# ---------------------------------------------------------------------------


def ensure_memory_collection() -> None:
    """Create the persistent Qdrant memory collection if it does not exist."""
    try:
        client = get_qdrant_client()
        if not client.collection_exists(collection_name=MEMORY_COLLECTION):
            client.create_collection(
                collection_name=MEMORY_COLLECTION,
                vectors_config=qdrant_models.VectorParams(
                    size=1536, distance=qdrant_models.Distance.COSINE
                ),
            )
    except Exception as exc:
        logger.warning("Memory collection init failed: %s", str(exc))


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------


async def save_memory(query: str, summary: str) -> None:
    """Persist a (query, summary) memory entry to Postgres and Qdrant.

    Postgres is written first (guaranteed durability).  The Qdrant upsert
    is best-effort; failures are logged but not re-raised.

    Args:
        query: The research query that produced the summary.
        summary: A short textual summary of the findings (≤ 2000 chars
            recommended).
    """
    mem_id = uuid.uuid4().hex
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO research_memory (id, query, summary, created_at)
                VALUES (%s, %s, %s, %s)
                """,
                (mem_id, query, summary, datetime.now(timezone.utc)),
            )
        conn.commit()
    finally:
        conn.close()

    # Embed and upsert to Qdrant for semantic search (best-effort).
    try:
        ensure_memory_collection()
        embeddings = build_embeddings()
        vector = await embeddings.aembed_query(query)
        client = get_qdrant_client()
        client.upsert(
            collection_name=MEMORY_COLLECTION,
            points=[
                qdrant_models.PointStruct(
                    id=mem_id,
                    vector=vector,
                    payload={"query": query, "summary": summary},
                )
            ],
        )
    except Exception as exc:
        logger.warning("Memory semantic indexing failed: %s", str(exc))


# ---------------------------------------------------------------------------
# Read path
# ---------------------------------------------------------------------------


def load_memory_sync(query: str, limit: int = 3) -> List[Dict[str, str]]:
    """Fallback: substring search via Postgres ILIKE.

    Args:
        query: Search term (matched with ``ILIKE %query%``).
        limit: Maximum number of results to return.

    Returns:
        List of ``{"query": ..., "summary": ...}`` dicts.
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT query, summary
                FROM research_memory
                WHERE query ILIKE %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (f"%{query}%", limit),
            )
            rows = cur.fetchall()
        return [{"query": row[0], "summary": row[1]} for row in rows]
    finally:
        conn.close()


async def load_memory(query: str, limit: int = 3) -> List[Dict[str, str]]:
    """Semantic memory search via Qdrant embeddings with SQL fallback.

    Args:
        query: The current research query to find related memories for.
        limit: Maximum number of results to return.

    Returns:
        List of ``{"query": ..., "summary": ...}`` dicts, or an empty
        list if nothing relevant is found.
    """
    try:
        ensure_memory_collection()
        embeddings = build_embeddings()
        vector = await embeddings.aembed_query(query)
        client = get_qdrant_client()
        response = client.query_points(
            collection_name=MEMORY_COLLECTION,
            query=vector,
            limit=limit,
        )
        results = []
        for hit in response.points:
            payload = hit.payload or {}
            if payload.get("query") and payload.get("summary"):
                results.append(
                    {"query": payload["query"], "summary": payload["summary"]}
                )
        if results:
            return results
    except Exception as exc:
        logger.warning(
            "Semantic memory search failed, falling back to SQL: %s", str(exc)
        )
    return load_memory_sync(query, limit)

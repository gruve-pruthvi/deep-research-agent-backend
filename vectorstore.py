"""Qdrant vector store operations: document chunking, indexing, and retrieval.

All per-session Qdrant collections are named ``research_{session_id}`` and
are recreated fresh for each research run.  Chunks are upserted in batches
of 100 to avoid payload-size limits.
"""

import logging
import os
import uuid
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from config import build_embeddings
from models import Document

logger = logging.getLogger(__name__)

# Maximum chunks kept per document and globally per run.
_MAX_CHUNKS_PER_DOC = 20
_MAX_CHUNKS_TOTAL = 200
_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 120
_CHUNK_TEXT_LIMIT = 1500
_UPSERT_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def get_qdrant_client() -> QdrantClient:
    """Return a Qdrant client configured from the ``QDRANT_URL`` env var."""
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    return QdrantClient(url=url)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_documents(documents: List[Document]) -> List[Dict[str, Any]]:
    """Split documents into overlapping text chunks suitable for embedding.

    Args:
        documents: List of extracted ``Document`` objects.

    Returns:
        List of chunk dicts with keys ``id``, ``text``, ``url``, ``title``.
        At most ``_MAX_CHUNKS_PER_DOC`` chunks per document and
        ``_MAX_CHUNKS_TOTAL`` chunks overall.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
    )
    chunks: List[Dict[str, Any]] = []
    for doc in documents:
        for idx, chunk_text in enumerate(splitter.split_text(doc.content)):
            if idx >= _MAX_CHUNKS_PER_DOC:
                break
            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk_text[:_CHUNK_TEXT_LIMIT],
                    "url": doc.url,
                    "title": doc.title,
                }
            )
    return chunks[:_MAX_CHUNKS_TOTAL]


# ---------------------------------------------------------------------------
# Index + retrieve
# ---------------------------------------------------------------------------


async def index_and_retrieve(
    session_id: str,
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """Embed chunks, upsert to Qdrant, then return the top-k nearest results.

    The per-session collection is dropped and recreated on each call to
    ensure a clean slate for every research run.

    Args:
        session_id: Unique session identifier used to name the collection.
        query: The research query used for the similarity search.
        chunks: List of chunk dicts from ``chunk_documents``.
        top_k: Number of results to return from the similarity search.

    Returns:
        List of dicts with keys ``text``, ``url``, ``title``.
    """
    if not chunks:
        return []

    embeddings = build_embeddings()
    vectors = await embeddings.aembed_documents(
        [chunk["text"] for chunk in chunks]
    )

    client = get_qdrant_client()
    collection_name = f"research_{session_id}"

    if client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=len(vectors[0]),
            distance=qdrant_models.Distance.COSINE,
        ),
    )

    points = [
        qdrant_models.PointStruct(
            id=chunk["id"],
            vector=vector,
            payload={
                "text": chunk["text"],
                "url": chunk["url"],
                "title": chunk["title"],
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    for i in range(0, len(points), _UPSERT_BATCH_SIZE):
        client.upsert(
            collection_name=collection_name,
            points=points[i : i + _UPSERT_BATCH_SIZE],
            wait=True,
        )

    query_vector = await embeddings.aembed_query(query)
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    )

    results: List[Dict[str, Any]] = []
    for hit in response.points:
        payload = hit.payload or {}
        results.append(
            {
                "text": payload.get("text", ""),
                "url": payload.get("url", ""),
                "title": payload.get("title", ""),
            }
        )
    return results

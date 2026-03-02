"""Researcher worker agent.

Distributes the document set across N parallel workers, each of which
summarises its assigned bucket and cites sources inline.
"""

import asyncio
import logging
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from config import research_llm
from models import Document, ResearchState

logger = logging.getLogger(__name__)

_DEFAULT_WORKERS = 3
_DEFAULT_SNIPPET_CHARS = 1200


# ---------------------------------------------------------------------------
# Snippet helper
# ---------------------------------------------------------------------------


def build_doc_snippets(
    documents: List[Document], max_chars: int = _DEFAULT_SNIPPET_CHARS
) -> List[str]:
    """Return truncated text snippets from a list of documents.

    Args:
        documents: Source documents to summarise.
        max_chars: Maximum characters taken from each document's content.

    Returns:
        List of formatted snippet strings ready for LLM input.
    """
    return [
        f"Title: {doc.title}\nURL: {doc.url}\nContent: {doc.content[:max_chars]}"
        for doc in documents
    ]


# ---------------------------------------------------------------------------
# Researcher worker
# ---------------------------------------------------------------------------


async def run_researchers(
    state: ResearchState, workers: int = _DEFAULT_WORKERS
) -> List[str]:
    """Run N parallel researcher workers, each summarising a document bucket.

    Documents are distributed round-robin across ``workers`` buckets.
    Empty buckets (when there are fewer documents than workers) are skipped.

    Args:
        state: Current ``ResearchState`` (reads ``documents``, ``query``).
        workers: Number of parallel summarisation workers.

    Returns:
        List of researcher note strings (one per non-empty bucket).
    """
    documents = state["documents"]
    if not documents:
        return []

    buckets: List[List[Document]] = [[] for _ in range(workers)]
    for idx, doc in enumerate(documents):
        buckets[idx % workers].append(doc)

    async def summarize(bucket: List[Document], index: int) -> str:
        """Summarise a single bucket of documents."""
        doc_snippets = build_doc_snippets(bucket)
        system = (
            "You are a research assistant. Summarize the provided documents "
            "with key facts and note important evidence. "
            "Cite sources by URL inline."
        )
        user = (
            f"Research question: {state['query']}\n"
            "Documents:\n\n" + "\n\n".join(doc_snippets)
        )
        response = await research_llm.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return f"Researcher {index + 1} notes:\n{response.content}"

    tasks = [
        summarize(bucket, idx)
        for idx, bucket in enumerate(buckets)
        if bucket
    ]
    if not tasks:
        return []
    return list(await asyncio.gather(*tasks))

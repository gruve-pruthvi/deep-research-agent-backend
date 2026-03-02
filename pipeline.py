"""Research pipeline orchestration.

Implements the iterative loop:
  plan → search → score_sources → extract → chunk → index → gaps → [repeat]

``run_research_loop`` drives up to ``max_iters`` iterations, emitting SSE
status events via the optional ``emit_status`` callback.
"""

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.planner import plan_queries
from config import research_llm
from extraction.documents import extract_document
from models import Document, ResearchState, SearchResult
from search.providers import SEARCH_PROVIDERS
from search.scoring import (
    apply_credibility_scores,
    provider_breakdown,
    score_sources_llm,
)
from vectorstore import chunk_documents, index_and_retrieve

logger = logging.getLogger(__name__)

# Type alias for the status-emission callback.
_EmitFn = Callable[[str, str, Optional[Dict[str, Any]]], Awaitable[None]]


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


async def run_search(state: ResearchState) -> Dict[str, Any]:
    """Execute all search providers in parallel for each planned query.

    Deduplicates results by URL and logs provider-level failures without
    aborting the pipeline.

    Args:
        state: Current ``ResearchState`` (reads ``search_queries``,
            ``max_results``).

    Returns:
        Dict with key ``"sources"`` containing a list of ``SearchResult``.
    """
    results: List[SearchResult] = []
    seen: set[str] = set()

    for query in state["search_queries"]:
        if len(query.strip()) < 2:
            logger.warning("Skipping short query=%s", query)
            continue
        logger.info("Searching web query=%s", query)

        provider_tasks = [
            provider(query, max_results=state["max_results"])
            for _, provider in SEARCH_PROVIDERS
        ]
        provider_results = await asyncio.gather(*provider_tasks, return_exceptions=True)

        for (name, _), hits in zip(SEARCH_PROVIDERS, provider_results):
            if isinstance(hits, Exception):
                logger.warning(
                    "Search provider failed provider=%s error=%s", name, str(hits)
                )
                continue
            for hit in hits:
                if hit.url and hit.url not in seen:
                    results.append(hit)
                    seen.add(hit.url)

    return {"sources": results}


async def fetch_sources(state: ResearchState) -> Dict[str, Any]:
    """Fetch and extract content for the top-N sources.

    Args:
        state: Current ``ResearchState`` (reads ``sources``, ``max_docs``).

    Returns:
        Dict with key ``"documents"`` containing extracted ``Document`` objects.
    """
    documents: List[Document] = []
    for source in state["sources"][: state["max_docs"]]:
        try:
            doc = await extract_document(source)
            if doc.content:
                documents.append(doc)
        except Exception as exc:
            logger.warning(
                "Failed to extract source url=%s error=%s", source.url, str(exc)
            )
    return {"documents": documents}


async def index_sources(state: ResearchState) -> Dict[str, Any]:
    """Chunk documents and upsert to Qdrant, then retrieve relevant chunks.

    Falls back to raw chunk snippets when Qdrant is unavailable.

    Args:
        state: Current ``ResearchState`` (reads ``documents``, ``session_id``,
            ``query``, ``top_k``).

    Returns:
        Dict with keys ``"chunks"`` and ``"retrieved"``.
    """
    chunks = chunk_documents(state["documents"])
    try:
        retrieved = await index_and_retrieve(
            session_id=state["session_id"],
            query=state["query"],
            chunks=chunks,
            top_k=state["top_k"],
        )
    except Exception as exc:
        logger.error(
            "Indexing failed session_id=%s error=%s", state["session_id"], str(exc)
        )
        retrieved = [
            {"text": c["text"], "url": c["url"], "title": c["title"]}
            for c in chunks[:8]
        ]
    return {"chunks": chunks, "retrieved": retrieved}


async def assess_gaps(state: ResearchState) -> Dict[str, Any]:
    """Decide whether additional research iterations are needed.

    Args:
        state: Current ``ResearchState`` (reads ``query``, ``sources``).

    Returns:
        Dict with keys ``"continue"`` (bool) and ``"gaps"`` (list of str).
    """
    system = (
        "You are a research analyst. Decide if more research is needed. "
        'Return JSON: {"continue": true/false, "gaps": ["..."]}'
    )
    source_titles = ", ".join(src.title for src in state["sources"][:8])
    prompt = (
        f"Question: {state['query']}\n"
        f"Current sources: {source_titles}\n"
        "If coverage is weak or missing perspectives, propose gaps."
    )
    response = await research_llm.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=prompt)]
    )
    try:
        data = json.loads(response.content)
        if isinstance(data, dict):
            return {
                "continue": bool(data.get("continue", False)),
                "gaps": [str(g) for g in data.get("gaps", []) if str(g)],
            }
    except Exception:
        logger.warning("Gap assessment parsing failed")
    return {"continue": False, "gaps": []}


# ---------------------------------------------------------------------------
# Main research loop
# ---------------------------------------------------------------------------


async def run_research_loop(
    state: ResearchState,
    emit_status: Optional[_EmitFn] = None,
    max_iters: int = 3,
    approved_queries: Optional[List[str]] = None,
) -> ResearchState:
    """Execute the iterative research loop (plan → search → index → gaps).

    Runs up to ``max_iters`` iterations.  After each iteration the gap
    assessor decides whether to continue; the loop also stops early when
    no gaps are identified.

    Args:
        state: Mutable ``ResearchState`` dict (modified in-place).
        emit_status: Optional async callback for SSE stage events.
            Signature: ``(stage, message, data) -> None``.
        max_iters: Hard upper bound on the number of iterations.
        approved_queries: Pre-approved queries to use on the first
            iteration, bypassing the planning stage.

    Returns:
        The updated ``ResearchState`` after all iterations complete.
    """
    for iteration in range(1, max_iters + 1):
        state["iteration"] = iteration

        # ── Plan ─────────────────────────────────────────────────────────
        if emit_status:
            await emit_status("plan", f"Planning iteration {iteration}", None)

        if iteration == 1 and approved_queries:
            state["search_queries"] = approved_queries
        else:
            try:
                plan_result = await plan_queries(state)
                state.update(plan_result)
            except Exception as exc:
                logger.error(
                    "plan_queries failed iteration=%d error=%s", iteration, str(exc)
                )
                if emit_status:
                    await emit_status(
                        "warning", f"Planning failed: {exc}", {"stage": "plan"}
                    )
                break

        if emit_status:
            await emit_status(
                "queries",
                "Generated search queries",
                {"queries": state["search_queries"]},
            )
            await emit_status(
                "plan_preview",
                "Research plan ready",
                {"queries": state["search_queries"], "iteration": iteration},
            )

        # ── Search ───────────────────────────────────────────────────────
        if emit_status:
            await emit_status("search", f"Searching (iteration {iteration})", None)
        try:
            search_result = await run_search(state)
            state.update(search_result)
        except Exception as exc:
            logger.error(
                "run_search failed iteration=%d error=%s", iteration, str(exc)
            )
            if emit_status:
                await emit_status(
                    "warning", f"Search failed: {exc}", {"stage": "search"}
                )
            state["sources"] = state.get("sources", [])

        # ── Score sources ─────────────────────────────────────────────────
        heuristic_ranked = apply_credibility_scores(state["sources"], {})
        llm_targets = heuristic_ranked[: min(8, len(heuristic_ranked))]
        try:
            llm_scores = await score_sources_llm(llm_targets)
        except Exception as exc:
            logger.warning("score_sources_llm failed: %s", str(exc))
            llm_scores = {}
        state["sources"] = apply_credibility_scores(state["sources"], llm_scores)

        if emit_status:
            await emit_status(
                "sources",
                f"Found {len(state['sources'])} sources",
                {
                    "top_sources": [
                        {
                            "title": src.title,
                            "url": src.url,
                            "provider": src.provider,
                            "credibility": src.credibility,
                        }
                        for src in state["sources"][:5]
                    ],
                    "providers": provider_breakdown(state["sources"]),
                },
            )

        # ── Extract ───────────────────────────────────────────────────────
        if emit_status:
            await emit_status("extract", "Extracting and indexing sources", None)
        try:
            fetch_result = await fetch_sources(state)
            state.update(fetch_result)
        except Exception as exc:
            logger.error("fetch_sources failed: %s", str(exc))
            if emit_status:
                await emit_status(
                    "warning",
                    f"Document extraction failed: {exc}",
                    {"stage": "extract"},
                )
        if emit_status:
            await emit_status(
                "documents",
                f"Extracted {len(state['documents'])} documents",
                {"documents": len(state["documents"])},
            )

        # ── Index + retrieve ──────────────────────────────────────────────
        try:
            index_result = await index_sources(state)
            state.update(index_result)
        except Exception as exc:
            logger.error("index_sources failed: %s", str(exc))
            if emit_status:
                await emit_status(
                    "warning",
                    f"Indexing failed, using raw chunks: {exc}",
                    {"stage": "retrieval"},
                )
        if emit_status:
            await emit_status(
                "retrieval",
                f"Retrieved {len(state['retrieved'])} relevant chunks",
                {"chunks": len(state["retrieved"])},
            )

        # ── Gap assessment ────────────────────────────────────────────────
        try:
            gap_result = await assess_gaps(state)
            state["gaps"] = gap_result.get("gaps", [])
        except Exception as exc:
            logger.warning("assess_gaps failed: %s", str(exc))
            if emit_status:
                await emit_status(
                    "warning",
                    f"Gap assessment failed: {exc}",
                    {"stage": "gaps"},
                )
            gap_result = {"continue": False, "gaps": []}
            state["gaps"] = []

        if emit_status:
            await emit_status(
                "gaps",
                "Gap assessment complete",
                {
                    "continue": gap_result.get("continue"),
                    "gaps": state["gaps"],
                },
            )

        if not gap_result.get("continue") or iteration >= max_iters:
            break

    return state

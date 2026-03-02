"""Search provider implementations: Tavily, DuckDuckGo, and Wikipedia.

Each provider exposes an ``async`` function with the signature
``(query: str, max_results: int) -> List[SearchResult]``.

``SEARCH_PROVIDERS`` is the ordered list used by the pipeline; Tavily is
primary, DuckDuckGo is the fallback, and Wikipedia is supplementary.
"""

import asyncio
import logging
import os
from typing import Any, List, Tuple

import httpx

from models import SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tavily (primary)
# ---------------------------------------------------------------------------


async def search_web(query: str, max_results: int = 5) -> List[SearchResult]:
    """Search using the Tavily API.

    Args:
        query: The search query string.
        max_results: Maximum number of results to request.

    Returns:
        List of ``SearchResult`` objects with ``provider="tavily"``.

    Raises:
        RuntimeError: When the API key is missing or the request fails.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("Missing TAVILY_API_KEY for web search")
        raise RuntimeError("Missing TAVILY_API_KEY for web search")

    payload = {
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }

    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.post(
                "https://api.tavily.com/search",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Tavily search failed. status=%s response=%s",
                exc.response.status_code,
                exc.response.text,
            )
            raise RuntimeError(
                f"Tavily search failed ({exc.response.status_code}). "
                "Check TAVILY_API_KEY and request payload."
            ) from exc
        except httpx.HTTPError as exc:
            logger.error("Tavily search http error query=%s error=%s", query, str(exc))
            raise RuntimeError("Tavily search request failed") from exc

    results: List[SearchResult] = []
    for item in data.get("results", []):
        results.append(
            SearchResult(
                title=item.get("title", "Untitled"),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                provider="tavily",
            )
        )
    return results


# ---------------------------------------------------------------------------
# DuckDuckGo (fallback, synchronous wrapper)
# ---------------------------------------------------------------------------


def _search_duckduckgo_sync(query: str, max_results: int = 5) -> List[SearchResult]:
    """Synchronous DuckDuckGo search using the ``ddgs`` library.

    Runs in a worker thread via ``asyncio.to_thread``; must not be called
    directly from async code.
    """
    try:
        from ddgs import DDGS
    except Exception as exc:
        logger.warning("ddgs not available: %s", str(exc))
        return []

    results: List[SearchResult] = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append(
                SearchResult(
                    title=item.get("title", "Untitled"),
                    url=item.get("href", ""),
                    snippet=item.get("body", ""),
                    provider="duckduckgo",
                )
            )
    return results


async def search_duckduckgo(query: str, max_results: int = 5) -> List[SearchResult]:
    """Search using DuckDuckGo (runs the sync implementation in a thread).

    Args:
        query: The search query string.
        max_results: Maximum number of results to request.

    Returns:
        List of ``SearchResult`` objects with ``provider="duckduckgo"``.
    """
    return await asyncio.to_thread(_search_duckduckgo_sync, query, max_results)


# ---------------------------------------------------------------------------
# Wikipedia (supplementary)
# ---------------------------------------------------------------------------


async def search_wikipedia(query: str, max_results: int = 5) -> List[SearchResult]:
    """Search Wikipedia via its public REST API.

    Args:
        query: The search query string.
        max_results: Maximum number of results to request.

    Returns:
        List of ``SearchResult`` objects with ``provider="wikipedia"``.
        Returns an empty list on HTTP errors (non-fatal).
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": max_results,
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params=params,
                headers={"User-Agent": "DeepResearchAgent/1.0 (research-bot)"},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        logger.warning("Wikipedia search failed query=%s error=%s", query, str(exc))
        return []

    results: List[SearchResult] = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title", "Untitled")
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=item.get("snippet", ""),
                provider="wikipedia",
            )
        )
    return results


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

SEARCH_PROVIDERS: List[Tuple[str, Any]] = [
    ("tavily", search_web),
    ("duckduckgo", search_duckduckgo),
    ("wikipedia", search_wikipedia),
]

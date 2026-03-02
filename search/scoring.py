"""Credibility scoring for search results.

Provides heuristic TLD-based scoring, LLM-assisted scoring, blended
combination, provider breakdown, and an overall confidence metric.
"""

import json
import logging
from typing import Dict, List
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage

from config import research_llm
from models import SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heuristic scoring
# ---------------------------------------------------------------------------


def heuristic_credibility(url: str) -> float:
    """Estimate source credibility from the URL's top-level domain.

    Scoring tiers:
    * ``.gov`` / ``.edu`` → 0.95
    * ``.org`` / ``.int`` → 0.80
    * ``.com`` / ``.net`` → 0.60
    * anything else      → 0.50

    Args:
        url: The source URL.

    Returns:
        Credibility score in [0.0, 1.0].
    """
    domain = urlparse(url).netloc.lower()
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 0.95
    if any(domain.endswith(tld) for tld in (".org", ".int")):
        return 0.80
    if any(domain.endswith(tld) for tld in (".com", ".net")):
        return 0.60
    return 0.50


# ---------------------------------------------------------------------------
# LLM-assisted scoring
# ---------------------------------------------------------------------------


async def score_sources_llm(sources: List[SearchResult]) -> Dict[str, float]:
    """Ask the LLM to score each source on authority, recency, and relevance.

    Args:
        sources: List of ``SearchResult`` objects to score.

    Returns:
        Dict mapping URL → credibility score (0.0–1.0).  Returns an empty
        dict when ``sources`` is empty or the LLM response cannot be parsed.
    """
    if not sources:
        return {}

    prompt = (
        "Score each source credibility from 0 to 1 based on publisher authority, "
        "recency, and relevance. Return JSON object mapping url -> score."
    )
    items = "\n".join(f"- {src.title} | {src.url}" for src in sources)
    response = await research_llm.ainvoke(
        [SystemMessage(content=prompt), HumanMessage(content=items)]
    )
    try:
        data = json.loads(response.content)
        if isinstance(data, dict):
            return {
                str(k): float(v)
                for k, v in data.items()
                if k and v is not None
            }
    except Exception:
        logger.warning("LLM credibility scoring failed")
    return {}


# ---------------------------------------------------------------------------
# Score application
# ---------------------------------------------------------------------------


def apply_credibility_scores(
    sources: List[SearchResult],
    llm_scores: Dict[str, float],
) -> List[SearchResult]:
    """Blend heuristic (60%) and LLM (40%) scores and sort by credibility.

    When no LLM score is available for a source the heuristic score is
    used as-is.

    Args:
        sources: List of ``SearchResult`` objects to score.
        llm_scores: Mapping of URL → LLM credibility score.

    Returns:
        The same list with ``.credibility`` populated, sorted descending.
    """
    for src in sources:
        heuristic = heuristic_credibility(src.url)
        llm_score = llm_scores.get(src.url)
        if llm_score is not None:
            src.credibility = round(0.6 * heuristic + 0.4 * llm_score, 3)
        else:
            src.credibility = round(heuristic, 3)
    return sorted(sources, key=lambda s: s.credibility, reverse=True)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def provider_breakdown(sources: List[SearchResult]) -> Dict[str, int]:
    """Count sources by provider name.

    Args:
        sources: List of ``SearchResult`` objects.

    Returns:
        Dict mapping provider name → count.
    """
    counts: Dict[str, int] = {}
    for src in sources:
        counts[src.provider] = counts.get(src.provider, 0) + 1
    return counts


def compute_confidence(sources: List[SearchResult], uncertainty: float) -> float:
    """Derive an overall confidence score (0–1) from credibility and uncertainty.

    Formula: ``(avg_credibility × 0.7) + ((1 − uncertainty) × 0.3)``,
    clamped to [0.0, 1.0].

    Args:
        sources: Scored ``SearchResult`` objects.
        uncertainty: Uncertainty score from the verifier (0 = confident,
            1 = uncertain).

    Returns:
        Confidence score rounded to three decimal places.
    """
    if not sources:
        return 0.3
    avg_cred = sum(src.credibility for src in sources) / max(len(sources), 1)
    score = max(0.0, min(1.0, (avg_cred * 0.7) + ((1.0 - uncertainty) * 0.3)))
    return round(score, 3)

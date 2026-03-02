"""Query-planning agent.

``plan_queries`` generates 3–5 focused web-search queries for the current
research state, incorporating known gaps and relevant session memories.
``clarify_query`` optionally surfaces 1–2 clarifying questions for
ambiguous user queries.
"""

import json
import logging
import re
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from config import orchestrator_llm
from memory import load_memory
from models import ResearchState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query parsing helper
# ---------------------------------------------------------------------------


def parse_search_queries(text: str) -> List[str]:
    """Parse a JSON array or bullet list of search queries from LLM output.

    Strips code fences, then attempts JSON parsing.  Falls back to
    splitting on newlines and stripping bullet characters.

    Args:
        text: Raw LLM output string.

    Returns:
        List of non-empty query strings.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned.strip())
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [
                str(item).strip()
                for item in data
                if isinstance(item, str) and str(item).strip()
            ]
    except json.JSONDecodeError:
        pass
    lines = [line.strip("- ").strip() for line in cleaned.splitlines()]
    return [line for line in lines if len(line) >= 2]


# ---------------------------------------------------------------------------
# Planning agent
# ---------------------------------------------------------------------------


async def plan_queries(state: ResearchState) -> Dict[str, Any]:
    """Generate focused search queries for the next research iteration.

    Incorporates any known gaps from previous iterations and relevant
    entries from persistent session memory.

    Args:
        state: Current ``ResearchState`` (reads ``query``, ``gaps``).

    Returns:
        Dict with key ``"search_queries"`` containing a list of strings.
    """
    system = (
        "You are a research planner. Generate 3-5 focused web search queries. "
        "Return ONLY a JSON array of strings. Do not include code fences."
    )

    gap_context = ""
    if state.get("gaps"):
        gap_context = f"Known gaps: {', '.join(state['gaps'])}"

    memory_items = await load_memory(state["query"])
    memory_context = ""
    if memory_items:
        summaries = "\n".join(f"- {item['summary']}" for item in memory_items)
        memory_context = f"Related memory:\n{summaries}"

    prompt = (
        f"Research question: {state['query']}\n"
        f"{gap_context}\n"
        f"{memory_context}"
    )
    response = await orchestrator_llm.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=prompt)]
    )
    queries = parse_search_queries(response.content)
    return {"search_queries": queries}


# ---------------------------------------------------------------------------
# Clarification agent
# ---------------------------------------------------------------------------


async def clarify_query(query: str) -> List[str]:
    """Return 1–2 clarifying questions if the query is ambiguous.

    Args:
        query: The raw user research query.

    Returns:
        A list of clarifying question strings, or an empty list when the
        query is already self-contained.
    """
    system = (
        "You are a research assistant. If the query is ambiguous and would "
        "meaningfully benefit from 1-2 targeted clarifying questions, return "
        "them as a JSON array of strings. If the query is clear and "
        "self-contained, return an empty JSON array []. "
        "Do not include code fences. Return ONLY a JSON array."
    )
    response = await orchestrator_llm.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=f"Query: {query}")]
    )
    try:
        data = json.loads(response.content.strip())
        if isinstance(data, list):
            return [str(q).strip() for q in data if str(q).strip()]
    except Exception:
        logger.warning(
            "clarify_query parsing failed content=%s", response.content[:200]
        )
    return []

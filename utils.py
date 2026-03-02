"""Serialisation and formatting utilities shared across the backend.

Contains helpers for converting arbitrary objects to JSON-safe forms,
converting chat message dicts to LangChain message objects, building
human-readable source lists, and compacting ResearchState for DB storage.
"""

import logging
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from models import ChatMessage, SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------


def to_jsonable(value: Any) -> Any:
    """Recursively convert any value to a JSON-serialisable form.

    Handles: None, primitives (strips null chars from str), lists/tuples,
    dicts, dataclasses, objects with ``__dict__``, and LangChain messages
    (which expose a ``content`` attribute).

    Args:
        value: Any Python object.

    Returns:
        A JSON-serialisable representation.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, str):
            return value.replace("\u0000", "")
        return value
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if is_dataclass(value):
        return {key: to_jsonable(val) for key, val in asdict(value).items()}
    if hasattr(value, "__dict__"):
        return {key: to_jsonable(val) for key, val in vars(value).items()}
    content = getattr(value, "content", None)
    if content is not None:
        return content
    return str(value)


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def to_messages(payload: List[ChatMessage]) -> List[AnyMessage]:
    """Convert a list of ChatMessage dicts into LangChain message objects.

    Args:
        payload: List of ``{"role": ..., "content": ...}`` dicts.

    Returns:
        Equivalent list of ``AIMessage``, ``HumanMessage``, or
        ``SystemMessage`` objects.
    """
    converted: List[AnyMessage] = []
    for item in payload:
        if item["role"] == "assistant":
            converted.append(AIMessage(content=item["content"]))
        elif item["role"] == "system":
            converted.append(SystemMessage(content=item["content"]))
        else:
            converted.append(HumanMessage(content=item["content"]))
    return converted


# ---------------------------------------------------------------------------
# Source formatting
# ---------------------------------------------------------------------------


def build_source_list(sources: List[SearchResult]) -> str:
    """Format a numbered source list string from a list of SearchResult objects.

    Args:
        sources: Ordered list of search results.

    Returns:
        Multi-line string with one ``N. Title — URL`` entry per source.
    """
    return "\n".join(
        f"{idx + 1}. {src.title} — {src.url}"
        for idx, src in enumerate(sources)
    )


# ---------------------------------------------------------------------------
# State compaction for storage
# ---------------------------------------------------------------------------


def compact_state_for_storage(state: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a compacted, JSON-safe snapshot of ResearchState for DB storage.

    Large lists are truncated and Document objects are reduced to
    ``{title, url}`` pairs to stay within Postgres JSONB size limits.

    Args:
        state: A ResearchState dict (or the inner ``"state"`` sub-dict).

    Returns:
        A dict that is safe to serialise with ``json.dumps``.
    """
    return {
        "session_id": state["session_id"],
        "query": state["query"],
        "depth": state["depth"],
        "iteration": state["iteration"],
        "max_results": state["max_results"],
        "max_docs": state["max_docs"],
        "top_k": state["top_k"],
        "search_queries": state["search_queries"],
        "sources": [to_jsonable(src) for src in state["sources"][:50]],
        "documents": [
            {"title": doc.title, "url": doc.url}
            for doc in state["documents"][:50]
        ],
        "retrieved": state["retrieved"][:20],
        "report": state.get("report", "")[:20000],
        "gaps": state["gaps"],
        "researcher_notes": state["researcher_notes"][:5],
        "analyst_summary": state["analyst_summary"],
        "critic_notes": state["critic_notes"],
        "verifier_notes": state.get("verifier_notes", ""),
        "uncertainty_score": state.get("uncertainty_score"),
        "confidence_score": state.get("confidence_score"),
        "evaluation": state.get("evaluation", {}),
        "transparency": state.get("transparency", {}),
    }

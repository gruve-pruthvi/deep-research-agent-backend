"""Data models for the Deep Research Agent backend.

Defines all shared TypedDicts, dataclasses, and type aliases used
across the pipeline, agents, and API layers.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated


# ---------------------------------------------------------------------------
# Primitive type aliases
# ---------------------------------------------------------------------------

Role = Literal["user", "assistant", "system"]


# ---------------------------------------------------------------------------
# Chat-mode models
# ---------------------------------------------------------------------------


class ChatMessage(TypedDict):
    """A single chat turn with a role and text content."""

    role: Role
    content: str


class GraphState(TypedDict):
    """State for the LangGraph conversational chat graph."""

    messages: Annotated[List[AnyMessage], add_messages]


# ---------------------------------------------------------------------------
# Research-pipeline dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """A single search result returned by any provider."""

    title: str
    url: str
    snippet: str
    provider: str
    credibility: float = 0.0


@dataclass
class Document:
    """A fetched and normalised source document."""

    title: str
    url: str
    content: str


# ---------------------------------------------------------------------------
# Research pipeline state
# ---------------------------------------------------------------------------


class ResearchState(TypedDict):
    """Full mutable state carried through every stage of the research loop."""

    session_id: str
    query: str
    depth: str
    iteration: int
    max_results: int
    max_docs: int
    top_k: int
    search_queries: List[str]
    sources: List[SearchResult]
    documents: List[Document]
    chunks: List[Dict[str, Any]]
    retrieved: List[Dict[str, Any]]
    report: str
    gaps: List[str]
    researcher_notes: List[str]
    analyst_summary: str
    critic_notes: str
    verifier_notes: str
    uncertainty_score: float
    confidence_score: float
    evaluation: Dict[str, Any]
    transparency: Dict[str, Any]

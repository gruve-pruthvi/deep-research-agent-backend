"""LangGraph graph builders for the chat and research-pipeline visualisation.

``compiled_graph`` is the live conversational graph used by
``/chat/stream``.  ``compiled_research_graph`` is a topology-only mirror
of the research pipeline used solely for Mermaid visualisation at
``/graph``.
"""

import logging
from typing import Any, Dict, List

from langchain_core.messages import AnyMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config import chat_llm
from models import GraphState, ResearchState
from tools import get_current_utc_time, run_python

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chat graph
# ---------------------------------------------------------------------------

CHAT_TOOLS = [get_current_utc_time, run_python]
_chat_llm_with_tools = chat_llm.bind_tools(CHAT_TOOLS)


def build_graph() -> Any:
    """Build and compile the conversational LangGraph chat graph.

    The graph has an ``assistant`` node backed by ``chat_llm`` and, when
    tools are registered, a ``tools`` node wired with conditional edges.

    Returns:
        A compiled ``StateGraph`` ready for ``astream_events``.
    """
    graph: StateGraph = StateGraph(GraphState)

    async def call_model(state: GraphState) -> Dict[str, List[AnyMessage]]:
        """Invoke the LLM with the current message history."""
        response = await _chat_llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    graph.add_node("assistant", call_model)

    if CHAT_TOOLS:
        graph.add_node("tools", ToolNode(CHAT_TOOLS))
        graph.add_edge(START, "assistant")
        graph.add_conditional_edges("assistant", tools_condition)
        graph.add_edge("tools", "assistant")
    else:
        graph.add_edge(START, "assistant")

    graph.add_edge("assistant", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Research pipeline graph (topology-only, for visualisation)
# ---------------------------------------------------------------------------


def build_research_graph() -> Any:
    """Build a topology-only StateGraph mirroring the research pipeline.

    All nodes are no-ops; the graph is compiled only to produce a Mermaid
    diagram.  It is not used to execute the actual pipeline.

    Returns:
        A compiled ``StateGraph`` that can render a Mermaid diagram.
    """
    rg: StateGraph = StateGraph(ResearchState)

    async def _noop(state: ResearchState) -> Dict[str, Any]:  # noqa: ARG001
        """Placeholder node; does nothing."""
        return {}

    for node_name in (
        "plan",
        "search",
        "score_sources",
        "extract",
        "chunk_docs",
        "retrieval",
        "gaps",
        "researchers",
        "analyst",
        "critic",
        "verify",
        "writer",
        "save",
    ):
        rg.add_node(node_name, _noop)

    rg.add_edge(START, "plan")
    rg.add_edge("plan", "search")
    rg.add_edge("search", "score_sources")
    rg.add_edge("score_sources", "extract")
    rg.add_edge("extract", "chunk_docs")
    rg.add_edge("chunk_docs", "retrieval")
    rg.add_edge("retrieval", "gaps")
    rg.add_conditional_edges(
        "gaps",
        lambda s: "plan" if s.get("gaps") else "researchers",
        {"plan": "plan", "researchers": "researchers"},
    )
    rg.add_edge("researchers", "analyst")
    rg.add_edge("analyst", "critic")
    rg.add_edge("critic", "verify")
    rg.add_edge("verify", "writer")
    rg.add_edge("writer", "save")
    rg.add_edge("save", END)

    return rg.compile()


# ---------------------------------------------------------------------------
# Module-level compiled instances
# ---------------------------------------------------------------------------

compiled_graph = build_graph()
compiled_research_graph = build_research_graph()

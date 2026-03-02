"""Writer agent.

Produces the final structured research report and runs a post-processing
citation-verification pass to correct mismatched ``[N]`` numbers.
"""

import logging
from typing import Any, Dict, List

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from config import orchestrator_llm
from models import ResearchState, SearchResult
from utils import build_source_list

logger = logging.getLogger(__name__)

_LENGTH_HINTS: Dict[str, str] = {
    "shallow": "Keep it concise (400-700 words).",
    "standard": "Provide a balanced report (700-1200 words).",
    "deep": "Provide a detailed report (1200-1800 words).",
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_writer_prompt(state: ResearchState) -> List[AnyMessage]:
    """Construct the full system + user prompt for the writer LLM.

    Args:
        state: Current ``ResearchState`` (reads ``query``, ``sources``,
            ``retrieved``, ``analyst_summary``, ``critic_notes``,
            ``verifier_notes``, ``uncertainty_score``, ``depth``).

    Returns:
        List of LangChain message objects ready for ``ainvoke`` or
        ``astream``.
    """
    source_list = build_source_list(state["sources"])
    evidence = "\n\n".join(
        f"Source: {item.get('title', '')} ({item.get('url', '')})\n"
        f"Excerpt: {item.get('text', '')}"
        for item in state["retrieved"]
    )
    length_hint = _LENGTH_HINTS.get(
        state.get("depth", "standard"), "Provide a balanced report."
    )

    system = (
        "You are a research writer. Produce a structured report with "
        "citations. Use [1], [2] etc matching the source list."
    )
    user = (
        f"Research question: {state['query']}\n\n"
        f"Analyst summary:\n{state['analyst_summary']}\n\n"
        f"Critic notes:\n{state['critic_notes']}\n\n"
        f"Verifier notes:\n{state.get('verifier_notes', '')}\n\n"
        f"Uncertainty score (0=high confidence, 1=low confidence): "
        f"{state.get('uncertainty_score', 0.5)}\n\n"
        f"Sources (for grounding only, do not output a Sources section):\n"
        f"{source_list}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Length guidance: {length_hint}\n\n"
        "Return sections:\n"
        "1. Executive Summary\n"
        "2. Key Findings\n"
        "3. Supporting Evidence\n"
        "4. Contradictions\n"
        "Do NOT include a Sources section in the final answer."
    )
    return [SystemMessage(content=system), HumanMessage(content=user)]


def build_synthesis_prompt(state: ResearchState) -> List[AnyMessage]:
    """Alias for ``build_writer_prompt`` used by the non-streaming path."""
    return build_writer_prompt(state)


# ---------------------------------------------------------------------------
# Non-streaming synthesis
# ---------------------------------------------------------------------------


async def synthesize(state: ResearchState) -> Dict[str, Any]:
    """Generate the research report in a single non-streaming call.

    Args:
        state: Current ``ResearchState``.

    Returns:
        Dict with key ``"report"`` containing the generated report string.
    """
    response = await orchestrator_llm.ainvoke(build_synthesis_prompt(state))
    return {"report": response.content}


# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------


async def verify_citations(report: str, sources: List[SearchResult]) -> str:
    """Post-process the report to fix mismatched ``[N]`` citation numbers.

    Args:
        report: The raw generated report text.
        sources: Ordered list of sources used to build citation numbers.

    Returns:
        The corrected report string.  Returns the original on any failure.
    """
    if not report or not sources:
        return report

    system = (
        "You are a citation verifier. Given a research report with inline "
        "[N] citations and a numbered source list, verify that each [N] "
        "reference matches the correct source. Correct any wrong citation "
        "numbers. Return the corrected report only, without explanation."
    )
    source_list = build_source_list(sources)
    user = f"Report:\n{report}\n\nSources:\n{source_list}"

    try:
        response = await orchestrator_llm.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return response.content or report
    except Exception as exc:
        logger.warning("Citation verification failed: %s", str(exc))
        return report

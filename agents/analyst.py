"""Analyst agent.

Synthesises researcher notes into key findings, contradictions, and open
questions.  Has access to the ``run_python`` sandboxed tool for quick
numerical analysis.
"""

import logging
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from config import orchestrator_llm
from models import ResearchState
from tools import run_python

logger = logging.getLogger(__name__)


async def run_analyst(state: ResearchState) -> str:
    """Combine researcher notes into structured findings via the orchestrator LLM.

    If the LLM requests a ``run_python`` tool call the result is fed back
    in a second invocation so the analyst can incorporate computed values.

    Args:
        state: Current ``ResearchState`` (reads ``query``,
            ``researcher_notes``).

    Returns:
        Analyst summary string with key findings and open questions.
    """
    system = (
        "You are an analyst. Combine researcher notes into key findings, "
        "contradictions, and open questions. Keep it structured."
    )
    notes = "\n\n".join(state["researcher_notes"])
    user = f"Research question: {state['query']}\n\nResearcher notes:\n{notes}"

    analyst_llm = orchestrator_llm.bind_tools([run_python])
    response = await analyst_llm.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )

    if response.tool_calls:
        tool_outputs: List[ToolMessage] = []
        for call in response.tool_calls:
            result = run_python.invoke(call["args"])
            tool_outputs.append(
                ToolMessage(content=result, tool_call_id=call["id"])
            )
        response = await analyst_llm.ainvoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=user),
                response,
                *tool_outputs,
            ]
        )

    return response.content

"""Critic agent.

Reviews the analyst summary for weak evidence, missing citations, and
potential biases, and surfaces actionable improvement suggestions.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from config import orchestrator_llm
from models import ResearchState
from utils import build_source_list

logger = logging.getLogger(__name__)


async def run_critic(state: ResearchState) -> str:
    """Identify weaknesses in the analyst summary and suggest fixes.

    Args:
        state: Current ``ResearchState`` (reads ``query``,
            ``analyst_summary``, ``sources``).

    Returns:
        Critic notes string with identified issues and actionable fixes.
    """
    system = (
        "You are a critical reviewer. Identify weak evidence, missing "
        "citations, and potential biases. Provide actionable fixes."
    )
    user = (
        f"Research question: {state['query']}\n\n"
        f"Analyst summary:\n{state['analyst_summary']}\n\n"
        f"Sources:\n{build_source_list(state['sources'])}"
    )
    response = await orchestrator_llm.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return response.content

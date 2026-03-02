"""Verifier agent.

Assesses the overall evidence quality and returns an uncertainty score
between 0 (high confidence) and 1 (low confidence).
"""

import json
import logging
from typing import Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from config import orchestrator_llm
from models import ResearchState
from utils import build_source_list

logger = logging.getLogger(__name__)


async def run_verifier(state: ResearchState) -> Tuple[str, float]:
    """Estimate evidence quality and produce an uncertainty score.

    Args:
        state: Current ``ResearchState`` (reads ``query``,
            ``analyst_summary``, ``critic_notes``, ``sources``).

    Returns:
        Tuple of ``(verifier_notes: str, uncertainty_score: float)``.
        ``uncertainty_score`` is in [0.0, 1.0]; defaults to 0.5 when the
        LLM response cannot be parsed.
    """
    system = (
        "You are a verifier. Assess the evidence quality and estimate an "
        "uncertainty score from 0 (high confidence) to 1 (low confidence). "
        'Return JSON: {"notes": "...", "uncertainty": 0.0}'
    )
    user = (
        f"Research question: {state['query']}\n\n"
        f"Analyst summary:\n{state['analyst_summary']}\n\n"
        f"Critic notes:\n{state['critic_notes']}\n\n"
        f"Sources:\n{build_source_list(state['sources'])}"
    )
    response = await orchestrator_llm.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    try:
        data = json.loads(response.content)
        return str(data.get("notes", "")), float(data.get("uncertainty", 0.5))
    except Exception:
        logger.warning("Verifier parsing failed")
        return response.content, 0.5

"""Evaluator agent.

Scores the final report on coverage, evidence quality, and clarity on a
1–5 scale each, and returns structured JSON feedback.
"""

import json
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from config import research_llm
from models import ResearchState
from utils import build_source_list

logger = logging.getLogger(__name__)


async def run_evaluation(state: ResearchState) -> Dict[str, Any]:
    """Score the research report on coverage, evidence, and clarity (1–5 each).

    Args:
        state: Current ``ResearchState`` (reads ``query``, ``report``,
            ``sources``).

    Returns:
        Dict with keys ``coverage``, ``evidence``, ``clarity`` (int 0–5)
        and ``notes`` (str).  Falls back to zeros with the raw LLM content
        in ``notes`` when parsing fails.
    """
    system = (
        "You are an evaluator. Score the report quality "
        "(coverage, evidence, clarity) from 1-5. "
        'Return JSON: {"coverage": 0-5, "evidence": 0-5, '
        '"clarity": 0-5, "notes": "..."}'
    )
    user = (
        f"Question: {state['query']}\n\n"
        f"Report:\n{state.get('report', '')}\n\n"
        f"Sources:\n{build_source_list(state['sources'])}"
    )
    response = await research_llm.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    try:
        data = json.loads(response.content)
        if isinstance(data, dict):
            return data
    except Exception:
        logger.warning("Evaluation parsing failed")
    return {
        "coverage": 0,
        "evidence": 0,
        "clarity": 0,
        "notes": response.content,
    }

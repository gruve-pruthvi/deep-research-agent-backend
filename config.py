"""Environment configuration and shared LLM/embedding instances.

All ChatOpenAI and OpenAIEmbeddings objects are constructed once here and
imported by the modules that need them, avoiding repeated instantiation.
"""

import os
from typing import Dict, Optional

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# ---------------------------------------------------------------------------
# LLM instances
# ---------------------------------------------------------------------------

# Worker-class LLM used for parallelisable, lower-stakes tasks.
research_llm = ChatOpenAI(
    model=os.getenv("WORKER_MODEL", "gpt-4o-mini"),
    temperature=0.2,
)

research_llm_stream = ChatOpenAI(
    model=os.getenv("WORKER_MODEL", "gpt-4o-mini"),
    temperature=0.2,
    streaming=True,
)

# Orchestrator-class LLM used for planning, synthesis, and critique.
orchestrator_llm = ChatOpenAI(
    model=os.getenv("ORCHESTRATOR_MODEL", "gpt-4o"),
    temperature=0.2,
)

orchestrator_llm_stream = ChatOpenAI(
    model=os.getenv("ORCHESTRATOR_MODEL", "gpt-4o"),
    temperature=0.2,
    streaming=True,
)

# Vision-capable LLM — built lazily so the GOOGLE_API_KEY only needs to
# be present when an image is actually processed, not at import time.
# Override the model via the VISION_MODEL env var (default: gemini-2.0-flash).
# Requires GOOGLE_API_KEY in the environment.
_vision_llm: Optional[ChatGoogleGenerativeAI] = None


def get_vision_llm() -> ChatGoogleGenerativeAI:
    """Return the shared Gemini vision LLM instance, constructing it on first call.

    Requires ``GOOGLE_API_KEY`` to be set in the environment.  Override the
    model with the ``VISION_MODEL`` env var
    (default: ``gemini-3.1-flash-image-preview``).
    """
    global _vision_llm
    if _vision_llm is None:
        _vision_llm = ChatGoogleGenerativeAI(
            model=os.getenv("VISION_MODEL", "gemini-3.1-flash-image-preview"),
            temperature=0.2,
        )
    return _vision_llm

# Chat-mode LLM (streaming, used by the conversational graph).
chat_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    streaming=True,
)


# ---------------------------------------------------------------------------
# Embedding factory
# ---------------------------------------------------------------------------


def build_embeddings() -> OpenAIEmbeddings:
    """Return an OpenAIEmbeddings instance (text-embedding-3-small, 1536-dim)."""
    return OpenAIEmbeddings(model="text-embedding-3-small")


# ---------------------------------------------------------------------------
# Depth presets
# ---------------------------------------------------------------------------


def depth_config(depth: str) -> Dict[str, int]:
    """Return the max_results / max_docs / top_k preset for a given depth label.

    Args:
        depth: One of ``"shallow"``, ``"standard"``, or ``"deep"``.

    Returns:
        Dict with keys ``max_results``, ``max_docs``, and ``top_k``.
        Falls back to ``"standard"`` for unknown depth values.
    """
    presets: Dict[str, Dict[str, int]] = {
        "shallow": {"max_results": 3, "max_docs": 6, "top_k": 6},
        "standard": {"max_results": 5, "max_docs": 10, "top_k": 8},
        "deep": {"max_results": 7, "max_docs": 14, "top_k": 10},
    }
    return presets.get(depth, presets["standard"])

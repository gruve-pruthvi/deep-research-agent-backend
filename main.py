"""FastAPI application entry point and route handlers.

All business logic lives in the sub-modules; this file wires together the
HTTP layer, SSE streaming, and startup/shutdown hooks.

Run from inside ``deep_research_backend/``:
    uvicorn main:app --reload --port 8000
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from agents.analyst import run_analyst
from agents.critic import run_critic
from agents.evaluator import run_evaluation
from agents.planner import clarify_query
from agents.researchers import run_researchers
from agents.verifier import run_verifier
from agents.writer import build_writer_prompt, synthesize, verify_citations
from config import depth_config, orchestrator_llm_stream
from database import fetch_history, fetch_run, init_db, save_run
from graphs import compiled_graph, compiled_research_graph
from memory import save_memory
from models import ChatMessage, ResearchState
from pipeline import run_research_loop
from search.scoring import compute_confidence
from utils import to_jsonable, to_messages

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("deep_research_backend")

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI()

_allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory chat history (non-persistent, per-process)
# ---------------------------------------------------------------------------

memory_store: Dict[str, List[Any]] = {}

# ---------------------------------------------------------------------------
# DB init at module load
# ---------------------------------------------------------------------------

init_db()


# ---------------------------------------------------------------------------
# Startup hook
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def log_graph_diagrams() -> None:
    """Log Mermaid diagrams for both graphs at server startup."""
    # logger.info(
    #     "=== Chat graph (Mermaid) ===\n%s",
    #     compiled_graph.get_graph().draw_mermaid(),
    # )
    # logger.info(
    #     "=== Research pipeline graph (Mermaid) ===\n%s",
    #     compiled_research_graph.get_graph().draw_mermaid(),
    # )


# ---------------------------------------------------------------------------
# Utility: build initial ResearchState
# ---------------------------------------------------------------------------


def _build_initial_state(
    session_id: str, query: str, depth: str
) -> ResearchState:
    """Create an empty ``ResearchState`` for a new research run.

    Args:
        session_id: Browser / client session identifier.
        query: The research query string.
        depth: Depth preset (``"shallow"`` / ``"standard"`` / ``"deep"``).

    Returns:
        Fully initialised ``ResearchState`` dict.
    """
    config = depth_config(depth)
    return ResearchState(
        session_id=session_id,
        query=query,
        depth=depth,
        iteration=0,
        max_results=config["max_results"],
        max_docs=config["max_docs"],
        top_k=config["top_k"],
        search_queries=[],
        sources=[],
        documents=[],
        chunks=[],
        retrieved=[],
        report="",
        gaps=[],
        researcher_notes=[],
        analyst_summary="",
        critic_notes="",
        verifier_notes="",
        uncertainty_score=0.5,
        confidence_score=0.0,
        evaluation={},
        transparency={},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/graph")
async def get_graph_diagrams() -> Dict[str, str]:
    """Return Mermaid diagrams for the chat and research pipeline graphs."""
    return {
        "chat_graph": compiled_graph.get_graph().draw_mermaid(),
        "research_graph": compiled_research_graph.get_graph().draw_mermaid(),
    }


@app.get("/graph/view", response_class=HTMLResponse)
async def view_graph_diagrams() -> HTMLResponse:
    """Render both graphs in the browser using Mermaid.js."""
    chat_mermaid = compiled_graph.get_graph().draw_mermaid()
    research_mermaid = compiled_research_graph.get_graph().draw_mermaid()
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Deep Research Agent — Graph View</title>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #f9f9fb;
           margin: 0; padding: 2rem; }}
    h1 {{ font-size: 1.4rem; margin-bottom: 2rem; color: #333; }}
    h2 {{ font-size: 1rem; font-weight: 600; color: #555;
          margin: 2rem 0 0.75rem; }}
    .graph-card {{
      background: #fff; border: 1px solid #e2e2e8; border-radius: 10px;
      padding: 1.5rem; margin-bottom: 2rem;
      box-shadow: 0 1px 4px rgba(0,0,0,.06);
    }}
    .mermaid svg {{ max-width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>Deep Research Agent — Graph View</h1>

  <h2>Chat Graph</h2>
  <div class="graph-card">
    <div class="mermaid">{chat_mermaid}</div>
  </div>

  <h2>Research Pipeline Graph</h2>
  <div class="graph-card">
    <div class="mermaid">{research_mermaid}</div>
  </div>

  <script>
    mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.post("/chat/stream")
async def chat_stream(body: Dict[str, Any], request: Request) -> StreamingResponse:
    """Stream a chat response using the LangGraph conversational graph.

    Accepts ``session_id``, ``messages`` (list), and/or ``message`` (str).
    Emits ``tool_start``, ``tool_end``, ``delta``, and ``[DONE]`` SSE events.
    """
    logger.info("chat_stream request")
    session_id: str = body.get("session_id") or "default"
    messages_payload: List[ChatMessage] = body.get("messages", [])
    message: Optional[str] = body.get("message")

    history = memory_store.get(session_id, [])
    if messages_payload:
        history = to_messages(messages_payload)
    if message:
        history = [*history, HumanMessage(content=message)]

    async def event_stream() -> AsyncGenerator[str, None]:
        assistant_chunks: List[str] = []
        disconnected = False
        try:
            async for event in compiled_graph.astream_events(
                {"messages": history}, version="v2"
            ):
                if await request.is_disconnected():
                    disconnected = True
                    break

                if event["event"] == "on_tool_start":
                    tool_name = event.get("name", "tool")
                    payload = json.dumps(
                        {
                            "type": "tool_start",
                            "name": tool_name,
                            "input": to_jsonable(
                                event.get("data", {}).get("input")
                            ),
                        }
                    )
                    yield f"data: {payload}\n\n"
                    continue

                if event["event"] == "on_tool_end":
                    tool_name = event.get("name", "tool")
                    payload = json.dumps(
                        {
                            "type": "tool_end",
                            "name": tool_name,
                            "output": to_jsonable(
                                event.get("data", {}).get("output")
                            ),
                        }
                    )
                    yield f"data: {payload}\n\n"
                    continue

                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if not chunk:
                        continue
                    delta = getattr(chunk, "content", None)
                    if not delta:
                        continue
                    assistant_chunks.append(delta)
                    yield f"data: {json.dumps({'delta': delta})}\n\n"

            if not await request.is_disconnected():
                yield "data: [DONE]\n\n"

        except Exception as exc:
            logger.exception("chat_stream failed session_id=%s", session_id)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        else:
            if assistant_chunks and not disconnected:
                memory_store[session_id] = [
                    *history,
                    AIMessage(content="".join(assistant_chunks)),
                ]

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/research")
async def run_research(body: Dict[str, Any]) -> Dict[str, Any]:
    """Run a non-streaming research request and return the completed report."""
    query: Optional[str] = body.get("query")
    session_id: str = body.get("session_id") or uuid.uuid4().hex
    depth: str = body.get("depth") or "standard"

    if not query:
        logger.warning("run_research missing query")
        return {"error": "Missing query"}

    logger.info("run_research session_id=%s", session_id)
    run_id = uuid.uuid4().hex
    state = _build_initial_state(session_id, query, depth)

    save_run(run_id, session_id, query, depth, "running", {"state": state})
    result = await run_research_loop(state)

    result["researcher_notes"] = await run_researchers(result)
    result["analyst_summary"] = await run_analyst(result)
    result["critic_notes"] = await run_critic(result)
    verifier_notes, uncertainty = await run_verifier(result)
    result["verifier_notes"] = verifier_notes
    result["uncertainty_score"] = uncertainty

    report_response = await synthesize(result)
    result.update(report_response)
    result["confidence_score"] = compute_confidence(
        result["sources"], result["uncertainty_score"]
    )
    result["evaluation"] = await run_evaluation(result)
    result["transparency"] = {
        "queries": result["search_queries"],
        "sources": [
            {"title": src.title, "url": src.url, "credibility": src.credibility}
            for src in result["sources"][:10]
        ],
    }

    await save_memory(query, result.get("report", "")[:2000])
    save_run(run_id, session_id, query, depth, "completed", {"state": result})

    return {
        "query": query,
        "session_id": session_id,
        "depth": depth,
        "report": result.get("report", ""),
        "sources": [
            {"title": src.title, "url": src.url, "snippet": src.snippet}
            for src in result.get("sources", [])
        ],
    }


@app.post("/research/clarify")
async def research_clarify(body: Dict[str, Any]) -> Dict[str, Any]:
    """Return clarifying questions for ambiguous queries, or proceed immediately."""
    query = body.get("query", "").strip()
    if not query:
        return {"questions": [], "proceed": True}
    questions = await clarify_query(query)
    if questions:
        return {"questions": questions, "proceed": False}
    return {"questions": [], "proceed": True}


@app.post("/research/stream")
async def run_research_stream(
    body: Dict[str, Any], request: Request
) -> StreamingResponse:
    """Stream a research response with per-stage SSE status events.

    Accepts ``query``, ``session_id``, ``depth``, ``max_iterations``
    (1–5), and ``approved_queries`` (optional pre-approved query list).

    SSE event types emitted: ``status`` (stage progress), ``delta``
    (report tokens), ``error``, ``[DONE]``.
    """
    query: Optional[str] = body.get("query")
    session_id: str = body.get("session_id") or uuid.uuid4().hex
    depth: str = body.get("depth") or "standard"
    max_iterations: int = max(1, min(5, int(body.get("max_iterations", 3))))
    approved_queries: Optional[List[str]] = body.get("approved_queries") or None

    if not query:
        logger.warning("run_research_stream missing query")
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': 'Missing query'})}\n\n"]),
            media_type="text/event-stream",
        )

    logger.info(
        "run_research_stream session_id=%s max_iterations=%d",
        session_id,
        max_iterations,
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        run_id = uuid.uuid4().hex
        queue: asyncio.Queue[str] = asyncio.Queue()

        async def emit_status(
            stage: str,
            message: str,
            data: Optional[Dict[str, Any]] = None,
        ) -> None:
            """Enqueue a status SSE event for the consumer."""
            payload = {
                "type": "status",
                "stage": stage,
                "message": message,
                "data": data,
            }
            await queue.put(f"data: {json.dumps(payload)}\n\n")

        try:
            state = _build_initial_state(session_id, query, depth)
            save_run(run_id, session_id, query, depth, "running", {"state": state})

            loop_task = asyncio.create_task(
                run_research_loop(
                    state,
                    emit_status=emit_status,
                    max_iters=max_iterations,
                    approved_queries=approved_queries,
                )
            )

            # Stream queued events while the loop runs.
            while True:
                if loop_task.done() and queue.empty():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.5)
                    yield event
                except asyncio.TimeoutError:
                    if loop_task.done():
                        break

            # Drain remaining queued events.
            while not queue.empty():
                yield queue.get_nowait()

            state = await loop_task

            # ── Post-loop agent stages ──────────────────────────────────
            await emit_status("researchers", "Summarizing sources", None)
            state["researcher_notes"] = await run_researchers(state)

            await emit_status("analyst", "Analyzing findings", None)
            state["analyst_summary"] = await run_analyst(state)

            await emit_status("critic", "Reviewing evidence", None)
            state["critic_notes"] = await run_critic(state)

            await emit_status("verify", "Verifying evidence", None)
            verifier_notes, uncertainty = await run_verifier(state)
            state["verifier_notes"] = verifier_notes
            state["uncertainty_score"] = uncertainty
            await emit_status(
                "verify",
                "Evidence verified",
                {
                    "uncertainty": state["uncertainty_score"],
                    "verifier_notes": state["verifier_notes"],
                },
            )

            # Drain status events before starting the writer stream.
            while not queue.empty():
                yield queue.get_nowait()

            save_run(
                run_id, session_id, query, depth, "synthesizing", {"state": state}
            )

            # ── Streaming writer ────────────────────────────────────────
            await emit_status("writer", "Writing report", None)
            while not queue.empty():
                yield queue.get_nowait()

            prompt = build_writer_prompt(state)
            report_parts: List[str] = []

            async for chunk in orchestrator_llm_stream.astream(prompt):
                if await request.is_disconnected():
                    break
                delta = getattr(chunk, "content", None)
                if not delta:
                    continue
                report_parts.append(delta)
                yield f"data: {json.dumps({'delta': delta})}\n\n"

            if not await request.is_disconnected():
                raw_report = "".join(report_parts)

                await emit_status("writer", "Verifying citations", None)
                state["report"] = await verify_citations(raw_report, state["sources"])
                state["confidence_score"] = compute_confidence(
                    state["sources"], state["uncertainty_score"]
                )
                state["evaluation"] = await run_evaluation(state)
                state["transparency"] = {
                    "queries": state["search_queries"],
                    "sources": [
                        {
                            "title": src.title,
                            "url": src.url,
                            "credibility": src.credibility,
                        }
                        for src in state["sources"][:10]
                    ],
                }

                await emit_status(
                    "transparency",
                    "Transparency report ready",
                    {
                        "confidence": state["confidence_score"],
                        "uncertainty": state["uncertainty_score"],
                        "verifier_notes": state["verifier_notes"],
                        "evaluation": state["evaluation"],
                        "transparency": state["transparency"],
                    },
                )

                # Drain final status events.
                while not queue.empty():
                    yield queue.get_nowait()

                save_run(
                    run_id, session_id, query, depth, "completed", {"state": state}
                )
                await save_memory(query, state.get("report", "")[:2000])
                await emit_status("done", "Report complete", None)

                while not queue.empty():
                    yield queue.get_nowait()

                yield "data: [DONE]\n\n"

        except Exception as exc:
            logger.exception(
                "run_research_stream failed session_id=%s", session_id
            )
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/research/history")
async def get_research_history(
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """List past research runs, optionally filtered by ``session_id``."""
    return {"runs": fetch_history(session_id)}


@app.get("/research/{run_id}")
async def get_research_run(run_id: str) -> Dict[str, Any]:
    """Retrieve a specific research run by ID including the stored report."""
    result = fetch_run(run_id)
    if result is None:
        return {"error": "Run not found"}
    return result

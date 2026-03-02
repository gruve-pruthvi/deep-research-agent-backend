# Backend — Deep Research Agent

FastAPI + LangGraph backend for streaming chat and deep research orchestration.

---

## Stack

| Component | Library / Service |
|-----------|------------------|
| API framework | FastAPI + Uvicorn |
| Orchestration | LangGraph, LangChain |
| LLMs | OpenAI (chat, vision, embeddings) — tiered orchestrator / worker |
| Vector store | Qdrant |
| Relational DB | Postgres (psycopg) |
| Search | Tavily, DuckDuckGo (`ddgs`), Wikipedia |
| Extraction | httpx, PyMuPDF, Playwright (optional) |
| Splitter | LangChain RecursiveCharacterTextSplitter |

---

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | All routes, pipeline stages, LLM wrappers, storage helpers |
| `requirements.txt` | Python dependencies |
| `docker-compose.yml` | Local Postgres + Qdrant containers |
| `.env` | Runtime configuration (see below) |

---

## Setup

**1. Create virtual environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure `.env`:**

```bash
# Required
OPENAI_API_KEY=...
TAVILY_API_KEY=...

# Infrastructure
QDRANT_URL=http://localhost:6333
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=postgres

# Model tiers
ORCHESTRATOR_MODEL=gpt-4o          # high-stakes stages: plan, analyst, critic, verifier, writer
WORKER_MODEL=gpt-4o-mini           # bulk stages: researchers, scoring, gap analysis

# Optional features
ENABLE_PLAYWRIGHT=false            # JS-rendered page extraction
ALLOWED_ORIGINS=http://localhost:5173
LOG_LEVEL=INFO
```

**3. Start infrastructure:**

```bash
docker compose up -d
```

**4. (Optional) Enable Playwright:**

```bash
playwright install chromium
```

**5. Run API server:**

```bash
uvicorn main:app --reload --port 8000
```

---

## Endpoints

### `GET /`
Health check. Returns `{"status": "ok"}`.

---

### `POST /chat/stream`
SSE conversational stream.

```json
{ "session_id": "optional", "message": "What time is it in UTC?" }
```

Events emitted: `tool_start`, `tool_end`, `delta`, `[DONE]`, `error`.

---

### `POST /research`
Non-streaming research. Waits for full pipeline completion.

```json
{
  "session_id": "optional",
  "query": "Compare OpenAI deep research vs Grok deep research",
  "depth": "standard"
}
```

Returns `{query, session_id, depth, report, sources}`.

---

### `POST /research/clarify`
Pre-research clarification. Call this before `/research/stream` to detect ambiguous queries.

```json
{ "query": "Tell me about transformer models" }
```

Response when questions are needed:
```json
{ "questions": ["Do you mean transformer neural networks or electrical transformers?"], "proceed": false }
```

Response when query is clear:
```json
{ "questions": [], "proceed": true }
```

---

### `POST /research/stream`
Main streaming research endpoint. SSE stream.

```json
{
  "session_id": "optional",
  "query": "Impact of LLM scaling laws on AI research",
  "depth": "deep",
  "max_iterations": 3,
  "approved_queries": ["optional", "pre-approved query list"]
}
```

Parameters:
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | required | Research question |
| `depth` | string | `"standard"` | `shallow` / `standard` / `deep` |
| `max_iterations` | int | `3` | 1–5; number of search-retrieve-gap cycles |
| `approved_queries` | string[] | `null` | Override first iteration's planned queries |
| `session_id` | string | auto UUID | Used for Qdrant collection naming and memory lookup |

SSE event reference:

| Stage | `data` payload |
|-------|---------------|
| `plan` | — |
| `plan_preview` | `{queries: string[], iteration: number}` |
| `queries` | `{queries: string[]}` |
| `search` | — |
| `sources` | `{top_sources: [{title, url, provider, credibility}], providers: {name: count}}` |
| `extract` | — |
| `documents` | `{documents: number}` |
| `retrieval` | `{chunks: number}` |
| `gaps` | `{continue: bool, gaps: string[]}` |
| `researchers` | — |
| `analyst` | — |
| `critic` | — |
| `verify` | `{uncertainty: number, verifier_notes: string}` |
| `writer` | — |
| `transparency` | `{confidence, uncertainty, verifier_notes, evaluation, transparency}` |
| `done` | — |
| `warning` | `{stage: string}` — emitted on non-fatal stage failure |

---

### `GET /research/history`
List past research runs. Optional `?session_id=` filter.

Response: `{ "runs": [{id, session_id, query, depth, status, updated_at}] }`

---

### `GET /research/{run_id}`
Retrieve a specific run including the full report.

Response: `{id, session_id, query, depth, status, updated_at, report, sources, evaluation, uncertainty_score, confidence_score}`

---

## Research Pipeline

```
1. clarify_query()          — optional; returns questions for ambiguous queries
2. plan_queries()           — orchestrator LLM; generates 3–5 search queries; uses semantic memory
3. [plan_preview event]     — emits planned queries before search starts
4. run_search()             — parallel Tavily + DuckDuckGo + Wikipedia
5. apply_credibility_scores() — hybrid heuristic + LLM scoring
6. fetch_sources()          — extract HTML / PDF / image; Playwright fallback if enabled
7. index_sources()          — chunk → embed → Qdrant upsert; fallback to raw chunks on failure
8. assess_gaps()            — worker LLM; decides whether to iterate
   └─ loop back to step 2 if continue=true and iterations remaining
9. run_researchers()        — 3 parallel workers (worker model); bucket-summarise documents
10. run_analyst()           — orchestrator LLM + Python sandbox tool
11. run_critic()            — orchestrator LLM; identifies weak evidence and biases
12. run_verifier()          — orchestrator LLM; returns uncertainty score (0–1) + notes
13. [verify event emitted]  — streams uncertainty + notes to frontend
14. build_writer_prompt()   — assembles full context for writer
15. orchestrator_llm_stream.astream() — streams report tokens as delta events
16. verify_citations()      — orchestrator LLM; post-pass to correct [N] citation numbers
17. run_evaluation()        — worker LLM; scores coverage / evidence / clarity (1–5)
18. save_memory()           — saves to Postgres + embeds to Qdrant memory collection
19. save_run()              — final state snapshot to Postgres
```

### Error Recovery

Each stage (plan, search, fetch, index, gaps) is wrapped in `try/except`. On failure:
- A `warning` SSE event is emitted with the stage name and error message.
- The pipeline continues with degraded state (e.g., empty sources, raw chunks bypassing Qdrant).
- The final report is still generated from whatever data was collected.

---

## LLM Model Tiers

| Variable | Default | Used for |
|----------|---------|----------|
| `ORCHESTRATOR_MODEL` | `gpt-4o` | `plan_queries`, `run_analyst`, `run_critic`, `run_verifier`, `synthesize`, streaming writer, `clarify_query`, `verify_citations` |
| `WORKER_MODEL` | `gpt-4o-mini` | `run_researchers`, `score_sources_llm`, `assess_gaps`, `run_evaluation` |

Both are configurable via environment variables.

---

## Memory System

### Session memory (Qdrant `research_<session_id>`)
- Created fresh each run; deleted and recreated on restart.
- Stores document chunks for RAG retrieval within the active research session.

### Cross-session semantic memory (Qdrant `research_memory_embeddings`)
- Persistent; never deleted between runs.
- `save_memory()` — embeds the query and upserts `{query, summary}` to both Qdrant and Postgres.
- `load_memory()` — embeds the incoming query, semantic search via Qdrant (limit 3); falls back to Postgres ILIKE if Qdrant is unavailable.
- Memory context is injected into `plan_queries()` so past related research informs new query planning.

---

## Tools

| Tool | Description |
|------|-------------|
| `get_current_utc_time` | Returns current UTC time (used in chat mode) |
| `run_python` | Sandboxed Python execution — no imports, no file/network access; only safe builtins |

---

## Data Stores

### Postgres tables

| Table | Columns | Purpose |
|-------|---------|---------|
| `research_runs` | id, session_id, query, depth, status, updated_at, data_json (JSONB) | Run snapshots; upserted on status changes |
| `research_memory` | id, query, summary, created_at | Query + summary for cross-session memory |

### Qdrant collections

| Collection | Dimensions | Distance | Lifecycle |
|------------|-----------|----------|-----------|
| `research_<session_id>` | 1536 | Cosine | Per-run; recreated each session |
| `research_memory_embeddings` | 1536 | Cosine | Persistent across all sessions |

---

## Smoke Test

```bash
# Test streaming research
curl -N -X POST http://127.0.0.1:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"n8n cloud vs self-hosted","depth":"shallow"}'

# Test clarification
curl -X POST http://127.0.0.1:8000/research/clarify \
  -H "Content-Type: application/json" \
  -d '{"query":"tell me about models"}'

# Test history
curl http://127.0.0.1:8000/research/history
```

---

## Common Issues

| Problem | Fix |
|---------|-----|
| `ddgs not available` | `pip install -r requirements.txt` in active venv |
| `playwright not installed` | `pip install playwright && playwright install chromium` |
| Wikipedia 403 | Non-fatal; pipeline continues with Tavily + DuckDuckGo |
| Port 8000 conflict | `lsof -i :8000` |
| Qdrant memory collection missing | Recreated automatically on next request via `ensure_memory_collection()` |

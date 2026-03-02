# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Deep Research Agent** is a full-stack AI-powered research system with:
- **Backend**: FastAPI + LangGraph orchestration (`deep_research_backend/main.py`)
- **Frontend**: React + Vite UI (`deep_research_frontend/src/App.tsx`)

## Commands

### Backend
```bash
cd deep_research_backend

# Start infrastructure (Postgres + Qdrant) — required before running the server
docker compose up -d

# Setup Python environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run dev server
uvicorn main:app --reload --port 8000

# Smoke test
curl -N -X POST http://127.0.0.1:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"your query here","depth":"shallow"}'
```

### Frontend
```bash
cd deep_research_frontend
npm install
npm run dev        # Dev server at http://localhost:5173
npm run build      # Production build (tsc -b && vite build)
npm run lint       # ESLint
npm run preview    # Preview production build
```

## Architecture

### Research Pipeline Flow

User query → FastAPI (`/research/stream`) → `run_research_loop()` → Multi-agent analysis → SSE stream to frontend

The loop runs up to 3 iterations, executing these stages in order:

| Stage | What happens |
|---|---|
| `plan` | LLM generates 3–5 focused search queries using session memory |
| `search` | Parallel searches: Tavily (primary), DuckDuckGo (fallback), Wikipedia (optional) |
| `sources` | Hybrid credibility scoring: 60% heuristic (TLD-based) + 40% LLM (GPT-4o-mini) |
| `extract` | Fetch top-N docs; handles HTML, PDFs (PyMuPDF), images (Gemini vision) |
| `documents` | Chunk docs via RecursiveCharacterTextSplitter (800 chars, 120 overlap, max 20/doc) |
| `retrieval` | Embed chunks (text-embedding-3-small) → upsert to Qdrant → semantic search top-k |
| `gaps` | LLM decides if additional iteration is needed |
| `researchers` | 3 parallel workers summarize bucketed documents |
| `analyst` | Synthesizes findings; has Python sandbox tool |
| `critic` | Identifies weaknesses and evidence gaps |
| `verify` | Assigns uncertainty score (0–1) |
| `writer` | Streams final structured report |
| `done` | Save run snapshot to Postgres |

### Depth Configuration
```python
{"shallow":  {"max_results": 3,  "max_docs": 6,  "top_k": 6},   # ~400–700 words
 "standard": {"max_results": 5,  "max_docs": 10, "top_k": 8},   # ~700–1200 words
 "deep":     {"max_results": 7,  "max_docs": 14, "top_k": 10}}  # ~1200–1800 words
```

### State Management

`ResearchState` (TypedDict) carries all data through the pipeline: queries, raw sources, extracted documents, vector-retrieved chunks, agent notes, and the final report. State is compacted (large fields truncated) before serialization to Postgres JSONB.

### Data Stores

- **Qdrant** (`:6333`): Per-session vector collections named `research_{session_id}`; deleted and recreated fresh each session. 1536-dim vectors, cosine distance. Batched upserts (100 chunks) to avoid payload limits.
- **Postgres** (`:5432`): Two tables — `research_runs` (run snapshots with compact `data_json`) and `research_memory` (query + report summary for context reuse).
- **In-memory**: Chat history stored per-process only (not persisted).

### API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/chat/stream` | Conversational mode, SSE stream |
| POST | `/research` | Non-streaming research |
| POST | `/research/stream` | Streaming research with stage events |

SSE event types: `status` (stage progress + data), `delta` (report token), `error`, `[DONE]`.

### Frontend

`App.tsx` is the single component handling both modes:
- **Chat mode**: Standard SSE stream with tool event display
- **Research mode**: Timeline visualization (14 stages), source cards with credibility scores, Markdown report rendering, transparency panel

SSE parsing handles: `status`, `delta`, `tool_start`, `tool_end`, `error` event types.

## Key Configuration

**Backend `.env`** (required):
```
OPENAI_API_KEY=...       # LLM + embeddings
GOOGLE_API_KEY=...       # Gemini vision model (image understanding)
TAVILY_API_KEY=...       # Primary search provider
QDRANT_URL=http://localhost:6333
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=postgres
```

Optional overrides:
```
VISION_MODEL=gemini-3.1-flash-image-preview   # Gemini model used for image description
```

## Important Implementation Details

- **Python sandbox**: Analyst's `run_python` tool restricts to safe builtins — no imports, no file/network access.
- **Qdrant payload limit**: Chunk text is stored in payload; batched upserts (100 at a time) prevent oversized requests.
- **JSON serialization**: Null characters are stripped and dataclasses converted before Postgres storage.
- **Wikipedia 403**: Handled gracefully — pipeline continues with other providers.
- **No auth layer**: No authentication or multi-tenant support.

## No Automated Tests

There is no test suite. Verification is done manually via curl or browser UI.

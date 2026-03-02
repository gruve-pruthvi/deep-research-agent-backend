# Deep Research Agent — Backend

FastAPI backend that runs a multi-agent research pipeline.
Give it a question; it searches the web, extracts sources, builds a vector index, and streams a structured report back to the client.

---

## How it works (30-second version)

```
Your query
  → plan search queries   (orchestrator LLM)
  → search web            (Tavily + DuckDuckGo + Wikipedia, parallel)
  → score & rank sources  (heuristic TLD + LLM credibility blend)
  → fetch & extract docs  (HTML / PDF / images)
  → chunk → embed → Qdrant (semantic retrieval)
  → gap check             (repeat up to 3×)
  → 3 researcher workers  (worker LLM, parallel)
  → analyst               (orchestrator LLM + Python sandbox)
  → critic → verifier     (orchestrator LLM)
  → streaming writer      (orchestrator LLM, token-by-token SSE)
  → save to Postgres
```

---

## Prerequisites

| Tool                    | Minimum version        |
| ----------------------- | ---------------------- |
| Docker + Docker Compose | Docker 24              |
| Python                  | 3.11+ (local dev only) |

---

## Option A — Docker (recommended)

Everything — the API, Postgres, and Qdrant — runs in containers.

**Step 1. Clone and enter the backend folder**

```bash
git clone <repo-url>
cd deep_research_backend
```

**Step 2. Create your `.env` file**

```bash
cp .env.example .env   # or create it manually (see Configuration section below)
```

Open `.env` and fill in your API keys.

**Step 3. Build and start all services**

```bash
docker compose up --build
```

The first build takes ~2 minutes (downloads Python image, installs deps).
Subsequent starts are fast:

```bash
docker compose up
```

**Step 4. Verify it's running**

```bash
curl http://localhost:8000/
# → {"status":"ok"}
```

**Step 5. Run a smoke test**

```bash
curl -N -X POST http://localhost:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"What is LangGraph?","depth":"shallow"}'
```

You should see SSE events streaming in your terminal.

**To stop everything:**

```bash
docker compose down
```

**To wipe all stored data (Qdrant + Postgres volumes):**

```bash
docker compose down -v
```

---

## Option B — Local development

Use this when you want fast hot-reloading without rebuilding Docker images.
You still need Docker running for Postgres and Qdrant.

**Step 1. Start infrastructure only**

```bash
docker compose up qdrant postgres -d
```

**Step 2. Create and activate a Python virtual environment**

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

**Step 3. Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 4. Create your `.env` file** (see Configuration section below)

**Step 5. Start the API server with hot-reload**

```bash
uvicorn main:app --reload --port 8000
```

The server restarts automatically whenever you save a `.py` file.

---

## Configuration

Create a `.env` file in the `deep_research_backend/` folder with the following:

```env
# ── Required API keys ───────────────────────────────────────────────────────
OPENAI_API_KEY=sk-...          # OpenAI — used for all LLM calls and embeddings
GOOGLE_API_KEY=AI...           # Google — used for image understanding (Gemini)
TAVILY_API_KEY=tvly-...        # Tavily — primary web search provider

# ── Infrastructure (defaults work with docker-compose) ─────────────────────
QDRANT_URL=http://localhost:6333
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=postgres

# ── Model selection (optional — these are the defaults) ────────────────────
ORCHESTRATOR_MODEL=gpt-4o                       # Used for: planning, analysis, critique, writing
WORKER_MODEL=gpt-4o-mini                        # Used for: researchers, scoring, gap checks
VISION_MODEL=gemini-3.1-flash-image-preview     # Used for: image description

# ── Optional features ───────────────────────────────────────────────────────
ENABLE_PLAYWRIGHT=false        # Set to true to enable JS-rendered page extraction
ALLOWED_ORIGINS=http://localhost:5173
LOG_LEVEL=INFO
```

> **Note:** When running with `docker compose up` (Option A), `QDRANT_URL` and
> `POSTGRES_HOST` are automatically overridden to use Docker's internal network.
> You do not need to change those values in `.env`.

---

## API Reference

### `GET /`

Health check.

```bash
curl http://localhost:8000/
```

Response: `{"status": "ok"}`

---

### `POST /research/stream` — main endpoint

Streams the research pipeline as SSE events.

```bash
curl -N -X POST http://localhost:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Impact of transformer models on NLP",
    "depth": "standard"
  }'
```

| Field              | Type     | Default      | Description                                    |
| ------------------ | -------- | ------------ | ---------------------------------------------- |
| `query`            | string   | **required** | Your research question                         |
| `depth`            | string   | `"standard"` | `shallow` / `standard` / `deep`                |
| `max_iterations`   | int      | `3`          | Search-retrieve cycles (1–5)                   |
| `approved_queries` | string[] | —            | Skip planning; use these queries directly      |
| `session_id`       | string   | auto UUID    | Ties memory and Qdrant collection to a session |

**Depth guide:**

| Depth      | Sources | Report length    |
| ---------- | ------- | ---------------- |
| `shallow`  | 3       | ~400–700 words   |
| `standard` | 5       | ~700–1200 words  |
| `deep`     | 7       | ~1200–1800 words |

**SSE events you'll receive:**

| Event type | When                                                       |
| ---------- | ---------------------------------------------------------- |
| `status`   | Each pipeline stage starts/completes (includes stage data) |
| `delta`    | Report token from the streaming writer                     |
| `warning`  | A non-fatal stage failure; pipeline continues              |
| `[DONE]`   | Report is complete                                         |

---

### `POST /research/clarify`

Check whether a query needs clarification before submitting to the pipeline.

```bash
curl -X POST http://localhost:8000/research/clarify \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about transformers"}'
```

Response (ambiguous): `{"questions": ["Do you mean transformer neural networks or electrical transformers?"], "proceed": false}`
Response (clear): `{"questions": [], "proceed": true}`

---

### `POST /research` — non-streaming

Same pipeline, waits for full completion before responding.

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "depth": "shallow"}'
```

---

### `POST /chat/stream`

Conversational chat mode (not the research pipeline). Streaming SSE.

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "What time is it in UTC?"}'
```

---

### `GET /research/history`

List past research runs.

```bash
curl http://localhost:8000/research/history
curl "http://localhost:8000/research/history?session_id=abc123"
```

---

### `GET /research/{run_id}`

Retrieve a specific run including the full report.

```bash
curl http://localhost:8000/research/abc123def456
```

---

### `GET /graph/view`

Open in a browser to see Mermaid diagrams of the pipeline and chat graphs:

```
http://localhost:8000/graph/view
```

---

## Project structure

```
deep_research_backend/
├── main.py              Route handlers (FastAPI)
├── config.py            LLM instances, depth presets
├── models.py            TypedDicts and dataclasses
├── pipeline.py          Research loop orchestration
├── graphs.py            LangGraph graph builders
├── database.py          Postgres helpers
├── vectorstore.py       Qdrant chunking + retrieval
├── memory.py            Cross-session semantic memory
├── utils.py             Serialisation helpers
├── tools.py             Sandboxed Python + UTC time tools
├── search/
│   ├── providers.py     Tavily, DuckDuckGo, Wikipedia
│   └── scoring.py       Credibility scoring
├── extraction/
│   └── documents.py     HTML / PDF / image extraction
├── agents/
│   ├── planner.py       Query planning + clarification
│   ├── researchers.py   Parallel document summarisers
│   ├── analyst.py       Findings synthesis
│   ├── critic.py        Evidence review
│   ├── verifier.py      Uncertainty scoring
│   ├── writer.py        Report generation + citation check
│   └── evaluator.py     Report quality scoring
├── Dockerfile
├── docker-compose.yml   API + Qdrant + Postgres
└── requirements.txt
```

---

## Troubleshooting

| Problem                             | Fix                                                                               |
| ----------------------------------- | --------------------------------------------------------------------------------- |
| `OPENAI_API_KEY not set`            | Add the key to your `.env` file                                                   |
| `GOOGLE_API_KEY required`           | Add `GOOGLE_API_KEY=...` to `.env` (needed for image URLs only)                   |
| Port 8000 already in use            | `lsof -i :8000` then kill the process, or change the port in `docker-compose.yml` |
| Qdrant/Postgres not ready           | Wait a few seconds and retry; containers have health checks                       |
| `ddgs not available`                | Run `pip install -r requirements.txt` inside your activated venv                  |
| Wikipedia returns 403               | Non-fatal — the pipeline continues with Tavily + DuckDuckGo results               |
| JS-heavy pages return empty content | Set `ENABLE_PLAYWRIGHT=true` and run `playwright install chromium`                |
| Want to reset all data              | `docker compose down -v` removes the Postgres and Qdrant volumes                  |

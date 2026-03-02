import json
import logging
import os
import re
import asyncio
import ast
import io
import contextlib
import uuid
from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, TypedDict, Awaitable
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from typing_extensions import Annotated
import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
import psycopg

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("deep_research_backend")

Role = Literal["user", "assistant", "system"]


class ChatMessage(TypedDict):
    role: Role
    content: str


class GraphState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


app = FastAPI()

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
allowed_origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    streaming=True,
)

tools: List[Any] = []
llm_with_tools = llm

@tool
def get_current_utc_time() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def build_graph() -> Any:
    graph = StateGraph(GraphState)

    async def call_model(state: GraphState) -> Dict[str, List[AnyMessage]]:
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    graph.add_node("assistant", call_model)
    if tools:
        graph.add_node("tools", ToolNode(tools))
        graph.add_edge(START, "assistant")
        graph.add_conditional_edges("assistant", tools_condition)
        graph.add_edge("tools", "assistant")
    else:
        graph.add_edge(START, "assistant")
    graph.add_edge("assistant", END)
    return graph.compile()


def build_research_graph() -> Any:
    """Build a LangGraph StateGraph mirroring the research pipeline topology for visualization."""
    rg: StateGraph = StateGraph(ResearchState)

    async def _noop(state: ResearchState) -> Dict[str, Any]:  # noqa: ARG001
        return {}

    for node_name in [
        "plan", "search", "score_sources", "extract",
        "chunk_docs", "retrieval", "gaps",
        "researchers", "analyst", "critic",
        "verify", "writer", "save",
    ]:
        rg.add_node(node_name, _noop)

    rg.add_edge(START, "plan")
    rg.add_edge("plan", "search")
    rg.add_edge("search", "score_sources")
    rg.add_edge("score_sources", "extract")
    rg.add_edge("extract", "chunk_docs")
    rg.add_edge("chunk_docs", "retrieval")
    rg.add_edge("retrieval", "gaps")
    rg.add_conditional_edges(
        "gaps",
        lambda s: "plan" if s.get("gaps") else "researchers",
        {"plan": "plan", "researchers": "researchers"},
    )
    rg.add_edge("researchers", "analyst")
    rg.add_edge("analyst", "critic")
    rg.add_edge("critic", "verify")
    rg.add_edge("verify", "writer")
    rg.add_edge("writer", "save")
    rg.add_edge("save", END)

    return rg.compile()


memory_store: Dict[str, List[AnyMessage]] = {}


def to_messages(payload: List[ChatMessage]) -> List[AnyMessage]:
    converted: List[AnyMessage] = []
    for item in payload:
        if item["role"] == "assistant":
            converted.append(AIMessage(content=item["content"]))
        elif item["role"] == "system":
            converted.append(SystemMessage(content=item["content"]))
        else:
            converted.append(HumanMessage(content=item["content"]))
    return converted


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, str):
            return value.replace("\u0000", "")
        return value
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_jsonable(val) for key, val in value.items()}
    if is_dataclass(value):
        return {key: to_jsonable(val) for key, val in asdict(value).items()}
    if hasattr(value, "__dict__"):
        return {key: to_jsonable(val) for key, val in vars(value).items()}
    content = getattr(value, "content", None)
    if content is not None:
        return content
    return str(value)


def compact_state_for_storage(state: ResearchState) -> Dict[str, Any]:
    return {
        "session_id": state["session_id"],
        "query": state["query"],
        "depth": state["depth"],
        "iteration": state["iteration"],
        "max_results": state["max_results"],
        "max_docs": state["max_docs"],
        "top_k": state["top_k"],
        "search_queries": state["search_queries"],
        "sources": [to_jsonable(src) for src in state["sources"][:50]],
        "documents": [
            {"title": doc.title, "url": doc.url}
            for doc in state["documents"][:50]
        ],
        "retrieved": state["retrieved"][:20],
        "report": state.get("report", "")[:20000],
        "gaps": state["gaps"],
        "researcher_notes": state["researcher_notes"][:5],
        "analyst_summary": state["analyst_summary"],
        "critic_notes": state["critic_notes"],
        "verifier_notes": state.get("verifier_notes", ""),
        "uncertainty_score": state.get("uncertainty_score", None),
        "confidence_score": state.get("confidence_score", None),
        "evaluation": state.get("evaluation", {}),
        "transparency": state.get("transparency", {}),
    }


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    provider: str
    credibility: float = 0.0


@dataclass
class Document:
    title: str
    url: str
    content: str


class ResearchState(TypedDict):
    session_id: str
    query: str
    depth: str
    iteration: int
    max_results: int
    max_docs: int
    top_k: int
    search_queries: List[str]
    sources: List[SearchResult]
    documents: List[Document]
    chunks: List[Dict[str, Any]]
    retrieved: List[Dict[str, Any]]
    report: str
    gaps: List[str]
    researcher_notes: List[str]
    analyst_summary: str
    critic_notes: str
    verifier_notes: str
    uncertainty_score: float
    confidence_score: float
    evaluation: Dict[str, Any]
    transparency: Dict[str, Any]


def strip_html(raw_html: str) -> str:
    cleaned = re.sub(r"<script.*?>.*?</script>", " ", raw_html, flags=re.DOTALL)
    cleaned = re.sub(r"<style.*?>.*?</style>", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_text(text: str) -> str:
    return text.replace("\u0000", "").strip()


@tool
def run_python(code: str) -> str:
    """Execute small Python snippets (no imports). Returns stdout or the last expression result."""
    if any(token in code for token in ["import", "__", "open(", "exec(", "eval(", "os.", "sys."]):
        return "Rejected: unsafe code."
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Attribute, ast.Call)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id not in {"print", "range", "len", "sum", "min", "max", "round"}:
                    return f"Rejected: function {node.func.id} not allowed."
            elif isinstance(node, ast.Attribute):
                return "Rejected: attribute access not allowed."
    safe_globals = {"__builtins__": {"print": print, "range": range, "len": len, "sum": sum, "min": min, "max": max, "round": round}}
    safe_locals: Dict[str, Any] = {}
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(tree, filename="<exec>", mode="exec"), safe_globals, safe_locals)
    output = stdout.getvalue().strip()
    return output or "OK"


async def describe_image(url: str) -> str:
    system = "You are a visual analyst. Describe the image content succinctly."
    user = [
        {"type": "text", "text": "Describe the image for research use."},
        {"type": "image_url", "image_url": {"url": url}},
    ]
    response = await vision_llm.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return response.content


tools = [get_current_utc_time, run_python]
llm_with_tools = llm.bind_tools(tools)

compiled_graph = build_graph()
compiled_research_graph = build_research_graph()


def init_db() -> None:
    conn = psycopg.connect(
        dbname=os.getenv("POSTGRES_DB", "postgres"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS research_runs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    query TEXT,
                    depth TEXT,
                    status TEXT,
                    updated_at TIMESTAMPTZ,
                    data_json JSONB
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS research_memory (
                    id TEXT PRIMARY KEY,
                    query TEXT,
                    summary TEXT,
                    created_at TIMESTAMPTZ
                )
                """
            )
        conn.commit()
    finally:
        conn.close()


def save_run(
    run_id: str,
    session_id: str,
    query: str,
    depth: str,
    status: str,
    data: Dict[str, Any],
) -> None:
    conn = psycopg.connect(
        dbname=os.getenv("POSTGRES_DB", "postgres"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
    )
    try:
        safe_data = {"state": compact_state_for_storage(data.get("state", data))}
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO research_runs (id, session_id, query, depth, status, updated_at, data_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(id) DO UPDATE SET
                    session_id=excluded.session_id,
                    query=excluded.query,
                    depth=excluded.depth,
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    data_json=excluded.data_json
                """,
                (
                    run_id,
                    session_id,
                    query,
                    depth,
                    status,
                    datetime.now(timezone.utc),
                    json.dumps(safe_data, default=to_jsonable),
                ),
            )
        conn.commit()
    finally:
        conn.close()
    # Attempt to init persistent memory collection (non-fatal)
    try:
        ensure_memory_collection()
    except Exception:
        pass


async def search_web(query: str, max_results: int = 5) -> List[SearchResult]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.error("Missing TAVILY_API_KEY for web search")
        raise RuntimeError("Missing TAVILY_API_KEY for web search")
    payload = {
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }
    safe_payload = {**payload}
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.post(
                "https://api.tavily.com/search",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Tavily search failed. status=%s payload=%s response=%s",
                exc.response.status_code,
                safe_payload,
                exc.response.text,
            )
            raise RuntimeError(
                f"Tavily search failed ({exc.response.status_code}). "
                "Check TAVILY_API_KEY and request payload."
            ) from exc
        except httpx.HTTPError as exc:
            logger.error("Tavily search http error payload=%s error=%s", safe_payload, str(exc))
            raise RuntimeError("Tavily search request failed") from exc
    results: List[SearchResult] = []
    for item in data.get("results", []):
        results.append(
            SearchResult(
                title=item.get("title", "Untitled"),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                provider="tavily",
            )
        )
    return results


def heuristic_credibility(url: str) -> float:
    domain = urlparse(url).netloc.lower()
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 0.95
    if any(domain.endswith(tld) for tld in [".org", ".int"]):
        return 0.8
    if any(domain.endswith(tld) for tld in [".com", ".net"]):
        return 0.6
    return 0.5


async def score_sources_llm(sources: List[SearchResult]) -> Dict[str, float]:
    if not sources:
        return {}
    prompt = (
        "Score each source credibility from 0 to 1 based on publisher authority, "
        "recency, and relevance. Return JSON object mapping url -> score."
    )
    items = "\n".join([f"- {src.title} | {src.url}" for src in sources])
    response = await research_llm.ainvoke(
        [SystemMessage(content=prompt), HumanMessage(content=items)]
    )
    try:
        data = json.loads(response.content)
        if isinstance(data, dict):
            return {str(k): float(v) for k, v in data.items() if k and v is not None}
    except Exception:
        logger.warning("LLM credibility scoring failed")
    return {}


def apply_credibility_scores(
    sources: List[SearchResult], llm_scores: Dict[str, float]
) -> List[SearchResult]:
    scored: List[SearchResult] = []
    for src in sources:
        heuristic = heuristic_credibility(src.url)
        llm_score = llm_scores.get(src.url)
        if llm_score is not None:
            combined = 0.6 * heuristic + 0.4 * llm_score
        else:
            combined = heuristic
        src.credibility = round(combined, 3)
        scored.append(src)
    return sorted(scored, key=lambda s: s.credibility, reverse=True)


def provider_breakdown(sources: List[SearchResult]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for src in sources:
        counts[src.provider] = counts.get(src.provider, 0) + 1
    return counts


def compute_confidence(sources: List[SearchResult], uncertainty: float) -> float:
    if not sources:
        return 0.3
    avg_cred = sum(src.credibility for src in sources) / max(len(sources), 1)
    score = max(0.0, min(1.0, (avg_cred * 0.7) + ((1 - uncertainty) * 0.3)))
    return round(score, 3)


async def run_evaluation(state: ResearchState) -> Dict[str, Any]:
    system = (
        "You are an evaluator. Score the report quality (coverage, evidence, clarity) from 1-5. "
        "Return JSON: {\"coverage\": 0-5, \"evidence\": 0-5, \"clarity\": 0-5, \"notes\": \"...\"}"
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
    return {"coverage": 0, "evidence": 0, "clarity": 0, "notes": response.content}


def _search_duckduckgo_sync(query: str, max_results: int = 5) -> List[SearchResult]:
    try:
        from ddgs import DDGS
    except Exception as exc:
        logger.warning("ddgs not available: %s", str(exc))
        return []
    results: List[SearchResult] = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append(
                SearchResult(
                    title=item.get("title", "Untitled"),
                    url=item.get("href", ""),
                    snippet=item.get("body", ""),
                    provider="duckduckgo",
                )
            )
    return results


async def search_duckduckgo(query: str, max_results: int = 5) -> List[SearchResult]:
    # ddgs is synchronous; run in a worker thread to avoid blocking the event loop.
    return await asyncio.to_thread(_search_duckduckgo_sync, query, max_results)


async def search_wikipedia(query: str, max_results: int = 5) -> List[SearchResult]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": max_results,
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params=params,
                headers={"User-Agent": "DeepResearchAgent/1.0 (research-bot)"},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        logger.warning("Wikipedia search failed query=%s error=%s", query, str(exc))
        return []
    results: List[SearchResult] = []
    for item in data.get("query", {}).get("search", []):
        title = item.get("title", "Untitled")
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        results.append(
            SearchResult(
                title=title,
                url=url,
                snippet=item.get("snippet", ""),
                provider="wikipedia",
            )
        )
    return results


SEARCH_PROVIDERS = [
    ("tavily", search_web),
    ("duckduckgo", search_duckduckgo),
    ("wikipedia", search_wikipedia),
]


async def fetch_document(url: str) -> tuple[bytes, str]:
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        try:
            resp = await client.get(url, headers={"User-Agent": "DeepResearchAgent/1.0"})
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            return resp.content, content_type
        except httpx.HTTPError as exc:
            logger.warning("Failed to fetch url=%s error=%s", url, str(exc))
            raise


async def extract_document(result: SearchResult) -> Document:
    raw, content_type = await fetch_document(result.url)
    url_lower = result.url.lower()
    if "application/pdf" in content_type or url_lower.endswith(".pdf"):
        try:
            import fitz  # type: ignore
            doc = fitz.open(stream=raw, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            return Document(title=result.title, url=result.url, content=normalize_text(text))
        except Exception as exc:
            logger.warning("PDF parsing failed url=%s error=%s", result.url, str(exc))
    if content_type.startswith("image/") or url_lower.endswith(
        (".png", ".jpg", ".jpeg", ".webp", ".gif")
    ):
        try:
            description = await describe_image(result.url)
            return Document(
                title=result.title,
                url=result.url,
                content=normalize_text(description),
            )
        except Exception as exc:
            logger.warning("Image parsing failed url=%s error=%s", result.url, str(exc))
    decoded = raw.decode("utf-8", "ignore")
    content = strip_html(decoded)
    # Fallback to Playwright for JS-heavy pages that returned near-empty HTML
    if len(content) < 200 and os.getenv("ENABLE_PLAYWRIGHT", "").lower() == "true":
        playwright_content = await extract_document_playwright(result.url)
        if playwright_content:
            content = playwright_content
    return Document(title=result.title, url=result.url, content=normalize_text(content))


async def extract_document_playwright(url: str) -> str:
    """Fetch JS-rendered content via Playwright headless browser."""
    try:
        from playwright.async_api import async_playwright  # type: ignore
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until="networkidle", timeout=15000)
            content = await page.content()
            await browser.close()
            return strip_html(content)
    except Exception as exc:
        logger.warning("Playwright extraction failed url=%s error=%s", url, str(exc))
        return ""


def build_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    return QdrantClient(url=url)


def chunk_documents(documents: List[Document]) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks: List[Dict[str, Any]] = []
    for doc in documents:
        for idx, chunk in enumerate(splitter.split_text(doc.content)):
            if idx >= 20:
                break
            chunk = chunk[:1500]
            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "url": doc.url,
                    "title": doc.title,
                }
            )
    return chunks[:200]


async def index_and_retrieve(
    session_id: str, query: str, chunks: List[Dict[str, Any]], top_k: int = 8
) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    embeddings = build_embeddings()
    vectors = await embeddings.aembed_documents([chunk["text"] for chunk in chunks])
    client = get_qdrant_client()
    collection_name = f"research_{session_id}"

    if client.collection_exists(collection_name=collection_name):
        client.delete_collection(collection_name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=len(vectors[0]), distance=qdrant_models.Distance.COSINE
        ),
    )
    points = [
        qdrant_models.PointStruct(
            id=chunk["id"],
            vector=vector,
            payload={
                "text": chunk["text"],
                "url": chunk["url"],
                "title": chunk["title"],
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    for i in range(0, len(points), 100):
        client.upsert(
            collection_name=collection_name,
            points=points[i : i + 100],
            wait=True,
        )
    query_vector = await embeddings.aembed_query(query)
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    )
    points = response.points
    results: List[Dict[str, Any]] = []
    for hit in points:
        payload = hit.payload or {}
        results.append(
            {
                "text": payload.get("text", ""),
                "url": payload.get("url", ""),
                "title": payload.get("title", ""),
            }
        )
    return results


research_llm = ChatOpenAI(
    model=os.getenv("WORKER_MODEL", "gpt-4o-mini"),
    temperature=0.2,
)

research_llm_stream = ChatOpenAI(
    model=os.getenv("WORKER_MODEL", "gpt-4o-mini"),
    temperature=0.2,
    streaming=True,
)

orchestrator_llm = ChatOpenAI(
    model=os.getenv("ORCHESTRATOR_MODEL", "gpt-4o"),
    temperature=0.2,
)

orchestrator_llm_stream = ChatOpenAI(
    model=os.getenv("ORCHESTRATOR_MODEL", "gpt-4o"),
    temperature=0.2,
    streaming=True,
)

vision_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
)


def parse_search_queries(text: str) -> List[str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned.strip())
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [
                str(item).strip()
                for item in data
                if isinstance(item, str) and str(item).strip()
            ]
    except json.JSONDecodeError:
        pass
    lines = [line.strip("- ").strip() for line in cleaned.splitlines()]
    return [line for line in lines if len(line) >= 2]


async def plan_queries(state: ResearchState) -> Dict[str, Any]:
    system = (
        "You are a research planner. Generate 3-5 focused web search queries. "
        "Return ONLY a JSON array of strings. Do not include code fences."
    )
    gap_context = ""
    if state.get("gaps"):
        gap_context = f"Known gaps: {', '.join(state['gaps'])}"
    memory_items = await load_memory(state["query"])
    memory_context = ""
    if memory_items:
        memory_context = "\n".join([f"- {item['summary']}" for item in memory_items])
        memory_context = f"Related memory:\n{memory_context}"
    prompt = f"Research question: {state['query']}\n{gap_context}\n{memory_context}"
    response = await orchestrator_llm.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=prompt)]
    )
    queries = parse_search_queries(response.content)
    return {"search_queries": queries}


async def run_search(state: ResearchState) -> Dict[str, Any]:
    results: List[SearchResult] = []
    seen: set[str] = set()
    for query in state["search_queries"]:
        if len(query.strip()) < 2:
            logger.warning("Skipping short query=%s", query)
            continue
        logger.info("Searching web query=%s", query)
        provider_tasks = [
            provider(query, max_results=state["max_results"])
            for _, provider in SEARCH_PROVIDERS
        ]
        provider_results = await asyncio.gather(*provider_tasks, return_exceptions=True)
        for (name, _), hits in zip(SEARCH_PROVIDERS, provider_results):
            if isinstance(hits, Exception):
                logger.warning("Search provider failed provider=%s error=%s", name, str(hits))
                continue
            for hit in hits:
                if hit.url and hit.url not in seen:
                    results.append(hit)
                    seen.add(hit.url)
    return {"sources": results}


async def fetch_sources(state: ResearchState) -> Dict[str, Any]:
    documents: List[Document] = []
    for source in state["sources"][: state["max_docs"]]:
        try:
            doc = await extract_document(source)
            if doc.content:
                documents.append(doc)
        except Exception as exc:
            logger.warning("Failed to extract source url=%s error=%s", source.url, str(exc))
            continue
    return {"documents": documents}


async def index_sources(state: ResearchState) -> Dict[str, Any]:
    chunks = chunk_documents(state["documents"])
    try:
        retrieved = await index_and_retrieve(
            session_id=state["session_id"],
            query=state["query"],
            chunks=chunks,
            top_k=state["top_k"],
        )
    except Exception as exc:
        logger.error("Indexing failed session_id=%s error=%s", state["session_id"], str(exc))
        retrieved = [
            {"text": chunk["text"], "url": chunk["url"], "title": chunk["title"]}
            for chunk in chunks[:8]
        ]
    return {"chunks": chunks, "retrieved": retrieved}


async def assess_gaps(state: ResearchState) -> Dict[str, Any]:
    system = (
        "You are a research analyst. Decide if more research is needed. "
        "Return JSON: {\"continue\": true/false, \"gaps\": [\"...\"]}"
    )
    source_titles = ", ".join([src.title for src in state["sources"][:8]])
    prompt = (
        f"Question: {state['query']}\n"
        f"Current sources: {source_titles}\n"
        "If coverage is weak or missing perspectives, propose gaps."
    )
    response = await research_llm.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=prompt)]
    )
    try:
        data = json.loads(response.content)
        if isinstance(data, dict):
            return {
                "continue": bool(data.get("continue", False)),
                "gaps": [str(item) for item in data.get("gaps", []) if str(item)],
            }
    except Exception:
        logger.warning("Gap assessment parsing failed")
    return {"continue": False, "gaps": []}


def build_source_list(sources: List[SearchResult]) -> str:
    return "\n".join(
        [f"{idx + 1}. {src.title} — {src.url}" for idx, src in enumerate(sources)]
    )


MEMORY_COLLECTION = "research_memory_embeddings"


def ensure_memory_collection() -> None:
    """Create persistent Qdrant collection for semantic memory if it doesn't exist."""
    try:
        client = get_qdrant_client()
        if not client.collection_exists(collection_name=MEMORY_COLLECTION):
            client.create_collection(
                collection_name=MEMORY_COLLECTION,
                vectors_config=qdrant_models.VectorParams(
                    size=1536, distance=qdrant_models.Distance.COSINE
                ),
            )
    except Exception as exc:
        logger.warning("Memory collection init failed: %s", str(exc))


async def save_memory(query: str, summary: str) -> None:
    mem_id = uuid.uuid4().hex
    conn = psycopg.connect(
        dbname=os.getenv("POSTGRES_DB", "postgres"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO research_memory (id, query, summary, created_at)
                VALUES (%s, %s, %s, %s)
                """,
                (mem_id, query, summary, datetime.now(timezone.utc)),
            )
        conn.commit()
    finally:
        conn.close()
    # Also embed and upsert to Qdrant for semantic search
    try:
        ensure_memory_collection()
        embeddings = build_embeddings()
        vector = await embeddings.aembed_query(query)
        client = get_qdrant_client()
        client.upsert(
            collection_name=MEMORY_COLLECTION,
            points=[
                qdrant_models.PointStruct(
                    id=mem_id,
                    vector=vector,
                    payload={"query": query, "summary": summary},
                )
            ],
        )
    except Exception as exc:
        logger.warning("Memory semantic indexing failed: %s", str(exc))


def load_memory_sync(query: str, limit: int = 3) -> List[Dict[str, str]]:
    """Fallback: substring search via Postgres ILIKE."""
    conn = psycopg.connect(
        dbname=os.getenv("POSTGRES_DB", "postgres"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT query, summary
                FROM research_memory
                WHERE query ILIKE %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (f"%{query}%", limit),
            )
            rows = cur.fetchall()
        return [{"query": row[0], "summary": row[1]} for row in rows]
    finally:
        conn.close()


async def load_memory(query: str, limit: int = 3) -> List[Dict[str, str]]:
    """Semantic memory search via Qdrant embeddings with SQL fallback."""
    try:
        ensure_memory_collection()
        embeddings = build_embeddings()
        vector = await embeddings.aembed_query(query)
        client = get_qdrant_client()
        response = client.query_points(
            collection_name=MEMORY_COLLECTION,
            query=vector,
            limit=limit,
        )
        results = []
        for hit in response.points:
            payload = hit.payload or {}
            if payload.get("query") and payload.get("summary"):
                results.append({"query": payload["query"], "summary": payload["summary"]})
        if results:
            return results
    except Exception as exc:
        logger.warning("Semantic memory search failed, falling back to SQL: %s", str(exc))
    return load_memory_sync(query, limit)


def build_doc_snippets(documents: List[Document], max_chars: int = 1200) -> List[str]:
    snippets: List[str] = []
    for doc in documents:
        text = doc.content[:max_chars]
        snippets.append(f"Title: {doc.title}\nURL: {doc.url}\nContent: {text}")
    return snippets


async def run_researchers(state: ResearchState, workers: int = 3) -> List[str]:
    documents = state["documents"]
    if not documents:
        return []
    buckets: List[List[Document]] = [[] for _ in range(workers)]
    for idx, doc in enumerate(documents):
        buckets[idx % workers].append(doc)

    async def summarize(bucket: List[Document], index: int) -> str:
        doc_snippets = build_doc_snippets(bucket)
        system = (
            "You are a research assistant. Summarize the provided documents with key facts "
            "and note important evidence. Cite sources by URL inline."
        )
        user = (
            f"Research question: {state['query']}\n"
            f"Documents:\n\n" + "\n\n".join(doc_snippets)
        )
        response = await research_llm.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return f"Researcher {index + 1} notes:\n{response.content}"

    tasks = [summarize(bucket, idx) for idx, bucket in enumerate(buckets) if bucket]
    if not tasks:
        return []
    return await asyncio.gather(*tasks)


async def run_analyst(state: ResearchState) -> str:
    system = (
        "You are an analyst. Combine researcher notes into key findings, contradictions, "
        "and open questions. Keep it structured."
    )
    notes = "\n\n".join(state["researcher_notes"])
    user = f"Research question: {state['query']}\n\nResearcher notes:\n{notes}"
    analyst_llm = orchestrator_llm.bind_tools([run_python])
    response = await analyst_llm.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    if response.tool_calls:
        tool_outputs = []
        for call in response.tool_calls:
            result = run_python.invoke(call["args"])
            tool_outputs.append(ToolMessage(content=result, tool_call_id=call["id"]))
        response = await analyst_llm.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=user), response, *tool_outputs]
        )
    return response.content


async def run_critic(state: ResearchState) -> str:
    system = (
        "You are a critical reviewer. Identify weak evidence, missing citations, "
        "and potential biases. Provide actionable fixes."
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


async def run_verifier(state: ResearchState) -> tuple[str, float]:
    system = (
        "You are a verifier. Assess the evidence quality and estimate an uncertainty score "
        "from 0 (high confidence) to 1 (low confidence). Return JSON: "
        "{\"notes\": \"...\", \"uncertainty\": 0.0}"
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


def build_writer_prompt(state: ResearchState) -> List[AnyMessage]:
    source_list = build_source_list(state["sources"])
    evidence = "\n\n".join(
        [
            f"Source: {item.get('title', '')} ({item.get('url', '')})\n"
            f"Excerpt: {item.get('text', '')}"
            for item in state["retrieved"]
        ]
    )
    length_hint = {
        "shallow": "Keep it concise (400-700 words).",
        "standard": "Provide a balanced report (700-1200 words).",
        "deep": "Provide a detailed report (1200-1800 words).",
    }.get(state.get("depth", "standard"), "Provide a balanced report.")
    system = (
        "You are a research writer. Produce a structured report with citations. "
        "Use [1], [2] etc matching the source list."
    )
    user = (
        f"Research question: {state['query']}\n\n"
        f"Analyst summary:\n{state['analyst_summary']}\n\n"
        f"Critic notes:\n{state['critic_notes']}\n\n"
        f"Verifier notes:\n{state.get('verifier_notes', '')}\n\n"
        f"Uncertainty score (0=high confidence,1=low confidence): {state.get('uncertainty_score', 0.5)}\n\n"
        f"Sources (for grounding only, do not output a Sources section):\n{source_list}\n\n"
        f"Evidence:\n{evidence}\n\n"
        f"Length guidance: {length_hint}\n\n"
        "Return sections:\n"
        "1. Executive Summary\n2. Key Findings\n3. Supporting Evidence\n"
        "4. Contradictions\n"
        "Do NOT include a Sources section in the final answer."
    )
    return [SystemMessage(content=system), HumanMessage(content=user)]


def build_synthesis_prompt(state: ResearchState) -> List[AnyMessage]:
    return build_writer_prompt(state)


async def synthesize(state: ResearchState) -> Dict[str, Any]:
    response = await orchestrator_llm.ainvoke(build_synthesis_prompt(state))
    return {"report": response.content}


async def clarify_query(query: str) -> List[str]:
    """Return 1–2 clarifying questions if the query is ambiguous, else empty list."""
    system = (
        "You are a research assistant. If the query is ambiguous and would meaningfully benefit "
        "from 1-2 targeted clarifying questions, return them as a JSON array of strings. "
        "If the query is clear and self-contained, return an empty JSON array []. "
        "Do not include code fences. Return ONLY a JSON array."
    )
    response = await orchestrator_llm.ainvoke(
        [SystemMessage(content=system), HumanMessage(content=f"Query: {query}")]
    )
    try:
        data = json.loads(response.content.strip())
        if isinstance(data, list):
            return [str(q).strip() for q in data if str(q).strip()]
    except Exception:
        logger.warning("clarify_query parsing failed content=%s", response.content[:200])
    return []


async def verify_citations(report: str, sources: List[SearchResult]) -> str:
    """Post-process report to fix mismatched [N] citation numbers."""
    if not report or not sources:
        return report
    system = (
        "You are a citation verifier. Given a research report with inline [N] citations and a "
        "numbered source list, verify that each [N] reference matches the correct source. "
        "Correct any wrong citation numbers. Return the corrected report only, without explanation."
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


init_db()


@app.on_event("startup")
async def log_graph_diagrams() -> None:
    logger.info("=== Chat graph (Mermaid) ===\n%s", compiled_graph.get_graph().draw_mermaid())
    logger.info("=== Research pipeline graph (Mermaid) ===\n%s", compiled_research_graph.get_graph().draw_mermaid())


def depth_config(depth: str) -> Dict[str, int]:
    presets = {
        "shallow": {"max_results": 3, "max_docs": 6, "top_k": 6},
        "standard": {"max_results": 5, "max_docs": 10, "top_k": 8},
        "deep": {"max_results": 7, "max_docs": 14, "top_k": 10},
    }
    return presets.get(depth, presets["standard"])


async def run_research_loop(
    state: ResearchState,
    emit_status: Callable[[str, str, Dict[str, Any] | None], Awaitable[None]] | None = None,
    max_iters: int = 3,
    approved_queries: List[str] | None = None,
) -> ResearchState:
    for iteration in range(1, max_iters + 1):
        state["iteration"] = iteration
        if emit_status:
            await emit_status("plan", f"Planning iteration {iteration}", None)

        # Use pre-approved queries on first iteration if provided
        if iteration == 1 and approved_queries:
            state["search_queries"] = approved_queries
        else:
            try:
                plan_result = await plan_queries(state)
                state.update(plan_result)
            except Exception as exc:
                logger.error("plan_queries failed iteration=%d error=%s", iteration, str(exc))
                if emit_status:
                    await emit_status("warning", f"Planning failed: {exc}", {"stage": "plan"})
                break

        if emit_status:
            await emit_status(
                "queries", "Generated search queries", {"queries": state["search_queries"]}
            )
        # Emit plan preview so frontend can display the planned queries
        if emit_status:
            await emit_status(
                "plan_preview",
                "Research plan ready",
                {"queries": state["search_queries"], "iteration": iteration},
            )

        if emit_status:
            await emit_status("search", f"Searching (iteration {iteration})", None)
        try:
            search_result = await run_search(state)
            state.update(search_result)
        except Exception as exc:
            logger.error("run_search failed iteration=%d error=%s", iteration, str(exc))
            if emit_status:
                await emit_status("warning", f"Search failed: {exc}", {"stage": "search"})
            # Continue with empty sources rather than aborting
            state["sources"] = state.get("sources", [])

        heuristic_ranked = apply_credibility_scores(state["sources"], {})
        llm_targets = heuristic_ranked[: min(8, len(heuristic_ranked))]
        try:
            llm_scores = await score_sources_llm(llm_targets)
        except Exception as exc:
            logger.warning("score_sources_llm failed: %s", str(exc))
            llm_scores = {}
        scored_sources = apply_credibility_scores(state["sources"], llm_scores)
        state["sources"] = scored_sources

        if emit_status:
            await emit_status(
                "sources",
                f"Found {len(state['sources'])} sources",
                {
                    "top_sources": [
                        {
                            "title": src.title,
                            "url": src.url,
                            "provider": src.provider,
                            "credibility": src.credibility,
                        }
                        for src in state["sources"][:5]
                    ],
                    "providers": provider_breakdown(state["sources"]),
                },
            )

        if emit_status:
            await emit_status("extract", "Extracting and indexing sources", None)
        try:
            fetch_result = await fetch_sources(state)
            state.update(fetch_result)
        except Exception as exc:
            logger.error("fetch_sources failed: %s", str(exc))
            if emit_status:
                await emit_status("warning", f"Document extraction failed: {exc}", {"stage": "extract"})
        if emit_status:
            await emit_status(
                "documents",
                f"Extracted {len(state['documents'])} documents",
                {"documents": len(state["documents"])},
            )

        try:
            index_result = await index_sources(state)
            state.update(index_result)
        except Exception as exc:
            logger.error("index_sources failed: %s", str(exc))
            if emit_status:
                await emit_status("warning", f"Indexing failed, using raw chunks: {exc}", {"stage": "retrieval"})
        if emit_status:
            await emit_status(
                "retrieval",
                f"Retrieved {len(state['retrieved'])} relevant chunks",
                {"chunks": len(state["retrieved"])},
            )

        try:
            gap_result = await assess_gaps(state)
            state["gaps"] = gap_result.get("gaps", [])
        except Exception as exc:
            logger.warning("assess_gaps failed: %s", str(exc))
            if emit_status:
                await emit_status("warning", f"Gap assessment failed: {exc}", {"stage": "gaps"})
            gap_result = {"continue": False, "gaps": []}
            state["gaps"] = []
        if emit_status:
            await emit_status(
                "gaps",
                "Gap assessment complete",
                {"continue": gap_result.get("continue"), "gaps": state["gaps"]},
            )

        if not gap_result.get("continue") or iteration >= max_iters:
            break

    return state


@app.get("/")
async def health() -> Dict[str, str]:
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
    body {{ font-family: system-ui, sans-serif; background: #f9f9fb; margin: 0; padding: 2rem; }}
    h1 {{ font-size: 1.4rem; margin-bottom: 2rem; color: #333; }}
    h2 {{ font-size: 1rem; font-weight: 600; color: #555; margin: 2rem 0 0.75rem; }}
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
    logger.info("chat_stream request")
    session_id = body.get("session_id") or "default"
    messages_payload: List[ChatMessage] = body.get("messages", [])
    message = body.get("message")

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
                {"messages": history},
                version="v2",
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
                            "input": to_jsonable(event.get("data", {}).get("input")),
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
                            "output": to_jsonable(event.get("data", {}).get("output")),
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
                    payload = json.dumps({"delta": delta})
                    yield f"data: {payload}\n\n"
            if not await request.is_disconnected():
                yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.exception("chat_stream failed session_id=%s", session_id)
            payload = json.dumps({"error": str(exc)})
            yield f"data: {payload}\n\n"
        else:
            if assistant_chunks and not disconnected:
                memory_store[session_id] = [
                    *history,
                    AIMessage(content="".join(assistant_chunks)),
                ]

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/research")
async def run_research(body: Dict[str, Any]) -> Dict[str, Any]:
    query = body.get("query")
    session_id = body.get("session_id") or uuid.uuid4().hex
    depth = body.get("depth") or "standard"
    if not query:
        logger.warning("run_research missing query")
        return {"error": "Missing query"}
    logger.info("run_research session_id=%s", session_id)
    config = depth_config(depth)
    run_id = uuid.uuid4().hex
    state: ResearchState = {
        "session_id": session_id,
        "query": query,
        "depth": depth,
        "iteration": 0,
        "max_results": config["max_results"],
        "max_docs": config["max_docs"],
        "top_k": config["top_k"],
        "search_queries": [],
        "sources": [],
        "documents": [],
        "chunks": [],
        "retrieved": [],
        "report": "",
        "gaps": [],
        "researcher_notes": [],
        "analyst_summary": "",
        "critic_notes": "",
        "verifier_notes": "",
        "uncertainty_score": 0.5,
        "confidence_score": 0.0,
        "evaluation": {},
        "transparency": {},
    }
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
    result["confidence_score"] = compute_confidence(result["sources"], result["uncertainty_score"])
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
async def run_research_stream(body: Dict[str, Any], request: Request) -> StreamingResponse:
    query = body.get("query")
    session_id = body.get("session_id") or uuid.uuid4().hex
    depth = body.get("depth") or "standard"
    raw_iters = body.get("max_iterations", 3)
    max_iterations = max(1, min(5, int(raw_iters)))
    approved_queries: List[str] | None = body.get("approved_queries") or None
    if not query:
        logger.warning("run_research_stream missing query")
        return StreamingResponse(
            iter([f"data: {json.dumps({'error': 'Missing query'})}\n\n"]),
            media_type="text/event-stream",
        )
    logger.info("run_research_stream session_id=%s max_iterations=%d", session_id, max_iterations)

    async def event_stream() -> AsyncGenerator[str, None]:
        run_id = uuid.uuid4().hex
        config = depth_config(depth)
        queue: asyncio.Queue[str] = asyncio.Queue()

        async def emit_status(stage: str, message: str, data: Dict[str, Any] | None = None) -> None:
            payload = {"type": "status", "stage": stage, "message": message, "data": data}
            await queue.put(f"data: {json.dumps(payload)}\n\n")

        try:
            state: ResearchState = {
                "session_id": session_id,
                "query": query,
                "depth": depth,
                "iteration": 0,
                "max_results": config["max_results"],
                "max_docs": config["max_docs"],
                "top_k": config["top_k"],
                "search_queries": [],
                "sources": [],
                "documents": [],
                "chunks": [],
                "retrieved": [],
                "report": "",
                "gaps": [],
                "researcher_notes": [],
                "analyst_summary": "",
                "critic_notes": "",
                "verifier_notes": "",
                "uncertainty_score": 0.5,
                "confidence_score": 0.0,
                "evaluation": {},
                "transparency": {},
            }

            save_run(run_id, session_id, query, depth, "running", {"state": state})

            loop_task = asyncio.create_task(
                run_research_loop(
                    state,
                    emit_status=emit_status,
                    max_iters=max_iterations,
                    approved_queries=approved_queries,
                )
            )

            while True:
                if loop_task.done() and queue.empty():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=0.5)
                    yield event
                except asyncio.TimeoutError:
                    if loop_task.done():
                        break
                    continue

            # Drain any remaining queued events
            while not queue.empty():
                yield queue.get_nowait()

            state = await loop_task
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
            # Emit verify result so frontend can display uncertainty + notes
            await emit_status(
                "verify",
                "Evidence verified",
                {
                    "uncertainty": state["uncertainty_score"],
                    "verifier_notes": state["verifier_notes"],
                },
            )
            save_run(run_id, session_id, query, depth, "synthesizing", {"state": state})
            prompt = build_writer_prompt(state)
            await emit_status("writer", "Writing report", None)
            report_parts: List[str] = []

            async for chunk in orchestrator_llm_stream.astream(prompt):
                if await request.is_disconnected():
                    break
                delta = getattr(chunk, "content", None)
                if not delta:
                    continue
                report_parts.append(delta)
                payload = json.dumps({"delta": delta})
                yield f"data: {payload}\n\n"

            if not await request.is_disconnected():
                raw_report = "".join(report_parts)
                # Citation verification pass
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
                save_run(run_id, session_id, query, depth, "completed", {"state": state})
                await save_memory(query, state.get("report", "")[:2000])
                await emit_status("done", "Report complete", None)
                yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.exception("run_research_stream failed session_id=%s", session_id)
            payload = json.dumps({"error": str(exc)})
            yield f"data: {payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/research/history")
async def get_research_history(session_id: str | None = None) -> Dict[str, Any]:
    """List past research runs, optionally filtered by session_id."""
    conn = psycopg.connect(
        dbname=os.getenv("POSTGRES_DB", "postgres"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
    )
    try:
        with conn.cursor() as cur:
            if session_id:
                cur.execute(
                    """
                    SELECT id, session_id, query, depth, status, updated_at
                    FROM research_runs
                    WHERE session_id = %s
                    ORDER BY updated_at DESC
                    LIMIT 50
                    """,
                    (session_id,),
                )
            else:
                cur.execute(
                    """
                    SELECT id, session_id, query, depth, status, updated_at
                    FROM research_runs
                    ORDER BY updated_at DESC
                    LIMIT 50
                    """
                )
            rows = cur.fetchall()
        runs = [
            {
                "id": row[0],
                "session_id": row[1],
                "query": row[2],
                "depth": row[3],
                "status": row[4],
                "updated_at": row[5].isoformat() if row[5] else None,
            }
            for row in rows
        ]
        return {"runs": runs}
    finally:
        conn.close()


@app.get("/research/{run_id}")
async def get_research_run(run_id: str) -> Dict[str, Any]:
    """Retrieve a specific research run by ID including the report."""
    conn = psycopg.connect(
        dbname=os.getenv("POSTGRES_DB", "postgres"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, session_id, query, depth, status, updated_at, data_json
                FROM research_runs
                WHERE id = %s
                """,
                (run_id,),
            )
            row = cur.fetchone()
        if not row:
            return {"error": "Run not found"}
        state = row[6] if row[6] else {}
        if isinstance(state, dict) and "state" in state:
            state = state["state"]
        return {
            "id": row[0],
            "session_id": row[1],
            "query": row[2],
            "depth": row[3],
            "status": row[4],
            "updated_at": row[5].isoformat() if row[5] else None,
            "report": state.get("report", ""),
            "sources": state.get("sources", []),
            "evaluation": state.get("evaluation", {}),
            "uncertainty_score": state.get("uncertainty_score"),
            "confidence_score": state.get("confidence_score"),
        }
    finally:
        conn.close()


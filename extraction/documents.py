"""Document fetching and content extraction.

Handles HTML (default), PDF (via PyMuPDF), images (via vision LLM), and
optionally JS-rendered pages (via Playwright).  All extracted text is
normalised through ``normalize_text`` before being returned.
"""

import base64
import logging
import os
import re
from typing import Tuple

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_vision_llm
from models import Document, SearchResult

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif")


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------


def strip_html(raw_html: str) -> str:
    """Remove script/style blocks, HTML tags, and collapse whitespace.

    Args:
        raw_html: Raw HTML string.

    Returns:
        Plain text with excess whitespace collapsed.
    """
    cleaned = re.sub(r"<script.*?>.*?</script>", " ", raw_html, flags=re.DOTALL)
    cleaned = re.sub(r"<style.*?>.*?</style>", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_text(text: str) -> str:
    """Strip null bytes and surrounding whitespace from text.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text safe for storage and embedding.
    """
    return text.replace("\u0000", "").strip()


# ---------------------------------------------------------------------------
# Vision helper
# ---------------------------------------------------------------------------


async def describe_image(raw: bytes, mime_type: str) -> str:
    """Return a textual description of image bytes via the Gemini vision LLM.

    The image is passed as a base64-encoded data-URI so the call works
    regardless of whether the source URL is publicly accessible.

    Args:
        raw: Raw image bytes already fetched by ``fetch_document``.
        mime_type: The ``Content-Type`` value from the HTTP response
            (e.g. ``"image/jpeg"``).  Any parameters (``; charset=...``)
            are stripped automatically.

    Returns:
        Textual description of the image content.
    """
    clean_mime = mime_type.split(";")[0].strip() or "image/jpeg"
    image_b64 = base64.standard_b64encode(raw).decode("utf-8")
    data_uri = f"data:{clean_mime};base64,{image_b64}"

    system = (
        "You are a visual analyst. Describe the image content in detail, "
        "highlighting any text, charts, diagrams, or key visual elements "
        "that are relevant for research."
    )
    user_content = [
        {"type": "text", "text": "Describe this image for research use."},
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]
    response = await get_vision_llm().ainvoke(
        [SystemMessage(content=system), HumanMessage(content=user_content)]
    )
    return response.content


# ---------------------------------------------------------------------------
# HTTP fetch
# ---------------------------------------------------------------------------


async def fetch_document(url: str) -> Tuple[bytes, str]:
    """Fetch raw bytes and the ``Content-Type`` header for a URL.

    Args:
        url: The URL to fetch.

    Returns:
        Tuple of ``(raw_bytes, content_type_string)``.

    Raises:
        httpx.HTTPError: On network or HTTP-level failures.
    """
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        try:
            resp = await client.get(
                url, headers={"User-Agent": "DeepResearchAgent/1.0"}
            )
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            return resp.content, content_type
        except httpx.HTTPError as exc:
            logger.warning("Failed to fetch url=%s error=%s", url, str(exc))
            raise


# ---------------------------------------------------------------------------
# Playwright fallback
# ---------------------------------------------------------------------------


async def extract_document_playwright(url: str) -> str:
    """Fetch JS-rendered page content via a headless Playwright browser.

    Only attempted when the ``ENABLE_PLAYWRIGHT`` env var is ``"true"`` and
    the HTML extraction produces fewer than 200 characters.

    Args:
        url: The URL to render.

    Returns:
        Plain text extracted from the rendered page, or an empty string on
        any failure.
    """
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


# ---------------------------------------------------------------------------
# Main extraction entry-point
# ---------------------------------------------------------------------------


async def extract_document(result: SearchResult) -> Document:
    """Convert a ``SearchResult`` into a ``Document`` by fetching its content.

    Dispatch order:
    1. PDF  → PyMuPDF text extraction.
    2. Image → vision LLM description.
    3. HTML → ``strip_html``; Playwright fallback if content < 200 chars.

    Args:
        result: The search result to fetch and parse.

    Returns:
        A ``Document`` with normalised content.
    """
    raw, content_type = await fetch_document(result.url)
    url_lower = result.url.lower()

    # --- PDF ---
    if "application/pdf" in content_type or url_lower.endswith(".pdf"):
        try:
            import fitz  # type: ignore  # PyMuPDF

            doc = fitz.open(stream=raw, filetype="pdf")
            text = "\n".join(page.get_text() for page in doc)
            return Document(
                title=result.title,
                url=result.url,
                content=normalize_text(text),
            )
        except Exception as exc:
            logger.warning(
                "PDF parsing failed url=%s error=%s", result.url, str(exc)
            )

    # --- Image ---
    if content_type.startswith("image/") or url_lower.endswith(_IMAGE_EXTENSIONS):
        try:
            # Pass already-fetched bytes — no second HTTP round-trip.
            description = await describe_image(raw, content_type)
            return Document(
                title=result.title,
                url=result.url,
                content=normalize_text(description),
            )
        except Exception as exc:
            logger.warning(
                "Image parsing failed url=%s error=%s", result.url, str(exc)
            )

    # --- HTML (default) ---
    decoded = raw.decode("utf-8", "ignore")
    content = strip_html(decoded)

    # Playwright fallback for JS-heavy pages with near-empty HTML output.
    if len(content) < 200 and os.getenv("ENABLE_PLAYWRIGHT", "").lower() == "true":
        playwright_content = await extract_document_playwright(result.url)
        if playwright_content:
            content = playwright_content

    return Document(
        title=result.title,
        url=result.url,
        content=normalize_text(content),
    )

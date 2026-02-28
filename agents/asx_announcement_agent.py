# agents/asx_announcement_agent.py
"""
ASX Market Announcement Agent

Fetches announcements directly from the ASX public API for low-latency reading.
No manual PDF upload required.

Two operation modes:
  1. Quick Interpret — fetch PDF → extract text → send to LLM immediately (no vector DB)
  2. Ingest to Knowledge Base — embed + store in LanceDB for future RAG queries

ASX public API (no authentication required):
  GET https://www.asx.com.au/asx/1/company/{TICKER}/announcements
      ?count={N}&market_sensitive=false

LLM routing via RouterChatClient — respects LLM_MODE / LLM_PRIMARY_PROVIDER env vars:
  paid    → OpenAI (gpt-5.1) or Bedrock (Claude)
  local   → Ollama
  hybrid  → Ollama with paid fallback
"""

from __future__ import annotations

import io
import os
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ASX API constants
# ---------------------------------------------------------------------------

_ASX_API_BASE = "https://www.asx.com.au/asx/1/company"

# ASX blocks plain bot requests; these headers mimic a browser.
_ASX_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.asx.com.au/",
    "Accept": "application/json, text/plain, */*",
}

# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

_SYSTEM_SUMMARY = (
    "You are a senior financial analyst specialising in ASX-listed companies. "
    "Analyse the provided ASX market announcement and return a structured report covering:\n"
    "1. **Executive Summary** — 2-3 sentence overview of what was announced\n"
    "2. **Key Financial Figures** — revenue, EBITDA, NPAT, EPS, DPS, guidance (if present)\n"
    "3. **Material Events** — acquisitions, capital raises, leadership changes, strategic updates\n"
    "4. **Market Sentiment** — overall tone: bullish / neutral / bearish, and why\n"
    "5. **Key Risks / Concerns** — any caveats, uncertainties, or negative signals\n\n"
    "Be concise, cite specific numbers and dates from the document. "
    "If a section is not covered in the announcement, say 'Not mentioned'."
)

_SYSTEM_QA = (
    "You are a financial analyst assistant specialising in ASX market announcements. "
    "Answer the question using ONLY information from the provided announcement text. "
    "Cite specific figures, dates, and sections. "
    "If the answer is not in the document, say so clearly."
)


# ---------------------------------------------------------------------------
# Fetch announcement list from ASX
# ---------------------------------------------------------------------------

def fetch_asx_announcements(
    ticker: str,
    count: int = 20,
    market_sensitive_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch latest announcements for an ASX-listed ticker.

    Args:
        ticker:               ASX ticker symbol (e.g. "BHP", "CBA")
        count:                Max number of announcements to return (1–100)
        market_sensitive_only: If True, only return market-sensitive announcements

    Returns:
        List of announcement metadata dicts with keys:
            id, ticker, document_date, headline, url,
            market_sensitive, number_of_pages, size_kb, doc_type

    Raises:
        ValueError: If ticker not found or API response format unexpected.
        requests.HTTPError: On HTTP errors.
    """
    ticker = ticker.upper().strip()
    url = f"{_ASX_API_BASE}/{ticker}/announcements"
    params: Dict[str, Any] = {
        "count": min(max(count, 1), 100),
        "market_sensitive": "true" if market_sensitive_only else "false",
    }

    try:
        resp = requests.get(url, params=params, headers=_ASX_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise ValueError(f"Ticker '{ticker}' not found on ASX.") from exc
        raise

    raw = resp.json()

    # ASX API wraps the list in {"data": [...]}
    announcements_raw: List[dict]
    if isinstance(raw, dict):
        announcements_raw = raw.get("data", [])
    elif isinstance(raw, list):
        announcements_raw = raw
    else:
        raise ValueError(
            f"Unexpected ASX API response format for {ticker}: {type(raw)}"
        )

    return [
        {
            "id":               a.get("id", ""),
            "ticker":           ticker,
            "document_date":    a.get("document_date", ""),
            "headline":         a.get("headline", "—"),
            "url":              a.get("url", ""),
            "market_sensitive": bool(a.get("market_sensitive", False)),
            "number_of_pages":  int(a.get("number_of_pages") or 0),
            "size_kb":          round(int(a.get("size") or 0) / 1024, 1),
            "doc_type":         a.get("doc_type", ""),
        }
        for a in announcements_raw
    ]


# ---------------------------------------------------------------------------
# PDF download + text extraction  (fully in-memory, no temp files)
# ---------------------------------------------------------------------------

def download_pdf_bytes(pdf_url: str, timeout: int = 30) -> bytes:
    """Download a PDF from an ASX URL and return its raw bytes."""
    resp = requests.get(pdf_url, headers=_ASX_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes using pdfplumber (in-memory, no temp file).

    Returns the extracted text or an empty string for image-only PDFs.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber not installed. Run: pip install pdfplumber"
        )

    text_chunks: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_chunks.append(page_text)

    return "\n\n".join(text_chunks)


# ---------------------------------------------------------------------------
# LanceDB ingestion
# ---------------------------------------------------------------------------

def ingest_announcement_to_lancedb(
    announcement: Dict[str, Any],
    pdf_text: str,
    db_uri: str = "vector_store",
) -> Dict[str, Any]:
    """
    Embed an ASX announcement and store it in LanceDB.

    Collection: 'asx_announcements' (separate from the generic 'pdfs' collection
    so ASX content can be searched/filtered independently).

    Returns:
        dict with keys: status ("ingested" | "skipped"), filename, collection/reason
    """
    import lancedb
    from embeddings.embed import embed_texts  # HuggingFace all-MiniLM-L6-v2

    ticker    = announcement.get("ticker", "UNK")
    doc_date  = (announcement.get("document_date") or "")[:10]   # YYYY-MM-DD
    headline  = announcement.get("headline", "Unknown")
    ann_id    = announcement.get("id", "")
    doc_type  = announcement.get("doc_type", "")
    pdf_url   = announcement.get("url", "")

    # Unique filename for deduplication
    filename = f"ASX_{ticker}_{doc_date}_{ann_id}.pdf"

    # Prefix text with announcement metadata so the LLM has context in every chunk
    full_text = (
        f"ASX Announcement — {ticker} | {doc_date} | {doc_type}\n"
        f"Headline: {headline}\n\n"
        f"{pdf_text or '(No readable text — may be an image-only PDF)'}"
    )

    db = lancedb.connect(db_uri)
    collection = "asx_announcements"

    # De-duplicate: skip if already ingested
    if collection in db.table_names():
        tbl = db.open_table(collection)
        existing = tbl.to_pandas()
        if "filename" in existing.columns and filename in existing["filename"].values:
            return {
                "status":   "skipped",
                "filename": filename,
                "reason":   "already ingested",
            }

        vector = embed_texts([full_text])[0]
        tbl.add([{
            "vector":     vector,
            "text":       full_text,
            "source_uri": pdf_url,
            "filename":   filename,
        }])
    else:
        vector = embed_texts([full_text])[0]
        db.create_table(collection, data=[{
            "vector":     vector,
            "text":       full_text,
            "source_uri": pdf_url,
            "filename":   filename,
        }])

    return {
        "status":     "ingested",
        "filename":   filename,
        "collection": collection,
    }


# ---------------------------------------------------------------------------
# Direct LLM interpretation  (low-latency — no vector DB round-trip)
#
# LLM chain (ASX-specific, isolated from the global RouterChatClient config):
#   Step 1 — Gemini flash-lite  : fast primary extraction from the document text
#   Step 2 — GPT-5.1 (reviewer) : quality gate; rewrites if figures/sentiment
#                                  are wrong or key sections are missing
#
# Why this order for document interpretation?
#   • The document text is injected verbatim → primary model needs to READ,
#     not KNOW — Gemini flash-lite handles extraction well at ~10x lower cost.
#   • GPT-5.1 as reviewer is a *stronger* quality gate than Gemini-lite
#     reviewing GPT (the global default), catching missed figures or
#     incorrect sentiment before the answer reaches the user.
#   • Aligns with the low-latency goal: Gemini responds in ~1-2s;
#     GPT review only fires when quality is genuinely substandard.
# ---------------------------------------------------------------------------

_REVIEWER_SYSTEM = (
    "You are a senior financial analyst reviewing a junior analyst's interpretation "
    "of an ASX market announcement. The full announcement text is provided below.\n\n"
    "Your task:\n"
    "1. Check the draft for accuracy — are all figures, dates, and facts correct "
    "   relative to the source document?\n"
    "2. Check completeness — are any material financial figures or events missed?\n"
    "3. Check sentiment — is the bullish/neutral/bearish assessment justified?\n\n"
    "If the draft is accurate and complete, reply EXACTLY:\n"
    "  APPROVED\n\n"
    "If you find errors or important omissions, reply with the corrected and "
    "complete analysis only (no preamble, no 'APPROVED' marker). "
    "Keep the same structured format as the draft."
)


def interpret_announcement(
    announcement: Dict[str, Any],
    pdf_text: str,
    question: Optional[str] = None,
    max_doc_chars: int = 8000,
) -> str:
    """
    Interpret an ASX announcement via a two-step LLM chain:
      1. Gemini flash-lite  — fast extraction / drafting  (~1-2s)
      2. GPT-5.1 (reviewer) — quality gate; rewrites only if needed

    This order is intentional: document text is provided verbatim so the
    primary model only needs to extract, not recall. GPT as reviewer provides
    a stronger quality gate than the inverse arrangement.

    Args:
        announcement:  Metadata dict from fetch_asx_announcements()
        pdf_text:      Extracted PDF text
        question:      Optional user question.
                       - None  → auto-generate structured summary
                       - str   → answer the specific question about the document
        max_doc_chars: Truncation limit to stay within LLM context window.

    Returns:
        The final interpretation (Gemini draft or GPT-reviewed rewrite).
    """
    from agents.llm_client import GoogleChatClient, OpenAIChatClient

    ticker   = announcement.get("ticker", "")
    doc_date = (announcement.get("document_date") or "")[:10]
    headline = announcement.get("headline", "")
    doc_type = announcement.get("doc_type", "")

    # Truncate to stay within context window
    text_body = pdf_text[:max_doc_chars]
    if len(pdf_text) > max_doc_chars:
        text_body += "\n\n[... document truncated to fit context window ...]"

    if not text_body.strip():
        text_body = "(No readable text extracted — this may be an image-only PDF)"

    doc_header = (
        f"ASX Announcement\n"
        f"Ticker: {ticker}  |  Date: {doc_date}  |  Type: {doc_type}\n"
        f"Headline: {headline}\n\n"
        f"{'─' * 60}\n"
        f"{text_body}\n"
        f"{'─' * 60}"
    )

    if question:
        system_msg = _SYSTEM_QA
        user_msg   = f"{doc_header}\n\nQuestion: {question}"
    else:
        system_msg = _SYSTEM_SUMMARY
        user_msg   = doc_header

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]

    # ── Step 1: Gemini flash-lite (fast primary) ──────────────────────────
    try:
        gemini  = GoogleChatClient()
        draft   = gemini.chat(messages, temperature=0.1).get("content", "")
    except Exception as exc:
        logger.warning("Gemini primary failed, falling back to GPT directly: %s", exc)
        # If Gemini is unavailable, fall back to GPT alone
        gpt    = OpenAIChatClient()
        return gpt.chat(messages, temperature=0.1).get("content", "No response returned.")

    if not draft.strip():
        draft = "(Gemini returned an empty response)"

    # ── Step 2: GPT-5.1 reviewer (quality gate) ───────────────────────────
    try:
        reviewer_messages = [
            {"role": "system", "content": _REVIEWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Source document:\n{doc_header}\n\n"
                    f"{'═' * 60}\n\n"
                    f"Draft analysis to review:\n{draft}"
                ),
            },
        ]
        gpt      = OpenAIChatClient()
        reviewed = gpt.chat(reviewer_messages, temperature=0.1).get("content", "").strip()

        if reviewed.upper().startswith("APPROVED"):
            logger.debug("ASX reviewer: APPROVED — returning Gemini draft")
            return draft

        # GPT rewrote it
        logger.debug("ASX reviewer: REWRITTEN by GPT")
        return reviewed if reviewed else draft

    except Exception as exc:
        # Reviewer unavailable — Gemini draft is still valid
        logger.warning("GPT reviewer failed, returning Gemini draft: %s", exc)
        return draft

"""
DataPAI RAG API — FastAPI service exposing LanceDB RAG to any client.

Serves as the shared backend for:
  1. Streamlit knowledge tab  — direct Python import (no HTTP needed)
  2. OpenWebUI Pipeline        — called via HTTP at POST /v1/rag/query
  3. Any external client       — standard REST API

Endpoints:
  POST /v1/rag/query      — ask a question, get RAG answer + source docs
  POST /v1/rag/ingest     — upload + ingest a document into LanceDB
  GET  /v1/rag/documents  — list all ingested documents
  DELETE /v1/rag/documents/{filename} — remove a document
  GET  /health            — health check (LanceDB + Ollama reachability)

OpenWebUI compatible:
  The /v1/rag/query response includes an 'openai_messages' field so the
  OpenWebUI pipeline can pass the augmented context directly to the LLM.

Run standalone:
  uvicorn agents.rag_api:app --host 0.0.0.0 --port 8100 --reload

Environment variables:
  LANCEDB_URI          LanceDB path (local or s3://)   default: s3://codepais3/lancedb_data/
  OLLAMA_HOST          Ollama base URL                  default: http://localhost:11434
  RAG_LLM_MODEL        Default Ollama model             default: llama3.2
  RAG_TOP_K            Documents to retrieve            default: 5
  RAG_MAX_CTX_CHARS    Max context chars sent to LLM    default: 6000
  RAG_API_KEY          Optional bearer token for API    default: (none)
"""

from __future__ import annotations

import os
import tempfile
import logging
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Security, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
_RAG_API_KEY  = os.getenv("RAG_API_KEY", "")          # empty = no auth
_TOP_K        = int(os.getenv("RAG_TOP_K", "5"))
_MAX_CTX      = int(os.getenv("RAG_MAX_CTX_CHARS", "6000"))
_DEFAULT_MODEL = os.getenv("RAG_LLM_MODEL", "llama3.2")

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DataPAI RAG API",
    description="LanceDB + Ollama RAG service — used by Streamlit and OpenWebUI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Optional API key auth ──────────────────────────────────────────────────────
_bearer = HTTPBearer(auto_error=False)

def _check_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> None:
    if not _RAG_API_KEY:
        return   # auth disabled
    if not credentials or credentials.credentials != _RAG_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ═══════════════════════════════════════════════════════════════════════════════
# Request / Response models
# ═══════════════════════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    role: str        # "user" | "assistant" | "system"
    content: str

class QueryRequest(BaseModel):
    question: str
    model: Optional[str] = None
    collections: Optional[List[str]] = None
    k: int = _TOP_K
    max_ctx_chars: int = _MAX_CTX
    chat_history: Optional[List[ChatMessage]] = None   # previous turns for context

class QueryResponse(BaseModel):
    answer: str
    model: str
    sources: List[dict]           # [{filename, collection, source_uri, score}]
    context_used: str             # raw context string sent to LLM
    openai_messages: List[dict]   # ready-to-use for OpenWebUI pipeline / OpenAI SDK

class IngestResponse(BaseModel):
    status: str
    filename: str
    collection: str
    message: str

class DocumentListResponse(BaseModel):
    documents: List[dict]
    total: int

class HealthResponse(BaseModel):
    status: str
    lancedb: str
    ollama: str
    collections: List[str]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _answer_with_history(
    question: str,
    chat_history: Optional[List[ChatMessage]],
    model: Optional[str],
    collections: Optional[List[str]],
    k: int,
    max_ctx_chars: int,
) -> QueryResponse:
    """
    Full RAG pipeline with optional multi-turn history injection.

    Steps:
      1. Search LanceDB for relevant documents
      2. Build context string from results
      3. Build message list (system + history + RAG-augmented user message)
      4. Call Ollama /api/chat
      5. Return structured response
    """
    import requests as _requests
    from agents.knowledge_query_agent import search_lancedb, build_context_from_results

    resolved_model = model or _DEFAULT_MODEL
    ollama_host = os.getenv("OLLAMA_HOST", os.getenv("STREAMLIT_OLLAMA_HOST", "http://localhost:11434"))

    # 1) Retrieve from LanceDB
    df = search_lancedb(question, collections=collections, k=k)
    context = build_context_from_results(df, max_chars=max_ctx_chars)

    # 2) Build sources list for response metadata
    sources: List[dict] = []
    if not df.empty:
        for _, row in df.iterrows():
            sources.append({
                "filename":   str(row.get("filename", "")),
                "collection": str(row.get("collection", "")),
                "source_uri": str(row.get("source_uri", "")),
                "score": float(row.get("_distance", row.get("score", 0.0))),
            })

    # 3) Build messages (system + optional history + augmented user message)
    system_msg = {
        "role": "system",
        "content": (
            "You are a helpful data and documentation assistant for DataPAI. "
            "Use ONLY the provided context to answer. "
            "If the answer is not in the context, say you don't know. "
            "Do NOT invent facts. Be concise and precise."
        ),
    }

    messages: List[dict] = [system_msg]

    # Inject previous conversation turns (for multi-turn chat)
    if chat_history:
        for turn in chat_history:
            messages.append({"role": turn.role, "content": turn.content})

    # Augment the current question with RAG context
    augmented_user = (
        f"Context from knowledge base:\n"
        f"{'─' * 60}\n"
        f"{context}\n"
        f"{'─' * 60}\n\n"
        f"Question: {question}"
    )
    messages.append({"role": "user", "content": augmented_user})

    # 4) Call Ollama
    resp = _requests.post(
        f"{ollama_host}/api/chat",
        json={"model": resolved_model, "messages": messages, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    answer = resp.json().get("message", {}).get("content", "No answer returned.")

    # 5) Build openai_messages (for OpenWebUI pipeline — passes clean context to next step)
    openai_messages = [system_msg] + (
        [{"role": t.role, "content": t.content} for t in (chat_history or [])]
    ) + [{"role": "user", "content": augmented_user}]

    return QueryResponse(
        answer=answer,
        model=resolved_model,
        sources=sources,
        context_used=context,
        openai_messages=openai_messages,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/v1/rag/query", response_model=QueryResponse, dependencies=[Depends(_check_api_key)])
def rag_query(req: QueryRequest) -> QueryResponse:
    """
    Ask a question against the LanceDB knowledge base.

    - Embeds the question
    - Retrieves top-k relevant documents
    - Builds RAG context (respects max_ctx_chars)
    - Calls Ollama with full multi-turn chat history if provided
    - Returns the answer, source documents, and OpenAI-compatible messages

    This endpoint is called by the OpenWebUI RAG pipeline.
    """
    try:
        return _answer_with_history(
            question=req.question,
            chat_history=req.chat_history,
            model=req.model,
            collections=req.collections,
            k=req.k,
            max_ctx_chars=req.max_ctx_chars,
        )
    except Exception as exc:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/rag/ingest", response_model=IngestResponse, dependencies=[Depends(_check_api_key)])
async def rag_ingest(file: UploadFile = File(...)) -> IngestResponse:
    """
    Upload and ingest a document into LanceDB.

    Supports: PDF, TXT, MD, CSV, PNG, JPG, JPEG
    The file is saved to a temp location, processed, embedded,
    and stored in the appropriate LanceDB collection.
    """
    from agents.knowledge_ingest_agent import ingest_files_to_lancedb

    db_uri = os.getenv("LANCEDB_URI", "s3://codepais3/lancedb_data/")

    suffix = os.path.splitext(file.filename or "upload")[1].lower()
    allowed = {".pdf", ".txt", ".md", ".csv", ".png", ".jpg", ".jpeg"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(allowed))}"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # collection is determined by file type inside ingest_files_to_lancedb
        ingest_files_to_lancedb([tmp_path], db_uri=db_uri)
        collection = "pdfs" if suffix == ".pdf" else (
            "images" if suffix in {".png", ".jpg", ".jpeg"} else "documents"
        )
        return IngestResponse(
            status="success",
            filename=file.filename or "unknown",
            collection=collection,
            message=f"Ingested '{file.filename}' into LanceDB collection '{collection}' at {db_uri}.",
        )
    except Exception as exc:
        logger.exception("Ingest failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.get("/v1/rag/documents", response_model=DocumentListResponse, dependencies=[Depends(_check_api_key)])
def list_documents() -> DocumentListResponse:
    """
    List all documents ingested into LanceDB across all collections.
    Returns filename, collection, source_uri for each document.
    """
    import lancedb

    db_uri = os.getenv("LANCEDB_URI", "s3://codepais3/lancedb_data/")
    docs: List[dict] = []

    try:
        db = lancedb.connect(db_uri)
        for col in db.table_names():
            tbl = db.open_table(col)
            df = tbl.to_pandas()
            if "filename" in df.columns:
                for _, row in df.iterrows():
                    docs.append({
                        "collection": col,
                        "filename":   str(row.get("filename", "")),
                        "source_uri": str(row.get("source_uri", "")),
                    })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LanceDB error: {exc}")

    return DocumentListResponse(documents=docs, total=len(docs))


@app.delete("/v1/rag/documents/{filename}", dependencies=[Depends(_check_api_key)])
def delete_document(filename: str) -> dict:
    """
    Remove a document from LanceDB by filename (across all collections).
    """
    import lancedb

    db_uri = os.getenv("LANCEDB_URI", "s3://codepais3/lancedb_data/")
    deleted = 0

    try:
        db = lancedb.connect(db_uri)
        for col in db.table_names():
            tbl = db.open_table(col)
            df = tbl.to_pandas()
            if "filename" in df.columns and filename in df["filename"].values:
                tbl.delete(f"filename = '{filename}'")
                deleted += 1
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")

    return {"status": "deleted", "filename": filename, "collections_affected": deleted}


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check — verifies LanceDB and Ollama are reachable."""
    import lancedb
    import requests as _requests

    db_uri  = os.getenv("LANCEDB_URI", "s3://codepais3/lancedb_data/")
    ollama  = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # LanceDB
    lance_status = "ok"
    collections: List[str] = []
    try:
        db = lancedb.connect(db_uri)
        collections = db.table_names()
    except Exception as exc:
        lance_status = f"error: {exc}"

    # Ollama
    ollama_status = "ok"
    try:
        r = _requests.get(f"{ollama}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as exc:
        ollama_status = f"error: {exc}"

    overall = "ok" if lance_status == "ok" and ollama_status == "ok" else "degraded"
    return HealthResponse(
        status=overall,
        lancedb=lance_status,
        ollama=ollama_status,
        collections=collections,
    )


# ── Dev runner ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agents.rag_api:app", host="0.0.0.0", port=8100, reload=True)

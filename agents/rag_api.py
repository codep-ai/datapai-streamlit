"""
DataPAI RAG API — FastAPI service exposing LanceDB RAG to any client.

Infrastructure context (3-EC2 setup):
  EC2 #1  Nginx reverse proxy / static host
  EC2 #2  Platform (8GB/2CPU) — THIS SERVICE RUNS HERE
            Handles: LanceDB retrieval (S3), Streamlit, Airbyte, Lightdash
            Business hours only
  EC2 #3  GPU instance (optional/demo) — Ollama + OpenWebUI
            Private IP, not always running

Design principle — split retrieval from generation:
  • /v1/rag/retrieve  → LanceDB search only, returns context docs (no LLM)
                        Called by OpenWebUI pipeline on EC2 #3 — GPU does generation locally
  • /v1/rag/query     → retrieval + LLM answer (Streamlit on EC2 #2 with
                        OLLAMA_HOST pointing to EC2 #3 private IP or local fallback)

This avoids the EC2 #3 → EC2 #2 → EC2 #3 round-trip for LLM calls.
The GPU on EC2 #3 is used directly by OpenWebUI; EC2 #2 only does retrieval.

Endpoints:
  POST /v1/rag/retrieve   — LanceDB search only → context docs + augmented messages
                            (NO LLM call — caller handles generation)
  POST /v1/rag/query      — full RAG: retrieval + Ollama answer
                            (used by Streamlit on EC2 #2)
  POST /v1/rag/ingest     — upload + ingest a document into LanceDB
  GET  /v1/rag/documents  — list all ingested documents
  DELETE /v1/rag/documents/{filename} — remove a document
  GET  /health            — health check (LanceDB + Ollama reachability)

Run standalone (on EC2 #2):
  uvicorn agents.rag_api:app --host 0.0.0.0 --port 8100 --reload

Environment variables (EC2 #2):
  LANCEDB_URI          LanceDB path (local or s3://)   default: s3://codepais3/lancedb_data/
  OLLAMA_HOST          EC2 #3 private IP when up        e.g. http://10.0.1.50:11434
                       Falls back to local Ollama if EC2 #3 is down
  OLLAMA_FALLBACK_HOST Local Ollama on EC2 #2 (smaller model fallback)
                                                        default: http://localhost:11434
  RAG_LLM_MODEL        Default Ollama model             default: llama3.2
  RAG_FALLBACK_MODEL   Smaller model on EC2 #2          default: llama3.2
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
_RAG_API_KEY     = os.getenv("RAG_API_KEY", "")          # empty = no auth
_TOP_K           = int(os.getenv("RAG_TOP_K", "5"))
_MAX_CTX         = int(os.getenv("RAG_MAX_CTX_CHARS", "6000"))
_DEFAULT_MODEL   = os.getenv("RAG_LLM_MODEL", "llama3.2")
_FALLBACK_MODEL  = os.getenv("RAG_FALLBACK_MODEL", "llama3.2")  # smaller model on EC2 #2
_OLLAMA_FALLBACK = os.getenv("OLLAMA_FALLBACK_HOST", "http://localhost:11434")

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


class RetrieveRequest(BaseModel):
    question: str
    collections: Optional[List[str]] = None
    k: int = _TOP_K
    max_ctx_chars: int = _MAX_CTX
    chat_history: Optional[List[ChatMessage]] = None   # previous turns for context


class RetrieveResponse(BaseModel):
    sources: List[dict]           # [{filename, collection, source_uri, score}]
    context_used: str             # raw context string
    openai_messages: List[dict]   # augmented messages ready to forward to any LLM
                                  # caller does generation — EC2 #3 GPU handles it

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
    # Primary: OLLAMA_HOST (EC2 #3 private IP when set, else localhost)
    primary_host   = os.getenv("OLLAMA_HOST", os.getenv("STREAMLIT_OLLAMA_HOST", "http://localhost:11434"))
    # Fallback: local Ollama on EC2 #2 (smaller model, may run even when EC2 #3 is down)
    fallback_host  = _OLLAMA_FALLBACK

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

    # 4) Call Ollama — try primary (EC2 #3), fall back to local (EC2 #2)
    def _call_ollama(host: str, mdl: str) -> str:
        r = _requests.post(
            f"{host}/api/chat",
            json={"model": mdl, "messages": messages, "stream": False},
            timeout=300,
        )
        r.raise_for_status()
        return r.json().get("message", {}).get("content", "No answer returned.")

    try:
        answer = _call_ollama(primary_host, resolved_model)
    except Exception as primary_exc:
        logger.warning(
            "Primary Ollama at %s failed (%s) — trying fallback at %s",
            primary_host, primary_exc, fallback_host,
        )
        # Only fall back if the fallback is a different host
        if fallback_host and fallback_host != primary_host:
            try:
                answer = _call_ollama(fallback_host, _FALLBACK_MODEL)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Both Ollama hosts failed.\n"
                    f"  Primary  ({primary_host}): {primary_exc}\n"
                    f"  Fallback ({fallback_host}): {fallback_exc}"
                ) from fallback_exc
        else:
            raise

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

@app.post("/v1/rag/retrieve", response_model=RetrieveResponse, dependencies=[Depends(_check_api_key)])
def rag_retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """
    LanceDB retrieval only — NO LLM call.

    Use this endpoint when the caller (e.g. OpenWebUI running on EC2 #3)
    wants to handle generation itself on its local GPU.

    Flow:
      1. Embed the question
      2. Search LanceDB for top-k relevant documents
      3. Build RAG context string
      4. Return context + sources + OpenAI-compatible augmented messages

    The returned `openai_messages` list is ready to pass directly to any
    OpenAI-compatible LLM API (Ollama /api/chat, OpenAI, etc.).
    This avoids the EC2 #3 → EC2 #2 → EC2 #3 round-trip for LLM calls.
    """
    from agents.knowledge_query_agent import search_lancedb, build_context_from_results

    try:
        # 1) Retrieve from LanceDB
        df = search_lancedb(req.question, collections=req.collections, k=req.k)
        context = build_context_from_results(df, max_chars=req.max_ctx_chars)

        # 2) Build sources metadata
        sources: List[dict] = []
        if not df.empty:
            for _, row in df.iterrows():
                sources.append({
                    "filename":   str(row.get("filename", "")),
                    "collection": str(row.get("collection", "")),
                    "source_uri": str(row.get("source_uri", "")),
                    "score": float(row.get("_distance", row.get("score", 0.0))),
                })

        # 3) Build system prompt
        system_msg = {
            "role": "system",
            "content": (
                "You are a helpful data and documentation assistant for DataPAI. "
                "Use ONLY the provided context to answer. "
                "If the answer is not in the context, say you don't know. "
                "Do NOT invent facts. Be concise and precise."
            ),
        }

        # 4) Build augmented messages (system + history + RAG-injected user question)
        messages: List[dict] = [system_msg]

        if req.chat_history:
            for turn in req.chat_history:
                messages.append({"role": turn.role, "content": turn.content})

        augmented_user = (
            f"Context from knowledge base:\n"
            f"{'─' * 60}\n"
            f"{context}\n"
            f"{'─' * 60}\n\n"
            f"Question: {req.question}"
        )
        messages.append({"role": "user", "content": augmented_user})

        return RetrieveResponse(
            sources=sources,
            context_used=context,
            openai_messages=messages,
        )

    except Exception as exc:
        logger.exception("RAG retrieve failed")
        raise HTTPException(status_code=500, detail=str(exc))


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

# agents/knowledge_query_agent.py

from __future__ import annotations

import os
from typing import List, Optional

import lancedb
import pandas as pd
import requests

from embeddings.embed import embed_texts  # uses your existing embedding logic


# -------------------------------------------------------------------
# LanceDB connection + search
# -------------------------------------------------------------------

# Default to your existing S3 LanceDB URI, but allow override via env
DEFAULT_DB_URI = "s3://codepais3/lancedb_data/"
DB_URI = os.environ.get("LANCEDB_URI", DEFAULT_DB_URI)


def _get_db():
    """
    Connect to LanceDB using DB_URI (S3 or local path).
    """
    return lancedb.connect(DB_URI)


def search_lancedb(
    query: str,
    collections: Optional[List[str]] = None,
    k: int = 5,
) -> pd.DataFrame:
    """
    Embed the query, search across one or more LanceDB collections,
    and return top-k rows as a pandas DataFrame.

    Collections typically: ["documents", "pdfs", "images"]
    """
    if not query or not query.strip():
        raise ValueError("Query is empty")

    if collections is None:
        collections = ["documents", "pdfs", "images", "asx_announcements"]

    db = _get_db()
    q_vec = embed_texts([query])[0]

    dfs: List[pd.DataFrame] = []

    for col in collections:
        if col not in db.table_names():
            # Skip missing collections (e.g. no images ingested yet)
            continue

        tbl = db.open_table(col)

        # LanceDB: nearest neighbor search using vector similarity
        result = (
            tbl.search(q_vec)
            .limit(k)
            .to_pandas()
        )

        if result.empty:
            continue

        result["collection"] = col
        dfs.append(result)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Prefer smaller distance or higher score depending on LanceDB version
    if "_distance" in combined.columns:
        combined = combined.sort_values("_distance", ascending=True)
    elif "score" in combined.columns:
        combined = combined.sort_values("score", ascending=False)

    # Now take top-k overall across all collections
    return combined.head(k)


# -------------------------------------------------------------------
# Context building for RAG
# -------------------------------------------------------------------

def build_context_from_results(
    df: pd.DataFrame,
    max_chars: int = 6000,
) -> str:
    """
    Build a text context from LanceDB search results for the LLM.

    Uses 'text' when available; falls back to 'source_uri' for non-text
    content, and includes filename + collection for traceability.
    """
    if df is None or df.empty:
        return "No relevant documents found."

    chunks: List[str] = []
    total_chars = 0

    # Make sure we don't exceed token/char limits
    for _, row in df.iterrows():
        text = str(row.get("text") or "").strip()
        src = str(row.get("source_uri") or "").strip()
        filename = str(row.get("filename") or "").strip()
        collection = str(row.get("collection") or "").strip()

        header = f"[{collection}] {filename} - {src}"

        if text:
            snippet = f"{header}\n{text}\n"
        else:
            # If no text (e.g. image with only URI), still include header
            snippet = f"{header}\n(No extracted text)\n"

        remaining = max_chars - total_chars
        if remaining <= 0:
            break

        snippet = snippet[:remaining]
        chunks.append(snippet)
        total_chars += len(snippet)

    if not chunks:
        return "No relevant documents found."

    return "\n\n".join(chunks)


# -------------------------------------------------------------------
# RAG answer via Ollama
# -------------------------------------------------------------------

def answer_with_ollama(
    question: str,
    model: Optional[str] = None,
    collections: Optional[List[str]] = None,
    k: int = 5,
    max_ctx_chars: int = 6000,
) -> str:
    """
    Full RAG pipeline:
    - embed question
    - retrieve top-k documents from LanceDB
    - build context
    - call Ollama chat API with context + question

    Env overrides:
      - RAG_LLM_MODEL: default model name (e.g. "llama3.2")
      - OLLAMA_HOST:   base URL (default "http://localhost:11434")
    """
    if not question or not question.strip():
        raise ValueError("Question is empty")

    # 1) Retrieve from LanceDB
    df = search_lancedb(question, collections=collections, k=k)

    # 2) Build context string
    context = build_context_from_results(df, max_chars=max_ctx_chars)

    # 3) Decide model + host
    default_model = os.environ.get("RAG_LLM_MODEL", "llama3.2")
    model = model or default_model
    ollama_host = os.environ.get("STREAMLIT_OLLAMA_HOST", "http://localhost:11434")

    # 4) Compose prompt
    prompt = f"""You are a helpful data and documentation assistant.

Use ONLY the context below to answer the question. If the answer is not
present in the context, say you don't know. Do NOT invent facts.

Context:
{context}

Question: {question}

Answer:"""

    # 5) Call Ollama chat API
    resp = requests.post(
        f"{ollama_host}/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful RAG assistant."},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()

    # Ollama chat API typically returns: { "message": { "role": "assistant", "content": "..."} }
    message = data.get("message", {})
    content = message.get("content", "")

    if not content:
        # (Optional) Debug info if something is off
        return "No answer returned by the model."

    return content


# -------------------------------------------------------------------
# Convenience: quick debug helper (optional)
# -------------------------------------------------------------------

def debug_search_and_print(
    query: str,
    collections: Optional[List[str]] = None,
    k: int = 5,
) -> None:
    """
    Simple helper to use from a Python shell to see what LanceDB returns.
    Example:

        from agents.knowledge_query_agent import debug_search_and_print
        debug_search_and_print("What is the Snowflake architecture?")
    """
    df = search_lancedb(query, collections=collections, k=k)
    if df.empty:
        print("No results.")
        return

    print(df[["collection", "filename", "source_uri"]].head(k))


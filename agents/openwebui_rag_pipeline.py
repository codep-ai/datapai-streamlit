"""
DataPAI RAG Pipeline for OpenWebUI
====================================
Adds your LanceDB knowledge base as a RAG-augmented "model" inside OpenWebUI.

HOW IT WORKS
────────────
OpenWebUI Pipelines intercept every chat message before it reaches the LLM.
This pipeline uses the SPLIT architecture to avoid wasteful EC2 round-trips:

  1. Takes the user's message from OpenWebUI
  2. Calls /v1/rag/retrieve on EC2 #2 → gets context docs + augmented messages
     (NO LLM call on EC2 #2 — pure LanceDB vector search)
  3. Calls Ollama directly at OLLAMA_HOST with the augmented messages
     (EC2 #3 GPU handles generation if configured, or local Ollama as fallback)
  4. Streams the answer back to OpenWebUI — exactly like a normal chat

This avoids the EC2 #3 → EC2 #2 → EC2 #3 round-trip:
  OLD: OpenWebUI → pipeline → /v1/rag/query (retrieval + Ollama call on EC2 #2) → return
  NEW: OpenWebUI → pipeline → /v1/rag/retrieve (retrieval only) → Ollama at OLLAMA_HOST

The user sees a model called "DataPAI RAG (LanceDB)" in OpenWebUI's model list.
They never need to leave OpenWebUI.

INSTALLATION
────────────
1. Start the RAG API on EC2 #2:
     uvicorn agents.rag_api:app --host 0.0.0.0 --port 8100

   Or add to your docker-compose.yml:
     rag-api:
       build: .
       command: uvicorn agents.rag_api:app --host 0.0.0.0 --port 8100
       environment:
         - LANCEDB_URI=s3://codepais3/lancedb_data/
         - RAG_LLM_MODEL=llama3.2
       ports:
         - "8100:8100"

2. In OpenWebUI → Admin → Pipelines → Add Pipeline:
     Paste this entire file.

3. Set the Pipeline environment variables:
     DATAPAI_RAG_API_URL = http://rag-api:8100   (or http://host.docker.internal:8100)
     DATAPAI_RAG_API_KEY = <your key>             (if RAG_API_KEY is set)
     DATAPAI_RAG_MODEL   = llama3.2              (override default model)
     OLLAMA_HOST         = http://localhost:11434  (EC2 #3 private IP if available)

4. Enable the pipeline for your workspace / users.

ENVIRONMENT VARIABLES (set in OpenWebUI Pipeline settings)
────────────────────────────────────────────────────────────
  DATAPAI_RAG_API_URL   URL of the DataPAI RAG FastAPI service (EC2 #2)
                        default: http://localhost:8100
  DATAPAI_RAG_API_KEY   Bearer token (if RAG_API_KEY is configured)
                        default: (empty — no auth)
  DATAPAI_RAG_MODEL     Ollama model to use for generation
                        default: llama3.2
  DATAPAI_RAG_TOP_K     Number of documents to retrieve
                        default: 5
  DATAPAI_RAG_FALLBACK  If True, answer from model memory when RAG API is unreachable
                        default: true
  OLLAMA_HOST           Where to send generation requests
                        default: http://localhost:11434
                        Set to EC2 #3 private IP for GPU inference, e.g. http://10.0.1.50:11434
"""

from __future__ import annotations

import os
from typing import AsyncGenerator, Iterator, List, Union

import requests

# OpenWebUI Pipeline SDK imports
# (these are available inside the OpenWebUI pipeline execution environment)
from pydantic import BaseModel


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline configuration — exposed in OpenWebUI's pipeline settings UI
# ═══════════════════════════════════════════════════════════════════════════════

class Pipeline:
    """
    DataPAI RAG Pipeline — connects OpenWebUI to LanceDB knowledge base.

    Uses the split retrieval/generation architecture:
      - EC2 #2: /v1/rag/retrieve (vector search, no LLM)
      - OLLAMA_HOST: generation (EC2 #3 GPU or local fallback)

    Appears as "DataPAI RAG (LanceDB)" in the OpenWebUI model selector.
    """

    class Valves(BaseModel):
        """
        Valves are user-configurable settings shown in the OpenWebUI pipeline UI.
        """
        DATAPAI_RAG_API_URL: str = os.getenv("DATAPAI_RAG_API_URL", "http://localhost:8100")
        DATAPAI_RAG_API_KEY: str = os.getenv("DATAPAI_RAG_API_KEY", "")
        DATAPAI_RAG_MODEL:   str = os.getenv("DATAPAI_RAG_MODEL",   "llama3.2")
        DATAPAI_RAG_TOP_K:   int = int(os.getenv("DATAPAI_RAG_TOP_K", "5"))
        DATAPAI_RAG_FALLBACK: bool = os.getenv("DATAPAI_RAG_FALLBACK", "true").lower() != "false"
        # Where to send generation requests — set to EC2 #3 private IP if available
        OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def __init__(self):
        self.name   = "DataPAI RAG (LanceDB)"
        self.valves = self.Valves()

    # ── OpenWebUI Pipeline lifecycle ───────────────────────────────────────

    async def on_startup(self):
        """Called when OpenWebUI starts. Verify RAG API and Ollama are reachable."""
        # Check RAG API
        try:
            r = requests.get(
                f"{self.valves.DATAPAI_RAG_API_URL}/health",
                timeout=5,
            )
            r.raise_for_status()
            health = r.json()
            print(
                f"[DataPAI RAG] Connected to RAG API — "
                f"LanceDB: {health.get('lancedb')}  "
                f"Collections: {health.get('collections', [])}"
            )
        except Exception as exc:
            print(f"[DataPAI RAG] ⚠ RAG API not reachable at startup: {exc}")

        # Check Ollama for generation
        try:
            r = requests.get(f"{self.valves.OLLAMA_HOST}/api/tags", timeout=5)
            r.raise_for_status()
            print(f"[DataPAI RAG] Ollama reachable at {self.valves.OLLAMA_HOST}")
        except Exception as exc:
            print(f"[DataPAI RAG] ⚠ Ollama not reachable at {self.valves.OLLAMA_HOST}: {exc}")

    async def on_shutdown(self):
        """Called when OpenWebUI shuts down."""
        print("[DataPAI RAG] Pipeline shutdown.")

    async def on_valves_updated(self):
        """Called when pipeline settings are saved in the UI."""
        print(
            f"[DataPAI RAG] Config updated — "
            f"RAG API: {self.valves.DATAPAI_RAG_API_URL}  "
            f"Ollama: {self.valves.OLLAMA_HOST}"
        )

    # ── Main pipeline entrypoint ───────────────────────────────────────────

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Iterator[str], AsyncGenerator[str, None]]:
        """
        Called for every chat message sent to this pipeline in OpenWebUI.

        Split architecture:
          Step 1 → POST /v1/rag/retrieve  (EC2 #2, LanceDB only, no LLM)
          Step 2 → POST /api/chat          (OLLAMA_HOST — EC2 #3 GPU or local)

        Args:
            user_message: The latest user message text.
            model_id:     The selected model ID in OpenWebUI.
            messages:     Full conversation history (OpenAI message format).
            body:         Full request body from OpenWebUI.

        Returns:
            The RAG-augmented answer string.
        """
        # ── Step 1: Retrieve context from LanceDB (no LLM call) ───────────

        # Extract conversation history (exclude the last user message)
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in messages[:-1]
            if m.get("role") in ("user", "assistant")
        ]

        retrieve_url = f"{self.valves.DATAPAI_RAG_API_URL}/v1/rag/retrieve"
        headers = {"Content-Type": "application/json"}
        if self.valves.DATAPAI_RAG_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.DATAPAI_RAG_API_KEY}"

        retrieve_payload = {
            "question":     user_message,
            "k":            self.valves.DATAPAI_RAG_TOP_K,
            "chat_history": chat_history,
        }

        try:
            retrieve_resp = requests.post(
                retrieve_url,
                json=retrieve_payload,
                headers=headers,
                timeout=30,           # retrieval only — should be fast
            )
            retrieve_resp.raise_for_status()
            retrieve_data = retrieve_resp.json()

            openai_messages = retrieve_data.get("openai_messages", [])
            sources = retrieve_data.get("sources", [])

        except requests.exceptions.ConnectionError:
            msg = (
                f"⚠️ DataPAI RAG API is not reachable at {self.valves.DATAPAI_RAG_API_URL}. "
                f"Please ensure the RAG service is running on EC2 #2."
            )
            if self.valves.DATAPAI_RAG_FALLBACK:
                return self._fallback_answer(user_message, messages)
            return msg

        except Exception as exc:
            if self.valves.DATAPAI_RAG_FALLBACK:
                return self._fallback_answer(user_message, messages)
            return f"⚠️ RAG retrieve error: {exc}"

        # ── Step 2: Generate answer via Ollama at OLLAMA_HOST ─────────────

        try:
            gen_resp = requests.post(
                f"{self.valves.OLLAMA_HOST}/api/chat",
                json={
                    "model":    self.valves.DATAPAI_RAG_MODEL,
                    "messages": openai_messages,
                    "stream":   False,
                },
                timeout=300,
            )
            gen_resp.raise_for_status()
            answer = gen_resp.json().get("message", {}).get("content", "No answer returned.")

        except Exception as exc:
            # Ollama unreachable — return context only so user can see what was found
            no_llm_msg = (
                f"⚠️ Ollama not reachable at `{self.valves.OLLAMA_HOST}` — "
                f"returning retrieved context only.\n\n"
                f"**Error:** `{exc}`\n\n"
            )
            if sources:
                no_llm_msg += "**Retrieved documents:**\n"
                for src in sources:
                    no_llm_msg += f"- **{src.get('filename','?')}** [{src.get('collection','?')}]\n"
            return no_llm_msg

        # ── Append source citations ────────────────────────────────────────

        if sources:
            citation_lines = ["\n\n---\n📎 **Sources from knowledge base:**"]
            for src in sources:
                name = src.get("filename", "?")
                coll = src.get("collection", "?")
                uri  = src.get("source_uri", "")
                citation_lines.append(f"- **{name}** [{coll}]  `{uri}`")
            answer += "\n".join(citation_lines)

        return answer

    # ── Fallback: plain Ollama (no RAG) ───────────────────────────────────

    def _fallback_answer(self, user_message: str, messages: List[dict]) -> str:
        """
        Answer directly via Ollama without RAG context.
        Used when the RAG API is unreachable and DATAPAI_RAG_FALLBACK=true.
        """
        model = self.valves.DATAPAI_RAG_MODEL

        try:
            resp = requests.post(
                f"{self.valves.OLLAMA_HOST}/api/chat",
                json={
                    "model":    model,
                    "messages": messages,
                    "stream":   False,
                },
                timeout=120,
            )
            resp.raise_for_status()
            answer = resp.json().get("message", {}).get("content", "No answer.")
            return f"*(RAG unavailable — answering from model memory only)*\n\n{answer}"
        except Exception as exc:
            return (
                f"⚠️ Both RAG API and Ollama fallback failed.\n\n"
                f"- RAG API: `{self.valves.DATAPAI_RAG_API_URL}` — unreachable\n"
                f"- Ollama: `{self.valves.OLLAMA_HOST}` — `{exc}`\n\n"
                f"Please check that the services are running."
            )

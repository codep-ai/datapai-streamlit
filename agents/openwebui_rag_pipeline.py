"""
DataPAI RAG Pipeline for OpenWebUI
====================================
Adds your LanceDB knowledge base as a RAG-augmented "model" inside OpenWebUI.

HOW IT WORKS
────────────
OpenWebUI Pipelines intercept every chat message before it reaches the LLM.
This pipeline:
  1. Takes the user's message from OpenWebUI
  2. Calls your DataPAI RAG API (agents/rag_api.py) to search LanceDB
  3. Injects the retrieved document context into the prompt
  4. Passes the augmented messages to the underlying Ollama model
  5. Streams the answer back to OpenWebUI — exactly like a normal chat

The user sees a model called "DataPAI RAG (LanceDB)" in OpenWebUI's model list.
They never need to leave OpenWebUI.

INSTALLATION
────────────
1. Start the RAG API:
     uvicorn agents.rag_api:app --host 0.0.0.0 --port 8100

   Or add to your docker-compose.yml:
     rag-api:
       build: .
       command: uvicorn agents.rag_api:app --host 0.0.0.0 --port 8100
       environment:
         - LANCEDB_URI=s3://codepais3/lancedb_data/
         - OLLAMA_HOST=http://ollama:11434
         - RAG_LLM_MODEL=llama3.2
       ports:
         - "8100:8100"

2. In OpenWebUI → Admin → Pipelines → Add Pipeline:
     Paste this entire file.

3. Set the Pipeline environment variable:
     DATAPAI_RAG_API_URL = http://rag-api:8100   (or http://host.docker.internal:8100)
     DATAPAI_RAG_API_KEY = <your key>             (if RAG_API_KEY is set)
     DATAPAI_RAG_MODEL   = llama3.2              (override default model)

4. Enable the pipeline for your workspace / users.

ENVIRONMENT VARIABLES (set in OpenWebUI Pipeline settings)
────────────────────────────────────────────────────────────
  DATAPAI_RAG_API_URL   URL of the DataPAI RAG FastAPI service
                        default: http://localhost:8100
  DATAPAI_RAG_API_KEY   Bearer token (if RAG_API_KEY is configured)
                        default: (empty — no auth)
  DATAPAI_RAG_MODEL     Ollama model to use for answers
                        default: llama3.2
  DATAPAI_RAG_TOP_K     Number of documents to retrieve
                        default: 5
  DATAPAI_RAG_FALLBACK  If True, fall back to plain LLM when RAG API is unreachable
                        default: true
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

    def __init__(self):
        self.name   = "DataPAI RAG (LanceDB)"
        self.valves = self.Valves()

    # ── OpenWebUI Pipeline lifecycle ───────────────────────────────────────

    async def on_startup(self):
        """Called when OpenWebUI starts. Verify RAG API is reachable."""
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
                f"Ollama: {health.get('ollama')}  "
                f"Collections: {health.get('collections', [])}"
            )
        except Exception as exc:
            print(f"[DataPAI RAG] ⚠ RAG API not reachable at startup: {exc}")

    async def on_shutdown(self):
        """Called when OpenWebUI shuts down."""
        print("[DataPAI RAG] Pipeline shutdown.")

    async def on_valves_updated(self):
        """Called when pipeline settings are saved in the UI."""
        print(f"[DataPAI RAG] Config updated — URL: {self.valves.DATAPAI_RAG_API_URL}")

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

        Args:
            user_message: The latest user message text.
            model_id:     The selected model ID in OpenWebUI.
            messages:     Full conversation history (OpenAI message format).
            body:         Full request body from OpenWebUI.

        Returns:
            The RAG-augmented answer string.
        """
        # Extract conversation history (exclude the last user message — that's user_message)
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in messages[:-1]   # everything except the latest turn
            if m.get("role") in ("user", "assistant")
        ]

        # Build request to DataPAI RAG API
        api_url = f"{self.valves.DATAPAI_RAG_API_URL}/v1/rag/query"
        headers = {"Content-Type": "application/json"}
        if self.valves.DATAPAI_RAG_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.DATAPAI_RAG_API_KEY}"

        payload = {
            "question":     user_message,
            "model":        self.valves.DATAPAI_RAG_MODEL,
            "k":            self.valves.DATAPAI_RAG_TOP_K,
            "chat_history": chat_history,
        }

        try:
            resp = requests.post(api_url, json=payload, headers=headers, timeout=300)
            resp.raise_for_status()
            data = resp.json()

            answer  = data.get("answer", "No answer returned.")
            sources = data.get("sources", [])

            # Append source citations to the answer (visible in OpenWebUI chat)
            if sources:
                citation_lines = ["\n\n---\n📎 **Sources from knowledge base:**"]
                for src in sources:
                    name = src.get("filename", "?")
                    coll = src.get("collection", "?")
                    uri  = src.get("source_uri", "")
                    citation_lines.append(f"- **{name}** [{coll}]  `{uri}`")
                answer += "\n".join(citation_lines)

            return answer

        except requests.exceptions.ConnectionError:
            msg = (
                f"⚠️ DataPAI RAG API is not reachable at {self.valves.DATAPAI_RAG_API_URL}. "
                f"Please ensure the RAG service is running."
            )
            if self.valves.DATAPAI_RAG_FALLBACK:
                # Fall back: answer without RAG context
                return self._fallback_answer(user_message, messages)
            return msg

        except Exception as exc:
            if self.valves.DATAPAI_RAG_FALLBACK:
                return self._fallback_answer(user_message, messages)
            return f"⚠️ RAG pipeline error: {exc}"

    # ── Fallback: plain Ollama (no RAG) ───────────────────────────────────

    def _fallback_answer(self, user_message: str, messages: List[dict]) -> str:
        """
        Answer directly via Ollama without RAG context.
        Used when the RAG API is unreachable and DATAPAI_RAG_FALLBACK=true.
        """
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model = self.valves.DATAPAI_RAG_MODEL

        try:
            resp = requests.post(
                f"{ollama_host}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                },
                timeout=120,
            )
            resp.raise_for_status()
            answer = resp.json().get("message", {}).get("content", "No answer.")
            return f"*(RAG unavailable — answering from model memory only)*\n\n{answer}"
        except Exception as exc:
            return f"⚠️ Both RAG API and Ollama fallback failed: {exc}"

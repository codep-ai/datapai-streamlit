"""
DataPAI Smart Router Pipeline for OpenWebUI
=============================================
A single "model" inside OpenWebUI that intelligently routes each message to:

  • SQL pipeline  — if the question is about data / metrics / reports
  • RAG pipeline  — if the question needs document / knowledge-base lookup
  • Ollama chat   — for everything else (general assistant)

HOW IT WORKS
────────────
1. First, a tiny Ollama call (~100ms) classifies the question:
      "sql"  → POST /v1/sql/query on EC2 #2 (Text2SQL API, port 8101)
      "rag"  → POST /v1/rag/retrieve on EC2 #2, then Ollama for generation
      "chat" → POST /api/chat on OLLAMA_HOST directly
2. The routed call is streamed back to OpenWebUI.

If the classifier call fails or times out, the pipeline falls back to the
keyword heuristic (no extra latency on failure).

INSTALLATION
────────────
1. Start both backend services on EC2 #2:
     uvicorn agents.rag_api:app      --host 0.0.0.0 --port 8100
     uvicorn agents.text2sql_api:app --host 0.0.0.0 --port 8101

2. In OpenWebUI → Admin → Pipelines → Add Pipeline: paste this file.

3. Set environment variables:
     OLLAMA_HOST              http://localhost:11434  (or EC2 #3 private IP)
     DATAPAI_RAG_API_URL      http://localhost:8100
     DATAPAI_SQL_API_URL      http://localhost:8101
     DATAPAI_ROUTER_MODEL     Small/fast Ollama model for routing  (default: llama3.2)
     DATAPAI_CHAT_MODEL       Model for general chat answers        (default: llama3.2)
     DATAPAI_RAG_MODEL        Model for RAG generation              (default: llama3.2)
     DATAPAI_SQL_DEFAULT_DB   Default SQL target DB                 (default: Snowflake)
     DATAPAI_RAG_API_KEY      Bearer token for RAG API   (empty = no auth)
     DATAPAI_SQL_API_KEY      Bearer token for SQL API   (empty = no auth)

ENVIRONMENT VARIABLES (all configurable as Valves in OpenWebUI UI)
────────────────────────────────────────────────────────────────────
  OLLAMA_HOST              Ollama endpoint for generation
  DATAPAI_RAG_API_URL      RAG FastAPI service URL (EC2 #2)
  DATAPAI_SQL_API_URL      Text2SQL FastAPI service URL (EC2 #2)
  DATAPAI_ROUTER_MODEL     Model used to classify intent (small = fast)
  DATAPAI_CHAT_MODEL       Model for plain chat answers
  DATAPAI_RAG_MODEL        Model for RAG-augmented answers
  DATAPAI_SQL_DEFAULT_DB   Default target database for SQL queries
  DATAPAI_RAG_TOP_K        Documents to retrieve for RAG (default: 5)
  DATAPAI_SQL_RUN_SQL      Execute the generated SQL (default: true)
  DATAPAI_SQL_MAX_ROWS     Max rows to show in SQL results (default: 50)
  DATAPAI_RAG_API_KEY      RAG API bearer token
  DATAPAI_SQL_API_KEY      SQL API bearer token
"""

from __future__ import annotations

import json
import os
import re
from typing import Generator, List, Optional, Union

import requests
from pydantic import BaseModel


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class Pipeline:
    """
    DataPAI Smart Router — routes questions to SQL, RAG, or plain chat.
    Appears as "DataPAI Smart Router" in the OpenWebUI model selector.
    """

    class Valves(BaseModel):
        # Shared
        OLLAMA_HOST:            str  = os.getenv("OLLAMA_HOST",             "http://localhost:11434")
        DATAPAI_RAG_API_URL:    str  = os.getenv("DATAPAI_RAG_API_URL",     "http://localhost:8100")
        DATAPAI_SQL_API_URL:    str  = os.getenv("DATAPAI_SQL_API_URL",     "http://localhost:8101")

        # Models
        DATAPAI_ROUTER_MODEL:   str  = os.getenv("DATAPAI_ROUTER_MODEL",    "llama3.2")
        DATAPAI_CHAT_MODEL:     str  = os.getenv("DATAPAI_CHAT_MODEL",      "llama3.2")
        DATAPAI_RAG_MODEL:      str  = os.getenv("DATAPAI_RAG_MODEL",       "llama3.2")

        # SQL
        DATAPAI_SQL_DEFAULT_DB: str  = os.getenv("DATAPAI_SQL_DEFAULT_DB",  "Snowflake")
        DATAPAI_SQL_RUN_SQL:    bool = os.getenv("DATAPAI_SQL_RUN_SQL",     "true").lower() != "false"
        DATAPAI_SQL_MAX_ROWS:   int  = int(os.getenv("DATAPAI_SQL_MAX_ROWS", "50"))

        # RAG
        DATAPAI_RAG_TOP_K:      int  = int(os.getenv("DATAPAI_RAG_TOP_K",   "5"))

        # Auth
        DATAPAI_RAG_API_KEY:    str  = os.getenv("DATAPAI_RAG_API_KEY",     "")
        DATAPAI_SQL_API_KEY:    str  = os.getenv("DATAPAI_SQL_API_KEY",     "")

    def __init__(self):
        self.name   = "DataPAI Smart Router"
        self.valves = self.Valves()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def on_startup(self):
        checks = [
            (f"{self.valves.DATAPAI_RAG_API_URL}/health", "RAG API"),
            (f"{self.valves.DATAPAI_SQL_API_URL}/health", "SQL API"),
            (f"{self.valves.OLLAMA_HOST}/api/tags",        "Ollama"),
        ]
        for url, label in checks:
            try:
                requests.get(url, timeout=5).raise_for_status()
                print(f"[DataPAI Router] ✓ {label} reachable")
            except Exception as exc:
                print(f"[DataPAI Router] ⚠ {label} not reachable: {exc}")

    async def on_shutdown(self):
        print("[DataPAI Router] Pipeline shutdown.")

    async def on_valves_updated(self):
        print(
            f"[DataPAI Router] Config updated — "
            f"RAG: {self.valves.DATAPAI_RAG_API_URL}  "
            f"SQL: {self.valves.DATAPAI_SQL_API_URL}  "
            f"Ollama: {self.valves.OLLAMA_HOST}"
        )

    # ── Main entrypoint ────────────────────────────────────────────────────

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Route the message to SQL, RAG, or plain chat, then stream the answer.
        """
        # Check for explicit DB override tag (e.g. "show revenue [db:Redshift]")
        db_override = self._parse_db_tag(user_message)
        clean_msg   = re.sub(r"\[db:\w+\]", "", user_message, flags=re.IGNORECASE).strip()

        # Classify intent
        route = self._classify(clean_msg)

        if route == "sql" or db_override:
            return self._route_sql(clean_msg, db_override)
        elif route == "rag":
            return self._route_rag(clean_msg, messages)
        else:
            return self._route_chat(messages)

    # ── Intent classifier ──────────────────────────────────────────────────

    # SQL-intent keywords (heuristic fallback)
    _SQL_KEYWORDS = re.compile(
        r"\b(select|from|where|group by|order by|join|having|count|sum|avg|max|min"
        r"|revenue|sales|orders|customers|products|metrics|kpi|report|dashboard"
        r"|breakdown|by (week|month|year|region|category|country|channel)"
        r"|how many|total|top \d+|bottom \d+|trend|growth|rate|volume"
        r"|compare|vs\.?|versus|over time|year over year|yoy|mom|wow"
        r"|run sql|query|table|database|schema)\b",
        re.IGNORECASE,
    )

    # RAG-intent keywords (heuristic fallback)
    _RAG_KEYWORDS = re.compile(
        r"\b(what is|what are|explain|describe|tell me about|documentation"
        r"|policy|process|procedure|how to|guide|tutorial|knowledge base"
        r"|our (data|schema|model|pipeline|architecture)|according to"
        r"|based on|in the docs?|from the docs?)\b",
        re.IGNORECASE,
    )

    def _classify(self, message: str) -> str:
        """
        Returns 'sql', 'rag', or 'chat'.
        Tries LLM classifier first; falls back to keyword heuristic.
        """
        # Try LLM classifier (fast, small model)
        try:
            resp = requests.post(
                f"{self.valves.OLLAMA_HOST}/api/chat",
                json={
                    "model":   self.valves.DATAPAI_ROUTER_MODEL,
                    "messages": [
                        {
                            "role":    "system",
                            "content": (
                                "You are a router. Classify the user question as exactly one of:\n"
                                "  sql  — requires a database query or data analysis\n"
                                "  rag  — requires document/knowledge-base lookup\n"
                                "  chat — general assistant question\n"
                                "Reply with ONE word only: sql, rag, or chat."
                            ),
                        },
                        {"role": "user", "content": message},
                    ],
                    "stream": False,
                    "options": {"num_predict": 5, "temperature": 0},
                },
                timeout=8,
            )
            resp.raise_for_status()
            label = resp.json().get("message", {}).get("content", "").strip().lower()
            if label in ("sql", "rag", "chat"):
                return label
        except Exception:
            pass  # fall through to heuristic

        # Keyword heuristic
        if self._SQL_KEYWORDS.search(message):
            return "sql"
        if self._RAG_KEYWORDS.search(message):
            return "rag"
        return "chat"

    # ── Routing handlers ───────────────────────────────────────────────────

    def _parse_db_tag(self, message: str) -> Optional[str]:
        m = re.search(r"\[db:(\w+)\]", message, re.IGNORECASE)
        return m.group(1) if m else None

    def _sql_headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.valves.DATAPAI_SQL_API_KEY:
            h["Authorization"] = f"Bearer {self.valves.DATAPAI_SQL_API_KEY}"
        return h

    def _rag_headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.valves.DATAPAI_RAG_API_KEY:
            h["Authorization"] = f"Bearer {self.valves.DATAPAI_RAG_API_KEY}"
        return h

    def _route_sql(self, question: str, db_override: Optional[str]) -> str:
        """Call the Text2SQL API and return formatted markdown."""
        db = db_override or self.valves.DATAPAI_SQL_DEFAULT_DB
        try:
            resp = requests.post(
                f"{self.valves.DATAPAI_SQL_API_URL}/v1/sql/query",
                json={
                    "question":       question,
                    "db":             db,
                    "run_sql":        self.valves.DATAPAI_SQL_RUN_SQL,
                    "generate_chart": False,
                    "generate_dbt":   False,
                },
                headers=self._sql_headers(),
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            return (
                f"⚠️ Text2SQL API not reachable at `{self.valves.DATAPAI_SQL_API_URL}`.\n\n"
                f"Please start it on EC2 #2:\n"
                f"```bash\nuvicorn agents.text2sql_api:app --host 0.0.0.0 --port 8101\n```"
            )
        except Exception as exc:
            return f"⚠️ SQL routing error: {exc}"

        return self._format_sql_answer(data)

    def _format_sql_answer(self, data: dict) -> str:
        lines: List[str] = []
        sql    = data.get("sql", "")
        db     = data.get("db", "")
        rows   = data.get("rows") or []
        count  = data.get("row_count")
        summ   = data.get("summary", "")
        follows = data.get("followup_questions") or []
        err    = data.get("error")
        valid  = data.get("is_valid", True)

        if sql:
            lines.append(f"```sql\n-- Target: {db}\n{sql}\n```")
        if not valid:
            lines.append(
                f"\n⚠️ **SQL validation warning** — review before running in production."
            )
        if err:
            lines.append(f"\n❌ **Execution error:** `{err}`")
            return "\n".join(lines)
        if rows:
            label = f"{count} row{'s' if count != 1 else ''}" if count else "Results"
            lines.append(f"\n**{label}:**\n")
            lines.append(self._markdown_table(rows, self.valves.DATAPAI_SQL_MAX_ROWS))
        if summ:
            lines.append(f"\n**Summary:** {summ}")
        if follows:
            lines.append("\n**Suggested follow-up questions:**")
            for q in follows[:4]:
                lines.append(f"- {q}")
        return "\n".join(lines)

    def _markdown_table(self, rows: List[dict], max_rows: int) -> str:
        if not rows:
            return "_No rows returned._"
        trimmed = rows[:max_rows]
        headers  = list(trimmed[0].keys())
        header_row = "| " + " | ".join(str(h) for h in headers) + " |"
        sep_row    = "| " + " | ".join("---" for _ in headers) + " |"
        data_rows  = [
            "| " + " | ".join(str(r.get(h, "")) for h in headers) + " |"
            for r in trimmed
        ]
        table = "\n".join([header_row, sep_row] + data_rows)
        if len(rows) > max_rows:
            table += f"\n\n_Showing {max_rows} of {len(rows)} rows._"
        return table

    def _route_rag(
        self,
        question: str,
        messages: List[dict],
    ) -> Generator[str, None, None]:
        """Retrieve from LanceDB, then stream generation from Ollama."""
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in messages[:-1]
            if m.get("role") in ("user", "assistant")
        ]

        # Step 1 — retrieve context (no LLM call)
        try:
            r = requests.post(
                f"{self.valves.DATAPAI_RAG_API_URL}/v1/rag/retrieve",
                json={
                    "question":     question,
                    "k":            self.valves.DATAPAI_RAG_TOP_K,
                    "chat_history": chat_history,
                },
                headers=self._rag_headers(),
                timeout=30,
            )
            r.raise_for_status()
            rd = r.json()
            openai_messages = rd.get("openai_messages", [])
            sources         = rd.get("sources", [])
        except requests.exceptions.ConnectionError:
            yield (
                f"⚠️ RAG API not reachable at `{self.valves.DATAPAI_RAG_API_URL}`. "
                f"Falling back to plain chat.\n\n"
            )
            yield from self._route_chat(messages)
            return
        except Exception as exc:
            yield f"⚠️ RAG retrieve error: {exc}\n\n"
            yield from self._route_chat(messages)
            return

        # Step 2 — stream generation from Ollama
        try:
            with requests.post(
                f"{self.valves.OLLAMA_HOST}/api/chat",
                json={
                    "model":    self.valves.DATAPAI_RAG_MODEL,
                    "messages": openai_messages,
                    "stream":   True,
                },
                stream=True,
                timeout=300,
            ) as gen_resp:
                gen_resp.raise_for_status()
                for raw_line in gen_resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except Exception as exc:
            yield (
                f"\n\n⚠️ Ollama not reachable at `{self.valves.OLLAMA_HOST}`. "
                f"Error: `{exc}`"
            )

        # Citations
        if sources:
            lines = ["\n\n---\n📎 **Sources from knowledge base:**"]
            for src in sources:
                name = src.get("filename", "?")
                coll = src.get("collection", "?")
                uri  = src.get("source_uri", "")
                lines.append(f"- **{name}** [{coll}]  `{uri}`")
            yield "\n".join(lines)

    def _route_chat(
        self,
        messages: List[dict],
    ) -> Generator[str, None, None]:
        """Stream a plain Ollama chat answer (no RAG, no SQL)."""
        try:
            with requests.post(
                f"{self.valves.OLLAMA_HOST}/api/chat",
                json={
                    "model":    self.valves.DATAPAI_CHAT_MODEL,
                    "messages": messages,
                    "stream":   True,
                },
                stream=True,
                timeout=300,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
        except Exception as exc:
            yield (
                f"⚠️ Ollama not reachable at `{self.valves.OLLAMA_HOST}`.\n\n"
                f"Error: `{exc}`\n\n"
                f"Please ensure Ollama is running."
            )

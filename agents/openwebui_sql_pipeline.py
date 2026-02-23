"""
DataPAI Text2SQL Pipeline for OpenWebUI
=========================================
Adds natural language → SQL as a model inside OpenWebUI.

HOW IT WORKS
────────────
User asks a business question in plain English.
Pipeline calls the DataPAI Text2SQL API (port 8101) which:
  1. Generates SQL via Vanna (RAG-based SQL generation)
  2. Validates the SQL
  3. Runs the SQL against the target database
  4. Summarises the result in plain English
  5. Generates follow-up question suggestions

The answer is returned as a formatted markdown table + summary.

INSTALLATION
────────────
1. Start the Text2SQL API on EC2 #2:
     uvicorn agents.text2sql_api:app --host 0.0.0.0 --port 8101

2. In OpenWebUI → Admin → Pipelines → Add Pipeline:
     Upload this file.

3. Set Pipeline environment variables:
     DATAPAI_SQL_API_URL = http://localhost:8101
     DATAPAI_SQL_API_KEY = <your key>      (if SQL_API_KEY is set)
     DATAPAI_SQL_DEFAULT_DB = Snowflake    (default target DB)

4. The model "DataPAI Text2SQL" appears in the OpenWebUI model selector.

ENVIRONMENT VARIABLES
─────────────────────
  DATAPAI_SQL_API_URL     URL of the Text2SQL FastAPI service
                          default: http://localhost:8101
  DATAPAI_SQL_API_KEY     Bearer token (if SQL_API_KEY is configured)
                          default: (empty — no auth)
  DATAPAI_SQL_DEFAULT_DB  Default target database
                          default: Snowflake
                          options: Snowflake, Redshift, DuckDB, SQLite3,
                                   Athena, BigQuery, dbt
  DATAPAI_SQL_RUN_SQL     Execute the generated SQL (True/False)
                          default: true
  DATAPAI_SQL_MAX_ROWS    Max rows to show in the table
                          default: 50
"""

from __future__ import annotations

import os
from typing import List, Optional, Union

import requests
from pydantic import BaseModel


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class Pipeline:
    """
    DataPAI Text2SQL — natural language → SQL → results inside OpenWebUI.

    Appears as "DataPAI Text2SQL" in the model selector.
    Supports multiple target databases via the DB valve.
    """

    class Valves(BaseModel):
        DATAPAI_SQL_API_URL:      str  = os.getenv("DATAPAI_SQL_API_URL",      "http://localhost:8101")
        DATAPAI_SQL_API_KEY:      str  = os.getenv("DATAPAI_SQL_API_KEY",      "")
        DATAPAI_SQL_DEFAULT_DB:   str  = os.getenv("DATAPAI_SQL_DEFAULT_DB",   "Snowflake")
        DATAPAI_SQL_RUN_SQL:      bool = os.getenv("DATAPAI_SQL_RUN_SQL",      "true").lower() != "false"
        DATAPAI_SQL_MAX_ROWS:     int  = int(os.getenv("DATAPAI_SQL_MAX_ROWS", "50"))
        DATAPAI_SQL_GENERATE_DBT: bool = os.getenv("DATAPAI_SQL_GENERATE_DBT", "false").lower() == "true"

    def __init__(self):
        self.name   = "DataPAI Text2SQL"
        self.valves = self.Valves()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def on_startup(self):
        try:
            r = requests.get(
                f"{self.valves.DATAPAI_SQL_API_URL}/health", timeout=5
            )
            r.raise_for_status()
            h = r.json()
            print(
                f"[DataPAI SQL] Connected — "
                f"vanna_model: {h.get('vanna_model')}  "
                f"supported_dbs: {h.get('supported_dbs', [])}"
            )
        except Exception as exc:
            print(f"[DataPAI SQL] ⚠ Text2SQL API not reachable at startup: {exc}")

    async def on_shutdown(self):
        print("[DataPAI SQL] Pipeline shutdown.")

    async def on_valves_updated(self):
        print(
            f"[DataPAI SQL] Config updated — "
            f"API: {self.valves.DATAPAI_SQL_API_URL}  "
            f"Default DB: {self.valves.DATAPAI_SQL_DEFAULT_DB}"
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.valves.DATAPAI_SQL_API_KEY:
            h["Authorization"] = f"Bearer {self.valves.DATAPAI_SQL_API_KEY}"
        return h

    def _parse_db_from_message(self, message: str) -> Optional[str]:
        """
        Allow user to override the target DB inline, e.g.:
          "show me sales by region [db:Redshift]"
        Returns the DB name if found, else None.
        """
        import re
        m = re.search(r"\[db:(\w+)\]", message, re.IGNORECASE)
        if m:
            return m.group(1)
        return None

    def _format_table(self, rows: List[dict], max_rows: int) -> str:
        """Render dict records as a markdown table."""
        if not rows:
            return "_No rows returned._"
        trimmed = rows[:max_rows]
        headers = list(trimmed[0].keys())
        header_row = "| " + " | ".join(str(h) for h in headers) + " |"
        sep_row    = "| " + " | ".join("---" for _ in headers) + " |"
        data_rows  = [
            "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |"
            for row in trimmed
        ]
        table = "\n".join([header_row, sep_row] + data_rows)
        if len(rows) > max_rows:
            table += f"\n\n_Showing {max_rows} of {len(rows)} rows._"
        return table

    def _format_answer(self, resp_data: dict) -> str:
        """Build the full markdown answer from the API response."""
        lines: List[str] = []

        sql      = resp_data.get("sql", "")
        db       = resp_data.get("db", "")
        rows     = resp_data.get("rows") or []
        count    = resp_data.get("row_count")
        summ     = resp_data.get("summary", "")
        follows  = resp_data.get("followup_questions") or []
        dbt_code = resp_data.get("dbt_code")
        err      = resp_data.get("error")
        valid    = resp_data.get("is_valid", True)

        # ── SQL block ──────────────────────────────────────────────────
        if sql:
            lines.append(f"```sql\n-- Target: {db}\n{sql}\n```")

        # ── Validation warning ─────────────────────────────────────────
        if not valid:
            lines.append(
                f"\n⚠️ **SQL validation warning** — the query may not be valid for `{db}`. "
                f"Review before running in production."
            )

        # ── Error ──────────────────────────────────────────────────────
        if err:
            lines.append(f"\n❌ **Execution error:** `{err}`")
            return "\n".join(lines)

        # ── Results table ──────────────────────────────────────────────
        if rows:
            row_label = f"{count} row{'s' if count != 1 else ''}" if count else "Results"
            lines.append(f"\n**{row_label}:**\n")
            lines.append(self._format_table(rows, self.valves.DATAPAI_SQL_MAX_ROWS))

        # ── Summary ────────────────────────────────────────────────────
        if summ:
            lines.append(f"\n**Summary:** {summ}")

        # ── Follow-up suggestions ──────────────────────────────────────
        if follows:
            lines.append("\n**Suggested follow-up questions:**")
            for q in follows[:4]:
                lines.append(f"- {q}")

        # ── dbt model code ─────────────────────────────────────────────
        if dbt_code:
            lines.append(f"\n**dbt model:**\n```sql\n{dbt_code}\n```")

        return "\n".join(lines)

    # ── Main entrypoint ────────────────────────────────────────────────────

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, None]:
        """
        Convert a natural language question to SQL, run it, and return results.

        DB selection (in priority order):
          1. Inline tag in message:  "show sales [db:Redshift]"
          2. Valve default:          DATAPAI_SQL_DEFAULT_DB
        """
        # Detect optional DB override in message
        db = self._parse_db_from_message(user_message) or self.valves.DATAPAI_SQL_DEFAULT_DB

        # Strip the [db:...] tag from the actual question
        import re
        clean_question = re.sub(r"\[db:\w+\]", "", user_message, flags=re.IGNORECASE).strip()

        payload = {
            "question":       clean_question,
            "db":             db,
            "run_sql":        self.valves.DATAPAI_SQL_RUN_SQL,
            "generate_chart": False,
            "generate_dbt":   self.valves.DATAPAI_SQL_GENERATE_DBT,
        }

        try:
            resp = requests.post(
                f"{self.valves.DATAPAI_SQL_API_URL}/v1/sql/query",
                json=payload,
                headers=self._headers(),
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

        except requests.exceptions.ConnectionError:
            return (
                f"⚠️ DataPAI Text2SQL API is not reachable at "
                f"`{self.valves.DATAPAI_SQL_API_URL}`.\n\n"
                f"Please ensure `agents/text2sql_api.py` is running on EC2 #2:\n"
                f"```bash\nuvicorn agents.text2sql_api:app --host 0.0.0.0 --port 8101\n```"
            )
        except requests.exceptions.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", str(exc))
            except Exception:
                detail = str(exc)
            return f"⚠️ Text2SQL API error: {detail}"
        except Exception as exc:
            return f"⚠️ Unexpected error: {exc}"

        return self._format_answer(data)

"""
ETL Audit System — Monitor & Refine

Full observability for every ETL agent and GenAI execution on data.

What gets captured automatically:
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Pipeline run     one row per run_etl_pipeline() call                  │
  │  Agent steps      every tool call: timing, success/failure, preview    │
  │  Data mutations   every DuckDB write: rows before/after, schema change │
  │  LLM calls        tokens, latency, cost, guardrail actions             │
  │  Compliance events PII detected/masked, policy violations              │
  └─────────────────────────────────────────────────────────────────────────┘

All data stored in:
  DuckDB tables  (same database as the pipeline — queryable by dbt/SQL)
  JSONL file     (append-only backup, path = ETL_AUDIT_LOG env var)

Monitoring & Refinement API:
  get_pipeline_runs(db, n)      recent runs with status + cost
  get_tool_performance(db)      success rate + p50/p95 latency per tool
  get_data_mutations(db, run_id) table-level change log for a run
  get_llm_cost_report(db)       cost breakdown by provider/model/run
  get_compliance_events(db)     PII detection + masking history
  get_failure_analysis(db)      failed steps → root cause → suggested fix
  get_refinement_report(db)     actionable recommendations to improve the pipeline

Usage:
    from agents.etl.audit import AuditLogger, audit_tool

    # In pipeline.py (done automatically by run_etl_pipeline):
    logger = AuditLogger(run_id="abc-123", db_path="datapai.duckdb")
    register_logger("abc-123", logger)

    # Wrap any SwarmResult tool:
    @audit_tool("quality_agent")
    def profile_table(context_variables: dict) -> SwarmResult:
        ...

    # After the run:
    report = get_refinement_report("datapai.duckdb")
    print(report)
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import duckdb
import pandas as pd
from autogen import SwarmResult

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
DB_PATH = os.getenv("DUCKDB_PATH", "datapai.duckdb")
AUDIT_LOG_PATH = os.getenv("ETL_AUDIT_LOG", "etl_audit.jsonl")

# ── Token cost table (input_per_1M, output_per_1M) in USD ────────────────────
_TOKEN_COSTS: dict[str, tuple[float, float]] = {
    "claude-3-5-sonnet-20241022":          (3.00, 15.00),
    "claude-3-5-sonnet-20241022-v2:0":     (3.00, 15.00),
    "anthropic.claude-3-5-sonnet-20241022-v2:0": (3.00, 15.00),
    "claude-3-haiku-20240307":             (0.25,  1.25),
    "gpt-4o":                              (5.00, 15.00),
    "gpt-4o-mini":                         (0.15,  0.60),
    "gpt-4-turbo":                         (10.0,  30.00),
    "llama3.1":                            (0.00,   0.00),   # local
    "llama3":                              (0.00,   0.00),   # local
    "mistral":                             (0.00,   0.00),   # local
}

_TOOL_FIX_HINTS: dict[str, str] = {
    "ingest_file":               "Verify file path, format (.csv/.xlsx/.parquet), and DuckDB write permissions.",
    "profile_table":             "Ensure table was loaded before profiling. Check DuckDB connection.",
    "check_primary_key":         "Column may not exist — verify schema after ingest_file succeeds.",
    "scan_pii":                  "Table may be empty or DuckDB connection failed. Run ingest_file first.",
    "mask_pii_columns":          "DuckDB write permission error or column type mismatch.",
    "generate_compliance_report":"Missing pii_scan context — ensure scan_pii ran first.",
    "write_audit_log":           "Check write permissions for audit JSONL file and DuckDB.",
    "generate_staging_sql":      "Missing table_name or schema — ensure ingest_file completed.",
    "generate_schema_yaml":      "Missing schema or model_name context — run generate_staging_sql first.",
    "save_dbt_artifacts":        "dbt project directory may not exist or not be writable.",
    "get_pipeline_summary":      "Context variables incomplete — check earlier pipeline steps.",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Schema initialisation
# ═══════════════════════════════════════════════════════════════════════════════

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS etl_pipeline_runs (
    run_id                VARCHAR,
    started_at            VARCHAR,
    completed_at          VARCHAR,
    duration_ms           BIGINT,
    status                VARCHAR,
    source_file           VARCHAR,
    table_name            VARCHAR,
    row_count             BIGINT,
    llm_provider          VARCHAR,
    llm_model             VARCHAR,
    compliance_status     VARCHAR,
    pii_columns_count     INTEGER,
    masked_columns_count  INTEGER,
    total_steps           INTEGER,
    failed_steps          INTEGER,
    total_tokens          INTEGER,
    estimated_cost_usd    DOUBLE,
    environment           VARCHAR
);

CREATE TABLE IF NOT EXISTS etl_agent_steps (
    step_id        VARCHAR,
    run_id         VARCHAR,
    sequence_num   INTEGER,
    agent_name     VARCHAR,
    tool_name      VARCHAR,
    tool_args_json VARCHAR,
    result_preview VARCHAR,
    success        BOOLEAN,
    error_message  VARCHAR,
    duration_ms    BIGINT,
    started_at     VARCHAR,
    completed_at   VARCHAR
);

CREATE TABLE IF NOT EXISTS etl_data_mutations (
    mutation_id    VARCHAR,
    run_id         VARCHAR,
    step_id        VARCHAR,
    table_name     VARCHAR,
    operation      VARCHAR,
    rows_before    BIGINT,
    rows_after     BIGINT,
    rows_affected  BIGINT,
    schema_changed BOOLEAN,
    agent_name     VARCHAR,
    tool_name      VARCHAR,
    timestamp      VARCHAR
);

CREATE TABLE IF NOT EXISTS etl_llm_calls (
    call_id                VARCHAR,
    run_id                 VARCHAR,
    agent_name             VARCHAR,
    provider               VARCHAR,
    model                  VARCHAR,
    prompt_tokens          INTEGER,
    completion_tokens      INTEGER,
    total_tokens           INTEGER,
    estimated_cost_usd     DOUBLE,
    latency_ms             BIGINT,
    guardrail_l1_applied   BOOLEAN,
    guardrail_l2_action    VARCHAR,
    guardrail_l3_findings  VARCHAR,
    timestamp              VARCHAR
);

CREATE TABLE IF NOT EXISTS etl_compliance_events (
    event_id     VARCHAR,
    run_id       VARCHAR,
    event_type   VARCHAR,
    column_name  VARCHAR,
    sensitivity  VARCHAR,
    pii_type     VARCHAR,
    action_taken VARCHAR,
    details_json VARCHAR,
    timestamp    VARCHAR
);
"""


def _init_schema(con: duckdb.DuckDBPyConnection) -> None:
    for stmt in _SCHEMA_SQL.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            con.execute(stmt)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    costs = _TOKEN_COSTS.get(model, (5.0, 15.0))
    return (prompt_tokens * costs[0] + completion_tokens * costs[1]) / 1_000_000


# ═══════════════════════════════════════════════════════════════════════════════
# AuditLogger — central event sink
# ═══════════════════════════════════════════════════════════════════════════════

class AuditLogger:
    """
    Central audit event sink for one pipeline run.

    Writes to two sinks simultaneously:
      1. DuckDB tables (queryable, dashboardable)
      2. JSONL file    (append-only backup, never modified)

    All inputs are sanitised (no raw PII values stored).
    """

    def __init__(
        self,
        run_id: str,
        db_path: str = DB_PATH,
        jsonl_path: str = AUDIT_LOG_PATH,
        llm_provider: str = "unknown",
        llm_model: str = "unknown",
    ) -> None:
        self.run_id = run_id
        self.db_path = db_path
        self.jsonl_path = Path(jsonl_path)
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self._step_counter = 0
        self._started_at = _now()
        self._steps_logged: list[dict] = []

        # Initialise schema
        try:
            con = duckdb.connect(db_path)
            _init_schema(con)
            con.close()
        except Exception as exc:
            logger.warning("AuditLogger: schema init failed (non-fatal): %s", exc)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _write_jsonl(self, record: dict) -> None:
        try:
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning("AuditLogger: JSONL write failed: %s", exc)

    def _exec(self, sql: str, params: list) -> None:
        try:
            con = duckdb.connect(self.db_path)
            con.execute(sql, params)
            con.close()
        except Exception as exc:
            logger.warning("AuditLogger: DuckDB write failed: %s\n  SQL: %s", exc, sql)

    # ── Step logging ───────────────────────────────────────────────────────

    def log_step(
        self,
        agent_name: str,
        tool_name: str,
        tool_args: dict,
        result_preview: str,
        success: bool,
        duration_ms: int,
        started_at: str,
        error_message: Optional[str] = None,
    ) -> str:
        """Log a single tool execution. Returns the step_id."""
        self._step_counter += 1
        step_id = f"{self.run_id}:{self._step_counter}"
        completed_at = _now()

        # Truncate for storage
        safe_args = json.dumps({k: str(v)[:200] for k, v in tool_args.items()})
        safe_preview = result_preview[:500] if result_preview else ""

        record = {
            "step_id": step_id,
            "run_id": self.run_id,
            "sequence_num": self._step_counter,
            "agent_name": agent_name,
            "tool_name": tool_name,
            "tool_args_json": safe_args,
            "result_preview": safe_preview,
            "success": success,
            "error_message": error_message,
            "duration_ms": duration_ms,
            "started_at": started_at,
            "completed_at": completed_at,
        }

        self._steps_logged.append(record)
        self._write_jsonl({"type": "agent_step", **record})
        self._exec(
            """INSERT INTO etl_agent_steps VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                step_id, self.run_id, self._step_counter,
                agent_name, tool_name, safe_args, safe_preview,
                success, error_message, duration_ms, started_at, completed_at,
            ],
        )

        if not success:
            logger.warning(
                "[AUDIT] Step FAILED  run=%s  tool=%s  error=%s  duration=%dms",
                self.run_id, tool_name, error_message, duration_ms,
            )
        else:
            logger.debug(
                "[AUDIT] Step OK  run=%s  tool=%s  duration=%dms",
                self.run_id, tool_name, duration_ms,
            )

        return step_id

    # ── Data mutation logging ──────────────────────────────────────────────

    def log_mutation(
        self,
        table_name: str,
        operation: str,
        rows_before: int,
        rows_after: int,
        schema_changed: bool,
        agent_name: str,
        tool_name: str,
        step_id: str = "",
    ) -> None:
        """Log a DuckDB table write operation."""
        mutation_id = str(uuid.uuid4())
        rows_affected = abs(rows_after - rows_before)
        ts = _now()

        record = {
            "mutation_id": mutation_id,
            "run_id": self.run_id,
            "step_id": step_id,
            "table_name": table_name,
            "operation": operation,
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_affected": rows_affected,
            "schema_changed": schema_changed,
            "agent_name": agent_name,
            "tool_name": tool_name,
            "timestamp": ts,
        }

        self._write_jsonl({"type": "data_mutation", **record})
        self._exec(
            "INSERT INTO etl_data_mutations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                mutation_id, self.run_id, step_id, table_name, operation,
                rows_before, rows_after, rows_affected,
                schema_changed, agent_name, tool_name, ts,
            ],
        )

        logger.info(
            "[AUDIT] Mutation  table=%s  op=%s  rows %d→%d  (Δ%d)",
            table_name, operation, rows_before, rows_after, rows_affected,
        )

    # ── LLM call logging ───────────────────────────────────────────────────

    def log_llm_call(
        self,
        agent_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        guardrail_l1_applied: bool = False,
        guardrail_l2_action: str = "NONE",
        guardrail_l3_findings: Optional[list] = None,
    ) -> None:
        """Log one LLM API call with token usage and guardrail metadata."""
        call_id = str(uuid.uuid4())
        total_tokens = prompt_tokens + completion_tokens
        cost = _estimate_cost(self.llm_model, prompt_tokens, completion_tokens)
        ts = _now()
        findings_str = json.dumps(guardrail_l3_findings or [])

        record = {
            "call_id": call_id,
            "run_id": self.run_id,
            "agent_name": agent_name,
            "provider": self.llm_provider,
            "model": self.llm_model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": cost,
            "latency_ms": latency_ms,
            "guardrail_l1_applied": guardrail_l1_applied,
            "guardrail_l2_action": guardrail_l2_action,
            "guardrail_l3_findings": findings_str,
            "timestamp": ts,
        }

        self._write_jsonl({"type": "llm_call", **record})
        self._exec(
            "INSERT INTO etl_llm_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                call_id, self.run_id, agent_name, self.llm_provider, self.llm_model,
                prompt_tokens, completion_tokens, total_tokens, cost,
                latency_ms, guardrail_l1_applied, guardrail_l2_action, findings_str, ts,
            ],
        )

    # ── Compliance event logging ───────────────────────────────────────────

    def log_compliance_event(
        self,
        event_type: str,
        column_name: str = "",
        sensitivity: str = "",
        pii_type: str = "",
        action_taken: str = "",
        details: Optional[dict] = None,
    ) -> None:
        """
        Log a compliance event.

        event_type: PII_DETECTED | MASKING_APPLIED | POLICY_VIOLATION |
                    AUDIT_WRITTEN | GUARDRAIL_INTERVENED | REVIEW_REQUIRED
        """
        event_id = str(uuid.uuid4())
        ts = _now()
        details_str = json.dumps(details or {})

        record = {
            "event_id": event_id,
            "run_id": self.run_id,
            "event_type": event_type,
            "column_name": column_name,
            "sensitivity": sensitivity,
            "pii_type": pii_type,
            "action_taken": action_taken,
            "details_json": details_str,
            "timestamp": ts,
        }

        self._write_jsonl({"type": "compliance_event", **record})
        self._exec(
            "INSERT INTO etl_compliance_events VALUES (?,?,?,?,?,?,?,?,?)",
            [
                event_id, self.run_id, event_type,
                column_name, sensitivity, pii_type,
                action_taken, details_str, ts,
            ],
        )

        logger.info(
            "[AUDIT] Compliance  event=%s  col=%s  sensitivity=%s  action=%s",
            event_type, column_name, sensitivity, action_taken,
        )

    # ── Pipeline run finalisation ──────────────────────────────────────────

    def finalise(self, context_variables: dict) -> None:
        """
        Write the pipeline-level summary row to etl_pipeline_runs.
        Call this at the end of run_etl_pipeline().
        """
        completed_at = _now()
        started_dt = datetime.fromisoformat(self._started_at)
        completed_dt = datetime.fromisoformat(completed_at)
        duration_ms = int((completed_dt - started_dt).total_seconds() * 1000)

        failed_steps = sum(1 for s in self._steps_logged if not s["success"])

        # Aggregate LLM cost from etl_llm_calls
        total_tokens = 0
        total_cost = 0.0
        try:
            con = duckdb.connect(self.db_path)
            row = con.execute(
                "SELECT SUM(total_tokens), SUM(estimated_cost_usd) "
                "FROM etl_llm_calls WHERE run_id = ?",
                [self.run_id],
            ).fetchone()
            con.close()
            if row and row[0]:
                total_tokens = int(row[0])
                total_cost = float(row[1])
        except Exception:
            pass

        record = {
            "run_id": self.run_id,
            "started_at": self._started_at,
            "completed_at": completed_at,
            "duration_ms": duration_ms,
            "status": context_variables.get("pipeline_status", "incomplete"),
            "source_file": context_variables.get("file_path", "unknown"),
            "table_name": context_variables.get("table_name", "unknown"),
            "row_count": context_variables.get("row_count", 0),
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "compliance_status": context_variables.get("compliance_status", "UNKNOWN"),
            "pii_columns_count": len(context_variables.get("pii_columns", [])),
            "masked_columns_count": len(context_variables.get("masked_columns", [])),
            "total_steps": self._step_counter,
            "failed_steps": failed_steps,
            "total_tokens": total_tokens,
            "estimated_cost_usd": total_cost,
            "environment": os.getenv("DATAPAI_ENV", "dev"),
        }

        self._write_jsonl({"type": "pipeline_run", **record})
        self._exec(
            "INSERT INTO etl_pipeline_runs VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                record["run_id"], record["started_at"], record["completed_at"],
                record["duration_ms"], record["status"],
                record["source_file"], record["table_name"], record["row_count"],
                record["llm_provider"], record["llm_model"],
                record["compliance_status"],
                record["pii_columns_count"], record["masked_columns_count"],
                record["total_steps"], record["failed_steps"],
                record["total_tokens"], record["estimated_cost_usd"],
                record["environment"],
            ],
        )

        logger.info(
            "[AUDIT] Pipeline finalised  run=%s  status=%s  "
            "steps=%d  failed=%d  cost=$%.4f  duration=%dms",
            self.run_id, record["status"], self._step_counter,
            failed_steps, total_cost, duration_ms,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level logger registry (keyed by run_id)
# ═══════════════════════════════════════════════════════════════════════════════

_ACTIVE_LOGGERS: dict[str, AuditLogger] = {}


def register_logger(run_id: str, audit_logger: AuditLogger) -> None:
    _ACTIVE_LOGGERS[run_id] = audit_logger


def unregister_logger(run_id: str) -> None:
    _ACTIVE_LOGGERS.pop(run_id, None)


def get_active_logger(run_id: str) -> Optional[AuditLogger]:
    return _ACTIVE_LOGGERS.get(run_id)


# ═══════════════════════════════════════════════════════════════════════════════
# @audit_tool decorator
# ═══════════════════════════════════════════════════════════════════════════════

def audit_tool(agent_name: str) -> Callable:
    """
    Decorator that automatically audits any SwarmResult tool function
    AND enforces the cost budget after every execution.

    Captures (written to etl_agent_steps):
      - timing (duration_ms)
      - success / failure
      - tool arguments (sanitised, first 200 chars per arg)
      - result preview (first 500 chars of SwarmResult.values)
      - error message on exception

    Cost enforcement (after each successful tool call):
      - Looks up the CostController via the '_cost_controller' key in
        the module-level _ACTIVE_CONTROLLERS registry (keyed by run_id).
      - Calls CostController.check_mid_run() which compares current spend
        against per-run / per-day / per-month limits.
      - On warning: appends a cost warning to the SwarmResult values.
      - On hard stop: raises CostBudgetExceeded, terminating the pipeline.

    The decorator looks up both registries via context_variables['_audit_run_id'].

    Usage (applied at registration time in pipeline.py — keeps tool code clean):
        audited = audit_tool("ingest_agent")(ingest_file)
        agent.register_for_execution()(audited)
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(context_variables: dict, *args, **kwargs) -> SwarmResult:
            # Late import to avoid circular dependency at module load time
            from .cost_control import CostBudgetExceeded  # noqa: PLC0415

            run_id = (
                context_variables.get("_audit_run_id")
                or context_variables.get("compliance_run_id", "")
            )
            audit       = _ACTIVE_LOGGERS.get(run_id)
            cost_ctrl   = _ACTIVE_CONTROLLERS.get(run_id)

            started_at = _now()
            t0 = time.monotonic()

            # Sanitise kwargs (no raw PII values in audit log)
            safe_args = {k: str(v)[:200] for k, v in kwargs.items()}

            try:
                result: SwarmResult = fn(context_variables, *args, **kwargs)
                success   = True
                error_msg = None
                preview   = str(getattr(result, "values", ""))[:500]
            except CostBudgetExceeded:
                # Re-raise immediately — do not swallow budget violations
                raise
            except Exception as exc:
                success   = False
                error_msg = str(exc)
                preview   = ""
                result = SwarmResult(
                    values=f"[{fn.__name__} failed: {exc}]",
                    context_variables=context_variables,
                )

            duration_ms = int((time.monotonic() - t0) * 1000)

            # ── Audit log ─────────────────────────────────────────────────
            if audit:
                audit.log_step(
                    agent_name=agent_name,
                    tool_name=fn.__name__,
                    tool_args=safe_args,
                    result_preview=preview,
                    success=success,
                    duration_ms=duration_ms,
                    started_at=started_at,
                    error_message=error_msg,
                )

            # ── Cost budget check (after every tool, raises on hard stop) ─
            if cost_ctrl and success:
                warning = cost_ctrl.check_mid_run()   # raises CostBudgetExceeded if over limit
                if warning:
                    # Append cost warning to the result so the orchestrator sees it
                    existing = getattr(result, "values", "")
                    result = SwarmResult(
                        values=f"{existing}\n\n{warning}",
                        context_variables=result.context_variables
                        if hasattr(result, "context_variables")
                        else context_variables,
                    )

            return result

        return wrapper
    return decorator


# Module-level CostController registry (parallel to _ACTIVE_LOGGERS)
_ACTIVE_CONTROLLERS: dict[str, Any] = {}


def register_controller(run_id: str, controller: Any) -> None:
    _ACTIVE_CONTROLLERS[run_id] = controller


def unregister_controller(run_id: str) -> None:
    _ACTIVE_CONTROLLERS.pop(run_id, None)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM usage parser — extract token counts from AG2 chat_result
# ═══════════════════════════════════════════════════════════════════════════════

def parse_and_log_llm_usage(
    chat_result: Any,
    audit: AuditLogger,
) -> dict:
    """
    Extract LLM token usage from an AG2 ChatResult and log it.

    AG2 stores cost in chat_result.cost as:
      {
        "usage_including_cached_inference": {
          "total_cost": 0.01,
          "<model>": {"cost": ..., "prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
        }
      }

    Returns a summary dict with totals.
    """
    summary = {"total_tokens": 0, "total_cost_usd": 0.0, "models_used": []}

    try:
        cost_info = getattr(chat_result, "cost", None)
        if not cost_info:
            return summary

        usage = cost_info.get("usage_including_cached_inference", {})
        for key, val in usage.items():
            if key == "total_cost" or not isinstance(val, dict):
                continue

            model = key
            prompt_tokens = val.get("prompt_tokens", 0)
            completion_tokens = val.get("completion_tokens", 0)
            total_tokens = val.get("total_tokens", prompt_tokens + completion_tokens)
            cost = val.get("cost", _estimate_cost(model, prompt_tokens, completion_tokens))

            audit.log_llm_call(
                agent_name="swarm",   # aggregate — not per-agent at this level
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=0,         # not available at aggregate level
            )

            summary["total_tokens"] += total_tokens
            summary["total_cost_usd"] += cost
            summary["models_used"].append(model)

    except Exception as exc:
        logger.warning("parse_and_log_llm_usage: could not parse chat_result.cost: %s", exc)

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Monitoring queries — return pandas DataFrames
# ═══════════════════════════════════════════════════════════════════════════════

def _query(db_path: str, sql: str, params: Optional[list] = None) -> pd.DataFrame:
    try:
        con = duckdb.connect(db_path)
        df = con.execute(sql, params or []).fetchdf()
        con.close()
        return df
    except Exception as exc:
        logger.warning("audit query failed: %s\nSQL: %s", exc, sql)
        return pd.DataFrame()


def get_pipeline_runs(
    db_path: str = DB_PATH,
    n: int = 20,
) -> pd.DataFrame:
    """
    Return the N most recent pipeline runs.

    Columns: run_id, started_at, duration_ms, status, table_name, row_count,
             llm_provider, compliance_status, total_steps, failed_steps,
             total_tokens, estimated_cost_usd
    """
    return _query(
        db_path,
        """
        SELECT run_id,
               started_at,
               duration_ms,
               status,
               table_name,
               row_count,
               llm_provider,
               llm_model,
               compliance_status,
               total_steps,
               failed_steps,
               total_tokens,
               ROUND(estimated_cost_usd, 6) AS estimated_cost_usd
        FROM   etl_pipeline_runs
        ORDER  BY started_at DESC
        LIMIT  ?
        """,
        [n],
    )


def get_tool_performance(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Return per-tool success rate and latency percentiles across all runs.

    Columns: tool_name, total_calls, success_rate_pct, avg_ms, p50_ms, p95_ms
    """
    return _query(
        db_path,
        """
        SELECT tool_name,
               COUNT(*)                                          AS total_calls,
               ROUND(AVG(CAST(success AS INTEGER)) * 100, 1)    AS success_rate_pct,
               ROUND(AVG(duration_ms))                          AS avg_ms,
               ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP
                     (ORDER BY duration_ms))                    AS p50_ms,
               ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP
                     (ORDER BY duration_ms))                    AS p95_ms
        FROM   etl_agent_steps
        GROUP  BY tool_name
        ORDER  BY total_calls DESC
        """,
    )


def get_data_mutations(
    db_path: str = DB_PATH,
    run_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return data mutation history.  Filter to a specific run_id if supplied.

    Columns: timestamp, run_id, table_name, operation, rows_before,
             rows_after, rows_affected, schema_changed, agent_name, tool_name
    """
    if run_id:
        return _query(
            db_path,
            """
            SELECT timestamp, run_id, table_name, operation,
                   rows_before, rows_after, rows_affected,
                   schema_changed, agent_name, tool_name
            FROM   etl_data_mutations
            WHERE  run_id = ?
            ORDER  BY timestamp ASC
            """,
            [run_id],
        )
    return _query(
        db_path,
        """
        SELECT timestamp, run_id, table_name, operation,
               rows_before, rows_after, rows_affected,
               schema_changed, agent_name, tool_name
        FROM   etl_data_mutations
        ORDER  BY timestamp DESC
        LIMIT  100
        """,
    )


def get_llm_cost_report(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    LLM cost breakdown by provider and model.

    Columns: provider, model, total_calls, total_tokens,
             total_cost_usd, avg_cost_per_call_usd
    """
    return _query(
        db_path,
        """
        SELECT provider,
               model,
               COUNT(*)                                AS total_calls,
               SUM(total_tokens)                       AS total_tokens,
               ROUND(SUM(estimated_cost_usd), 6)       AS total_cost_usd,
               ROUND(AVG(estimated_cost_usd), 6)       AS avg_cost_per_call_usd
        FROM   etl_llm_calls
        GROUP  BY provider, model
        ORDER  BY total_cost_usd DESC
        """,
    )


def get_compliance_events(
    db_path: str = DB_PATH,
    run_id: Optional[str] = None,
    event_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compliance event history with optional filters.

    Columns: timestamp, run_id, event_type, column_name, sensitivity,
             pii_type, action_taken
    """
    where_clauses = []
    params = []
    if run_id:
        where_clauses.append("run_id = ?")
        params.append(run_id)
    if event_type:
        where_clauses.append("event_type = ?")
        params.append(event_type)
    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    return _query(
        db_path,
        f"""
        SELECT timestamp, run_id, event_type, column_name,
               sensitivity, pii_type, action_taken
        FROM   etl_compliance_events
        {where}
        ORDER  BY timestamp DESC
        LIMIT  200
        """,
        params,
    )


def get_pii_trends(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    PII detection frequency by column name — helps refine detection heuristics.

    Columns: column_name, pii_type, sensitivity, detection_count,
             masked_count, mask_rate_pct
    """
    return _query(
        db_path,
        """
        SELECT column_name,
               pii_type,
               sensitivity,
               COUNT(*) FILTER (WHERE event_type = 'PII_DETECTED')  AS detection_count,
               COUNT(*) FILTER (WHERE event_type = 'MASKING_APPLIED') AS masked_count,
               ROUND(
                   COUNT(*) FILTER (WHERE event_type = 'MASKING_APPLIED') * 100.0 /
                   NULLIF(COUNT(*) FILTER (WHERE event_type = 'PII_DETECTED'), 0),
               1) AS mask_rate_pct
        FROM   etl_compliance_events
        WHERE  event_type IN ('PII_DETECTED', 'MASKING_APPLIED')
          AND  column_name != ''
        GROUP  BY column_name, pii_type, sensitivity
        ORDER  BY detection_count DESC
        """,
    )


def get_failure_analysis(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Failed tool executions with root-cause grouping and suggested fix.

    Columns: tool_name, failure_count, last_seen, sample_error, suggested_fix
    """
    df = _query(
        db_path,
        """
        SELECT tool_name,
               COUNT(*)        AS failure_count,
               MAX(started_at) AS last_seen,
               FIRST(error_message) AS sample_error
        FROM   etl_agent_steps
        WHERE  success = false
          AND  error_message IS NOT NULL
        GROUP  BY tool_name
        ORDER  BY failure_count DESC
        """,
    )

    if not df.empty:
        df["suggested_fix"] = df["tool_name"].map(
            lambda t: _TOOL_FIX_HINTS.get(t, "Review tool logs and DuckDB connection.")
        )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Refinement report — actionable improvement summary
# ═══════════════════════════════════════════════════════════════════════════════

def get_refinement_report(db_path: str = DB_PATH) -> str:
    """
    Generate a human-readable refinement report across all pipeline runs.

    Identifies:
      1. Reliability issues   — tools with success rate < 95%
      2. Performance issues   — tools with p95 latency > 10s
      3. Cost outliers        — runs costing more than 2× the median
      4. Compliance gaps      — columns frequently left unmasked
      5. PII blind spots      — common column names not caught by current heuristics
      6. Data quality patterns — most common quality issues across runs

    Returns a plain-text report suitable for printing or storing.
    """
    lines = [
        "═" * 70,
        "ETL AGENT REFINEMENT REPORT",
        f"Generated: {_now()}",
        "═" * 70,
    ]

    # 1. Reliability
    perf = get_tool_performance(db_path)
    if not perf.empty:
        low_reliability = perf[perf["success_rate_pct"] < 95]
        lines.append("\n── RELIABILITY (tools with success rate < 95%) ──────────────────────")
        if low_reliability.empty:
            lines.append("  ✓ All tools performing above 95% success rate")
        else:
            for _, row in low_reliability.iterrows():
                hint = _TOOL_FIX_HINTS.get(row["tool_name"], "Review logs.")
                lines.append(
                    f"  ⚠ {row['tool_name']:<35} "
                    f"{row['success_rate_pct']}% success  "
                    f"({row['total_calls']} calls)"
                )
                lines.append(f"    → {hint}")

    # 2. Performance
    if not perf.empty:
        slow_tools = perf[perf["p95_ms"] > 10_000]
        lines.append("\n── PERFORMANCE (tools with p95 latency > 10s) ──────────────────────")
        if slow_tools.empty:
            lines.append("  ✓ No slow tool executions detected")
        else:
            for _, row in slow_tools.iterrows():
                lines.append(
                    f"  ⚠ {row['tool_name']:<35} "
                    f"p95={row['p95_ms']}ms  avg={row['avg_ms']}ms"
                )
            lines.append(
                "  → Consider: caching schema results, sampling larger "
                "files before profiling, or chunked ingestion."
            )

    # 3. Cost
    runs = get_pipeline_runs(db_path, n=100)
    if not runs.empty and "estimated_cost_usd" in runs.columns:
        median_cost = runs["estimated_cost_usd"].median()
        expensive = runs[runs["estimated_cost_usd"] > median_cost * 2]
        lines.append("\n── COST (runs costing > 2× median) ─────────────────────────────────")
        lines.append(f"  Median run cost: ${median_cost:.4f}")
        if expensive.empty:
            lines.append("  ✓ No cost outliers detected")
        else:
            for _, row in expensive.head(5).iterrows():
                lines.append(
                    f"  ⚠ run={row['run_id'][:16]}…  "
                    f"${row['estimated_cost_usd']:.4f}  "
                    f"table={row['table_name']}  "
                    f"steps={row['total_steps']}"
                )
            lines.append(
                "  → Consider: using Ollama/Bedrock for large files, "
                "reducing max_rounds, or caching repeated schema lookups."
            )

    # 4. Compliance gaps
    events = get_compliance_events(db_path, event_type="POLICY_VIOLATION")
    lines.append("\n── COMPLIANCE GAPS (unmasked high-risk columns) ─────────────────────")
    if events.empty:
        lines.append("  ✓ No policy violations recorded")
    else:
        for _, row in events.head(10).iterrows():
            lines.append(
                f"  ⚠ col={row['column_name']}  "
                f"sensitivity={row['sensitivity']}  "
                f"run={row['run_id'][:16]}…"
            )
        lines.append(
            "  → Enable auto-masking for PII_HIGH/FINANCIAL in the compliance_agent "
            "or add column to the whitelist with documented lawful basis."
        )

    # 5. PII trends
    pii_df = get_pii_trends(db_path)
    if not pii_df.empty:
        low_mask = pii_df[pii_df["mask_rate_pct"].fillna(0) < 50]
        lines.append("\n── PII MASKING GAPS (detected but masked < 50% of the time) ─────────")
        if low_mask.empty:
            lines.append("  ✓ Detected PII is being masked consistently")
        else:
            for _, row in low_mask.iterrows():
                lines.append(
                    f"  ⚠ col={row['column_name']:<30} "
                    f"type={row['pii_type']:<15} "
                    f"detected={row['detection_count']}  "
                    f"masked={row['masked_count']}  "
                    f"({row['mask_rate_pct']}%)"
                )

    # 6. Failure analysis
    failures = get_failure_analysis(db_path)
    if not failures.empty:
        lines.append("\n── TOP FAILURES (tool error analysis) ───────────────────────────────")
        for _, row in failures.head(5).iterrows():
            lines.append(f"  ✗ {row['tool_name']}  ×{row['failure_count']}  last={row['last_seen']}")
            lines.append(f"    Error:  {str(row['sample_error'])[:100]}")
            lines.append(f"    Fix:    {row['suggested_fix']}")

    lines += [
        "",
        "═" * 70,
        "Run get_pipeline_runs() / get_tool_performance() for raw data.",
        "═" * 70,
    ]

    return "\n".join(lines)

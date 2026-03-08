"""
traceability/etl_bridge.py

Bridge between agents/etl/audit.py (DuckDB-based ETL audit) and the
Datap.ai Trace Ledger.

When the ETL pipeline runs, AuditLogger captures tool-level events in
DuckDB + JSONL.  This bridge mirrors those events into the Trace Ledger
so that:

  1. ETL runs appear in the unified governance timeline.
  2. Admin console can search ETL events alongside Text2SQL events.
  3. dbt-traceability models (and Lightdash) can report on ETL activity.
  4. The `etl_run_id` field in trace events links back to DuckDB audit tables
     for deep-dive debugging.

Usage — wrap run_etl_pipeline() with bridge_etl_run():

    from traceability.etl_bridge import bridge_etl_run
    from traceability import get_ledger
    from traceability.models import IdentityContext

    identity = IdentityContext.from_env()
    result = bridge_etl_run(
        identity   = identity,
        request    = "ingest sales_q4.csv",
        run_fn     = run_etl_pipeline,      # the original function
        run_kwargs = {"request": "ingest sales_q4.csv"},
    )

Or use the decorator form:

    @trace_etl_pipeline(identity=IdentityContext.from_env())
    def my_pipeline(...):
        ...

Design:
  - Non-invasive: agents/etl/audit.py is NOT modified.
  - Bridge calls get_ledger() at runtime — backend is from env vars.
  - All trace events carry etl_run_id so they link back to DuckDB tables.
  - Graceful fallback: if the trace ledger fails, the ETL pipeline continues.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Optional

from traceability.ledger import get_ledger
from traceability.models import (
    EventType,
    ActorType,
    TraceStatus,
    IdentityContext,
)

log = logging.getLogger(__name__)


def bridge_etl_run(
    *,
    identity:   IdentityContext,
    request:    str,
    run_fn:     Callable,
    run_kwargs: Optional[dict] = None,
) -> dict:
    """
    Run an ETL pipeline function while emitting trace events to the ledger.

    Emits:
      - REQUEST_RECEIVED  at start
      - TOOL_INVOKED      for each tool step (if audit data available post-run)
      - RESPONSE_RETURNED at end (success or failure)

    Returns the original run_fn result dict.
    """
    ledger     = get_ledger()
    run_kwargs = run_kwargs or {}
    request_id = _new_request_id()

    # ── Request received ───────────────────────────────────────────────────
    parent_trace_id = ledger.emit_request_received(
        identity   = identity,
        request_id = request_id,
        input_text = request,
        actor_id   = identity.user_id,
    )

    t0 = time.monotonic()
    status      = TraceStatus.OK
    error_msg   = None
    result: dict = {}

    try:
        result = run_fn(**run_kwargs)
    except Exception as exc:
        status    = TraceStatus.FAILED
        error_msg = str(exc)
        log.error("bridge_etl_run: ETL pipeline failed: %s", exc)
        raise
    finally:
        duration_ms = int((time.monotonic() - t0) * 1000)
        etl_run_id  = result.get("audit_run_id") if isinstance(result, dict) else None

        # ── Response returned ──────────────────────────────────────────────
        try:
            ledger.emit(
                identity        = identity,
                event_type      = EventType.RESPONSE_RETURNED,
                actor_type      = ActorType.SYSTEM,
                actor_id        = "etl_pipeline",
                request_id      = request_id,
                parent_trace_id = parent_trace_id,
                output_text     = (
                    result.get("compliance_status", "") if isinstance(result, dict) else ""
                ),
                status          = status,
                error_message   = error_msg,
                etl_run_id      = etl_run_id,
            )

            # ── Mirror step-level events if audit data available ───────────
            if etl_run_id and isinstance(result, dict):
                _mirror_etl_steps(
                    identity    = identity,
                    request_id  = request_id,
                    etl_run_id  = etl_run_id,
                    result      = result,
                    ledger_ref  = ledger,
                    parent_trace_id = parent_trace_id,
                )

        except Exception as exc:
            log.warning("bridge_etl_run: trace emit failed (non-fatal): %s", exc)

    return result


def _mirror_etl_steps(
    *,
    identity:        IdentityContext,
    request_id:      str,
    etl_run_id:      str,
    result:          dict,
    ledger_ref:      Any,
    parent_trace_id: str,
) -> None:
    """
    Read step data from the AuditLogger result context and emit TOOL_INVOKED
    events for each step.  This connects ETL tool calls to the trace ledger.
    """
    # The run_etl_pipeline() result dict typically includes:
    #   cost_report, compliance_status, pipeline_status, ...
    # Step-level data is in DuckDB — we emit a summary tool event per pipeline.
    compliance_status = result.get("compliance_status", "UNKNOWN")
    pipeline_status   = result.get("pipeline_status", "unknown")
    cost_report       = result.get("cost_report", {})

    ledger_ref.emit(
        identity        = identity,
        event_type      = EventType.TOOL_INVOKED,
        actor_type      = ActorType.TOOL,
        actor_id        = "etl_agent_swarm",
        request_id      = request_id,
        parent_trace_id = parent_trace_id,
        tool_name       = "run_etl_pipeline",
        output_text     = f"compliance={compliance_status} status={pipeline_status}",
        status          = (
            TraceStatus.OK if pipeline_status in ("completed", "ok")
            else TraceStatus.FAILED
        ),
        etl_run_id      = etl_run_id,
    )

    # If SQL was generated during ETL, emit a SQL_GENERATED event
    generated_sql = result.get("generated_sql") or result.get("staging_sql")
    if generated_sql:
        ledger_ref.emit_sql_generated(
            identity        = identity,
            request_id      = request_id,
            sql_text        = str(generated_sql),
            model_name      = result.get("llm_model", "unknown"),
            datasource_name = "duckdb_etl",
            datasource_type = "duckdb",
            parent_trace_id = parent_trace_id,
            etl_run_id      = etl_run_id,
        )


def trace_etl_pipeline(
    identity:   IdentityContext,
    request_key: str = "request",
) -> Callable:
    """
    Decorator factory.  Wraps an ETL pipeline function with trace emission.

    Usage:
        @trace_etl_pipeline(identity=IdentityContext.from_env())
        def run_etl_pipeline(request: str, ...) -> dict:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> dict:
            request_text = kwargs.get(request_key) or (args[0] if args else "")
            return bridge_etl_run(
                identity   = identity,
                request    = str(request_text),
                run_fn     = fn,
                run_kwargs = dict(zip(fn.__code__.co_varnames, args), **kwargs),
            )
        return wrapper
    return decorator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _new_request_id() -> str:
    import uuid
    return str(uuid.uuid4())

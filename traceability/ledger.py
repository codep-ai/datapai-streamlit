"""
traceability/ledger.py

TraceLedger — the central, storage-agnostic trace ledger for Datap.ai.

Every meaningful AI action in the platform calls TraceLedger.emit(...) or
one of the convenience methods.  The ledger:

  1. Applies redaction (never stores raw secrets / PII).
  2. Delegates persistence to whichever backend is configured.
  3. Is safe to call from any thread (Streamlit, AG2 swarms, FastAPI, etc.).
  4. Degrades gracefully — if the backend is unavailable, it logs the error
     and continues so the calling service is not blocked.

Factory:
  ledger = TraceLedger.from_env()   # reads DATAPAI_TRACE_BACKEND env var

Supported backends (set DATAPAI_TRACE_BACKEND):
  sqlite     — local development (default)
  snowflake  — cloud production
  null       — disabled / testing

Usage:
    from traceability import get_ledger, EventType, ActorType
    from traceability.models import IdentityContext

    identity = IdentityContext(
        tenant_id="acme",
        workspace_id="analytics",
        user_id="alice",
        session_id="sess-123",
    )
    ledger = get_ledger()

    trace_id = ledger.emit(
        identity     = identity,
        event_type   = EventType.REQUEST_RECEIVED,
        actor_type   = ActorType.USER,
        actor_id     = "alice",
        input_summary = "Show me revenue by region last quarter",
    )
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from traceability.backends import TraceLedgerBackend, NullTraceLedgerBackend
from traceability.models import (
    EventType,
    ActorType,
    IdentityContext,
    TraceEvent,
    TraceStatus,
)
from traceability.redaction import summarise, hash_payload, safe_sql_summary

log = logging.getLogger(__name__)


class TraceLedger:
    """
    Storage-agnostic trace ledger.

    Instantiate via TraceLedger.from_env() or pass a backend directly.
    All public methods are thread-safe.
    """

    def __init__(self, backend: TraceLedgerBackend) -> None:
        self._backend = backend
        self._lock    = threading.Lock()

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "TraceLedger":
        """
        Build a TraceLedger from environment variables.

        DATAPAI_TRACE_BACKEND controls the backend:
          sqlite     — SQLiteTraceLedgerBackend (default)
          snowflake  — SnowflakeTraceLedgerBackend
          null       — NullTraceLedgerBackend (disabled)
        """
        backend_name = os.getenv("DATAPAI_TRACE_BACKEND", "sqlite").lower()

        if backend_name == "null" or os.getenv("DATAPAI_TRACE_ENABLED", "true").lower() == "false":
            backend: TraceLedgerBackend = NullTraceLedgerBackend()

        elif backend_name == "snowflake":
            from traceability.backends.snowflake_backend import SnowflakeTraceLedgerBackend
            backend = SnowflakeTraceLedgerBackend()

        else:  # default: sqlite
            from traceability.backends.sqlite_backend import SQLiteTraceLedgerBackend
            backend = SQLiteTraceLedgerBackend()

        try:
            backend.initialise()
        except Exception as exc:
            log.warning(
                "TraceLedger: backend %s initialisation failed (%s). "
                "Falling back to NullTraceLedgerBackend.",
                backend_name, exc,
            )
            backend = NullTraceLedgerBackend()

        return cls(backend)

    # ── Core emit ─────────────────────────────────────────────────────────

    def emit(
        self,
        *,
        identity:        IdentityContext,
        event_type:      EventType,
        actor_type:      ActorType        = ActorType.SYSTEM,
        actor_id:        str              = "system",
        request_id:      str              = "",
        parent_trace_id: Optional[str]   = None,
        datasource_type: Optional[str]   = None,
        datasource_name: Optional[str]   = None,
        model_name:      Optional[str]   = None,
        tool_name:       Optional[str]   = None,
        policy_result:   Optional[str]   = None,
        input_text:      Optional[str]   = None,
        output_text:     Optional[str]   = None,
        sql_text:        Optional[str]   = None,
        prompt_text:     Optional[str]   = None,
        context_refs:    Optional[str]   = None,
        status:          TraceStatus     = TraceStatus.OK,
        error_code:      Optional[str]   = None,
        error_message:   Optional[str]   = None,
        etl_run_id:      Optional[str]   = None,
    ) -> str:
        """
        Emit a trace event and return its trace_id.

        - input_text / output_text are summarised + secrets masked before storage.
        - sql_text is stored as a structural summary + SHA-256 hash.
        - prompt_text is stored as SHA-256 hash only (no raw prompt stored).
        - Never raises — errors are logged and swallowed.
        """
        event = TraceEvent.new(
            identity        = identity,
            event_type      = event_type,
            actor_type      = actor_type,
            actor_id        = actor_id,
            request_id      = request_id,
            parent_trace_id = parent_trace_id,
            datasource_type = datasource_type,
            datasource_name = datasource_name,
            model_name      = model_name,
            tool_name       = tool_name,
            policy_result   = policy_result,
            input_summary   = summarise(input_text),
            output_summary  = summarise(output_text),
            sql_text        = sql_text,           # hashed inside TraceEvent.new()
            prompt_text     = prompt_text,        # hashed inside TraceEvent.new()
            context_refs    = context_refs,
            status          = status,
            error_code      = error_code,
            error_message   = summarise(error_message, max_len=300) if error_message else None,
            etl_run_id      = etl_run_id,
        )

        # Override sql_hash with structural summary approach
        if sql_text:
            event = _replace_sql_summary(event, sql_text)

        try:
            with self._lock:
                self._backend.append(event.to_dict())
        except Exception as exc:
            log.error("TraceLedger.emit failed: %s | event_type=%s", exc, event_type)

        return event.trace_id

    # ── Convenience methods ───────────────────────────────────────────────

    def emit_request_received(
        self,
        identity:     IdentityContext,
        request_id:   str,
        input_text:   Optional[str] = None,
        actor_id:     str           = "",
    ) -> str:
        return self.emit(
            identity    = identity,
            event_type  = EventType.REQUEST_RECEIVED,
            actor_type  = ActorType.USER,
            actor_id    = actor_id or identity.user_id,
            request_id  = request_id,
            input_text  = input_text,
        )

    def emit_sql_generated(
        self,
        identity:        IdentityContext,
        request_id:      str,
        sql_text:        str,
        model_name:      Optional[str] = None,
        datasource_name: Optional[str] = None,
        datasource_type: Optional[str] = None,
        parent_trace_id: Optional[str] = None,
        etl_run_id:      Optional[str] = None,
    ) -> str:
        return self.emit(
            identity        = identity,
            event_type      = EventType.SQL_GENERATED,
            actor_type      = ActorType.ASSISTANT,
            actor_id        = model_name or "llm",
            request_id      = request_id,
            parent_trace_id = parent_trace_id,
            model_name      = model_name,
            datasource_type = datasource_type,
            datasource_name = datasource_name,
            sql_text        = sql_text,
            output_text     = safe_sql_summary(sql_text),
            etl_run_id      = etl_run_id,
        )

    def emit_sql_validated(
        self,
        identity:        IdentityContext,
        request_id:      str,
        sql_text:        str,
        policy_result:   str,
        status:          TraceStatus     = TraceStatus.OK,
        error_message:   Optional[str]  = None,
        parent_trace_id: Optional[str]  = None,
        etl_run_id:      Optional[str]  = None,
    ) -> str:
        event_type = (
            EventType.SQL_VALIDATED if status == TraceStatus.OK
            else EventType.SQL_BLOCKED
        )
        return self.emit(
            identity        = identity,
            event_type      = event_type,
            actor_type      = ActorType.SYSTEM,
            actor_id        = "sql_validator",
            request_id      = request_id,
            parent_trace_id = parent_trace_id,
            sql_text        = sql_text,
            policy_result   = policy_result,
            status          = status,
            error_message   = error_message,
            etl_run_id      = etl_run_id,
        )

    def emit_sql_executed(
        self,
        identity:        IdentityContext,
        request_id:      str,
        sql_text:        str,
        datasource_name: Optional[str] = None,
        datasource_type: Optional[str] = None,
        output_summary:  Optional[str] = None,
        status:          TraceStatus   = TraceStatus.OK,
        error_message:   Optional[str] = None,
        parent_trace_id: Optional[str] = None,
        etl_run_id:      Optional[str] = None,
    ) -> str:
        return self.emit(
            identity        = identity,
            event_type      = EventType.SQL_EXECUTED,
            actor_type      = ActorType.TOOL,
            actor_id        = "sql_executor",
            request_id      = request_id,
            parent_trace_id = parent_trace_id,
            datasource_type = datasource_type,
            datasource_name = datasource_name,
            sql_text        = sql_text,
            output_text     = output_summary,
            status          = status,
            error_message   = error_message,
            etl_run_id      = etl_run_id,
        )

    def emit_policy_check(
        self,
        identity:        IdentityContext,
        request_id:      str,
        policy_result:   str,
        passed:          bool,
        reason:          Optional[str]  = None,
        parent_trace_id: Optional[str] = None,
        etl_run_id:      Optional[str] = None,
    ) -> str:
        event_type = (
            EventType.POLICY_CHECK_PASSED if passed
            else EventType.POLICY_CHECK_FAILED
        )
        return self.emit(
            identity        = identity,
            event_type      = event_type,
            actor_type      = ActorType.SYSTEM,
            actor_id        = "policy_engine",
            request_id      = request_id,
            parent_trace_id = parent_trace_id,
            policy_result   = policy_result,
            output_text     = reason,
            status          = TraceStatus.OK if passed else TraceStatus.BLOCKED,
            etl_run_id      = etl_run_id,
        )

    def emit_model_invoked(
        self,
        identity:        IdentityContext,
        request_id:      str,
        model_name:      str,
        prompt_text:     Optional[str]  = None,
        output_text:     Optional[str]  = None,
        parent_trace_id: Optional[str] = None,
        etl_run_id:      Optional[str] = None,
    ) -> str:
        return self.emit(
            identity        = identity,
            event_type      = EventType.MODEL_INVOKED,
            actor_type      = ActorType.ASSISTANT,
            actor_id        = model_name,
            request_id      = request_id,
            parent_trace_id = parent_trace_id,
            model_name      = model_name,
            prompt_text     = prompt_text,
            output_text     = output_text,
            etl_run_id      = etl_run_id,
        )

    def emit_tool_invoked(
        self,
        identity:        IdentityContext,
        request_id:      str,
        tool_name:       str,
        input_text:      Optional[str]  = None,
        output_text:     Optional[str]  = None,
        status:          TraceStatus    = TraceStatus.OK,
        error_message:   Optional[str]  = None,
        parent_trace_id: Optional[str] = None,
        etl_run_id:      Optional[str] = None,
    ) -> str:
        return self.emit(
            identity        = identity,
            event_type      = EventType.TOOL_INVOKED,
            actor_type      = ActorType.TOOL,
            actor_id        = tool_name,
            request_id      = request_id,
            parent_trace_id = parent_trace_id,
            tool_name       = tool_name,
            input_text      = input_text,
            output_text     = output_text,
            status          = status,
            error_message   = error_message,
            etl_run_id      = etl_run_id,
        )

    def emit_human_feedback(
        self,
        identity:        IdentityContext,
        request_id:      str,
        feedback_text:   str,
        parent_trace_id: Optional[str] = None,
    ) -> str:
        return self.emit(
            identity        = identity,
            event_type      = EventType.HUMAN_FEEDBACK_RECEIVED,
            actor_type      = ActorType.USER,
            actor_id        = identity.user_id,
            request_id      = request_id,
            parent_trace_id = parent_trace_id,
            input_text      = feedback_text,
        )

    # ── Read / query ──────────────────────────────────────────────────────

    def get_session_timeline(
        self,
        tenant_id:  str,
        session_id: str,
        limit:      int = 200,
    ) -> list[dict]:
        """Return all trace events for a session, ordered ASC by timestamp."""
        return self._backend.fetch_by_session(tenant_id, session_id, limit)

    def get_request_timeline(self, tenant_id: str, request_id: str) -> list[dict]:
        """Return all trace events for a single request."""
        return self._backend.fetch_by_request(tenant_id, request_id)

    def get_event(self, trace_id: str) -> Optional[dict]:
        """Fetch a single trace event by trace_id."""
        rows = self._backend.fetch_by_trace_id(trace_id)
        return rows[0] if rows else None

    def search(self, **kwargs) -> list[dict]:
        """Flexible search — pass any TraceLedgerBackend.search() kwargs."""
        return self._backend.search(**kwargs)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def close(self) -> None:
        self._backend.close()


# ── Module-level singleton ──────────────────────────────────────────────────

_ledger: Optional[TraceLedger] = None
_ledger_lock = threading.Lock()


def get_ledger() -> TraceLedger:
    """
    Return the module-level TraceLedger singleton.
    Initialised from environment variables on first call.
    Thread-safe.
    """
    global _ledger
    if _ledger is None:
        with _ledger_lock:
            if _ledger is None:
                _ledger = TraceLedger.from_env()
    return _ledger


def reset_ledger(backend: Optional[TraceLedgerBackend] = None) -> None:
    """
    Reset the module-level singleton.  Pass a backend for testing.
    Closes the existing backend connection first.
    """
    global _ledger
    with _ledger_lock:
        if _ledger is not None:
            _ledger.close()
        _ledger = TraceLedger(backend) if backend else None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _replace_sql_summary(event: TraceEvent, sql_text: str) -> TraceEvent:
    """
    Replace the event's output_summary with the structural SQL summary
    (which is more useful for replay/debug than a generic text summary).
    """
    from dataclasses import replace
    return replace(event, output_summary=safe_sql_summary(sql_text))

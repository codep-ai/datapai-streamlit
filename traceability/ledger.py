"""
traceability/ledger.py

TraceLedger — storage-agnostic compliance trace ledger for Datap.ai.

Compliance behaviour:
  - question_text stored verbatim (credentials masked, content preserved)
  - sql_text stored verbatim (the query, NOT the result rows)
  - sensitivity_level, pii_detected, pii_fields stored per event
  - ai_action_summary records what the AI did, not what data it returned
  - prompt_text stored as hash only (full prompts may contain system instructions)

Backend selection via DATAPAI_TRACE_BACKEND:
  sqlite     — local development (default)
  snowflake  — cloud production
  null       — disabled / testing
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

from traceability.backends import TraceLedgerBackend, NullTraceLedgerBackend
from traceability.models import (
    EventType, ActorType, IdentityContext, TraceEvent, TraceStatus,
)
from traceability.redaction import mask_credentials_only, summarise

log = logging.getLogger(__name__)


class TraceLedger:
    """Storage-agnostic compliance trace ledger. All methods are thread-safe."""

    def __init__(self, backend: TraceLedgerBackend) -> None:
        self._backend = backend
        self._lock    = threading.Lock()

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "TraceLedger":
        backend_name = os.getenv("DATAPAI_TRACE_BACKEND", "sqlite").lower()

        if backend_name == "null" or os.getenv("DATAPAI_TRACE_ENABLED", "true").lower() == "false":
            backend: TraceLedgerBackend = NullTraceLedgerBackend()
        elif backend_name == "snowflake":
            from traceability.backends.snowflake_backend import SnowflakeTraceLedgerBackend
            backend = SnowflakeTraceLedgerBackend()
        else:
            from traceability.backends.sqlite_backend import SQLiteTraceLedgerBackend
            backend = SQLiteTraceLedgerBackend()

        try:
            backend.initialise()
        except Exception as exc:
            log.warning("TraceLedger: backend init failed (%s). Using NullBackend.", exc)
            backend = NullTraceLedgerBackend()

        return cls(backend)

    # ── Core emit ─────────────────────────────────────────────────────────

    def emit(
        self,
        *,
        identity:          IdentityContext,
        event_type:        EventType,
        actor_type:        ActorType           = ActorType.SYSTEM,
        actor_id:          str                 = "system",
        request_id:        str                 = "",
        parent_trace_id:   Optional[str]       = None,
        datasource_type:   Optional[str]       = None,
        datasource_name:   Optional[str]       = None,
        model_name:        Optional[str]       = None,
        tool_name:         Optional[str]       = None,
        policy_result:     Optional[str]       = None,
        # Compliance fields
        question_text:     Optional[str]       = None,   # verbatim user question
        sql_text:          Optional[str]       = None,   # verbatim SQL (not results)
        sensitivity_level: Optional[str]       = None,
        pii_detected:      Optional[bool]      = None,
        pii_fields:        Optional[List[str]] = None,
        ai_action_summary: Optional[str]       = None,
        # AI agentic security
        agent_name:        Optional[str]       = None,
        boundary_violated: Optional[bool]      = None,
        risk_flags:        Optional[List[str]] = None,
        # Fingerprint only — not stored verbatim
        prompt_text:       Optional[str]       = None,
        # Misc
        context_refs:      Optional[str]       = None,
        status:            TraceStatus         = TraceStatus.OK,
        error_code:        Optional[str]       = None,
        error_message:     Optional[str]       = None,
        etl_run_id:        Optional[str]       = None,
        # Legacy compat
        input_text:        Optional[str]       = None,   # maps to question_text if set
        output_text:       Optional[str]       = None,   # maps to ai_action_summary if set
    ) -> str:
        """
        Emit a compliance trace event and return its trace_id.

        question_text: verbatim user question — credentials masked, content kept.
        sql_text:      verbatim SQL — stored as-is for audit (NOT the result rows).
        input_text:    legacy alias for question_text.
        output_text:   legacy alias for ai_action_summary.
        Never raises — errors are logged and swallowed to not block the caller.
        """
        # Legacy aliases
        effective_question  = question_text or input_text
        effective_ai_action = ai_action_summary or output_text

        # Mask credentials in question and SQL — preserve all substantive content
        if effective_question:
            effective_question = mask_credentials_only(effective_question)
        if sql_text:
            sql_text = mask_credentials_only(sql_text)

        event = TraceEvent.new(
            identity          = identity,
            event_type        = event_type,
            actor_type        = actor_type,
            actor_id          = actor_id,
            request_id        = request_id,
            parent_trace_id   = parent_trace_id,
            datasource_type   = datasource_type,
            datasource_name   = datasource_name,
            model_name        = model_name,
            tool_name         = tool_name,
            policy_result     = policy_result,
            question_text     = effective_question,
            sql_text          = sql_text,
            sensitivity_level = sensitivity_level,
            pii_detected      = pii_detected,
            pii_fields        = pii_fields,
            ai_action_summary = effective_ai_action,
            agent_name        = agent_name,
            boundary_violated = boundary_violated,
            risk_flags        = risk_flags,
            prompt_text       = prompt_text,
            context_refs      = context_refs,
            status            = status,
            error_code        = error_code,
            error_message     = summarise(error_message, max_len=500) if error_message else None,
            etl_run_id        = etl_run_id,
        )

        try:
            with self._lock:
                self._backend.append(event.to_dict())
        except Exception as exc:
            log.error("TraceLedger.emit failed: %s | event_type=%s", exc, event_type)

        return event.trace_id

    # ── Convenience methods ───────────────────────────────────────────────

    def emit_request_received(
        self,
        identity:          IdentityContext,
        request_id:        str,
        question_text:     Optional[str]  = None,
        actor_id:          str            = "",
        sensitivity_level: Optional[str]  = None,
        pii_detected:      Optional[bool] = None,
        pii_fields:        Optional[List[str]] = None,
    ) -> str:
        """Record a user question verbatim for compliance audit."""
        return self.emit(
            identity          = identity,
            event_type        = EventType.REQUEST_RECEIVED,
            actor_type        = ActorType.USER,
            actor_id          = actor_id or identity.user_id,
            request_id        = request_id,
            question_text     = question_text,
            sensitivity_level = sensitivity_level,
            pii_detected      = pii_detected,
            pii_fields        = pii_fields,
        )

    def emit_sql_generated(
        self,
        identity:          IdentityContext,
        request_id:        str,
        sql_text:          str,
        model_name:        Optional[str]       = None,
        datasource_name:   Optional[str]       = None,
        datasource_type:   Optional[str]       = None,
        sensitivity_level: Optional[str]       = None,
        pii_detected:      Optional[bool]      = None,
        pii_fields:        Optional[List[str]] = None,
        parent_trace_id:   Optional[str]       = None,
        etl_run_id:        Optional[str]       = None,
    ) -> str:
        """Record verbatim SQL that was generated. NOT the result rows."""
        return self.emit(
            identity          = identity,
            event_type        = EventType.SQL_GENERATED,
            actor_type        = ActorType.ASSISTANT,
            actor_id          = model_name or "llm",
            request_id        = request_id,
            parent_trace_id   = parent_trace_id,
            model_name        = model_name,
            datasource_type   = datasource_type,
            datasource_name   = datasource_name,
            sql_text          = sql_text,
            sensitivity_level = sensitivity_level,
            pii_detected      = pii_detected,
            pii_fields        = pii_fields,
            ai_action_summary = f"Generated SQL for {datasource_name or 'datasource'}",
            etl_run_id        = etl_run_id,
        )

    def emit_sql_validated(
        self,
        identity:          IdentityContext,
        request_id:        str,
        sql_text:          str,
        policy_result:     str,
        status:            TraceStatus         = TraceStatus.OK,
        error_message:     Optional[str]       = None,
        sensitivity_level: Optional[str]       = None,
        pii_detected:      Optional[bool]      = None,
        pii_fields:        Optional[List[str]] = None,
        parent_trace_id:   Optional[str]       = None,
        etl_run_id:        Optional[str]       = None,
    ) -> str:
        event_type = (
            EventType.SQL_VALIDATED if status == TraceStatus.OK
            else EventType.SQL_BLOCKED
        )
        return self.emit(
            identity          = identity,
            event_type        = event_type,
            actor_type        = ActorType.SYSTEM,
            actor_id          = "sql_validator",
            request_id        = request_id,
            parent_trace_id   = parent_trace_id,
            sql_text          = sql_text,
            policy_result     = policy_result,
            sensitivity_level = sensitivity_level,
            pii_detected      = pii_detected,
            pii_fields        = pii_fields,
            ai_action_summary = f"SQL validation: {policy_result}",
            status            = status,
            error_message     = error_message,
            etl_run_id        = etl_run_id,
        )

    def emit_sql_executed(
        self,
        identity:          IdentityContext,
        request_id:        str,
        sql_text:          str,
        datasource_name:   Optional[str]       = None,
        datasource_type:   Optional[str]       = None,
        sensitivity_level: Optional[str]       = None,
        pii_detected:      Optional[bool]      = None,
        pii_fields:        Optional[List[str]] = None,
        ai_action_summary: Optional[str]       = None,
        status:            TraceStatus         = TraceStatus.OK,
        error_message:     Optional[str]       = None,
        parent_trace_id:   Optional[str]       = None,
        etl_run_id:        Optional[str]       = None,
    ) -> str:
        """Record verbatim SQL that was executed. NOT the result rows."""
        return self.emit(
            identity          = identity,
            event_type        = EventType.SQL_EXECUTED,
            actor_type        = ActorType.TOOL,
            actor_id          = "sql_executor",
            request_id        = request_id,
            parent_trace_id   = parent_trace_id,
            datasource_type   = datasource_type,
            datasource_name   = datasource_name,
            sql_text          = sql_text,
            sensitivity_level = sensitivity_level,
            pii_detected      = pii_detected,
            pii_fields        = pii_fields,
            ai_action_summary = ai_action_summary or f"Executed SQL on {datasource_name or 'datasource'}",
            status            = status,
            error_message     = error_message,
            etl_run_id        = etl_run_id,
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
            identity          = identity,
            event_type        = event_type,
            actor_type        = ActorType.SYSTEM,
            actor_id          = "policy_engine",
            request_id        = request_id,
            parent_trace_id   = parent_trace_id,
            policy_result     = policy_result,
            ai_action_summary = reason,
            status            = TraceStatus.OK if passed else TraceStatus.BLOCKED,
            etl_run_id        = etl_run_id,
        )

    def emit_model_invoked(
        self,
        identity:        IdentityContext,
        request_id:      str,
        model_name:      str,
        prompt_text:     Optional[str]  = None,
        ai_action_summary: Optional[str] = None,
        parent_trace_id: Optional[str]  = None,
        etl_run_id:      Optional[str]  = None,
    ) -> str:
        return self.emit(
            identity          = identity,
            event_type        = EventType.MODEL_INVOKED,
            actor_type        = ActorType.ASSISTANT,
            actor_id          = model_name,
            request_id        = request_id,
            parent_trace_id   = parent_trace_id,
            model_name        = model_name,
            prompt_text       = prompt_text,
            ai_action_summary = ai_action_summary,
            etl_run_id        = etl_run_id,
        )

    def emit_tool_invoked(
        self,
        identity:          IdentityContext,
        request_id:        str,
        tool_name:         str,
        question_text:     Optional[str]  = None,
        ai_action_summary: Optional[str]  = None,
        status:            TraceStatus    = TraceStatus.OK,
        error_message:     Optional[str]  = None,
        parent_trace_id:   Optional[str]  = None,
        etl_run_id:        Optional[str]  = None,
    ) -> str:
        return self.emit(
            identity          = identity,
            event_type        = EventType.TOOL_INVOKED,
            actor_type        = ActorType.TOOL,
            actor_id          = tool_name,
            request_id        = request_id,
            parent_trace_id   = parent_trace_id,
            tool_name         = tool_name,
            question_text     = question_text,
            ai_action_summary = ai_action_summary,
            status            = status,
            error_message     = error_message,
            etl_run_id        = etl_run_id,
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
            question_text   = feedback_text,
        )

    # ── Read / query ──────────────────────────────────────────────────────

    # ── AI Agentic methods ────────────────────────────────────────────────

    def emit_agent_action(
        self,
        identity:          IdentityContext,
        request_id:        str,
        agent_name:        str,
        ai_action_summary: str,
        tool_name:         Optional[str]       = None,
        datasource_name:   Optional[str]       = None,
        datasource_type:   Optional[str]       = None,
        sql_text:          Optional[str]       = None,
        sensitivity_level: Optional[str]       = None,
        pii_detected:      Optional[bool]      = None,
        pii_fields:        Optional[List[str]] = None,
        risk_flags:        Optional[List[str]] = None,
        status:            TraceStatus         = TraceStatus.OK,
        error_message:     Optional[str]       = None,
        parent_trace_id:   Optional[str]       = None,
        etl_run_id:        Optional[str]       = None,
    ) -> str:
        """
        Record a single autonomous AI agent action.

        Every agent tool call, data access, write operation, or decision
        should emit this event so it is auditable.

        ai_action_summary: describe what the agent DID (e.g. 'Called ingest_file
          on sales_q4.csv, wrote 1500 rows to staging.sales'), not the data content.
        risk_flags: list any risks detected (e.g. ['WRITE_OPERATION', 'PII_SCHEMA_ACCESS'])
        """
        return self.emit(
            identity          = identity,
            event_type        = EventType.AGENT_ACTION,
            actor_type        = ActorType.AGENT,
            actor_id          = agent_name,
            request_id        = request_id,
            parent_trace_id   = parent_trace_id,
            tool_name         = tool_name,
            agent_name        = agent_name,
            datasource_type   = datasource_type,
            datasource_name   = datasource_name,
            sql_text          = sql_text,
            sensitivity_level = sensitivity_level,
            pii_detected      = pii_detected,
            pii_fields        = pii_fields,
            ai_action_summary = ai_action_summary,
            risk_flags        = risk_flags,
            status            = status,
            error_message     = error_message,
            etl_run_id        = etl_run_id,
        )

    def emit_agent_boundary_violation(
        self,
        identity:          IdentityContext,
        request_id:        str,
        agent_name:        str,
        violation_summary: str,
        risk_flags:        Optional[List[str]] = None,
        datasource_name:   Optional[str]       = None,
        sql_text:          Optional[str]       = None,
        parent_trace_id:   Optional[str]       = None,
        etl_run_id:        Optional[str]       = None,
    ) -> str:
        """
        Record that an AI agent attempted an action outside its allowed scope.

        Use this when:
        - An agent tried to access a schema/table outside its permission set
        - An agent attempted a write when running in read-only mode
        - An agent tried to cross tenant/workspace boundaries
        - An agent tried to execute DDL, DROP, TRUNCATE, or other dangerous statements
        - An agent attempted to access PII columns it is not authorised for

        This event always has status=VIOLATION and boundary_violated=True.
        It is always stored regardless of whether the action was blocked.
        """
        return self.emit(
            identity          = identity,
            event_type        = EventType.AGENT_BOUNDARY_VIOLATION,
            actor_type        = ActorType.AGENT,
            actor_id          = agent_name,
            request_id        = request_id,
            parent_trace_id   = parent_trace_id,
            agent_name        = agent_name,
            datasource_name   = datasource_name,
            sql_text          = sql_text,
            ai_action_summary = violation_summary,
            boundary_violated = True,
            risk_flags        = risk_flags or ["BOUNDARY_VIOLATION"],
            status            = TraceStatus.VIOLATION,
            etl_run_id        = etl_run_id,
        )

    def get_session_timeline(self, tenant_id: str, session_id: str,
                             limit: int = 200) -> list[dict]:
        return self._backend.fetch_by_session(tenant_id, session_id, limit)

    def get_request_timeline(self, tenant_id: str, request_id: str) -> list[dict]:
        return self._backend.fetch_by_request(tenant_id, request_id)

    def get_event(self, trace_id: str) -> Optional[dict]:
        rows = self._backend.fetch_by_trace_id(trace_id)
        return rows[0] if rows else None

    def search(self, **kwargs) -> list[dict]:
        return self._backend.search(**kwargs)

    def close(self) -> None:
        self._backend.close()


# ── Module singleton ──────────────────────────────────────────────────────────

_ledger: Optional[TraceLedger] = None
_ledger_lock = threading.Lock()


def get_ledger() -> TraceLedger:
    global _ledger
    if _ledger is None:
        with _ledger_lock:
            if _ledger is None:
                _ledger = TraceLedger.from_env()
    return _ledger


def reset_ledger(backend: Optional[TraceLedgerBackend] = None) -> None:
    global _ledger
    with _ledger_lock:
        if _ledger is not None:
            _ledger.close()
        _ledger = TraceLedger(backend) if backend else None


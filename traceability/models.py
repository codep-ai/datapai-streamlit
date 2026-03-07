"""
traceability/models.py

Core data models for the Datap.ai Trace Ledger.

Design: immutable, append-only event records.
Every meaningful AI action emits one or more TraceEvent records.
Events are never mutated — corrections append new events.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ── Event types ───────────────────────────────────────────────────────────────

class EventType(str, Enum):
    """All recognised trace event types. Stored as strings for portability."""

    REQUEST_RECEIVED      = "request_received"
    SESSION_STARTED       = "session_started"
    MEMORY_RETRIEVED      = "memory_retrieved"
    RAG_RETRIEVED         = "rag_retrieved"
    POLICY_CHECK_STARTED  = "policy_check_started"
    POLICY_CHECK_PASSED   = "policy_check_passed"
    POLICY_CHECK_FAILED   = "policy_check_failed"
    MODEL_INVOKED         = "model_invoked"
    TOOL_INVOKED          = "tool_invoked"
    SQL_GENERATED         = "sql_generated"
    SQL_VALIDATED         = "sql_validated"
    SQL_BLOCKED           = "sql_blocked"
    SQL_EXECUTED          = "sql_executed"
    DOCUMENT_EXTRACTED    = "document_extracted"
    RESPONSE_RETURNED     = "response_returned"
    HUMAN_FEEDBACK_RECEIVED = "human_feedback_received"
    ACTION_REPLAYED       = "action_replayed"


class ActorType(str, Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"
    TOOL      = "tool"


class TraceStatus(str, Enum):
    OK      = "ok"
    BLOCKED = "blocked"
    FAILED  = "failed"
    PENDING = "pending"


# ── Identity context ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IdentityContext:
    """
    Immutable identity tuple passed through every governed action.

    All trace events and memory operations are scoped by this tuple.
    Anonymous execution against protected systems is forbidden.
    """
    tenant_id:    str
    workspace_id: str
    user_id:      str
    session_id:   str

    @classmethod
    def from_env(cls) -> "IdentityContext":
        """
        Build an identity context from environment variables.
        Suitable for local dev / ETL pipelines where no HTTP session exists.
        """
        import os
        return cls(
            tenant_id    = os.getenv("DATAPAI_TENANT_ID", "default"),
            workspace_id = os.getenv("DATAPAI_WORKSPACE_ID", "default"),
            user_id      = os.getenv("DATAPAI_USER_ID", "system"),
            session_id   = os.getenv("DATAPAI_SESSION_ID", str(uuid.uuid4())),
        )

    @classmethod
    def system(cls) -> "IdentityContext":
        """Minimal system-level identity for background / migration tasks."""
        return cls(
            tenant_id    = "system",
            workspace_id = "system",
            user_id      = "system",
            session_id   = str(uuid.uuid4()),
        )


# ── Trace event ───────────────────────────────────────────────────────────────

@dataclass
class TraceEvent:
    """
    A single immutable trace record.

    Fields follow the spec minimum plus extended governance fields.
    Use TraceEvent.new(...) as the canonical constructor — it auto-generates
    trace_id and event_timestamp.
    """

    # ── Core identity ─────────────────────────────────────────────────────
    trace_id:       str
    parent_trace_id: Optional[str]
    tenant_id:      str
    workspace_id:   str
    user_id:        str
    session_id:     str
    request_id:     str

    # ── Event classification ──────────────────────────────────────────────
    event_type:       EventType
    event_timestamp:  str           # ISO-8601 UTC
    actor_type:       ActorType
    actor_id:         str

    # ── Data source context ───────────────────────────────────────────────
    datasource_type:  Optional[str]
    datasource_name:  Optional[str]

    # ── Model / tool context ──────────────────────────────────────────────
    model_name:   Optional[str]
    tool_name:    Optional[str]

    # ── Governance ────────────────────────────────────────────────────────
    policy_result:  Optional[str]   # PASSED | BLOCKED | SKIPPED

    # ── Content (summarised / hashed, never raw PII) ──────────────────────
    input_summary:  Optional[str]
    output_summary: Optional[str]
    sql_hash:       Optional[str]   # SHA-256 of SQL text
    prompt_hash:    Optional[str]   # SHA-256 of full prompt

    # ── References ────────────────────────────────────────────────────────
    context_refs:   Optional[str]   # JSON list of retrieved context IDs

    # ── Outcome ───────────────────────────────────────────────────────────
    status:        TraceStatus
    error_code:    Optional[str]
    error_message: Optional[str]

    # ── Auto-set by new() ─────────────────────────────────────────────────
    etl_run_id: Optional[str] = None   # bridge to agents/etl/audit.py run_id

    @classmethod
    def new(
        cls,
        *,
        identity:        IdentityContext,
        event_type:      EventType,
        actor_type:      ActorType       = ActorType.SYSTEM,
        actor_id:        str             = "system",
        request_id:      str             = "",
        parent_trace_id: Optional[str]  = None,
        datasource_type: Optional[str]  = None,
        datasource_name: Optional[str]  = None,
        model_name:      Optional[str]  = None,
        tool_name:       Optional[str]  = None,
        policy_result:   Optional[str]  = None,
        input_summary:   Optional[str]  = None,
        output_summary:  Optional[str]  = None,
        sql_text:        Optional[str]  = None,
        prompt_text:     Optional[str]  = None,
        context_refs:    Optional[str]  = None,
        status:          TraceStatus    = TraceStatus.OK,
        error_code:      Optional[str]  = None,
        error_message:   Optional[str]  = None,
        etl_run_id:      Optional[str]  = None,
    ) -> "TraceEvent":
        """
        Canonical constructor.  Auto-generates trace_id and timestamp.
        Accepts raw sql_text / prompt_text and hashes them automatically.
        """
        return cls(
            trace_id        = str(uuid.uuid4()),
            parent_trace_id = parent_trace_id,
            tenant_id       = identity.tenant_id,
            workspace_id    = identity.workspace_id,
            user_id         = identity.user_id,
            session_id      = identity.session_id,
            request_id      = request_id or str(uuid.uuid4()),
            event_type      = event_type,
            event_timestamp = datetime.now(timezone.utc).isoformat(),
            actor_type      = actor_type,
            actor_id        = actor_id,
            datasource_type = datasource_type,
            datasource_name = datasource_name,
            model_name      = model_name,
            tool_name       = tool_name,
            policy_result   = policy_result,
            input_summary   = input_summary,
            output_summary  = output_summary,
            sql_hash        = _sha256(sql_text) if sql_text else None,
            prompt_hash     = _sha256(prompt_text) if prompt_text else None,
            context_refs    = context_refs,
            status          = status,
            error_code      = error_code,
            error_message   = error_message,
            etl_run_id      = etl_run_id,
        )

    def to_dict(self) -> dict:
        """Serialize to flat dict — suitable for any storage backend."""
        return {
            "trace_id":        self.trace_id,
            "parent_trace_id": self.parent_trace_id,
            "tenant_id":       self.tenant_id,
            "workspace_id":    self.workspace_id,
            "user_id":         self.user_id,
            "session_id":      self.session_id,
            "request_id":      self.request_id,
            "event_type":      self.event_type.value,
            "event_timestamp": self.event_timestamp,
            "actor_type":      self.actor_type.value,
            "actor_id":        self.actor_id,
            "datasource_type": self.datasource_type,
            "datasource_name": self.datasource_name,
            "model_name":      self.model_name,
            "tool_name":       self.tool_name,
            "policy_result":   self.policy_result,
            "input_summary":   self.input_summary,
            "output_summary":  self.output_summary,
            "sql_hash":        self.sql_hash,
            "prompt_hash":     self.prompt_hash,
            "context_refs":    self.context_refs,
            "status":          self.status.value,
            "error_code":      self.error_code,
            "error_message":   self.error_message,
            "etl_run_id":      self.etl_run_id,
        }


# ── Schema DDL (portable SQL:2003 subset) ─────────────────────────────────────

TRACE_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS datapai_trace_events (
    trace_id         VARCHAR(36)   NOT NULL,
    parent_trace_id  VARCHAR(36),
    tenant_id        VARCHAR(255)  NOT NULL,
    workspace_id     VARCHAR(255)  NOT NULL,
    user_id          VARCHAR(255)  NOT NULL,
    session_id       VARCHAR(36)   NOT NULL,
    request_id       VARCHAR(36)   NOT NULL,
    event_type       VARCHAR(64)   NOT NULL,
    event_timestamp  VARCHAR(32)   NOT NULL,
    actor_type       VARCHAR(32)   NOT NULL,
    actor_id         VARCHAR(255)  NOT NULL,
    datasource_type  VARCHAR(64),
    datasource_name  VARCHAR(255),
    model_name       VARCHAR(128),
    tool_name        VARCHAR(128),
    policy_result    VARCHAR(32),
    input_summary    TEXT,
    output_summary   TEXT,
    sql_hash         VARCHAR(64),
    prompt_hash      VARCHAR(64),
    context_refs     TEXT,
    status           VARCHAR(32)   NOT NULL DEFAULT 'ok',
    error_code       VARCHAR(64),
    error_message    TEXT,
    etl_run_id       VARCHAR(36),
    PRIMARY KEY (trace_id)
)
"""

# Index DDL — applied separately (SQLite and Snowflake both support these)
TRACE_EVENTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_te_tenant_session ON datapai_trace_events (tenant_id, session_id)",
    "CREATE INDEX IF NOT EXISTS idx_te_user_ts       ON datapai_trace_events (user_id, event_timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_te_event_type    ON datapai_trace_events (event_type)",
    "CREATE INDEX IF NOT EXISTS idx_te_etl_run       ON datapai_trace_events (etl_run_id)",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

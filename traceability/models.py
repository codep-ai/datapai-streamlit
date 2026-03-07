"""
traceability/models.py — Datap.ai Trace Ledger core models.

Compliance storage model (financial / government / AI agentic):
  STORED verbatim:
    question_text      — exact user question (credentials masked, content preserved)
    sql_text           — exact SQL generated or executed (NOT the result rows)
    sensitivity_level  — LOW | MEDIUM | HIGH | CRITICAL
    pii_detected       — boolean: was PII found in this event
    pii_fields         — JSON list of PII column/field names (names only, not values)
    ai_action_summary  — what the AI/agent did (tools called, decisions made)
    boundary_violated  — did an AI agent attempt to access outside allowed scope
    risk_flags         — JSON list of detected security/compliance risks
    full identity      — tenant / workspace / user / session / request
    event_timestamp    — UTC ISO-8601

  NOT stored (by design):
    SQL result rows    (the actual data returned)
    document content   (only extraction metadata)
    full model output  (only ai_action_summary)
    secrets / tokens   (masked before storage, even in question_text)

Regulators (SOX, MiFID II, APRA CPS 234, FedRAMP) require: WHO asked WHAT
sensitive question at WHAT TIME, WHAT SQL was run against WHAT datasource,
and WHETHER PII was involved — but NOT the result data itself.

AI agentic compliance: every agent action, tool call, and boundary crossing
must be traceable to support security review and incident response.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional


class EventType(str, Enum):
    # User / session events
    REQUEST_RECEIVED        = "request_received"
    SESSION_STARTED         = "session_started"
    # Context retrieval
    MEMORY_RETRIEVED        = "memory_retrieved"
    RAG_RETRIEVED           = "rag_retrieved"
    # Policy
    POLICY_CHECK_STARTED    = "policy_check_started"
    POLICY_CHECK_PASSED     = "policy_check_passed"
    POLICY_CHECK_FAILED     = "policy_check_failed"
    # AI model
    MODEL_INVOKED           = "model_invoked"
    # Tool / agent actions
    TOOL_INVOKED            = "tool_invoked"
    AGENT_ACTION            = "agent_action"      # AI agent autonomous action
    AGENT_BOUNDARY_VIOLATION = "agent_boundary_violation"  # agent exceeded scope
    # SQL lifecycle
    SQL_GENERATED           = "sql_generated"
    SQL_VALIDATED           = "sql_validated"
    SQL_BLOCKED             = "sql_blocked"
    SQL_EXECUTED            = "sql_executed"
    # Document
    DOCUMENT_EXTRACTED      = "document_extracted"
    # Response / feedback
    RESPONSE_RETURNED       = "response_returned"
    HUMAN_FEEDBACK_RECEIVED = "human_feedback_received"
    ACTION_REPLAYED         = "action_replayed"


class ActorType(str, Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"
    TOOL      = "tool"
    AGENT     = "agent"     # autonomous AI agent


class TraceStatus(str, Enum):
    OK        = "ok"
    BLOCKED   = "blocked"
    FAILED    = "failed"
    PENDING   = "pending"
    VIOLATION = "violation"  # boundary/security violation by an agent


class SensitivityLevel(str, Enum):
    LOW      = "LOW"       # general business, no regulated data
    MEDIUM   = "MEDIUM"    # internal data, moderate sensitivity
    HIGH     = "HIGH"      # PII, financial records, health data
    CRITICAL = "CRITICAL"  # classified / legally privileged


@dataclass(frozen=True)
class IdentityContext:
    """Immutable identity tuple — scopes all trace events and memory."""
    tenant_id:    str
    workspace_id: str
    user_id:      str
    session_id:   str

    @classmethod
    def from_env(cls) -> "IdentityContext":
        import os
        return cls(
            tenant_id    = os.getenv("DATAPAI_TENANT_ID", "default"),
            workspace_id = os.getenv("DATAPAI_WORKSPACE_ID", "default"),
            user_id      = os.getenv("DATAPAI_USER_ID", "system"),
            session_id   = os.getenv("DATAPAI_SESSION_ID", str(uuid.uuid4())),
        )

    @classmethod
    def system(cls) -> "IdentityContext":
        return cls(tenant_id="system", workspace_id="system",
                   user_id="system", session_id=str(uuid.uuid4()))


@dataclass
class TraceEvent:
    """
    Single immutable compliance-grade audit record.

    Covers both human-initiated requests and autonomous AI agent actions.
    Every agent tool call, boundary check, and security decision is traceable.
    """

    # Identity
    trace_id:        str
    parent_trace_id: Optional[str]
    tenant_id:       str
    workspace_id:    str
    user_id:         str
    session_id:      str
    request_id:      str

    # Event classification
    event_type:      EventType
    event_timestamp: str
    actor_type:      ActorType
    actor_id:        str          # user_id, model name, agent name, tool name

    # Data source
    datasource_type: Optional[str]
    datasource_name: Optional[str]

    # Model / tool / agent
    model_name:  Optional[str]
    tool_name:   Optional[str]
    agent_name:  Optional[str]    # name of the AI agent if actor_type=AGENT

    # Governance
    policy_result: Optional[str]

    # ── COMPLIANCE FIELDS — verbatim content for audit ────────────────────

    # Verbatim original question. Credentials masked; content preserved.
    question_text: Optional[str]

    # Verbatim SQL generated or executed. NOT the result rows.
    sql_text: Optional[str]

    # Sensitivity classification
    sensitivity_level: Optional[str]

    # PII detection
    pii_detected: Optional[bool]
    pii_fields:   Optional[str]   # JSON list of PII field names

    # What the AI/agent did. NOT the data it returned.
    ai_action_summary: Optional[str]

    # ── AI AGENTIC SECURITY FIELDS ────────────────────────────────────────

    # Did this agent action attempt to exceed allowed scope?
    # e.g. cross-tenant access, write when read-only, schema outside permissions
    boundary_violated: Optional[bool]

    # JSON list of detected security/compliance risk flags.
    # e.g. ["CROSS_TENANT_ACCESS", "WRITE_ATTEMPT", "PII_SCHEMA_ACCESS",
    #        "MISSING_LIMIT", "SELECT_STAR_SENSITIVE", "DDL_ATTEMPT"]
    risk_flags: Optional[str]

    # ── Fingerprints ──────────────────────────────────────────────────────
    sql_hash:    Optional[str]
    prompt_hash: Optional[str]

    # References
    context_refs: Optional[str]

    # Outcome
    status:        TraceStatus
    error_code:    Optional[str]
    error_message: Optional[str]

    # Bridge to agents/etl/audit.py
    etl_run_id: Optional[str] = None

    @classmethod
    def new(
        cls,
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
        agent_name:        Optional[str]       = None,
        policy_result:     Optional[str]       = None,
        question_text:     Optional[str]       = None,
        sql_text:          Optional[str]       = None,
        sensitivity_level: Optional[str]       = None,
        pii_detected:      Optional[bool]      = None,
        pii_fields:        Optional[List[str]] = None,
        ai_action_summary: Optional[str]       = None,
        boundary_violated: Optional[bool]      = None,
        risk_flags:        Optional[List[str]] = None,
        prompt_text:       Optional[str]       = None,
        context_refs:      Optional[str]       = None,
        status:            TraceStatus         = TraceStatus.OK,
        error_code:        Optional[str]       = None,
        error_message:     Optional[str]       = None,
        etl_run_id:        Optional[str]       = None,
    ) -> "TraceEvent":
        return cls(
            trace_id           = str(uuid.uuid4()),
            parent_trace_id    = parent_trace_id,
            tenant_id          = identity.tenant_id,
            workspace_id       = identity.workspace_id,
            user_id            = identity.user_id,
            session_id         = identity.session_id,
            request_id         = request_id or str(uuid.uuid4()),
            event_type         = event_type,
            event_timestamp    = datetime.now(timezone.utc).isoformat(),
            actor_type         = actor_type,
            actor_id           = actor_id,
            datasource_type    = datasource_type,
            datasource_name    = datasource_name,
            model_name         = model_name,
            tool_name          = tool_name,
            agent_name         = agent_name,
            policy_result      = policy_result,
            question_text      = question_text,
            sql_text           = sql_text,
            sensitivity_level  = sensitivity_level,
            pii_detected       = pii_detected,
            pii_fields         = json.dumps(pii_fields) if pii_fields else None,
            ai_action_summary  = ai_action_summary,
            boundary_violated  = boundary_violated,
            risk_flags         = json.dumps(risk_flags) if risk_flags else None,
            sql_hash           = _sha256(sql_text) if sql_text else None,
            prompt_hash        = _sha256(prompt_text) if prompt_text else None,
            context_refs       = context_refs,
            status             = status,
            error_code         = error_code,
            error_message      = error_message,
            etl_run_id         = etl_run_id,
        )

    def to_dict(self) -> dict:
        return {
            "trace_id":           self.trace_id,
            "parent_trace_id":    self.parent_trace_id,
            "tenant_id":          self.tenant_id,
            "workspace_id":       self.workspace_id,
            "user_id":            self.user_id,
            "session_id":         self.session_id,
            "request_id":         self.request_id,
            "event_type":         self.event_type.value,
            "event_timestamp":    self.event_timestamp,
            "actor_type":         self.actor_type.value,
            "actor_id":           self.actor_id,
            "datasource_type":    self.datasource_type,
            "datasource_name":    self.datasource_name,
            "model_name":         self.model_name,
            "tool_name":          self.tool_name,
            "agent_name":         self.agent_name,
            "policy_result":      self.policy_result,
            "question_text":      self.question_text,
            "sql_text":           self.sql_text,
            "sensitivity_level":  self.sensitivity_level,
            "pii_detected":       self.pii_detected,
            "pii_fields":         self.pii_fields,
            "ai_action_summary":  self.ai_action_summary,
            "boundary_violated":  self.boundary_violated,
            "risk_flags":         self.risk_flags,
            "sql_hash":           self.sql_hash,
            "prompt_hash":        self.prompt_hash,
            "context_refs":       self.context_refs,
            "status":             self.status.value,
            "error_code":         self.error_code,
            "error_message":      self.error_message,
            "etl_run_id":         self.etl_run_id,
        }


TRACE_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS datapai_trace_events (
    trace_id           VARCHAR(36)    NOT NULL,
    parent_trace_id    VARCHAR(36),
    tenant_id          VARCHAR(255)   NOT NULL,
    workspace_id       VARCHAR(255)   NOT NULL,
    user_id            VARCHAR(255)   NOT NULL,
    session_id         VARCHAR(36)    NOT NULL,
    request_id         VARCHAR(36)    NOT NULL,
    event_type         VARCHAR(64)    NOT NULL,
    event_timestamp    VARCHAR(32)    NOT NULL,
    actor_type         VARCHAR(32)    NOT NULL,
    actor_id           VARCHAR(255)   NOT NULL,
    datasource_type    VARCHAR(64),
    datasource_name    VARCHAR(255),
    model_name         VARCHAR(128),
    tool_name          VARCHAR(128),
    agent_name         VARCHAR(128),
    policy_result      VARCHAR(32),
    question_text      TEXT,
    sql_text           TEXT,
    sensitivity_level  VARCHAR(16),
    pii_detected       BOOLEAN,
    pii_fields         TEXT,
    ai_action_summary  TEXT,
    boundary_violated  BOOLEAN,
    risk_flags         TEXT,
    sql_hash           VARCHAR(64),
    prompt_hash        VARCHAR(64),
    context_refs       TEXT,
    status             VARCHAR(32)    NOT NULL DEFAULT 'ok',
    error_code         VARCHAR(64),
    error_message      TEXT,
    etl_run_id         VARCHAR(36),
    PRIMARY KEY (trace_id)
)
"""

TRACE_EVENTS_MIGRATION_DDL_SNOWFLAKE = [
    "ALTER TABLE datapai_trace_events ADD COLUMN IF NOT EXISTS question_text     TEXT",
    "ALTER TABLE datapai_trace_events ADD COLUMN IF NOT EXISTS sql_text          TEXT",
    "ALTER TABLE datapai_trace_events ADD COLUMN IF NOT EXISTS sensitivity_level VARCHAR(16)",
    "ALTER TABLE datapai_trace_events ADD COLUMN IF NOT EXISTS pii_detected      BOOLEAN",
    "ALTER TABLE datapai_trace_events ADD COLUMN IF NOT EXISTS pii_fields        TEXT",
    "ALTER TABLE datapai_trace_events ADD COLUMN IF NOT EXISTS ai_action_summary TEXT",
    "ALTER TABLE datapai_trace_events ADD COLUMN IF NOT EXISTS agent_name        VARCHAR(128)",
    "ALTER TABLE datapai_trace_events ADD COLUMN IF NOT EXISTS boundary_violated BOOLEAN",
    "ALTER TABLE datapai_trace_events ADD COLUMN IF NOT EXISTS risk_flags        TEXT",
]

TRACE_EVENTS_MIGRATION_DDL_SQLITE = [
    ("question_text",     "TEXT"),
    ("sql_text",          "TEXT"),
    ("sensitivity_level", "VARCHAR(16)"),
    ("pii_detected",      "BOOLEAN"),
    ("pii_fields",        "TEXT"),
    ("ai_action_summary", "TEXT"),
    ("agent_name",        "VARCHAR(128)"),
    ("boundary_violated", "BOOLEAN"),
    ("risk_flags",        "TEXT"),
]

TRACE_EVENTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_te_tenant_session   ON datapai_trace_events (tenant_id, session_id)",
    "CREATE INDEX IF NOT EXISTS idx_te_user_ts          ON datapai_trace_events (user_id, event_timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_te_event_type       ON datapai_trace_events (event_type)",
    "CREATE INDEX IF NOT EXISTS idx_te_etl_run          ON datapai_trace_events (etl_run_id)",
    "CREATE INDEX IF NOT EXISTS idx_te_sensitivity      ON datapai_trace_events (sensitivity_level)",
    "CREATE INDEX IF NOT EXISTS idx_te_pii              ON datapai_trace_events (pii_detected)",
    "CREATE INDEX IF NOT EXISTS idx_te_boundary         ON datapai_trace_events (boundary_violated)",
    "CREATE INDEX IF NOT EXISTS idx_te_agent            ON datapai_trace_events (agent_name)",
]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

"""
traceability — Datap.ai Trace Ledger

Every meaningful AI action in the platform emits a TraceEvent here.
Events are immutable, append-only, and scoped by tenant/workspace/user/session.

Quick start:
    from traceability import get_ledger, EventType, ActorType, IdentityContext

    identity = IdentityContext(
        tenant_id    = "acme",
        workspace_id = "analytics",
        user_id      = "alice",
        session_id   = "sess-abc",
    )
    ledger = get_ledger()          # initialised from DATAPAI_TRACE_BACKEND env var

    # Emit a request event
    trace_id = ledger.emit_request_received(
        identity   = identity,
        request_id = "req-001",
        input_text = "Show me revenue by region last quarter",
        actor_id   = "alice",
    )

    # Emit SQL generation
    ledger.emit_sql_generated(
        identity        = identity,
        request_id      = "req-001",
        sql_text        = "SELECT region, SUM(revenue) FROM orders ...",
        model_name      = "claude-sonnet-4-6",
        datasource_name = "snowflake_prod",
        parent_trace_id = trace_id,
    )

Replay:
    from traceability.replay import TraceReplayer

    replayer = TraceReplayer(get_ledger())
    timeline = replayer.get_request_timeline("acme", "req-001")
    print(replayer.format_timeline(timeline))

Environment variables:
  DATAPAI_TRACE_BACKEND    sqlite | snowflake | null  (default: sqlite)
  DATAPAI_TRACE_ENABLED    true | false               (default: true)
  DATAPAI_TRACE_SQLITE_PATH  path to .db file         (default: datapai_traces.db)

  For Snowflake backend — see .env.traceability.example
"""

from traceability.models import (
    EventType,
    ActorType,
    TraceStatus,
    IdentityContext,
    TraceEvent,
)
from traceability.ledger import TraceLedger, get_ledger, reset_ledger
from traceability.redaction import mask_secrets, summarise, hash_payload
from traceability.replay import TraceReplayer

__all__ = [
    # Models
    "EventType",
    "ActorType",
    "TraceStatus",
    "IdentityContext",
    "TraceEvent",
    # Ledger
    "TraceLedger",
    "get_ledger",
    "reset_ledger",
    # Replay
    "TraceReplayer",
    # Redaction
    "mask_secrets",
    "summarise",
    "hash_payload",
]

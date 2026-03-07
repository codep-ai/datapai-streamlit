"""
traceability/replay.py

Trace replay and timeline reconstruction for the Datap.ai Trace Ledger.

Capabilities:
  - Fetch all events for a trace_id / request_id / session_id.
  - Reconstruct a chronological event timeline.
  - Show the model/tool invocation chain.
  - Show SQL generation → validation → execution sequence.
  - Compare first answer vs corrected answer (human feedback loop).
  - Format timeline for Streamlit trace viewer (spec section E.2).

Usage:
    from traceability.replay import TraceReplayer
    from traceability import get_ledger

    replayer = TraceReplayer(get_ledger())
    timeline = replayer.get_request_timeline(tenant_id="acme", request_id="req-123")
    print(replayer.format_timeline(timeline))
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from traceability.ledger import TraceLedger
from traceability.models import EventType


@dataclass
class TimelineEvent:
    """A single event in a reconstructed trace timeline."""
    trace_id:        str
    event_type:      str
    event_timestamp: str
    actor_type:      str
    actor_id:        str
    model_name:      Optional[str]
    tool_name:       Optional[str]
    status:          str
    policy_result:   Optional[str]
    input_summary:   Optional[str]
    output_summary:  Optional[str]
    sql_hash:        Optional[str]
    error_message:   Optional[str]
    duration_ms:     Optional[int]    # computed between consecutive events

    @property
    def is_error(self) -> bool:
        return self.status in ("failed", "blocked")

    @property
    def is_sql_event(self) -> bool:
        return self.event_type in (
            EventType.SQL_GENERATED.value,
            EventType.SQL_VALIDATED.value,
            EventType.SQL_BLOCKED.value,
            EventType.SQL_EXECUTED.value,
        )

    @property
    def label(self) -> str:
        """Short human-readable label for this event."""
        base = self.event_type.replace("_", " ").title()
        if self.tool_name:
            return f"{base} ({self.tool_name})"
        if self.model_name:
            return f"{base} — {self.model_name}"
        return base


class TraceReplayer:
    """
    Reconstruct and display trace timelines from the ledger.
    """

    def __init__(self, ledger: TraceLedger) -> None:
        self._ledger = ledger

    # ── Timeline fetch ────────────────────────────────────────────────────

    def get_request_timeline(
        self,
        tenant_id:  str,
        request_id: str,
    ) -> list[TimelineEvent]:
        """Return ordered timeline events for a single request."""
        rows = self._ledger.get_request_timeline(tenant_id, request_id)
        return _rows_to_timeline(rows)

    def get_session_timeline(
        self,
        tenant_id:  str,
        session_id: str,
        limit:      int = 200,
    ) -> list[TimelineEvent]:
        """Return ordered timeline events for an entire session."""
        rows = self._ledger.get_session_timeline(tenant_id, session_id, limit)
        return _rows_to_timeline(rows)

    # ── Derived views ─────────────────────────────────────────────────────

    def get_sql_chain(self, timeline: list[TimelineEvent]) -> list[TimelineEvent]:
        """Extract only the SQL generation/validation/execution sequence."""
        return [e for e in timeline if e.is_sql_event]

    def get_tool_chain(self, timeline: list[TimelineEvent]) -> list[TimelineEvent]:
        """Extract all model and tool invocations."""
        return [
            e for e in timeline
            if e.event_type in (
                EventType.MODEL_INVOKED.value,
                EventType.TOOL_INVOKED.value,
            )
        ]

    def get_policy_decisions(self, timeline: list[TimelineEvent]) -> list[TimelineEvent]:
        """Extract all policy check events."""
        return [
            e for e in timeline
            if e.event_type in (
                EventType.POLICY_CHECK_STARTED.value,
                EventType.POLICY_CHECK_PASSED.value,
                EventType.POLICY_CHECK_FAILED.value,
            )
        ]

    def get_human_corrections(self, timeline: list[TimelineEvent]) -> list[TimelineEvent]:
        """Extract human feedback events (corrections, approvals)."""
        return [
            e for e in timeline
            if e.event_type == EventType.HUMAN_FEEDBACK_RECEIVED.value
        ]

    def compare_first_vs_corrected(
        self,
        timeline: list[TimelineEvent],
    ) -> dict:
        """
        Compare the first generated SQL against any human-corrected SQL.

        Returns:
          {
            "first_sql_hash": str | None,
            "corrected_sql_hash": str | None,
            "has_correction": bool,
            "feedback_events": list[TimelineEvent],
          }
        """
        sql_events   = self.get_sql_chain(timeline)
        feedback     = self.get_human_corrections(timeline)

        first_sql_hash = next(
            (e.sql_hash for e in sql_events if e.event_type == EventType.SQL_GENERATED.value),
            None,
        )
        # After feedback, look for a subsequent SQL_GENERATED event
        corrected_sql_hash = None
        if feedback and len(sql_events) > 1:
            feedback_ts = feedback[0].event_timestamp
            later_sql = [
                e for e in sql_events
                if e.event_type == EventType.SQL_GENERATED.value
                and e.event_timestamp > feedback_ts
            ]
            if later_sql:
                corrected_sql_hash = later_sql[0].sql_hash

        return {
            "first_sql_hash":     first_sql_hash,
            "corrected_sql_hash": corrected_sql_hash,
            "has_correction":     corrected_sql_hash is not None,
            "feedback_events":    feedback,
        }

    # ── Formatting ────────────────────────────────────────────────────────

    def format_timeline(self, timeline: list[TimelineEvent]) -> str:
        """
        Return a plain-text trace timeline for logging / debugging.

        Each line: timestamp | event_type | actor | status | summary
        """
        if not timeline:
            return "(no trace events)"

        lines = [
            "─" * 80,
            "TRACE TIMELINE",
            "─" * 80,
        ]
        for i, evt in enumerate(timeline):
            status_tag = f"[{evt.status.upper()}]" if evt.is_error else f"[{evt.status}]"
            summary    = (evt.output_summary or evt.input_summary or "")[:80]
            dur        = f" +{evt.duration_ms}ms" if evt.duration_ms else ""
            lines.append(
                f"{i+1:3}. {evt.event_timestamp[:19]}  "
                f"{evt.event_type:<30}  "
                f"{evt.actor_id:<20}  "
                f"{status_tag:<10}"
                f"{dur:>10}  {summary}"
            )
            if evt.is_error and evt.error_message:
                lines.append(f"      ERROR: {evt.error_message[:120]}")

        lines.append("─" * 80)
        return "\n".join(lines)

    def to_streamlit_cards(self, timeline: list[TimelineEvent]) -> list[dict]:
        """
        Convert timeline to a list of dicts suitable for rendering in
        Streamlit (spec section E.2 trace viewer).

        Each dict has keys: label, status, actor, timestamp, summary,
        sql_hash, error, duration_ms, is_sql_event, is_error.
        """
        return [
            {
                "label":       evt.label,
                "event_type":  evt.event_type,
                "status":      evt.status,
                "actor":       evt.actor_id,
                "timestamp":   evt.event_timestamp,
                "summary":     evt.output_summary or evt.input_summary or "",
                "sql_hash":    evt.sql_hash,
                "error":       evt.error_message,
                "duration_ms": evt.duration_ms,
                "is_sql":      evt.is_sql_event,
                "is_error":    evt.is_error,
                "policy":      evt.policy_result,
            }
            for evt in timeline
        ]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _rows_to_timeline(rows: list[dict]) -> list[TimelineEvent]:
    """Convert raw backend row dicts → sorted TimelineEvent list with durations."""
    events: list[TimelineEvent] = []
    for row in rows:
        events.append(TimelineEvent(
            trace_id        = row.get("trace_id", ""),
            event_type      = row.get("event_type", ""),
            event_timestamp = row.get("event_timestamp", ""),
            actor_type      = row.get("actor_type", ""),
            actor_id        = row.get("actor_id", ""),
            model_name      = row.get("model_name"),
            tool_name       = row.get("tool_name"),
            status          = row.get("status", "ok"),
            policy_result   = row.get("policy_result"),
            input_summary   = row.get("input_summary"),
            output_summary  = row.get("output_summary"),
            sql_hash        = row.get("sql_hash"),
            error_message   = row.get("error_message"),
            duration_ms     = None,
        ))

    # Sort ascending by timestamp
    events.sort(key=lambda e: e.event_timestamp)

    # Compute inter-event durations
    for i in range(1, len(events)):
        try:
            t0 = datetime.fromisoformat(events[i - 1].event_timestamp)
            t1 = datetime.fromisoformat(events[i].event_timestamp)
            events[i].duration_ms = int((t1 - t0).total_seconds() * 1000)
        except Exception:
            pass

    return events

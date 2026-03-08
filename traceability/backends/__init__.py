"""
traceability/backends/__init__.py

Storage-agnostic backend interface for the Datap.ai Trace Ledger.

All backends must implement TraceLedgerBackend.  Business logic in
TraceLedger (ledger.py) is completely storage-agnostic — it only
calls methods on this interface.

Available backends:
  sqlite     — local development (default)
  snowflake  — cloud production
  null       — no-op / testing
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class TraceLedgerBackend(ABC):
    """
    Abstract storage backend for trace events.

    Implementations:
      - SQLiteTraceLedgerBackend   (traceability.backends.sqlite_backend)
      - SnowflakeTraceLedgerBackend (traceability.backends.snowflake_backend)
      - NullTraceLedgerBackend     (this file — for testing)
    """

    @abstractmethod
    def initialise(self) -> None:
        """Create schema / tables if they don't exist. Idempotent."""

    @abstractmethod
    def append(self, event: dict) -> None:
        """
        Persist a single trace event dict.
        Must be append-only — never update or delete existing records.
        """

    @abstractmethod
    def fetch_by_trace_id(self, trace_id: str) -> list[dict]:
        """Return the single event matching trace_id (or empty list)."""

    @abstractmethod
    def fetch_by_session(
        self,
        tenant_id: str,
        session_id: str,
        limit: int = 200,
    ) -> list[dict]:
        """
        Return all events for a session, ordered by event_timestamp ASC.
        Always filter by tenant_id first to prevent cross-tenant bleed.
        """

    @abstractmethod
    def fetch_by_request(
        self,
        tenant_id: str,
        request_id: str,
    ) -> list[dict]:
        """Return all events for a request, ordered by event_timestamp ASC."""

    @abstractmethod
    def search(
        self,
        *,
        tenant_id:    str,
        user_id:      Optional[str]  = None,
        workspace_id: Optional[str]  = None,
        event_type:   Optional[str]  = None,
        datasource:   Optional[str]  = None,
        status:       Optional[str]  = None,
        from_ts:      Optional[str]  = None,
        to_ts:        Optional[str]  = None,
        etl_run_id:   Optional[str]  = None,
        limit:        int            = 100,
        offset:       int            = 0,
    ) -> list[dict]:
        """
        Flexible search. tenant_id is mandatory to prevent cross-tenant bleed.
        All other filters are optional.  Returns rows ordered by event_timestamp DESC.
        """

    @abstractmethod
    def count(self, tenant_id: str, **filters: Any) -> int:
        """Return row count matching filters (for pagination)."""

    @abstractmethod
    def close(self) -> None:
        """Release any held connections or resources."""


# ── Null backend (testing / disabled tracing) ──────────────────────────────────

class NullTraceLedgerBackend(TraceLedgerBackend):
    """
    No-op backend.  All writes are silently discarded.
    Reads return empty lists.  Use when DATAPAI_TRACE_ENABLED=false.
    """

    def initialise(self) -> None:
        pass

    def append(self, event: dict) -> None:
        pass

    def fetch_by_trace_id(self, trace_id: str) -> list[dict]:
        return []

    def fetch_by_session(self, tenant_id: str, session_id: str, limit: int = 200) -> list[dict]:
        return []

    def fetch_by_request(self, tenant_id: str, request_id: str) -> list[dict]:
        return []

    def search(self, *, tenant_id: str, **kwargs: Any) -> list[dict]:
        return []

    def count(self, tenant_id: str, **filters: Any) -> int:
        return 0

    def close(self) -> None:
        pass


__all__ = ["TraceLedgerBackend", "NullTraceLedgerBackend"]

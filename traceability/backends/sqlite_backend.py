"""
traceability/backends/sqlite_backend.py

SQLite implementation of TraceLedgerBackend.

Used for local development.  No extra dependencies beyond the Python
standard library.  Thread-safe via check_same_thread=False + a simple lock.

Environment variables:
  DATAPAI_TRACE_SQLITE_PATH   Path to the SQLite database file.
                               Default: datapai_traces.db
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from typing import Any, Optional

from traceability.backends import TraceLedgerBackend
from traceability.models import TRACE_EVENTS_DDL, TRACE_EVENTS_INDEXES

log = logging.getLogger(__name__)

_DEFAULT_PATH = os.getenv("DATAPAI_TRACE_SQLITE_PATH", "datapai_traces.db")


class SQLiteTraceLedgerBackend(TraceLedgerBackend):
    """
    SQLite trace storage backend for local development.

    All writes go through a single sqlite3 connection protected by a
    threading.Lock so this class is safe to use from multiple Streamlit
    threads or AG2 swarm threads.
    """

    def __init__(self, db_path: str = _DEFAULT_PATH) -> None:
        self.db_path = db_path
        self._lock  = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,   # autocommit
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def initialise(self) -> None:
        """Create the trace_events table and indexes if they don't exist."""
        try:
            con = self._connect()
            with self._lock:
                # SQLite uses slightly different DDL — adapt the portable DDL
                ddl = _adapt_ddl_for_sqlite(TRACE_EVENTS_DDL)
                con.execute(ddl)
                for idx in TRACE_EVENTS_INDEXES:
                    con.execute(idx)
                con.commit()
            log.debug("SQLiteTraceLedgerBackend: schema ready at %s", self.db_path)
        except Exception as exc:
            log.error("SQLiteTraceLedgerBackend.initialise failed: %s", exc)
            raise

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ── Write ─────────────────────────────────────────────────────────────

    def append(self, event: dict) -> None:
        """Insert a single trace event row.  Never updates existing rows."""
        cols = list(event.keys())
        placeholders = ", ".join("?" for _ in cols)
        sql = (
            f"INSERT OR IGNORE INTO datapai_trace_events "
            f"({', '.join(cols)}) VALUES ({placeholders})"
        )
        values = [event.get(c) for c in cols]
        try:
            con = self._connect()
            with self._lock:
                con.execute(sql, values)
        except Exception as exc:
            log.error("SQLiteTraceLedgerBackend.append failed: %s | event=%s", exc, event.get("trace_id"))

    # ── Read ──────────────────────────────────────────────────────────────

    def fetch_by_trace_id(self, trace_id: str) -> list[dict]:
        return self._query(
            "SELECT * FROM datapai_trace_events WHERE trace_id = ?",
            [trace_id],
        )

    def fetch_by_session(
        self,
        tenant_id: str,
        session_id: str,
        limit: int = 200,
    ) -> list[dict]:
        return self._query(
            "SELECT * FROM datapai_trace_events "
            "WHERE tenant_id = ? AND session_id = ? "
            "ORDER BY event_timestamp ASC LIMIT ?",
            [tenant_id, session_id, limit],
        )

    def fetch_by_request(self, tenant_id: str, request_id: str) -> list[dict]:
        return self._query(
            "SELECT * FROM datapai_trace_events "
            "WHERE tenant_id = ? AND request_id = ? "
            "ORDER BY event_timestamp ASC",
            [tenant_id, request_id],
        )

    def search(
        self,
        *,
        tenant_id:    str,
        user_id:      Optional[str] = None,
        workspace_id: Optional[str] = None,
        event_type:   Optional[str] = None,
        datasource:   Optional[str] = None,
        status:       Optional[str] = None,
        from_ts:      Optional[str] = None,
        to_ts:        Optional[str] = None,
        etl_run_id:   Optional[str] = None,
        limit:        int           = 100,
        offset:       int           = 0,
    ) -> list[dict]:
        clauses = ["tenant_id = ?"]
        params: list[Any] = [tenant_id]

        if user_id:
            clauses.append("user_id = ?"); params.append(user_id)
        if workspace_id:
            clauses.append("workspace_id = ?"); params.append(workspace_id)
        if event_type:
            clauses.append("event_type = ?"); params.append(event_type)
        if datasource:
            clauses.append("datasource_name = ?"); params.append(datasource)
        if status:
            clauses.append("status = ?"); params.append(status)
        if from_ts:
            clauses.append("event_timestamp >= ?"); params.append(from_ts)
        if to_ts:
            clauses.append("event_timestamp <= ?"); params.append(to_ts)
        if etl_run_id:
            clauses.append("etl_run_id = ?"); params.append(etl_run_id)

        where = " AND ".join(clauses)
        params += [limit, offset]

        return self._query(
            f"SELECT * FROM datapai_trace_events "
            f"WHERE {where} "
            f"ORDER BY event_timestamp DESC "
            f"LIMIT ? OFFSET ?",
            params,
        )

    def count(self, tenant_id: str, **filters: Any) -> int:
        rows = self._query(
            "SELECT COUNT(*) AS n FROM datapai_trace_events WHERE tenant_id = ?",
            [tenant_id],
        )
        return rows[0]["n"] if rows else 0

    # ── Internal ──────────────────────────────────────────────────────────

    def _query(self, sql: str, params: list) -> list[dict]:
        try:
            con = self._connect()
            with self._lock:
                cur = con.execute(sql, params)
                rows = cur.fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            log.error("SQLiteTraceLedgerBackend._query failed: %s\nSQL: %s", exc, sql)
            return []


# ── DDL adaptation ─────────────────────────────────────────────────────────────

def _adapt_ddl_for_sqlite(ddl: str) -> str:
    """
    Adjust the portable DDL for SQLite quirks:
    - Remove VARCHAR lengths (SQLite is typeless anyway)
    - Remove unsupported DEFAULT expressions in PRIMARY KEY constraints
    """
    # SQLite accepts VARCHAR(n) fine — no adaptation needed for current DDL.
    return ddl

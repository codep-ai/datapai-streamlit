"""
traceability/backends/sqlite_backend.py

SQLite implementation of TraceLedgerBackend for local development.
Thread-safe via threading.Lock.

Handles schema migration automatically — adds new compliance columns
(question_text, sql_text, sensitivity_level, pii_detected, pii_fields,
ai_action_summary) to existing databases without data loss.

Environment variables:
  DATAPAI_TRACE_SQLITE_PATH   Path to the SQLite file (default: datapai_traces.db)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from typing import Any, Optional

from traceability.backends import TraceLedgerBackend
from traceability.models import (
    TRACE_EVENTS_DDL,
    TRACE_EVENTS_INDEXES,
    TRACE_EVENTS_MIGRATION_DDL_SQLITE,
)

log = logging.getLogger(__name__)

_DEFAULT_PATH = os.getenv("DATAPAI_TRACE_SQLITE_PATH", "datapai_traces.db")


class SQLiteTraceLedgerBackend(TraceLedgerBackend):
    """SQLite trace storage backend for local development."""

    def __init__(self, db_path: str = _DEFAULT_PATH) -> None:
        self.db_path = db_path
        self._lock  = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def initialise(self) -> None:
        """Create table + apply migrations idempotently."""
        try:
            con = self._connect()
            with self._lock:
                con.execute(TRACE_EVENTS_DDL)
                self._apply_migrations(con)
                for idx in TRACE_EVENTS_INDEXES:
                    con.execute(idx)
            log.debug("SQLiteTraceLedgerBackend: schema ready at %s", self.db_path)
        except Exception as exc:
            log.error("SQLiteTraceLedgerBackend.initialise failed: %s", exc)
            raise

    def _apply_migrations(self, con: sqlite3.Connection) -> None:
        """
        Add compliance columns that may not exist in older databases.
        SQLite does not support ALTER TABLE ADD COLUMN IF NOT EXISTS,
        so we check PRAGMA table_info first.
        """
        cur = con.execute("PRAGMA table_info(datapai_trace_events)")
        existing = {row[1] for row in cur.fetchall()}

        for col_name, col_type in TRACE_EVENTS_MIGRATION_DDL_SQLITE:
            if col_name not in existing:
                try:
                    con.execute(
                        f"ALTER TABLE datapai_trace_events "
                        f"ADD COLUMN {col_name} {col_type}"
                    )
                    log.info("SQLiteTraceLedgerBackend: added column %s", col_name)
                except Exception as exc:
                    log.warning("Migration column %s failed: %s", col_name, exc)

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def append(self, event: dict) -> None:
        cols         = list(event.keys())
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
            log.error("SQLiteTraceLedgerBackend.append failed: %s | trace_id=%s",
                      exc, event.get("trace_id"))

    def fetch_by_trace_id(self, trace_id: str) -> list[dict]:
        return self._query(
            "SELECT * FROM datapai_trace_events WHERE trace_id = ?",
            [trace_id],
        )

    def fetch_by_session(self, tenant_id: str, session_id: str,
                         limit: int = 200) -> list[dict]:
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
        tenant_id:         str,
        user_id:           Optional[str] = None,
        workspace_id:      Optional[str] = None,
        event_type:        Optional[str] = None,
        datasource:        Optional[str] = None,
        status:            Optional[str] = None,
        sensitivity_level: Optional[str] = None,
        pii_detected:      Optional[bool] = None,
        from_ts:           Optional[str] = None,
        to_ts:             Optional[str] = None,
        etl_run_id:        Optional[str] = None,
        limit:             int           = 100,
        offset:            int           = 0,
    ) -> list[dict]:
        clauses = ["tenant_id = ?"]
        params: list[Any] = [tenant_id]

        if user_id:
            clauses.append("user_id = ?");            params.append(user_id)
        if workspace_id:
            clauses.append("workspace_id = ?");       params.append(workspace_id)
        if event_type:
            clauses.append("event_type = ?");         params.append(event_type)
        if datasource:
            clauses.append("datasource_name = ?");    params.append(datasource)
        if status:
            clauses.append("status = ?");             params.append(status)
        if sensitivity_level:
            clauses.append("sensitivity_level = ?");  params.append(sensitivity_level)
        if pii_detected is not None:
            clauses.append("pii_detected = ?");       params.append(int(pii_detected))
        if from_ts:
            clauses.append("event_timestamp >= ?");   params.append(from_ts)
        if to_ts:
            clauses.append("event_timestamp <= ?");   params.append(to_ts)
        if etl_run_id:
            clauses.append("etl_run_id = ?");         params.append(etl_run_id)

        where   = " AND ".join(clauses)
        params += [limit, offset]
        return self._query(
            f"SELECT * FROM datapai_trace_events WHERE {where} "
            f"ORDER BY event_timestamp DESC LIMIT ? OFFSET ?",
            params,
        )

    def count(self, tenant_id: str, **filters: Any) -> int:
        rows = self._query(
            "SELECT COUNT(*) AS n FROM datapai_trace_events WHERE tenant_id = ?",
            [tenant_id],
        )
        return rows[0]["n"] if rows else 0

    def _query(self, sql: str, params: list) -> list[dict]:
        try:
            con = self._connect()
            with self._lock:
                cur  = con.execute(sql, params)
                rows = cur.fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            log.error("SQLiteTraceLedgerBackend._query failed: %s", exc)
            return []

"""
traceability/backends/snowflake_backend.py

Snowflake implementation of TraceLedgerBackend.

Used in cloud / production deployments.  Connects via Snowflake Connector
for Python (snowflake-connector-python).

The trace table is also exposed via dbt models (see dbt_traceability/)
so that Lightdash can query it directly for governance reporting dashboards.

Environment variables (required for cloud mode):
  SNOWFLAKE_ACCOUNT        e.g. xy12345.us-east-1
  SNOWFLAKE_USER           service account user
  SNOWFLAKE_PASSWORD       service account password (or use key-pair below)
  SNOWFLAKE_PRIVATE_KEY_PATH  path to private key file (alternative to password)
  SNOWFLAKE_PRIVATE_KEY_PASSPHRASE  passphrase for private key (if encrypted)
  SNOWFLAKE_DATABASE       target database
  SNOWFLAKE_SCHEMA         target schema (default: DATAPAI_TRACES)
  SNOWFLAKE_WAREHOUSE      virtual warehouse to use
  SNOWFLAKE_ROLE           role with INSERT + CREATE TABLE privileges

Optional:
  DATAPAI_TRACE_SNOWFLAKE_TABLE   Override table name (default: DATAPAI_TRACE_EVENTS)
  DATAPAI_TRACE_SNOWFLAKE_BATCH   Batch size for bulk inserts (default: 50)

Notes:
  - Uses INSERT with MERGE (UPSERT) to ensure idempotency on retry.
  - Connection pooling via connection reuse per backend instance.
  - Graceful fallback: if Snowflake is unreachable, logs an error and
    continues (traces are best-effort in degraded mode).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from traceability.backends import TraceLedgerBackend

log = logging.getLogger(__name__)

_TABLE = os.getenv("DATAPAI_TRACE_SNOWFLAKE_TABLE", "DATAPAI_TRACE_EVENTS")
_BATCH_SIZE = int(os.getenv("DATAPAI_TRACE_SNOWFLAKE_BATCH", "50"))


class SnowflakeTraceLedgerBackend(TraceLedgerBackend):
    """
    Snowflake trace storage backend for cloud / production.

    Connection is lazy — opened on first use and reused across calls.
    If Snowflake is unavailable the backend degrades gracefully (logs error,
    returns empty results) so the calling service keeps running.

    The same table is the source of truth for the dbt_traceability/ dbt
    project, which exposes staging models and reporting views consumable
    by Lightdash.
    """

    def __init__(
        self,
        account:    Optional[str] = None,
        user:       Optional[str] = None,
        password:   Optional[str] = None,
        database:   Optional[str] = None,
        schema:     Optional[str] = None,
        warehouse:  Optional[str] = None,
        role:       Optional[str] = None,
        table:      str           = _TABLE,
    ) -> None:
        self._account   = account   or os.getenv("SNOWFLAKE_ACCOUNT", "")
        self._user      = user      or os.getenv("SNOWFLAKE_USER", "")
        self._password  = password  or os.getenv("SNOWFLAKE_PASSWORD", "")
        self._database  = database  or os.getenv("SNOWFLAKE_DATABASE", "")
        self._schema    = schema    or os.getenv("SNOWFLAKE_SCHEMA", "DATAPAI_TRACES")
        self._warehouse = warehouse or os.getenv("SNOWFLAKE_WAREHOUSE", "")
        self._role      = role      or os.getenv("SNOWFLAKE_ROLE", "")
        self._table     = table
        self._conn      = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def _connect(self):
        """Return an open Snowflake connection, creating one if needed."""
        if self._conn is not None:
            return self._conn
        try:
            import snowflake.connector  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "snowflake-connector-python is required for the Snowflake backend. "
                "Install it with: pip install snowflake-connector-python"
            ) from exc

        connect_kwargs: dict[str, Any] = {
            "account":   self._account,
            "user":      self._user,
            "database":  self._database,
            "schema":    self._schema,
            "warehouse": self._warehouse,
        }
        if self._role:
            connect_kwargs["role"] = self._role

        # Key-pair auth takes precedence over password
        private_key_path = os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH", "")
        if private_key_path:
            connect_kwargs["private_key"] = _load_private_key(private_key_path)
        else:
            connect_kwargs["password"] = self._password

        self._conn = snowflake.connector.connect(**connect_kwargs)
        log.info(
            "SnowflakeTraceLedgerBackend: connected to %s/%s/%s",
            self._account, self._database, self._schema,
        )
        return self._conn

    def initialise(self) -> None:
        """Create the trace events table in Snowflake if it doesn't exist."""
        snowflake_ddl = _snowflake_ddl(self._table)
        try:
            con = self._connect()
            cur = con.cursor()
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")
            cur.execute(f"USE SCHEMA {self._schema}")
            cur.execute(snowflake_ddl)
            cur.close()
            log.info("SnowflakeTraceLedgerBackend: table %s ready", self._table)
        except Exception as exc:
            log.error("SnowflakeTraceLedgerBackend.initialise failed: %s", exc)

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ── Write ─────────────────────────────────────────────────────────────

    def append(self, event: dict) -> None:
        """
        Insert a single trace event using MERGE to ensure idempotency.
        On duplicate trace_id (e.g. retry), silently skips.
        """
        cols   = list(event.keys())
        values = [event.get(c) for c in cols]
        col_list = ", ".join(cols)
        src_cols = ", ".join(f"s.{c}" for c in cols)
        bind     = ", ".join("%s" for _ in cols)

        # MERGE ensures append-only semantics on retry
        merge_sql = f"""
            MERGE INTO {self._schema}.{self._table} AS t
            USING (SELECT {bind}) AS s ({col_list})
            ON t.trace_id = s.trace_id
            WHEN NOT MATCHED THEN INSERT ({col_list}) VALUES ({src_cols})
        """
        try:
            con = self._connect()
            cur = con.cursor()
            cur.execute(merge_sql, values)
            cur.close()
        except Exception as exc:
            log.error(
                "SnowflakeTraceLedgerBackend.append failed: %s | trace_id=%s",
                exc, event.get("trace_id"),
            )

    # ── Read ──────────────────────────────────────────────────────────────

    def fetch_by_trace_id(self, trace_id: str) -> list[dict]:
        return self._query(
            f"SELECT * FROM {self._schema}.{self._table} WHERE trace_id = %s",
            [trace_id],
        )

    def fetch_by_session(
        self,
        tenant_id: str,
        session_id: str,
        limit: int = 200,
    ) -> list[dict]:
        return self._query(
            f"SELECT * FROM {self._schema}.{self._table} "
            f"WHERE tenant_id = %s AND session_id = %s "
            f"ORDER BY event_timestamp ASC LIMIT %s",
            [tenant_id, session_id, limit],
        )

    def fetch_by_request(self, tenant_id: str, request_id: str) -> list[dict]:
        return self._query(
            f"SELECT * FROM {self._schema}.{self._table} "
            f"WHERE tenant_id = %s AND request_id = %s "
            f"ORDER BY event_timestamp ASC",
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
        clauses = ["tenant_id = %s"]
        params: list[Any] = [tenant_id]

        if user_id:
            clauses.append("user_id = %s");       params.append(user_id)
        if workspace_id:
            clauses.append("workspace_id = %s");  params.append(workspace_id)
        if event_type:
            clauses.append("event_type = %s");    params.append(event_type)
        if datasource:
            clauses.append("datasource_name = %s"); params.append(datasource)
        if status:
            clauses.append("status = %s");         params.append(status)
        if from_ts:
            clauses.append("event_timestamp >= %s"); params.append(from_ts)
        if to_ts:
            clauses.append("event_timestamp <= %s"); params.append(to_ts)
        if etl_run_id:
            clauses.append("etl_run_id = %s");    params.append(etl_run_id)

        where = " AND ".join(clauses)
        params += [limit, offset]

        return self._query(
            f"SELECT * FROM {self._schema}.{self._table} "
            f"WHERE {where} "
            f"ORDER BY event_timestamp DESC "
            f"LIMIT %s OFFSET %s",
            params,
        )

    def count(self, tenant_id: str, **filters: Any) -> int:
        rows = self._query(
            f"SELECT COUNT(*) AS n FROM {self._schema}.{self._table} WHERE tenant_id = %s",
            [tenant_id],
        )
        return rows[0].get("N", rows[0].get("n", 0)) if rows else 0

    # ── Internal ──────────────────────────────────────────────────────────

    def _query(self, sql: str, params: list) -> list[dict]:
        try:
            con = self._connect()
            cur = con.cursor()
            cur.execute(sql, params)
            cols  = [desc[0].lower() for desc in cur.description]
            rows  = [dict(zip(cols, row)) for row in cur.fetchall()]
            cur.close()
            return rows
        except Exception as exc:
            log.error(
                "SnowflakeTraceLedgerBackend._query failed: %s\nSQL: %s",
                exc, sql,
            )
            return []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _snowflake_ddl(table: str) -> str:
    """
    Snowflake-specific CREATE TABLE DDL for the trace events table.

    Snowflake uses VARCHAR without length limits, TIMESTAMP_TZ for timestamps,
    and VARIANT for JSON fields.  The schema matches the portable DDL in
    models.py but uses Snowflake native types.

    This table is also referenced from dbt_traceability/models/sources.yml
    as the raw source for all dbt staging models.
    """
    return f"""
    CREATE TABLE IF NOT EXISTS {table} (
        trace_id         VARCHAR        NOT NULL,
        parent_trace_id  VARCHAR,
        tenant_id        VARCHAR        NOT NULL,
        workspace_id     VARCHAR        NOT NULL,
        user_id          VARCHAR        NOT NULL,
        session_id       VARCHAR        NOT NULL,
        request_id       VARCHAR        NOT NULL,
        event_type       VARCHAR        NOT NULL,
        event_timestamp  VARCHAR        NOT NULL,
        actor_type       VARCHAR        NOT NULL,
        actor_id         VARCHAR        NOT NULL,
        datasource_type  VARCHAR,
        datasource_name  VARCHAR,
        model_name       VARCHAR,
        tool_name        VARCHAR,
        policy_result    VARCHAR,
        input_summary    VARCHAR,
        output_summary   VARCHAR,
        sql_hash         VARCHAR,
        prompt_hash      VARCHAR,
        context_refs     VARCHAR,
        status           VARCHAR        NOT NULL DEFAULT 'ok',
        error_code       VARCHAR,
        error_message    VARCHAR,
        etl_run_id       VARCHAR,
        PRIMARY KEY (trace_id)
    )
    """


def _load_private_key(path: str) -> bytes:
    """Load and decrypt an RSA private key for Snowflake key-pair auth."""
    from cryptography.hazmat.backends import default_backend  # type: ignore[import]
    from cryptography.hazmat.primitives import serialization  # type: ignore[import]

    passphrase_str = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE", "")
    passphrase = passphrase_str.encode() if passphrase_str else None

    with open(path, "rb") as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=passphrase,
            backend=default_backend(),
        )

    return private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

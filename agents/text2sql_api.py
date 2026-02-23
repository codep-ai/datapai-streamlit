"""
DataPAI Text2SQL API
=====================
FastAPI service that wraps Vanna RAG-based SQL generation.
Called by openwebui_sql_pipeline.py running on EC2 #3 (OpenWebUI).

Runs on EC2 #2 (platform), port 8101:
  uvicorn agents.text2sql_api:app --host 0.0.0.0 --port 8101

Endpoints:
  POST /v1/sql/query  — question → SQL → optional execution → results
  GET  /health        — health + supported DBs

Environment variables:
  SQL_API_KEY              Bearer token (empty = no auth)
  VANNA_API_KEY            Vanna.ai API key
  VANNA_MODEL              Vanna model name          default: chinook
  SQL_DEFAULT_DB           Default target DB         default: Snowflake
  SQL_MAX_ROWS             Max rows in response      default: 100

  Per-DB connection (used when run_sql=true):
  SNOWFLAKE_ACCOUNT / _USERNAME / _PASSWORD / _DATABASE / _SCHEMA / _WAREHOUSE
  REDSHIFT_HOST / _PORT / _USER / _PASSWORD / _DBNAME / _SCHEMA
  SQLITE_DB_PATH           Path to SQLite file
  DUCKDB_DB_PATH           Path to DuckDB file (empty = in-memory)
  ATHENA_DATABASE / _DATABASES (comma-separated) / _S3_STAGING_DIR / _WORKGROUP
  BIGQUERY_PROJECT / _DATASET  (uses ADC / instance service account)
  AWS_REGION               for Athena + Glue     default: us-east-1
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
_SQL_API_KEY  = os.getenv("SQL_API_KEY", "")
_VANNA_KEY    = os.getenv("VANNA_API_KEY", "")
_VANNA_MODEL  = os.getenv("VANNA_MODEL", "chinook")
_DEFAULT_DB   = os.getenv("SQL_DEFAULT_DB", "Snowflake")
_MAX_ROWS     = int(os.getenv("SQL_MAX_ROWS", "100"))

SUPPORTED_DBS = [
    "Snowflake", "Redshift", "Athena", "DuckDB", "SQLite3", "BigQuery", "dbt"
]

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="DataPAI Text2SQL API",
    description="Vanna RAG-based SQL generation — called by OpenWebUI SQL pipeline",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_bearer = HTTPBearer(auto_error=False)

def _check_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer),
) -> None:
    if not _SQL_API_KEY:
        return
    if not credentials or credentials.credentials != _SQL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Vanna singleton ───────────────────────────────────────────────────────────
_vanna_instance = None
_vanna_connected_db: Optional[str] = None   # which DB vanna is currently wired to


def _get_vanna(db: str):
    """
    Return (and lazily create) the Vanna instance.
    Re-connects if a different DB is requested.
    """
    global _vanna_instance, _vanna_connected_db

    if _vanna_instance is None:
        try:
            # prefer local fork (same path as vanna_calls.py)
            import sys, os as _os
            sys.path.insert(0, _os.path.dirname(_os.path.dirname(__file__)))
            from remote import VannaDefault       # type: ignore
        except ImportError:
            from vanna.remote import VannaDefault  # type: ignore

        _vanna_instance = VannaDefault(api_key=_VANNA_KEY, model=_VANNA_MODEL)

    # Reconnect to DB if needed
    if _vanna_connected_db != db:
        _connect_vanna(_vanna_instance, db)
        _vanna_connected_db = db

    return _vanna_instance


def _connect_vanna(vn, db: str) -> None:
    """Best-effort DB connection. Silently skips if env vars are missing."""
    try:
        if db == "Snowflake":
            vn.connect_to_snowflake(
                account=os.environ["SNOWFLAKE_ACCOUNT"],
                username=os.environ["SNOWFLAKE_USERNAME"],
                password=os.environ["SNOWFLAKE_PASSWORD"],
                database=os.environ["SNOWFLAKE_DATABASE"],
                warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", ""),
            )
        elif db in ("Redshift", "dbt"):
            vn.connect_to_postgres(
                host=os.environ["REDSHIFT_HOST"],
                dbname=os.environ["REDSHIFT_DBNAME"],
                user=os.environ["REDSHIFT_USER"],
                password=os.environ["REDSHIFT_PASSWORD"],
                port=int(os.getenv("REDSHIFT_PORT", "5439")),
            )
        elif db == "SQLite3":
            path = os.getenv("SQLITE_DB_PATH", "")
            if path:
                vn.connect_to_sqlite(path)
        elif db == "DuckDB":
            import duckdb
            path = os.getenv("DUCKDB_DB_PATH", ":memory:")
            conn = duckdb.connect(path)
            vn.connect_to_duckdb(url=f"duckdb:///{path}" if path != ":memory:" else "duckdb:///:memory:")
        elif db == "Athena":
            from pyathena import connect as _ac
            vn.run_sql = lambda sql: _athena_run_sql(sql)   # override run_sql
        elif db == "BigQuery":
            project = os.environ["BIGQUERY_PROJECT"]
            dataset = os.getenv("BIGQUERY_DATASET", "")
            vn.connect_to_bigquery(project_id=project)
    except KeyError as e:
        print(f"[text2sql_api] Missing env var for {db}: {e} — run_sql will be disabled")
    except Exception as exc:
        print(f"[text2sql_api] Could not connect Vanna to {db}: {exc}")


def _athena_run_sql(sql: str):
    """Execute SQL on Athena, returning a pandas DataFrame."""
    import pandas as pd
    from pyathena import connect as athena_connect

    conn = athena_connect(
        s3_staging_dir=os.environ["ATHENA_S3_STAGING_DIR"],
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        work_group=os.getenv("ATHENA_WORKGROUP", "primary"),
        schema_name=os.getenv("ATHENA_DATABASE", "default"),
    )
    return pd.read_sql(sql, conn)


# ── DB-specific SQL assumption strings ───────────────────────────────────────

def _build_assumption(db: str) -> str:
    if db == "Snowflake":
        database = os.getenv("SNOWFLAKE_DATABASE", "")
        schema   = os.getenv("SNOWFLAKE_SCHEMA", "")
        return (
            f" Use Snowflake SQL. Assume database is {database} and schema is {schema}. "
            f'Note: "_" may be used to split fields in names.'
        )

    if db == "dbt":
        database = os.getenv("SNOWFLAKE_DATABASE", "")
        schema   = os.getenv("SNOWFLAKE_SCHEMA", "")
        return (
            f" Generate dbt SQL (dbt model style) for Snowflake. "
            f"Assume database is {database} and schema is {schema}."
        )

    if db == "Redshift":
        database = os.getenv("REDSHIFT_DBNAME", "")
        schema   = os.getenv("REDSHIFT_SCHEMA", "")
        return (
            f" Use Amazon Redshift SQL. "
            f"Assume database is {database} and schema is {schema}."
        )

    if db in ("Athena", "Athena (Iceberg)"):
        databases_str = os.getenv("ATHENA_DATABASES", "")
        default_db    = os.getenv("ATHENA_DATABASE", "")

        if databases_str:
            dbs     = [d.strip() for d in databases_str.split(",") if d.strip()]
            db_hint = f"Available databases include: {', '.join(dbs)}."
        elif default_db:
            db_hint = f"Assume default database is {default_db}."
        else:
            db_hint = ""

        # Optional: inject Glue schema context
        schema_context = ""
        try:
            from athena_metadata import build_schema_context   # type: ignore
            databases = (
                [d.strip() for d in databases_str.split(",") if d.strip()]
                if databases_str
                else ([default_db] if default_db else [])
            )
            if databases:
                schema_context = build_schema_context(
                    databases=databases,
                    region_name=os.getenv("AWS_REGION"),
                    max_tables_per_db=int(os.getenv("ATHENA_MAX_TABLES_PER_DB", "25")),
                    max_columns_per_table=int(os.getenv("ATHENA_MAX_COLS_PER_TABLE", "30")),
                )
        except Exception:
            pass

        return (
            f"Use Amazon Athena dialect (Presto/Trino-style), LIMIT 20 rows. "
            f"{db_hint} {schema_context} "
            "Prefer partition filters and LIMIT when possible to reduce scanned data."
        )

    if db == "DuckDB":
        return " Use DuckDB SQL dialect."

    if db == "BigQuery":
        project = os.getenv("BIGQUERY_PROJECT", "")
        dataset = os.getenv("BIGQUERY_DATASET", "")
        return (
            f" Use Google BigQuery standard SQL. "
            f"Project: {project}, Dataset: {dataset}."
        )

    return ""


# ── Request / Response models ─────────────────────────────────────────────────

class SqlQueryRequest(BaseModel):
    question:       str
    db:             Optional[str] = None
    run_sql:        bool          = True
    generate_chart: bool          = False
    generate_dbt:   bool          = False


class SqlQueryResponse(BaseModel):
    sql:                str
    db:                 str
    rows:               Optional[List[Dict[str, Any]]] = None
    row_count:          Optional[int]                  = None
    summary:            Optional[str]                  = None
    followup_questions: Optional[List[str]]            = None
    error:              Optional[str]                  = None
    is_valid:           bool                           = True


class HealthResponse(BaseModel):
    status:        str
    vanna_model:   str
    default_db:    str
    supported_dbs: List[str]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        vanna_model=_VANNA_MODEL,
        default_db=_DEFAULT_DB,
        supported_dbs=SUPPORTED_DBS,
    )


@app.post(
    "/v1/sql/query",
    response_model=SqlQueryResponse,
    dependencies=[Depends(_check_api_key)],
)
def sql_query(req: SqlQueryRequest) -> SqlQueryResponse:
    """
    Convert a natural language question to SQL, optionally execute it,
    and return a structured result with summary + follow-up suggestions.
    """
    db = (req.db or _DEFAULT_DB).strip()

    try:
        vn = _get_vanna(db)
    except Exception as exc:
        return SqlQueryResponse(sql="", db=db, error=f"Vanna init failed: {exc}", is_valid=False)

    # ── Generate SQL ──────────────────────────────────────────────────────────
    assumption    = _build_assumption(db)
    question_full = req.question + assumption

    try:
        sql = vn.generate_sql(question=question_full, allow_llm_to_see_data=True)
    except Exception as exc:
        return SqlQueryResponse(sql="", db=db, error=str(exc), is_valid=False)

    # ── Validate ──────────────────────────────────────────────────────────────
    try:
        is_valid = bool(vn.is_sql_valid(sql=sql))
    except Exception:
        is_valid = True   # assume valid if the check itself throws

    # ── Execute (optional) ────────────────────────────────────────────────────
    rows: Optional[List[Dict[str, Any]]] = None
    row_count: Optional[int]             = None
    error: Optional[str]                 = None

    if req.run_sql and is_valid:
        try:
            df    = vn.run_sql(sql=sql)
            rows  = df.to_dict(orient="records") if df is not None else []
            rows  = rows[:_MAX_ROWS]
            row_count = len(rows)
        except Exception as exc:
            error = str(exc)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary: Optional[str] = None
    if rows is not None and not error:
        try:
            import pandas as pd
            df_s   = pd.DataFrame(rows)
            summary = vn.generate_summary(question=req.question, df=df_s)
        except Exception:
            pass

    # ── Follow-up questions ───────────────────────────────────────────────────
    followup_questions: Optional[List[str]] = None
    if rows is not None:
        try:
            import pandas as pd
            df_f              = pd.DataFrame(rows) if rows else pd.DataFrame()
            followup_questions = vn.generate_followup_questions(
                question=req.question, sql=sql, df=df_f
            )
        except Exception:
            pass

    return SqlQueryResponse(
        sql=sql,
        db=db,
        rows=rows,
        row_count=row_count,
        summary=summary,
        followup_questions=followup_questions,
        error=error,
        is_valid=is_valid,
    )

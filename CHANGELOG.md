# Changelog

All notable changes to DataPAI are documented here.

---

## [0.4.0] — 2026-02-23

### OpenWebUI Pipeline Suite

Full OpenWebUI integration — the entire Streamlit feature set is now available as pipelines inside OpenWebUI, backed by two FastAPI services running on EC2 #2.

#### New backend services (EC2 #2)

| Service | Port | File |
|---|---|---|
| RAG API (LanceDB) | 8100 | `agents/rag_api.py` |
| Text2SQL API (Vanna) | 8101 | `agents/text2sql_api.py` |

**RAG API** (`agents/rag_api.py`)
- `POST /v1/rag/retrieve` — LanceDB vector search only (no LLM call); used by OpenWebUI pipeline so EC2 #3 GPU handles generation locally
- `POST /v1/rag/query` — full RAG: retrieval + Ollama answer (used by Streamlit)
- `POST /v1/rag/ingest` — upload a document into LanceDB
- `GET  /v1/rag/documents` — list ingested documents
- `DELETE /v1/rag/documents/{filename}` — remove a document
- `GET  /health` — liveness + LanceDB + Ollama reachability

**Text2SQL API** (`agents/text2sql_api.py`)
- `POST /v1/sql/query` — NL question → Vanna SQL → optional execution → rows + summary + follow-up questions + optional dbt model code
- `GET  /health` — liveness + supported DBs
- Supports: Snowflake, Redshift, Athena, DuckDB, SQLite3, BigQuery, dbt
- Injects dbt FAISS metadata context before SQL generation (mirrors Streamlit behaviour)
- `DBT_ENABLED` / `DBT_METADATA_TOP_K` env vars control context injection

#### New OpenWebUI pipelines (upload to EC2 #3 via Admin → Pipelines)

**`agents/openwebui_rag_pipeline.py`** — "DataPAI RAG (LanceDB)"
- Split architecture: calls `/v1/rag/retrieve` on EC2 #2 (no LLM), then streams generation via `OLLAMA_HOST` (EC2 #3 GPU)
- `DATAPAI_RAG_STREAM` valve — token streaming on/off (default: on)
- Appends source citations after the streamed answer
- Fallback to plain Ollama if RAG API is unreachable

**`agents/openwebui_sql_pipeline.py`** — "DataPAI Text2SQL"
- NL → SQL → results table + summary + follow-up questions
- `DATAPAI_SQL_GENERATE_DBT` valve — also return a dbt model (default: off)
- `[db:Redshift]` inline tag overrides the default target database
- Renders SQL, result table, summary, follow-ups, and optional dbt model in markdown

**`agents/openwebui_combined_pipeline.py`** — "DataPAI Smart Router"
- Single model that intelligently routes each message:
  - `sql` → Text2SQL API (EC2 #2) — for data/metrics/report questions
  - `rag` → RAG retrieve + Ollama stream — for knowledge-base/doc questions
  - `chat` → Ollama stream directly — for general assistant questions
- LLM classifier (fast, `num_predict=5`) with regex keyword fallback
- All three paths stream tokens back to OpenWebUI
- `[db:Athena]` inline tag forces SQL on a specific database
- All valves configurable in OpenWebUI pipeline settings UI

#### Feature parity: Streamlit vs OpenWebUI

| Feature | Streamlit | OpenWebUI |
|---|---|---|
| SQL generation + validation + execution | yes | yes |
| Results table + row count | yes | yes |
| Summary | yes | yes |
| Follow-up question suggestions | yes | yes |
| Multi-DB: Snowflake, Redshift, Athena, DuckDB, SQLite3, BigQuery, dbt | yes | yes |
| Athena: Glue schema context + Iceberg detection | yes | yes |
| dbt FAISS metadata context injection | yes | yes |
| dbt code generation | yes | yes (opt-in valve) |
| RAG chat (LanceDB knowledge base) | yes | yes |
| Smart SQL / RAG / chat routing | no | yes |
| Token streaming | no | yes |
| Plotly interactive charts | yes | no (no renderer) |
| Lightdash integration | yes | no |

---

### Athena support

- `connect_db.py` — `connect_athena()` using PyAthena + EC2 IAM role (no access keys required)
- `athena_metadata.py` — Glue Data Catalog schema inspector with automatic Iceberg format detection; Iceberg-specific SQL hints only appear when Iceberg tables are present
- `vanna_calls.py` — Athena branch: Presto/Trino dialect, Glue schema context injection, partition filter hints, LIMIT advice
- `app.py`, `app_text2sql.py` — "Athena" added to database selector

---

### Agents package refactor

- `agents/__init__.py` — package marker; all agents now importable as a package
- Converted bare imports to relative imports throughout `agents/`
- `agents/tooling/registry.py` — `@tool` decorator and `call_tool_from_json_call`
- `agents/tooling/loader.py` — idempotent tool loader called on agent init
- `agents/llm_client.py` — `DATAPAI_LLM_ENABLED` env var for deterministic executor mode
- `agents/dbt_agent.py` — rewritten as planner + executor (LLM plans, Python writes files deterministically)
- `agents/supervisor_agent.py` — integrated `WorkflowPlan` + `PLAN_SYSTEM_PROMPT`

---

### ETL workflow service

- `etl/contracts.py` — data contract models
- `etl/plans.py` — workflow plan schema
- `etl/run_context.py` — runtime execution context
- `etl/workflow_runner.py` — AG2-style workflow runner
- `etl/workflow_service.py` — workflow service layer

---

### Bug fixes

- `vanna_calls.py` — removed duplicate `generate_sql_cached` function; fixed `else default_db:` syntax error; fixed undefined `database` variable in Athena branch
- `connect_db.py` — fixed db_type string mismatch (`"Athena (support Parquet/Iceberg)"` → `"Athena"`); fixed wrong function name (`connect_to_athena` → `connect_athena`)
- `dbt_metadata.py` — guarded `update_dbt_metadata()` behind `if __name__ == "__main__"` to prevent execution on import
- `README.md` — updated entry point to `app_all.py`
- `.gitignore` — fixed `__pycache__/` → `__pycache__` pattern

---

## [0.3.0] — 2026-02-xx

### AG2 ETL Swarm

Compliance-first, audited, cost-controlled end-to-end pipeline using AG2 multi-agent framework.

---

## [0.2.0] — earlier

### AI Agents

Initial AI agent framework with dbt, Snowflake ingest, and file ingest agents.

---

## [0.1.0] — earlier

### Initial release

Streamlit Text2SQL app with Vanna RAG-based SQL generation for Snowflake, Redshift, DuckDB, SQLite3, and BigQuery.

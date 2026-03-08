# dbt-traceability

dbt project for the **Datap.ai Trace Ledger** — governance and audit reporting
on Snowflake, queryable via Lightdash.

## Project structure

```
dbt-traceability/
├── dbt_project.yml          # project config (profile: datapai_snowflake)
├── packages.yml             # dbt_utils dependency
├── models/
│   ├── sources.yml          # raw source: DATAPAI_TRACES.DATAPAI_TRACE_EVENTS
│   ├── staging/
│   │   ├── stg_trace_events.sql     # typed, quality-filtered view over raw table
│   │   └── schema.yml
│   ├── intermediate/
│   │   ├── int_request_timeline.sql  # per-event timeline with timing
│   │   ├── int_session_summary.sql   # per-session aggregation
│   │   └── schema.yml
│   └── reporting/           # Lightdash-facing tables
│       ├── rpt_request_summary.sql   # one row per request
│       ├── rpt_policy_violations.sql # blocked / failed events
│       ├── rpt_sql_risk_summary.sql  # SQL risk by user/datasource/day
│       ├── rpt_model_usage.sql       # LLM model usage by user/day
│       ├── rpt_session_overview.sql  # one row per session
│       └── schema.yml                # Lightdash meta (dimensions, metrics)
└── tests/                   # generic dbt tests
```

## Prerequisites

Uses the **`datapai_snowflake`** profile from `../dbt-demo/profiles.yml`.
Set these environment variables (same as `dbt-demo`):

```bash
SNOWFLAKE_ACCOUNT=<your-account>
SNOWFLAKE_USER=<service-user>
SNOWFLAKE_PASSWORD=<password>
SNOWFLAKE_ROLE=<role-with-insert-create>
SNOWFLAKE_DATABASE=<database>
SNOWFLAKE_WAREHOUSE=<warehouse>
SNOWFLAKE_SCHEMA=DATAPAI_TRACES     # schema where the Python backend writes
```

## Running

```bash
cd dbt-traceability
dbt deps                      # install dbt_utils
dbt run                       # build all models
dbt test                      # run tests
dbt docs generate && dbt docs serve   # view docs
```

Or run specific layers:

```bash
dbt run --select staging       # just staging views
dbt run --select reporting     # just reporting tables (for Lightdash)
dbt run --select +rpt_request_summary   # with all upstream deps
```

## Lightdash integration

1. Connect Lightdash to Snowflake using the same credentials.
2. Point Lightdash at this dbt project (`dbt-traceability/`).
3. The `reporting/` models have `meta.label` and `meta.group_label` set for
   clean grouping in the Lightdash UI under **"Traceability"**.
4. Key dashboards to build in Lightdash:
   - **Request Activity** — `rpt_request_summary` (filter by tenant/user/date)
   - **Governance Violations** — `rpt_policy_violations` (risk review queue)
   - **SQL Risk** — `rpt_sql_risk_summary` (block rates per datasource)
   - **Model Usage** — `rpt_model_usage` (LLM cost/usage governance)
   - **Session Overview** — `rpt_session_overview` (user activity)

## Raw source

The raw `DATAPAI_TRACE_EVENTS` table is written by:
- `SnowflakeTraceLedgerBackend` in `traceability/backends/snowflake_backend.py`

The Python backend is the **only writer**.  dbt models are **read-only** on top
of the raw table.  Never write to `DATAPAI_TRACE_EVENTS` from dbt.

## Governance notes

- All models filter by `tenant_id` to prevent cross-tenant data bleed.
- Raw `input_summary` / `output_summary` fields are already redacted by the
  Python backend — no PII or secrets are stored in the raw table.
- `sql_hash` stores SHA-256 of the SQL, not the SQL itself.
- `prompt_hash` stores SHA-256 of the prompt, not the prompt itself.

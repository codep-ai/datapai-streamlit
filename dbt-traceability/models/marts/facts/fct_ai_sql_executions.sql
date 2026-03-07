{{
  config(
    materialized = 'table',
    tags         = ['traceability', 'marts', 'fact']
  )
}}

/*
  fct_ai_sql_executions

  One row per SQL execution event (event_type = 'sql_executed').
  Audit trail for every query that actually ran against a datasource.

  Compliance:
    sql_text — exact SQL that ran (NOT result rows — data never stored)
    This table is the definitive record for data access audits.

  Note: A SQL may appear in fct_ai_sql_generations but NOT here (if blocked).
        Compliance auditors should join both tables to see generation vs execution.
*/

with sql_events as (

    select *
    from {{ ref('stg_trace_events') }}
    where event_type = 'sql_executed'

)

select
    -- Surrogate key
    trace_id,
    parent_trace_id,

    -- Foreign keys
    tenant_id,
    workspace_id,
    user_id,
    session_id,
    request_id,

    -- Event timing
    event_timestamp           as executed_at,
    date_trunc('day', event_timestamp) as executed_date,

    -- ── COMPLIANCE: verbatim SQL that actually ran ─────────────────────
    sql_text,
    sql_hash,

    -- ── COMPLIANCE: sensitivity and PII at execution time ─────────────
    sensitivity_level,
    pii_detected,
    pii_fields,

    -- Datasource context (FK → dim_ai_datasources)
    datasource_type,
    datasource_name,

    -- Model that generated this SQL (FK → dim_ai_models)
    model_name,

    -- Governance
    policy_result,
    status,
    error_code,
    error_message,
    etl_run_id

from sql_events

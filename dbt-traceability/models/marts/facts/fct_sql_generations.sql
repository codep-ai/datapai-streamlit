{{
  config(
    materialized = 'table',
    tags         = ['traceability', 'marts', 'fact']
  )
}}

/*
  fct_sql_generations

  One row per SQL generation event (event_type = 'sql_generated').
  Event-grain fact table for SQL audit and risk analysis.

  Compliance:
    sql_text — verbatim SQL for regulatory audit (NOT result rows)
    sql_hash — fingerprint for deduplication and pattern analysis
    sensitivity_level, pii_detected, pii_fields — data risk classification

  Foreign keys:
    tenant_id + user_id         → dim_users
    tenant_id + datasource_name → dim_datasources
    tenant_id + model_name      → dim_models
*/

with sql_events as (

    select *
    from {{ ref('stg_trace_events') }}
    where event_type = 'sql_generated'

)

select
    -- Surrogate key
    trace_id,
    parent_trace_id,

    -- Foreign keys (join to dimensions)
    tenant_id,
    workspace_id,
    user_id,
    session_id,
    request_id,

    -- Event timing
    event_timestamp           as generated_at,
    date_trunc('day', event_timestamp) as generated_date,

    -- ── COMPLIANCE: verbatim SQL for audit ─────────────────────────────
    sql_text,
    sql_hash,

    -- ── COMPLIANCE: sensitivity and PII ───────────────────────────────
    sensitivity_level,
    pii_detected,
    pii_fields,

    -- What the model decided / did
    ai_action_summary,

    -- Datasource context (FK → dim_datasources)
    datasource_type,
    datasource_name,

    -- Model used (FK → dim_models)
    model_name,

    -- Derived risk level for this SQL
    case
        when sensitivity_level = 'CRITICAL'     then 'CRITICAL'
        when sensitivity_level = 'HIGH'         then 'HIGH'
        when pii_detected = true                then 'MEDIUM'
        else 'LOW'
    end                       as sql_risk_level,

    -- Governance
    policy_result,
    status,
    error_code,
    error_message,
    etl_run_id

from sql_events

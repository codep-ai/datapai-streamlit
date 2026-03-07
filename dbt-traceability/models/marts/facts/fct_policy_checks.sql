{{
  config(
    materialized = 'table',
    tags         = ['traceability', 'marts', 'fact']
  )
}}

/*
  fct_policy_checks

  One row per governance policy check event.
  Includes both passed and failed/blocked checks.

  Powers:
    - Lightdash "Policy Enforcement" dashboard
    - Admin "Policy Check Audit" (spec G)
    - Compliance: what was checked, what was decided, why it was blocked

  Grain: one policy evaluation event
*/

with policy_events as (

    select *
    from {{ ref('stg_trace_events') }}
    where event_type in (
        'policy_check_started',
        'policy_check_passed',
        'policy_check_failed',
        'sql_blocked',
        'agent_boundary_violation'
    )

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
    event_timestamp           as checked_at,
    date_trunc('day', event_timestamp) as check_date,

    -- What was checked
    event_type,
    policy_result,

    -- ── COMPLIANCE: context for the check ─────────────────────────────
    -- Verbatim SQL or question that triggered the check
    question_text,
    sql_text,
    sql_hash,
    sensitivity_level,
    pii_detected,
    pii_fields,

    -- ── AI AGENTIC: boundary violations ───────────────────────────────
    agent_name,
    ai_action_summary,
    boundary_violated,
    risk_flags,

    -- Outcome
    is_policy_blocked,
    is_boundary_violation,
    status,
    error_code,
    error_message,

    -- Classification
    case
        when is_boundary_violation              then 'AGENT_BOUNDARY_VIOLATION'
        when event_type = 'sql_blocked'         then 'SQL_BLOCKED'
        when event_type = 'policy_check_failed' then 'POLICY_FAILED'
        when event_type = 'policy_check_passed' then 'PASSED'
        else 'OTHER'
    end                       as check_outcome,

    datasource_name,
    model_name,
    etl_run_id

from policy_events

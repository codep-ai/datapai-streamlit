{{
  config(
    materialized = 'table',
    tags         = ['traceability', 'reporting', 'published']
  )
}}

/*
  rpt_session_overview

  One row per session for the governance console and Lightdash dashboard.

  Powers:
    - Lightdash "Session Overview" dashboard
    - Admin user/workspace overview (spec G)
*/

select
    tenant_id,
    workspace_id,
    user_id,
    session_id,
    session_started_at,
    session_last_activity_at,
    session_duration_minutes,
    total_requests,
    total_events,
    distinct_models_used,
    models_used,
    datasources_accessed,
    sql_generated_count,
    sql_executed_count,
    sql_blocked_count,
    policy_blocks,
    error_events,
    human_corrections,
    session_risk_level
from {{ ref('int_session_summary') }}
order by session_started_at desc

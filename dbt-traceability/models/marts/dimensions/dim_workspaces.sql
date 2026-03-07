{{
  config(
    materialized = 'table',
    tags         = ['traceability', 'marts', 'dimension']
  )
}}

/*
  dim_workspaces

  One row per (tenant_id, workspace_id).
  Aggregates activity and risk profile across the workspace lifetime.

  Powers:
    - Lightdash workspace-level governance overview
    - Tenant admin dashboard
*/

with events as (

    select * from {{ ref('stg_trace_events') }}

),

workspaces as (

    select
        tenant_id,
        workspace_id,

        -- Lifecycle
        min(event_timestamp)               as first_active_at,
        max(event_timestamp)               as last_active_at,

        -- Users in workspace
        count(distinct user_id)            as distinct_users,
        count(distinct session_id)         as total_sessions,
        count(distinct request_id)         as total_requests,

        -- Models used
        count(distinct model_name)         as distinct_models_used,
        listagg(distinct model_name, ', ')
            within group (order by model_name) as models_used,

        -- Datasources accessed
        count(distinct datasource_name)    as distinct_datasources,
        listagg(distinct datasource_name, ', ')
            within group (order by datasource_name) as datasources_accessed,

        -- SQL profile
        count_if(event_type = 'sql_generated') as sql_generated,
        count_if(event_type = 'sql_executed')  as sql_executed,
        count_if(event_type = 'sql_blocked')   as sql_blocked,

        -- AI agentic
        count_if(is_agent_event)           as agent_events,
        count_if(is_boundary_violation)    as boundary_violations,

        -- Risk
        count_if(is_high_sensitivity)      as high_sensitivity_events,
        count_if(pii_detected = true)      as pii_events,
        count_if(is_policy_blocked)        as policy_blocks,

        case
            when count_if(is_boundary_violation)      > 0 then 'CRITICAL'
            when count_if(event_type = 'sql_blocked') > 0 then 'HIGH'
            when count_if(is_policy_blocked)          > 0 then 'MEDIUM'
            when count_if(pii_detected = true)        > 0 then 'MEDIUM'
            else 'LOW'
        end                                as workspace_risk_tier

    from events
    group by 1, 2

)

select * from workspaces

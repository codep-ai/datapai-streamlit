{{
  config(
    materialized = 'table',
    tags         = ['traceability', 'marts', 'dimension']
  )
}}

/*
  dim_ai_datasources

  One row per (tenant_id, datasource_name).
  Tracks every datasource connected to the platform with usage and risk profile.

  Powers:
    - Lightdash "Datasource Risk" panel
    - Admin datasource access audit
    - Compliance: which datasources were accessed, by whom, how often
*/

with events as (

    select *
    from {{ ref('stg_trace_events') }}
    where datasource_name is not null

),

datasources as (

    select
        tenant_id,
        workspace_id,
        datasource_type,
        datasource_name,

        -- Lifecycle
        min(event_timestamp)               as first_accessed_at,
        max(event_timestamp)               as last_accessed_at,

        -- Access volume
        count(distinct user_id)            as distinct_users,
        count(distinct session_id)         as distinct_sessions,
        count(distinct request_id)         as distinct_requests,

        -- SQL profile
        count_if(event_type = 'sql_generated') as sql_generated,
        count_if(event_type = 'sql_executed')  as sql_executed,
        count_if(event_type = 'sql_blocked')   as sql_blocked,
        count(distinct sql_hash)               as distinct_sql_patterns,

        -- AI agentic usage
        count_if(is_agent_event)               as agent_events,
        count_if(is_boundary_violation)        as boundary_violations,

        -- Risk / sensitivity
        count_if(sensitivity_level = 'CRITICAL') as critical_events,
        count_if(sensitivity_level = 'HIGH')     as high_sensitivity_events,
        count_if(pii_detected = true)            as pii_events,

        case
            when count_if(is_boundary_violation)      > 0 then 'CRITICAL'
            when count_if(event_type = 'sql_blocked') > 0 then 'HIGH'
            when count_if(sensitivity_level = 'HIGH') > 0 then 'HIGH'
            when count_if(pii_detected = true)        > 0 then 'MEDIUM'
            else 'LOW'
        end                                    as datasource_risk_tier

    from events
    group by 1, 2, 3, 4

)

select * from datasources

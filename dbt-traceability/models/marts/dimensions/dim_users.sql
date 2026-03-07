{{
  config(
    materialized = 'table',
    tags         = ['traceability', 'marts', 'dimension']
  )
}}

/*
  dim_users

  One row per (tenant_id, user_id).
  Aggregates lifetime activity for each user across all workspaces.

  Powers:
    - Lightdash "User Activity" drill-through
    - Admin user-level risk profile
    - Compliance: identify which users asked sensitive questions
*/

with events as (

    select * from {{ ref('stg_trace_events') }}

),

users as (

    select
        tenant_id,
        workspace_id,
        user_id,

        -- Lifecycle
        min(event_timestamp)               as first_seen_at,
        max(event_timestamp)               as last_seen_at,

        -- Activity volume
        count(distinct session_id)         as total_sessions,
        count(distinct request_id)         as total_requests,
        count(*)                           as total_events,

        -- SQL activity
        count_if(event_type = 'sql_generated') as sql_generated_count,
        count_if(event_type = 'sql_executed')  as sql_executed_count,
        count_if(event_type = 'sql_blocked')   as sql_blocked_count,

        -- AI agentic
        count_if(is_agent_event)           as agent_events,
        count_if(is_boundary_violation)    as boundary_violations,

        -- Compliance risk profile
        count_if(is_high_sensitivity)      as high_sensitivity_events,
        count_if(pii_detected = true)      as pii_events,
        count_if(is_policy_blocked)        as policy_blocks,

        -- Highest sensitivity level ever seen for this user
        case
            when count_if(sensitivity_level = 'CRITICAL') > 0 then 'CRITICAL'
            when count_if(sensitivity_level = 'HIGH')     > 0 then 'HIGH'
            when count_if(sensitivity_level = 'MEDIUM')   > 0 then 'MEDIUM'
            when count_if(sensitivity_level = 'LOW')      > 0 then 'LOW'
            else null
        end                                as highest_sensitivity_seen,

        -- Overall user risk tier
        case
            when count_if(is_boundary_violation)      > 0 then 'CRITICAL'
            when count_if(event_type = 'sql_blocked') > 0 then 'HIGH'
            when count_if(is_policy_blocked)          > 0 then 'MEDIUM'
            when count_if(pii_detected = true)        > 0 then 'MEDIUM'
            else 'LOW'
        end                                as user_risk_tier

    from events
    group by 1, 2, 3

)

select * from users

{{
  config(
    materialized = 'table',
    tags         = ['traceability', 'intermediate']
  )
}}

/*
  int_session_summary

  One row per (tenant_id, session_id) summarising all activity in that session.
  Used by admin governance console and Lightdash session dashboards.
*/

with events as (

    select * from {{ ref('stg_trace_events') }}

),

session_agg as (

    select
        tenant_id,
        workspace_id,
        user_id,
        session_id,

        -- Timing
        min(event_timestamp)                              as session_started_at,
        max(event_timestamp)                              as session_last_activity_at,
        datediff(
            'minute',
            min(event_timestamp),
            max(event_timestamp)
        )                                                 as session_duration_minutes,

        -- Volume
        count(distinct request_id)                        as total_requests,
        count(*)                                          as total_events,

        -- Models used
        count(distinct model_name)                        as distinct_models_used,
        listagg(distinct model_name, ', ')
            within group (order by model_name)            as models_used,

        -- Datasources accessed
        listagg(distinct datasource_name, ', ')
            within group (order by datasource_name)       as datasources_accessed,

        -- SQL activity
        count_if(is_sql_event)                            as sql_events,
        count_if(event_type = 'sql_generated')            as sql_generated_count,
        count_if(event_type = 'sql_executed')             as sql_executed_count,
        count_if(event_type = 'sql_blocked')              as sql_blocked_count,

        -- Governance
        count_if(is_policy_blocked)                       as policy_blocks,
        count_if(is_error)                                as error_events,
        count_if(is_human_correction)                     as human_corrections,

        -- AI agentic activity
        count_if(is_agent_event)                          as agent_events,
        count_if(is_boundary_violation)                   as boundary_violations,
        listagg(
            case when is_boundary_violation then risk_flags end, ', '
        ) within group (order by event_timestamp)         as boundary_violation_risks,

        -- Sensitivity / PII
        count_if(is_high_sensitivity)                     as high_sensitivity_events,
        count_if(pii_detected = true)                     as pii_events,

        -- Risk indicator (accounts for agentic boundary violations)
        case
            when count_if(is_boundary_violation)      > 0  then 'CRITICAL'
            when count_if(event_type = 'sql_blocked') > 0  then 'HIGH'
            when count_if(is_policy_blocked)          > 0  then 'MEDIUM'
            when count_if(is_high_sensitivity)        > 0  then 'MEDIUM'
            when count_if(pii_detected = true)        > 0  then 'MEDIUM'
            when count_if(is_error)                   > 0  then 'LOW'
            else 'NONE'
        end                                               as session_risk_level

    from events
    group by 1, 2, 3, 4

)

select * from session_agg

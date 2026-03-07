{{
  config(
    materialized = 'table',
    tags         = ['traceability', 'marts', 'dimension']
  )
}}

/*
  dim_ai_models

  One row per (tenant_id, model_name).
  Tracks LLM model usage, quality, and risk profile across the platform.

  Powers:
    - Lightdash "Model Performance" panel
    - Admin model cost / quality tracking
    - Compliance: which AI model made which decisions
*/

with events as (

    select *
    from {{ ref('stg_trace_events') }}
    where model_name is not null

),

models as (

    select
        tenant_id,
        workspace_id,
        model_name,

        -- Lifecycle
        min(event_timestamp)               as first_used_at,
        max(event_timestamp)               as last_used_at,

        -- Usage volume
        count(distinct user_id)            as distinct_users,
        count(distinct session_id)         as distinct_sessions,
        count(distinct request_id)         as distinct_requests,
        count(*)                           as total_invocations,

        -- SQL generation profile
        count_if(event_type = 'sql_generated') as sql_generated,
        count_if(event_type = 'sql_blocked')   as sql_blocked,

        -- Quality signals
        count_if(is_human_correction)      as human_corrections,
        count_if(is_error)                 as error_events,
        round(
            count_if(is_error) * 100.0
            / nullif(count(*), 0),
        1)                                 as error_rate_pct,

        -- Risk profile — does this model generate high-risk output?
        count_if(is_boundary_violation)    as boundary_violations,
        count_if(sensitivity_level = 'CRITICAL') as critical_sensitivity_events,
        count_if(pii_detected = true)      as pii_events

    from events
    group by 1, 2, 3

)

select * from models

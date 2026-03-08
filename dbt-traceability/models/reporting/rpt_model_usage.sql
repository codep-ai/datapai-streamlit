{{
  config(
    materialized = 'table',
    tags         = ['traceability', 'reporting', 'published']
  )
}}

/*
  rpt_model_usage

  LLM model usage breakdown per tenant, workspace, and user.

  Powers:
    - Lightdash "Model Usage" dashboard
    - Admin "Model Usage Summary" (spec G)
*/

with model_events as (

    select *
    from {{ ref('stg_trace_events') }}
    where event_type = 'model_invoked'
      and model_name is not null

),

summary as (

    select
        tenant_id,
        workspace_id,
        user_id,
        model_name,
        date_trunc('day', event_timestamp) as usage_date,

        count(*)                           as invocations,
        count(distinct session_id)         as distinct_sessions,
        count(distinct request_id)         as distinct_requests,
        count_if(is_error)                 as failed_invocations,
        round(
            count_if(is_error) * 100.0
            / nullif(count(*), 0),
        1)                                 as failure_rate_pct

    from model_events
    group by 1, 2, 3, 4, 5

)

select * from summary
order by usage_date desc, invocations desc

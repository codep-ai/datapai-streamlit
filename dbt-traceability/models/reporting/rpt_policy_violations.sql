{{
  config(
    materialized = 'table',
    tags         = ['traceability', 'reporting', 'published', 'governance']
  )
}}

/*
  rpt_policy_violations

  All policy-blocked and failed events, with context for review.

  Powers:
    - Lightdash "Governance Violations" dashboard
    - Admin "Blocked Action Review" queue (spec G)
    - Compliance audit export
*/

with events as (

    select * from {{ ref('stg_trace_events') }}

),

violations as (

    select
        trace_id,
        parent_trace_id,
        tenant_id,
        workspace_id,
        user_id,
        session_id,
        request_id,
        event_type,
        event_timestamp,
        actor_id,
        datasource_type,
        datasource_name,
        model_name,
        tool_name,
        policy_result,
        input_summary,
        output_summary,
        sql_hash,
        status,
        error_code,
        error_message,
        etl_run_id,

        -- Classification
        case
            when event_type = 'sql_blocked'        then 'SQL_BLOCKED'
            when event_type = 'policy_check_failed' then 'POLICY_FAILED'
            when status = 'blocked'                 then 'ACTION_BLOCKED'
            when status = 'failed'                  then 'ACTION_FAILED'
            else 'UNKNOWN'
        end as violation_type,

        date_trunc('day', event_timestamp) as violation_date

    from events
    where
        is_policy_blocked = true
        or is_error = true
        or event_type in ('sql_blocked', 'policy_check_failed')

)

select * from violations
order by event_timestamp desc

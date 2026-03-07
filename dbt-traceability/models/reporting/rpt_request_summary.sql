{{
  config(
    materialized = 'table',
    tags         = ['traceability', 'reporting', 'published']
  )
}}

/*
  rpt_request_summary

  One row per (tenant_id, request_id) — a summary of every user request.

  Powers:
    - Lightdash "Request Activity" dashboard
    - Admin governance console request search
    - Audit export
*/

with timeline as (

    select * from {{ ref('int_request_timeline') }}

),

request_agg as (

    select
        tenant_id,
        workspace_id,
        user_id,
        session_id,
        request_id,
        etl_run_id,

        -- Timing
        min(event_timestamp)            as request_started_at,
        max(event_timestamp)            as request_ended_at,
        max(total_request_ms)           as total_request_ms,

        -- First question asked
        max(case when event_type = 'request_received'
            then input_summary end)     as user_question_summary,

        -- Final response
        max(case when event_type = 'response_returned'
            then output_summary end)    as response_summary,

        -- Model used
        max(model_name)                 as primary_model,

        -- SQL activity
        max(case when event_type = 'sql_generated'
            then output_summary end)    as sql_generated_summary,
        max(case when event_type = 'sql_generated'
            then sql_hash end)          as sql_hash,
        count_if(event_type = 'sql_generated')  as sql_generated_count,
        count_if(event_type = 'sql_executed')   as sql_executed_count,
        count_if(event_type = 'sql_blocked')    as sql_blocked_count,
        count_if(event_type = 'sql_validated')  as sql_validated_count,

        -- Datasources touched
        listagg(distinct datasource_name, ', ')
            within group (order by datasource_name) as datasources_touched,

        -- Governance
        count_if(is_policy_blocked)     as policy_blocks,
        max(case when is_policy_blocked
            then error_message end)     as policy_block_reason,
        count_if(is_error)              as error_count,
        count_if(is_human_correction)   as human_corrections,

        -- Risk level
        case
            when count_if(event_type = 'sql_blocked')  > 0 then 'HIGH'
            when count_if(is_policy_blocked)            > 0 then 'MEDIUM'
            when count_if(is_error)                     > 0 then 'LOW'
            else 'NONE'
        end                             as risk_level,

        -- Final status
        case
            when count_if(event_type = 'sql_blocked')   > 0 then 'blocked'
            when count_if(is_error)                      > 0 then 'failed'
            else 'ok'
        end                             as request_status

    from timeline
    group by 1, 2, 3, 4, 5, 6

)

select * from request_agg

{{
  config(
    materialized = 'table',
    tags         = ['traceability', 'reporting', 'published']
  )
}}

/*
  rpt_request_summary

  One row per (tenant_id, request_id) — full compliance record of a user request.

  Includes:
    - verbatim question_text (what the user actually asked)
    - verbatim sql_text of the primary SQL generated
    - sensitivity classification and PII detection
    - AI/agent actions taken
    - boundary violations if any
    - governance decisions

  SQL result rows are NEVER stored — only the query and metadata.

  Powers:
    - Lightdash "Request Activity" dashboard
    - Admin governance console request search
    - Compliance audit export (financial / government)
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
        min(event_timestamp)               as request_started_at,
        max(event_timestamp)               as request_ended_at,
        max(total_request_ms)              as total_request_ms,

        -- ── COMPLIANCE: verbatim question ──────────────────────────────────
        -- The exact question the user typed, for regulatory audit.
        max(case when event_type = 'request_received'
            then question_text end)        as question_text,

        -- ── COMPLIANCE: verbatim SQL ──────────────────────────────────────
        -- The exact SQL generated. Auditors can verify what was queried.
        max(case when event_type = 'sql_generated'
            then sql_text end)             as sql_generated_text,
        max(case when event_type = 'sql_executed'
            then sql_text end)             as sql_executed_text,
        max(case when event_type = 'sql_generated'
            then sql_hash end)             as sql_hash,

        -- ── COMPLIANCE: sensitivity and PII ───────────────────────────────
        max(sensitivity_level)             as highest_sensitivity_level,
        max(case when pii_detected = true then 1 else 0 end)
                                           as pii_detected,
        listagg(distinct pii_fields, ', ')
            within group (order by pii_fields)
                                           as all_pii_fields,

        -- ── AI AGENTIC: what the AI did ────────────────────────────────────
        listagg(case when is_agent_event then ai_action_summary end, ' | ')
            within group (order by event_timestamp)
                                           as agent_actions_taken,
        count_if(is_agent_event)           as agent_event_count,

        -- ── AI AGENTIC: boundary violations ───────────────────────────────
        max(case when is_boundary_violation then 1 else 0 end)
                                           as boundary_violated,
        listagg(case when is_boundary_violation then risk_flags end, ', ')
            within group (order by event_timestamp)
                                           as boundary_violation_risks,

        -- Model used
        max(model_name)                    as primary_model,

        -- Datasources accessed
        listagg(distinct datasource_name, ', ')
            within group (order by datasource_name)
                                           as datasources_touched,

        -- SQL volume
        count_if(event_type = 'sql_generated')  as sql_generated_count,
        count_if(event_type = 'sql_executed')   as sql_executed_count,
        count_if(event_type = 'sql_blocked')    as sql_blocked_count,

        -- Governance
        count_if(is_policy_blocked)        as policy_blocks,
        max(case when is_policy_blocked
            then error_message end)        as policy_block_reason,
        count_if(is_error)                 as error_count,
        count_if(is_human_correction)      as human_corrections,

        -- Risk level (accounts for both SQL and agentic risks)
        case
            when max(case when is_boundary_violation then 1 else 0 end) > 0 then 'CRITICAL'
            when max(sensitivity_level) = 'CRITICAL'                         then 'CRITICAL'
            when count_if(event_type = 'sql_blocked')                  > 0   then 'HIGH'
            when count_if(is_policy_blocked)                           > 0   then 'HIGH'
            when max(sensitivity_level) = 'HIGH'                             then 'HIGH'
            when max(case when pii_detected = true then 1 else 0 end)  > 0   then 'MEDIUM'
            when count_if(is_error)                                    > 0   then 'LOW'
            else 'NONE'
        end                                as risk_level,

        -- Final request status
        case
            when max(case when is_boundary_violation then 1 else 0 end) > 0 then 'violation'
            when count_if(event_type = 'sql_blocked')                   > 0 then 'blocked'
            when count_if(is_error)                                      > 0 then 'failed'
            else 'ok'
        end                                as request_status

    from timeline
    group by 1, 2, 3, 4, 5, 6

)

select * from request_agg

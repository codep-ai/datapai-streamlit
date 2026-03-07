{{
  config(
    materialized = 'view',
    tags         = ['traceability', 'staging']
  )
}}

/*
  stg_trace_events

  Staging model over the raw Snowflake trace events table.
  Casts types, renames nothing (keeping the raw field names stable),
  and applies lightweight quality filters.

  This view is the single entry point for all downstream traceability models.
  Lightdash connects to the reporting layer, not directly to this staging view.
*/

with source as (

    select * from {{ source('datapai_traces', 'datapai_trace_events') }}

),

staged as (

    select
        -- Identity
        trace_id,
        parent_trace_id,
        tenant_id,
        workspace_id,
        user_id,
        session_id,
        request_id,

        -- Event classification
        event_type,
        try_to_timestamp(event_timestamp)   as event_timestamp,
        actor_type,
        actor_id,

        -- Data source
        datasource_type,
        datasource_name,

        -- Model / tool
        model_name,
        tool_name,

        -- Governance
        policy_result,

        -- Content (already summarised / hashed by the Python backend)
        input_summary,
        output_summary,
        sql_hash,
        prompt_hash,
        context_refs,

        -- Outcome
        status,
        error_code,
        error_message,

        -- ETL bridge
        etl_run_id,

        -- Derived convenience columns
        case
            when event_type in (
                'sql_generated', 'sql_validated', 'sql_blocked', 'sql_executed'
            ) then true
            else false
        end                                 as is_sql_event,

        case
            when status in ('failed', 'blocked') then true
            else false
        end                                 as is_error,

        case
            when event_type = 'policy_check_failed' then true
            when status = 'blocked'                  then true
            else false
        end                                 as is_policy_blocked,

        case
            when event_type = 'human_feedback_received' then true
            else false
        end                                 as is_human_correction

    from source
    where
        -- Basic sanity — must have core identity fields
        trace_id    is not null
        and tenant_id   is not null
        and user_id     is not null
        and event_type  is not null

)

select * from staged

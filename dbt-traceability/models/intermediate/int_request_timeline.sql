{{
  config(
    materialized = 'table',
    tags         = ['traceability', 'intermediate']
  )
}}

/*
  int_request_timeline

  One row per (request_id, event) with cumulative timing and derived flags.
  Groups all events for a single user request into an ordered timeline.

  Used by:
    - rpt_request_summary (reporting)
    - rpt_sql_risk_summary (reporting)
    - Streamlit trace viewer (via direct Snowflake query)
*/

with events as (

    select * from {{ ref('stg_trace_events') }}

),

with_row_num as (

    select
        *,
        row_number() over (
            partition by tenant_id, request_id
            order by event_timestamp asc
        ) as event_sequence,

        min(event_timestamp) over (
            partition by tenant_id, request_id
        ) as request_started_at,

        max(event_timestamp) over (
            partition by tenant_id, request_id
        ) as request_ended_at,

        datediff(
            'millisecond',
            min(event_timestamp) over (partition by tenant_id, request_id),
            event_timestamp
        ) as ms_since_request_start

    from events

),

final as (

    select
        -- Identity
        tenant_id,
        workspace_id,
        user_id,
        session_id,
        request_id,

        -- Event
        trace_id,
        parent_trace_id,
        event_sequence,
        event_type,
        event_timestamp,
        actor_type,
        actor_id,

        -- Request-level timing
        request_started_at,
        request_ended_at,
        datediff('millisecond', request_started_at, request_ended_at) as total_request_ms,
        ms_since_request_start,

        -- Context
        datasource_type,
        datasource_name,
        model_name,
        tool_name,
        policy_result,

        -- Content
        input_summary,
        output_summary,
        sql_hash,

        -- Outcome flags
        status,
        is_sql_event,
        is_error,
        is_policy_blocked,
        is_human_correction,
        error_code,
        error_message,

        -- ETL bridge
        etl_run_id

    from with_row_num

)

select * from final

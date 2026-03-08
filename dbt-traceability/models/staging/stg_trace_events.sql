{{
  config(
    materialized = 'view',
    tags         = ['traceability', 'staging']
  )
}}

/*
  stg_trace_events

  Typed, quality-filtered view over the raw Snowflake trace events table.
  Exposes ALL compliance and agentic fields verbatim.

  Compliance design:
    question_text  — stored verbatim for regulatory audit (credentials pre-masked)
    sql_text       — stored verbatim (the query, NOT result rows)
    sensitivity_level, pii_detected, pii_fields — classification fields
    ai_action_summary — what the AI/agent did, not what data it returned
    boundary_violated, risk_flags — AI agentic security fields
*/

with source as (

    select * from {{ source('datapai_traces', 'datapai_trace_events') }}

),

staged as (

    select
        -- Identity (all events must have these)
        trace_id,
        parent_trace_id,
        tenant_id,
        workspace_id,
        user_id,
        session_id,
        request_id,

        -- Event classification
        event_type,
        try_to_timestamp(event_timestamp)    as event_timestamp,
        actor_type,
        actor_id,

        -- Data source
        datasource_type,
        datasource_name,

        -- Model / tool / agent
        model_name,
        tool_name,
        agent_name,

        -- Governance
        policy_result,

        -- ── COMPLIANCE FIELDS (verbatim for audit) ──────────────────────
        -- The original user question, exactly as asked.
        -- Credentials masked at ingestion; substantive content preserved.
        question_text,

        -- The exact SQL generated or executed.
        -- Never the result rows — the query only.
        sql_text,

        -- Sensitivity classification
        sensitivity_level,

        -- PII detection
        pii_detected,
        pii_fields,        -- JSON list of PII field names, e.g. ["email","ssn"]

        -- What the AI/agent did (action description, not data content)
        ai_action_summary,

        -- ── AI AGENTIC SECURITY FIELDS ───────────────────────────────────
        -- Did this agent action attempt to exceed allowed scope?
        boundary_violated,
        -- JSON list of detected risks, e.g. ["DDL_ATTEMPT","PII_SCHEMA_ACCESS"]
        risk_flags,

        -- Fingerprints
        sql_hash,
        prompt_hash,

        -- References
        context_refs,

        -- Outcome
        status,
        error_code,
        error_message,

        -- ETL bridge
        etl_run_id,

        -- ── Derived convenience flags ────────────────────────────────────
        case
            when event_type in (
                'sql_generated', 'sql_validated', 'sql_blocked', 'sql_executed'
            ) then true else false
        end                                  as is_sql_event,

        case
            when event_type in (
                'agent_action', 'agent_boundary_violation', 'tool_invoked'
            ) then true else false
        end                                  as is_agent_event,

        case
            when status in ('failed', 'blocked', 'violation') then true
            else false
        end                                  as is_error,

        case
            when event_type = 'policy_check_failed'      then true
            when event_type = 'agent_boundary_violation' then true
            when status in ('blocked', 'violation')       then true
            else false
        end                                  as is_policy_blocked,

        case
            when boundary_violated = true then true
            else false
        end                                  as is_boundary_violation,

        case
            when sensitivity_level in ('HIGH', 'CRITICAL') then true
            else false
        end                                  as is_high_sensitivity,

        case
            when event_type = 'human_feedback_received' then true
            else false
        end                                  as is_human_correction

    from source
    where
        trace_id   is not null
        and tenant_id  is not null
        and user_id    is not null
        and event_type is not null

)

select * from staged

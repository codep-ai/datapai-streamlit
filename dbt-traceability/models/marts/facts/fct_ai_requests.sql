{{
  config(
    materialized = 'table',
    tags         = ['traceability', 'marts', 'fact']
  )
}}

/*
  fct_ai_requests

  One row per (tenant_id, request_id) — the core fact table for all AI requests.

  Grain: one user request (may span many events internally).

  This is the primary table for Lightdash "Request Activity" dashboards and
  joins to all dimension tables via foreign keys.

  Compliance fields:
    question_text       — verbatim user question (credentials masked)
    sql_generated_text  — verbatim SQL generated (for audit)
    sql_executed_text   — verbatim SQL actually executed
    sensitivity_level   — highest sensitivity across the request
    pii_detected        — any PII detected in the request
    boundary_violated   — any agent boundary violation

  Foreign keys:
    tenant_id + user_id         → dim_users
    tenant_id + workspace_id    → dim_workspaces
    tenant_id + datasources_touched (multi-value, denormalised)
    tenant_id + primary_model   → dim_models
*/

select
    tenant_id,
    workspace_id,
    user_id,
    session_id,
    request_id,
    etl_run_id,

    -- Timing
    request_started_at,
    request_ended_at,
    total_request_ms,

    -- ── COMPLIANCE: verbatim question ──────────────────────────────────
    question_text,

    -- ── COMPLIANCE: verbatim SQL ───────────────────────────────────────
    sql_generated_text,
    sql_executed_text,
    sql_hash,

    -- ── COMPLIANCE: sensitivity ────────────────────────────────────────
    highest_sensitivity_level,
    pii_detected,
    all_pii_fields,

    -- ── AI AGENTIC ─────────────────────────────────────────────────────
    agent_actions_taken,
    agent_event_count,
    boundary_violated,
    boundary_violation_risks,

    -- Model and datasources
    primary_model,
    datasources_touched,

    -- SQL volume
    sql_generated_count,
    sql_executed_count,
    sql_blocked_count,

    -- Governance
    policy_blocks,
    policy_block_reason,
    error_count,
    human_corrections,

    -- Risk
    risk_level,
    request_status

from {{ ref('rpt_request_summary') }}

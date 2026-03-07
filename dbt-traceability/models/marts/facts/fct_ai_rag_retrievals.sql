{{
  config(
    materialized = 'table',
    tags         = ['traceability', 'marts', 'fact']
  )
}}

/*
  fct_ai_rag_retrievals

  One row per RAG (Retrieval-Augmented Generation) or memory retrieval event.
  Covers rag_retrieved and memory_retrieved events.

  Powers:
    - Lightdash "RAG Activity" panel
    - Audit of what context was retrieved for each request
    - Performance: retrieval latency and hit rate analysis

  Grain: one retrieval event (RAG chunk lookup or memory fetch)
*/

with retrieval_events as (

    select *
    from {{ ref('stg_trace_events') }}
    where event_type in (
        'rag_retrieved',
        'memory_retrieved'
    )

)

select
    -- Surrogate key
    trace_id,
    parent_trace_id,

    -- Foreign keys
    tenant_id,
    workspace_id,
    user_id,
    session_id,
    request_id,

    -- Event timing
    event_timestamp           as retrieved_at,
    date_trunc('day', event_timestamp) as retrieved_date,

    -- Retrieval type
    event_type,

    -- ── What was retrieved ─────────────────────────────────────────────
    -- context_refs: JSON list of retrieved chunk IDs / memory IDs
    context_refs,

    -- What the model was asking for (context for retrieval)
    ai_action_summary,

    -- ── COMPLIANCE: sensitivity of retrieved context ───────────────────
    sensitivity_level,
    pii_detected,
    pii_fields,

    -- Datasource the retrieval hit (vector store, memory store, etc.)
    datasource_type,
    datasource_name,

    -- Tool used for retrieval
    tool_name,
    model_name,

    -- Outcome
    status,
    error_code,
    error_message,
    etl_run_id

from retrieval_events

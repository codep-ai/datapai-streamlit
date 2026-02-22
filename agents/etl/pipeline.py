"""
AG2 Swarm-based ETL Pipeline with compliance-first architecture.

Agent flow (compliance is mandatory, not optional):

  orchestrator
    └─→ ingest_agent         load file → DuckDB
          └─→ compliance_agent  PII scan → mask → audit log → compliance report
                └─→ quality_agent   profile data, flag issues
                      └─→ transform_agent  generate dbt governance YAML + SQL model
                            └─→ orchestrator  final summary → TERMINATE

Guardrail layers applied before every LLM interaction:
  [L1] Prompt sanitization   — strip PII patterns from context sent to any LLM
  [L2] Bedrock Guardrails    — AWS-managed policy (Bedrock provider only)
  [L3] Response validation   — detect PII leakage in LLM outputs

Audit layers applied to every tool call automatically:
  ● Timing (duration_ms per tool)
  ● Success / failure + error message
  ● Data mutations (rows before/after every DuckDB write)
  ● LLM token usage + estimated cost (parsed from AG2 chat_result)
  ● Compliance events (PII detected/masked per column)

Cost control (three-level budget enforcement):
  ● Pre-run:  check day/month aggregate spend before starting
  ● Mid-run:  check per-run / day / month after every tool call
  ● Post-run: cost report with per-agent breakdown
  ● Auto model downgrade when approaching budget (paid → cheaper → ollama)
  ● Hard stop with CostBudgetExceeded when limit is reached

Usage:
    from agents.etl.pipeline import run_etl_pipeline

    result = run_etl_pipeline(
        request="Load /tmp/transactions.csv into DuckDB and generate a dbt model",
        context_variables={"dbt_project_dir": "/path/to/dbt_project"},
        budget=BudgetConfig(per_run_usd=0.50, per_day_usd=5.00),
    )
    print(result["compliance_status"])   # "COMPLIANT" | "NEEDS_REVIEW" | "NON_COMPLIANT"
    print(result["llm_provider"])        # which LLM was used
    print(result["cost_report"])         # per-agent token + cost breakdown
    print(result["audit_run_id"])        # UUID — query etl_agent_steps WHERE run_id=...
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from autogen import (
    AfterWorkOption,
    ConversableAgent,
    ON_CONDITION,
    AFTER_WORK,
    initiate_swarm_chat,
    register_hand_off,
)

from .audit import (
    AuditLogger,
    audit_tool,
    parse_and_log_llm_usage,
    register_controller,
    register_logger,
    unregister_controller,
    unregister_logger,
)
from .cost_control import (
    BudgetConfig,
    CostBudgetExceeded,
    CostController,
    estimate_run_cost,
)
from .llm_config import (
    build_cost_aware_llm_config,
    build_llm_config,
    provider_label,
    sanitize_context_for_llm,
)
from .prompts import (
    COMPLIANCE_AGENT_PROMPT,
    INGEST_AGENT_PROMPT,
    ORCHESTRATOR_PROMPT,
    QUALITY_AGENT_PROMPT,
    TRANSFORM_AGENT_PROMPT,
)
from .tools import (
    get_pipeline_summary,
    generate_schema_yaml,
    generate_staging_sql,
    ingest_file,
    profile_table,
    check_primary_key,
    save_dbt_artifacts,
)
from .compliance_tools import (
    generate_compliance_report,
    mask_pii_columns,
    scan_pii,
    write_audit_log,
)
from .airbyte_tools import register_airbyte_tools

logger = logging.getLogger(__name__)


# ── Swarm builder ─────────────────────────────────────────────────────────────

def build_etl_swarm(
    llm_config: Optional[dict] = None,
    provider: Optional[str] = None,
) -> tuple[ConversableAgent, list[ConversableAgent]]:
    """
    Construct the AG2 Swarm: create agents, register audited tools, wire handoffs.

    Every tool is wrapped with @audit_tool at registration time, so timing,
    success/failure, and result previews are captured without modifying
    the tool functions themselves.

    Args:
        llm_config: Optional custom AG2 llm_config dict. Auto-detected if None.
        provider:   Force LLM provider: bedrock | anthropic | openai | ollama.

    Returns:
        (orchestrator, [orchestrator, ingest_agent, compliance_agent,
                        quality_agent, transform_agent])
    """
    cfg = llm_config or build_llm_config(provider)
    info = provider_label()
    logger.info(
        "Building ETL Swarm — provider=%s model=%s residency=%s guardrails=%s",
        info["provider"], info["model"], info["data_residency"], info["guardrails"],
    )

    # ── Agent definitions ──────────────────────────────────────────────────
    orchestrator = ConversableAgent(
        name="orchestrator",
        system_message=ORCHESTRATOR_PROMPT,
        llm_config=cfg,
    )
    ingest_agent = ConversableAgent(
        name="ingest_agent",
        system_message=INGEST_AGENT_PROMPT,
        llm_config=cfg,
    )
    compliance_agent = ConversableAgent(
        name="compliance_agent",
        system_message=COMPLIANCE_AGENT_PROMPT,
        llm_config=cfg,
    )
    quality_agent = ConversableAgent(
        name="quality_agent",
        system_message=QUALITY_AGENT_PROMPT,
        llm_config=cfg,
    )
    transform_agent = ConversableAgent(
        name="transform_agent",
        system_message=TRANSFORM_AGENT_PROMPT,
        llm_config=cfg,
    )

    # ── Tool registration (all wrapped with @audit_tool) ───────────────────
    #
    # Pattern for each tool:
    #   audited_fn = audit_tool("<agent_name>")(original_fn)
    #   agent.register_for_llm(description="...")(audited_fn)
    #   agent.register_for_execution()(audited_fn)
    #
    # This wraps execution only — the LLM sees the original function signature
    # via register_for_llm; the audit wrapper activates on actual execution.

    # ingest_agent ── local file loading (CSV / Excel / Parquet → DuckDB)
    _ingest_file = audit_tool("ingest_agent")(ingest_file)
    ingest_agent.register_for_llm(
        description=(
            "Load a local CSV, Excel, or Parquet file directly into DuckDB. "
            "Use this for local files. For database sources, SaaS APIs, or cloud "
            "storage use the Airbyte tools below."
        )
    )(_ingest_file)
    ingest_agent.register_for_execution()(_ingest_file)

    # ingest_agent ── Airbyte Docker tools (E2E source → destination ingestion)
    #   Covers: Postgres, MySQL, MSSQL, Salesforce, S3, HubSpot, Stripe, GitHub,
    #           Google Sheets, MongoDB, REST API
    #   → Snowflake, BigQuery, Redshift, Postgres, DuckDB, S3
    register_airbyte_tools(
        ingest_agent=ingest_agent,
        audit_decorator=audit_tool("ingest_agent"),
    )

    # compliance_agent ── PII scan, masking, audit, report
    _scan_pii = audit_tool("compliance_agent")(scan_pii)
    compliance_agent.register_for_llm(
        description=(
            "Scan all columns in the loaded DuckDB table for PII and sensitive data. "
            "Classifies each column as PII_HIGH, FINANCIAL, PII_MEDIUM, PII_LOW, or PUBLIC."
        )
    )(_scan_pii)
    compliance_agent.register_for_execution()(_scan_pii)

    _mask_pii = audit_tool("compliance_agent")(mask_pii_columns)
    compliance_agent.register_for_llm(
        description=(
            "Mask sensitive columns in the DuckDB table. "
            "strategy='hash' (default, SHA-256, preserves referential integrity) "
            "or strategy='redact'. Defaults to high_risk_columns from the PII scan."
        )
    )(_mask_pii)
    compliance_agent.register_for_execution()(_mask_pii)

    _compliance_report = audit_tool("compliance_agent")(generate_compliance_report)
    compliance_agent.register_for_llm(
        description=(
            "Generate a structured compliance report covering PII classification, "
            "masking summary, regulatory applicability, and recommended actions."
        )
    )(_compliance_report)
    compliance_agent.register_for_execution()(_compliance_report)

    _write_audit = audit_tool("compliance_agent")(write_audit_log)
    compliance_agent.register_for_llm(
        description=(
            "Write an append-only audit log record for this pipeline run "
            "to both a JSONL file and the DuckDB etl_audit_log table."
        )
    )(_write_audit)
    compliance_agent.register_for_execution()(_write_audit)

    # quality_agent ── profiling
    _profile = audit_tool("quality_agent")(profile_table)
    quality_agent.register_for_llm(
        description="Profile a DuckDB table: null rates, distinct counts, duplicate rows"
    )(_profile)
    quality_agent.register_for_execution()(_profile)

    _check_pk = audit_tool("quality_agent")(check_primary_key)
    quality_agent.register_for_llm(
        description="Check if a specific column is a valid primary key (unique + not null)"
    )(_check_pk)
    quality_agent.register_for_execution()(_check_pk)

    # transform_agent ── dbt model generation
    _gen_sql = audit_tool("transform_agent")(generate_staging_sql)
    transform_agent.register_for_llm(
        description="Generate a dbt staging SQL model for the loaded DuckDB table"
    )(_gen_sql)
    transform_agent.register_for_execution()(_gen_sql)

    _gen_yaml = audit_tool("transform_agent")(generate_schema_yaml)
    transform_agent.register_for_llm(
        description=(
            "Generate a dbt schema.yml — governance guide with PII classification, "
            "masking status, regulations, and business-user review checklist"
        )
    )(_gen_yaml)
    transform_agent.register_for_execution()(_gen_yaml)

    _save_dbt = audit_tool("transform_agent")(save_dbt_artifacts)
    transform_agent.register_for_llm(
        description="Save dbt SQL and governance YAML files to the dbt project directory"
    )(_save_dbt)
    transform_agent.register_for_execution()(_save_dbt)

    # orchestrator ── final summary
    _summary = audit_tool("orchestrator")(get_pipeline_summary)
    orchestrator.register_for_llm(
        description="Generate a human-readable summary of the completed ETL pipeline run"
    )(_summary)
    orchestrator.register_for_execution()(_summary)

    # ── Handoff wiring ─────────────────────────────────────────────────────
    register_hand_off(
        agent=orchestrator,
        hand_to=[
            ON_CONDITION(
                ingest_agent,
                "Transfer to ingest_agent when a source file needs to be loaded into DuckDB",
            ),
            AFTER_WORK(AfterWorkOption.TERMINATE),
        ],
    )
    register_hand_off(
        agent=ingest_agent,
        hand_to=[
            ON_CONDITION(
                compliance_agent,
                "Transfer to compliance_agent after the file has been successfully ingested — "
                "compliance check is mandatory before any further processing",
            ),
            AFTER_WORK(AfterWorkOption.REVERT_TO_USER),
        ],
    )
    register_hand_off(
        agent=compliance_agent,
        hand_to=[
            ON_CONDITION(
                quality_agent,
                "Transfer to quality_agent after PII scan, masking, compliance report, "
                "and audit log are all complete",
            ),
            AFTER_WORK(AfterWorkOption.REVERT_TO_USER),
        ],
    )
    register_hand_off(
        agent=quality_agent,
        hand_to=[
            ON_CONDITION(
                transform_agent,
                "Transfer to transform_agent after data quality profiling is complete",
            ),
            AFTER_WORK(AfterWorkOption.REVERT_TO_USER),
        ],
    )
    register_hand_off(
        agent=transform_agent,
        hand_to=[
            ON_CONDITION(
                orchestrator,
                "Transfer back to orchestrator after dbt artifacts have been saved",
            ),
            AFTER_WORK(AfterWorkOption.TERMINATE),
        ],
    )

    agents = [orchestrator, ingest_agent, compliance_agent, quality_agent, transform_agent]
    return orchestrator, agents


# ── Public entry point ────────────────────────────────────────────────────────

def run_etl_pipeline(
    request: str,
    context_variables: Optional[dict] = None,
    llm_config: Optional[dict] = None,
    provider: Optional[str] = None,
    budget: Optional[BudgetConfig] = None,
    max_rounds: int = 40,
) -> dict:
    """
    Run the full compliance-first, cost-controlled, fully-audited ETL pipeline.

    Guardrail layers:
      [L1] context_variables PII sanitization before the swarm starts
      [L2] Bedrock Guardrails (when BEDROCK_GUARDRAIL_ID is configured)
      [L3] Response validation warnings via Python logger

    Cost control (three-level enforcement):
      Pre-run:  day/month aggregate budget check — stops before wasting tokens
      Mid-run:  per-run / day / month checked after every tool call
      Auto:     model downgrade (expensive → cheap → ollama) when near limit
      Hard stop: CostBudgetExceeded terminates gracefully if limit exceeded

    Audit layers (automatic, no code changes to tools required):
      ● Every tool call → etl_agent_steps
      ● LLM token usage → etl_llm_calls (tokens, cost, latency)
      ● Compliance events → etl_compliance_events
      ● Pipeline summary → etl_pipeline_runs

    Args:
        request:           Natural language ETL request.
        context_variables: Seed context (db_path, dbt_project_dir, …).
        llm_config:        AG2 llm_config override. Auto-detected if None.
        provider:          Force provider: bedrock | anthropic | openai | ollama.
        budget:            BudgetConfig override. Reads env vars if None.
        max_rounds:        Safety cap on agent rounds (default 40).

    Returns:
        {
          status, table_name, row_count,
          pii_columns, high_risk_columns, masked_columns,
          compliance_status, quality_issues, dbt_artifacts,
          audit_run_id,     ← query etl_agent_steps WHERE run_id = audit_run_id
          llm_provider,     ← {provider, model, data_residency, cost, guardrails}
          llm_usage,        ← {total_tokens, total_cost_usd, models_used}
          cost_report,      ← per-agent token + cost breakdown string
          cost_estimate,    ← pre-run estimate dict (populated even if pipeline aborted)
          budget_exceeded,  ← True if pipeline was terminated by a budget limit
          context_variables,
          messages,
        }
    """
    ctx = context_variables or {}
    budget_cfg = budget or BudgetConfig()

    # ── Determine run_id ──────────────────────────────────────────────────
    run_id = ctx.get("compliance_run_id") or str(uuid.uuid4())
    ctx["_audit_run_id"] = run_id

    # ── [L1] Sanitize context before any LLM sees it ──────────────────────
    safe_ctx = sanitize_context_for_llm(ctx)

    # ── Resolve provider + model info ─────────────────────────────────────
    info = provider_label()
    db_path = ctx.get("db_path", "datapai.duckdb")

    logger.info(
        "run_etl_pipeline start  run_id=%s  provider=%s  model=%s  "
        "budget=%s",
        run_id, info["provider"], info["model"], budget_cfg.summary(),
    )

    # ── Pre-run cost estimation ───────────────────────────────────────────
    schema = ctx.get("schema", {})
    row_count_hint = ctx.get("row_count", 0)
    cost_estimate = estimate_run_cost(
        schema=schema,
        row_count=row_count_hint,
        provider=info["provider"].lower().replace(" ", "_"),
        model=info["model"],
    )
    logger.info(
        "[COST] Pre-run estimate: %d tokens  $%.4f–$%.4f  (%s)",
        cost_estimate["token_estimate"],
        cost_estimate["cost_low_usd"],
        cost_estimate["cost_high_usd"],
        cost_estimate["recommendation"],
    )

    # ── Create + register AuditLogger and CostController ──────────────────
    audit = AuditLogger(
        run_id=run_id,
        db_path=db_path,
        llm_provider=info["provider"],
        llm_model=info["model"],
    )
    cost_ctrl = CostController(
        run_id=run_id,
        db_path=db_path,
        model=info["model"],
        budget=budget_cfg,
    )
    register_logger(run_id, audit)
    register_controller(run_id, cost_ctrl)

    # ── Pre-run aggregate budget check (day / month) ───────────────────────
    pre_run_status: dict = {}
    budget_exceeded = False
    final_ctx: dict = safe_ctx
    chat_result = None
    llm_usage: dict = {}
    cost_report = ""

    try:
        pre_run_status = cost_ctrl.pre_run_check()
        for w in pre_run_status.get("warnings", []):
            logger.warning(w)

        # ── Build cost-aware LLM config (may downgrade model) ─────────────
        current_run_pct = (
            cost_ctrl.ledger.get_run_spend(run_id)
            / budget_cfg.per_run_usd * 100
            if budget_cfg.per_run_usd else 0
        )
        effective_cfg = llm_config or build_cost_aware_llm_config(
            provider=provider,
            budget_pct_used=current_run_pct,
        )

        # ── Run the Swarm ─────────────────────────────────────────────────
        orchestrator, agents = build_etl_swarm(effective_cfg, provider)

        chat_result, final_ctx, last_agent = initiate_swarm_chat(
            initial_agent=orchestrator,
            agents=agents,
            messages=request,
            context_variables=safe_ctx,
            max_rounds=max_rounds,
            after_work=AfterWorkOption.TERMINATE,
        )

        # ── Parse LLM token usage ─────────────────────────────────────────
        llm_usage = parse_and_log_llm_usage(chat_result, audit)

        # ── Emit compliance events ────────────────────────────────────────
        for col in final_ctx.get("pii_columns", []):
            col_info = final_ctx.get("pii_scan", {}).get(col, {})
            audit.log_compliance_event(
                event_type="PII_DETECTED",
                column_name=col,
                sensitivity=col_info.get("sensitivity", ""),
                pii_type=col_info.get("pii_type", ""),
                action_taken="flagged",
            )
        for m in final_ctx.get("masked_columns", []):
            if isinstance(m, dict):
                audit.log_compliance_event(
                    event_type="MASKING_APPLIED",
                    column_name=m.get("column", ""),
                    sensitivity=m.get("sensitivity", ""),
                    action_taken=m.get("strategy", "hash"),
                )

    except CostBudgetExceeded as exc:
        budget_exceeded = True
        logger.error("[COST] %s", exc)
        final_ctx["pipeline_status"] = "budget_exceeded"
        final_ctx["budget_exceeded_reason"] = str(exc)
        audit.log_compliance_event(
            event_type="BUDGET_EXCEEDED",
            action_taken="pipeline_terminated",
            details={"reason": str(exc)},
        )

    finally:
        # ── Cost report (always generated, even on budget exceeded) ────────
        cost_report = cost_ctrl.report()
        logger.info("\n%s", cost_report)

        # ── Finalise audit record ─────────────────────────────────────────
        audit.finalise(final_ctx)

        # ── Clean up registries ───────────────────────────────────────────
        unregister_logger(run_id)
        unregister_controller(run_id)

    return {
        "status": final_ctx.get("pipeline_status", "incomplete"),
        "table_name": final_ctx.get("table_name"),
        "row_count": final_ctx.get("row_count"),
        "pii_columns": final_ctx.get("pii_columns", []),
        "high_risk_columns": final_ctx.get("high_risk_columns", []),
        "masked_columns": [
            m["column"] if isinstance(m, dict) else m
            for m in final_ctx.get("masked_columns", [])
        ],
        "compliance_status": final_ctx.get("compliance_status", "UNKNOWN"),
        "quality_issues": final_ctx.get("quality_issues", []),
        "dbt_artifacts": final_ctx.get("dbt_artifacts_saved", []),
        "audit_run_id": run_id,
        "llm_provider": info,
        "llm_usage": llm_usage,
        "cost_report": cost_report,
        "cost_estimate": cost_estimate,
        "budget_exceeded": budget_exceeded,
        "pre_run_budget_status": pre_run_status,
        "context_variables": final_ctx,
        "messages": chat_result.chat_history if chat_result else [],
    }

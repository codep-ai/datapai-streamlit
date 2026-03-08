"""
agents/dbt_guardrail_agent.py
=============================

dbt AI Guardrail Agent for Datap.ai.

This agent is responsible for:
  - Interpreting dbt AI governance metadata
  - Compiling and refreshing the AI policy catalog
  - Explaining why a model/field/action is allowed, masked, aggregate-only,
    approval-required, or denied
  - Surfacing allowed and blocked models and fields for a given use case
  - Validating SQL, RAG, summarization, and tool actions against the policy catalog
  - Helping data teams apply the governance standard to their dbt projects

Architecture:
  - Inherits from BaseAgent (same pattern as the ETL agent)
  - Uses the @tool registry (agents/tooling/registry.py)
  - Delegates all policy logic to guardrail.policy_compiler and guardrail.validators
  - Stateless: the policy catalog is compiled on first use and cached per session

Usage (Streamlit / API):
    from agents.dbt_guardrail_agent import DbtGuardrailAgent
    from llm_client import RouterChatClient

    agent = DbtGuardrailAgent(llm=RouterChatClient())
    result = agent.run(
        "Which models are AI-eligible for text2sql in the analytics workspace?",
        context={"workspace_id": "analytics", "tenant_id": "acme"}
    )

Tool-only usage (no LLM required for deterministic lookups):
    agent = DbtGuardrailAgent(llm=None)
    summary = agent.catalog_summary()
    explanation = agent.explain("customers", "email")
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .tooling.registry import tool

logger = logging.getLogger(__name__)

# ── Guardrail imports ─────────────────────────────────────────────────────────
try:
    from guardrail.policy_compiler import PolicyCompiler
    from guardrail.validators import (
        validate_sql_against_policy,
        validate_summary_against_policy,
        validate_retrieval_against_policy,
        validate_ai_action_against_policy,
    )
    from guardrail.context_filter import (
        filter_context,
        build_safe_schema_context,
        get_allowed_assets_for_use_case,
        get_allowed_fields_for_asset,
        summarize_filtered_context,
    )
    _GUARDRAIL_AVAILABLE = True
except ImportError as exc:
    logger.warning("guardrail module not available: %s", exc)
    _GUARDRAIL_AVAILABLE = False
    PolicyCompiler = None  # type: ignore

# ── Default manifest path ─────────────────────────────────────────────────────
DBT_MANIFEST_PATH = os.environ.get(
    "DBT_MANIFEST_PATH",
    os.path.join("dbt-demo", "target", "manifest.json"),
)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level compiler singleton (shared across tool calls in a process)
# ──────────────────────────────────────────────────────────────────────────────

_compiler: Optional[PolicyCompiler] = None


def _get_compiler(manifest_path: Optional[str] = None) -> Optional[PolicyCompiler]:
    global _compiler
    if not _GUARDRAIL_AVAILABLE:
        return None
    if _compiler is None or manifest_path:
        path = manifest_path or DBT_MANIFEST_PATH
        _compiler = PolicyCompiler(manifest_path=path)
    return _compiler


# ──────────────────────────────────────────────────────────────────────────────
# @tool-registered functions (used by the BaseAgent tool loop)
# ──────────────────────────────────────────────────────────────────────────────

@tool("guardrail_catalog_summary")
def guardrail_catalog_summary(manifest_path: str = "") -> Dict[str, Any]:
    """
    Compile the AI policy catalog and return a summary.

    Returns counts of AI-eligible models, certified models, denied models,
    total columns, PII columns, and output-safe columns.

    Args:
        manifest_path: Optional path to dbt manifest.json.
    """
    compiler = _get_compiler(manifest_path or None)
    if compiler is None:
        return {"error": "guardrail module not available"}
    return compiler.catalog_summary()


@tool("guardrail_explain_model")
def guardrail_explain_model(
    model_name:    str,
    use_case:      str = "",
    manifest_path: str = "",
) -> Dict[str, Any]:
    """
    Explain the AI governance policy for a specific dbt model.

    Returns whether the model is allowed, the answer mode, reasons,
    sensitivity details, and workspace/tenant constraints.

    Args:
        model_name:    The dbt model name (e.g. 'customers', 'orders').
        use_case:      Optional AI use case (e.g. 'text2sql', 'rag', 'summarization').
        manifest_path: Optional path to dbt manifest.json.
    """
    compiler = _get_compiler(manifest_path or None)
    if compiler is None:
        return {"error": "guardrail module not available"}
    return compiler.explain_policy_decision(model_name=model_name, use_case=use_case)


@tool("guardrail_explain_column")
def guardrail_explain_column(
    model_name:    str,
    column_name:   str,
    use_case:      str = "",
    manifest_path: str = "",
) -> Dict[str, Any]:
    """
    Explain the AI governance policy for a specific column/field.

    Returns PII class, masking rule, answer mode, usage permissions,
    and reasons for the allow/deny decision.

    Args:
        model_name:    The dbt model containing the column.
        column_name:   The column name.
        use_case:      Optional AI use case.
        manifest_path: Optional path to dbt manifest.json.
    """
    compiler = _get_compiler(manifest_path or None)
    if compiler is None:
        return {"error": "guardrail module not available"}
    return compiler.explain_policy_decision(
        model_name=model_name, column_name=column_name, use_case=use_case
    )


@tool("guardrail_list_eligible_models")
def guardrail_list_eligible_models(
    use_case:      str = "",
    workspace_id:  str = "",
    tenant_id:     str = "",
    manifest_path: str = "",
) -> List[Dict[str, Any]]:
    """
    List all AI-eligible dbt models for a given use case and context.

    Returns model name, access level, answer mode, sensitivity,
    PII flags, and approved use cases.

    Args:
        use_case:      AI use case filter (e.g. 'text2sql', 'rag', 'summarization').
        workspace_id:  Filter by workspace.
        tenant_id:     Filter by tenant.
        manifest_path: Optional path to dbt manifest.json.
    """
    compiler = _get_compiler(manifest_path or None)
    if compiler is None:
        return [{"error": "guardrail module not available"}]
    catalog   = compiler.compile()
    eligible  = get_allowed_assets_for_use_case(catalog, use_case, workspace_id, tenant_id)
    return [m.to_dict() for m in eligible]


@tool("guardrail_list_eligible_columns")
def guardrail_list_eligible_columns(
    model_name:    str,
    use_case:      str = "",
    output_only:   bool = False,
    manifest_path: str = "",
) -> List[Dict[str, Any]]:
    """
    List AI-eligible columns for a dbt model.

    Args:
        model_name:    The dbt model name.
        use_case:      AI use case filter.
        output_only:   If true, return only columns safe for AI output.
        manifest_path: Optional path to dbt manifest.json.
    """
    compiler = _get_compiler(manifest_path or None)
    if compiler is None:
        return [{"error": "guardrail module not available"}]
    catalog = compiler.compile()
    cols    = get_allowed_fields_for_asset(catalog, model_name, use_case, output_only)
    return [c.to_dict() for c in cols]


@tool("guardrail_validate_sql")
def guardrail_validate_sql(
    sql:           str,
    use_case:      str = "text2sql",
    workspace_id:  str = "",
    tenant_id:     str = "",
    manifest_path: str = "",
) -> Dict[str, Any]:
    """
    Validate a SQL query against the AI policy catalog.

    Checks for dangerous keywords, model eligibility, workspace/tenant rules,
    column output permissions, and aggregate-only enforcement.

    Returns allowed (bool), violations list, blocked models/columns, and answer mode.

    Args:
        sql:           The SQL query string to validate.
        use_case:      AI use case (default 'text2sql').
        workspace_id:  Current workspace context.
        tenant_id:     Current tenant context.
        manifest_path: Optional path to dbt manifest.json.
    """
    compiler = _get_compiler(manifest_path or None)
    if compiler is None:
        return {"error": "guardrail module not available"}
    catalog = compiler.compile()
    result  = validate_sql_against_policy(
        sql=sql, catalog=catalog, use_case=use_case,
        workspace_id=workspace_id, tenant_id=tenant_id,
    )
    return result.to_dict()


@tool("guardrail_validate_action")
def guardrail_validate_action(
    use_case:      str,
    target_models: str,  # JSON array string, e.g. '["customers", "orders"]'
    workspace_id:  str = "",
    tenant_id:     str = "",
    manifest_path: str = "",
) -> Dict[str, Any]:
    """
    Validate a general AI action against the policy catalog.

    Supports any use case: text2sql, summarization, rag, export, explanation, etc.

    Args:
        use_case:      The AI use case being performed.
        target_models: JSON array of model names involved.
        workspace_id:  Current workspace.
        tenant_id:     Current tenant.
        manifest_path: Optional path to dbt manifest.json.
    """
    compiler = _get_compiler(manifest_path or None)
    if compiler is None:
        return {"error": "guardrail module not available"}
    try:
        models_list = json.loads(target_models) if isinstance(target_models, str) else target_models
    except json.JSONDecodeError:
        models_list = [target_models]
    catalog = compiler.compile()
    result  = validate_ai_action_against_policy(
        use_case=use_case, target_models=models_list, catalog=catalog,
        workspace_id=workspace_id, tenant_id=tenant_id,
    )
    return result.to_dict()


@tool("guardrail_build_safe_context")
def guardrail_build_safe_context(
    use_case:      str = "",
    workspace_id:  str = "",
    tenant_id:     str = "",
    model_names:   str = "",  # JSON array string or comma-separated
    manifest_path: str = "",
) -> Dict[str, Any]:
    """
    Build a policy-filtered schema context for use in AI prompts.

    Returns only the tables and columns permitted by the AI policy catalog
    for the given use case, workspace, and tenant context.

    Args:
        use_case:      AI use case (text2sql, rag, summarization, etc.).
        workspace_id:  Current workspace.
        tenant_id:     Current tenant.
        model_names:   Optional JSON array or comma-separated list of model names.
        manifest_path: Optional path to dbt manifest.json.
    """
    compiler = _get_compiler(manifest_path or None)
    if compiler is None:
        return {"error": "guardrail module not available"}
    catalog = compiler.compile()

    # Parse model_names
    names_list: Optional[List[str]] = None
    if model_names:
        try:
            names_list = json.loads(model_names)
        except json.JSONDecodeError:
            names_list = [n.strip() for n in model_names.split(",") if n.strip()]

    safe_ctx = build_safe_schema_context(
        catalog=catalog, use_case=use_case,
        workspace_id=workspace_id, tenant_id=tenant_id,
        model_names=names_list,
    )
    summary = summarize_filtered_context(safe_ctx)
    return {"context": safe_ctx, "summary": summary}


@tool("guardrail_refresh_catalog")
def guardrail_refresh_catalog(manifest_path: str = "") -> Dict[str, Any]:
    """
    Force a refresh of the AI policy catalog from the dbt manifest.

    Use this after running 'dbt compile' or 'dbt build' to pick up
    updated governance metadata.

    Args:
        manifest_path: Optional path to dbt manifest.json.
    """
    global _compiler
    path      = manifest_path or DBT_MANIFEST_PATH
    _compiler = PolicyCompiler(manifest_path=path)
    catalog   = _compiler.refresh()
    return {
        "status":          "refreshed",
        "catalog_version": catalog.version,
        "model_count":     len(catalog.models),
        "column_count":    len(catalog.columns),
    }


@tool("guardrail_list_blocked_models")
def guardrail_list_blocked_models(manifest_path: str = "") -> List[Dict[str, Any]]:
    """
    List all dbt models that are denied or not AI-eligible.

    Useful for governance audits and diagnosing missing metadata.

    Args:
        manifest_path: Optional path to dbt manifest.json.
    """
    compiler = _get_compiler(manifest_path or None)
    if compiler is None:
        return [{"error": "guardrail module not available"}]
    catalog  = compiler.compile()
    blocked  = [m for m in catalog.models.values() if not m.is_ai_eligible()]
    return [
        {
            "model_name":      m.model_name,
            "ai_enabled":      m.ai_enabled,
            "ai_access_level": m.ai_access_level.value,
            "reason":          (
                "access=private" if m.access == "private" else
                "ai_enabled=false" if not m.ai_enabled else
                f"ai_access_level={m.ai_access_level.value}"
            ),
        }
        for m in blocked
    ]


@tool("guardrail_list_restricted_columns")
def guardrail_list_restricted_columns(
    model_name:    str = "",
    manifest_path: str = "",
) -> List[Dict[str, Any]]:
    """
    List columns that are restricted (PII, PHI, or not output-safe).

    Args:
        model_name:    If provided, limit to this model. Otherwise list all.
        manifest_path: Optional path to dbt manifest.json.
    """
    compiler = _get_compiler(manifest_path or None)
    if compiler is None:
        return [{"error": "guardrail module not available"}]
    catalog = compiler.compile()

    cols = catalog.columns.values()
    if model_name:
        cols = [c for c in cols if c.model_name == model_name]

    restricted = [c for c in cols if c.is_pii_or_phi() or not c.is_output_safe()]
    return [
        {
            "key":          c.key,
            "model":        c.model_name,
            "column":       c.column_name,
            "pii":          c.pii.value,
            "phi":          c.phi,
            "masking_rule": c.masking_rule.value,
            "answer_mode":  c.answer_mode.value,
        }
        for c in restricted
    ]


# ──────────────────────────────────────────────────────────────────────────────
# DbtGuardrailAgent class
# ──────────────────────────────────────────────────────────────────────────────

DBT_GUARDRAIL_SYSTEM_PROMPT = """
You are the Datap.ai dbt AI Guardrail Agent.

Your role is to help users, developers, and data stewards understand and apply
the Datap.ai dbt AI governance framework.

You can:
- Compile and inspect the AI policy catalog from dbt manifest metadata
- Explain why a model, table, or column is allowed, masked, aggregate-only, or denied
- List AI-eligible models and columns for specific use cases and workspaces
- Validate SQL queries, RAG retrievals, summarizations, and tool actions against policy
- Build policy-filtered schema context for AI prompt construction
- List blocked or restricted assets for governance review

You MUST respond in strict JSON only, with no additional text.

Response types:

1) Tool call:
{
  "type": "tool_call",
  "tool_name": "<tool_name>",
  "args": { ... }
}

2) Final answer:
{
  "type": "final_answer",
  "result": "Human-readable governance explanation or result summary."
}

Available tools cover: guardrail_catalog_summary, guardrail_explain_model,
guardrail_explain_column, guardrail_list_eligible_models,
guardrail_list_eligible_columns, guardrail_validate_sql,
guardrail_validate_action, guardrail_build_safe_context,
guardrail_refresh_catalog, guardrail_list_blocked_models,
guardrail_list_restricted_columns.

Always cite the specific policy reason (ai_access_level, pii class, answer mode,
blocked_ai_use_cases) in your explanations.
""".strip()


class DbtGuardrailAgent:
    """
    dbt AI Guardrail Agent — explains and enforces dbt-driven AI governance.

    Inherits the BaseAgent tool loop pattern.  Registers guardrail tools
    via the @tool decorator system.

    Parameters
    ----------
    llm : BaseChatClient or None
        LLM client for natural-language explanations.
        Pass None for deterministic tool-only usage.
    manifest_path : str, optional
        Path to the dbt manifest.json.  Defaults to dbt-demo/target/manifest.json.
    max_steps : int
        Maximum LLM tool-call steps (default 6).
    """

    AGENT_NAME = "DbtGuardrailAgent"

    def __init__(
        self,
        llm:           Optional[Any] = None,
        manifest_path: Optional[str] = None,
        max_steps:     int           = 6,
    ):
        self.llm           = llm
        self.manifest_path = manifest_path or DBT_MANIFEST_PATH
        self.max_steps     = max_steps
        self._compiler: Optional[PolicyCompiler] = None

        if manifest_path:
            os.environ["DBT_MANIFEST_PATH"] = manifest_path

        # If llm provided, wire into BaseAgent loop
        if llm is not None:
            try:
                from .agent_base import BaseAgent
                self._base = BaseAgent(
                    name          = self.AGENT_NAME,
                    llm           = llm,
                    system_prompt = DBT_GUARDRAIL_SYSTEM_PROMPT,
                    max_steps     = max_steps,
                )
            except ImportError:
                self._base = None
        else:
            self._base = None

    def run(
        self,
        goal:    str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the guardrail agent.

        If an LLM is available, uses the full BaseAgent tool loop.
        Otherwise returns a structured error asking to set up an LLM.
        """
        if self._base is not None:
            return self._base.run(goal=goal, context=context or {})
        return {
            "status": "error",
            "reason": (
                "DbtGuardrailAgent requires an LLM for natural-language queries. "
                "Use the direct methods (catalog_summary, explain, validate_sql, etc.) "
                "for programmatic access without an LLM."
            ),
        }

    # ── Convenience methods (no LLM required) ────────────────────────────────

    def get_compiler(self) -> Optional[PolicyCompiler]:
        if not _GUARDRAIL_AVAILABLE:
            return None
        if self._compiler is None:
            self._compiler = PolicyCompiler(manifest_path=self.manifest_path)
        return self._compiler

    def catalog_summary(self) -> Dict[str, Any]:
        """Return a catalog summary without needing an LLM."""
        return guardrail_catalog_summary(self.manifest_path)

    def explain(
        self,
        model_name:  str,
        column_name: Optional[str] = None,
        use_case:    str = "",
    ) -> Dict[str, Any]:
        """Explain the policy for a model or column."""
        if column_name:
            return guardrail_explain_column(model_name, column_name, use_case, self.manifest_path)
        return guardrail_explain_model(model_name, use_case, self.manifest_path)

    def list_eligible_models(
        self,
        use_case:    str = "",
        workspace_id: str = "",
        tenant_id:   str = "",
    ) -> List[Dict[str, Any]]:
        return guardrail_list_eligible_models(use_case, workspace_id, tenant_id, self.manifest_path)

    def validate_sql(
        self,
        sql:          str,
        use_case:     str = "text2sql",
        workspace_id: str = "",
        tenant_id:    str = "",
    ) -> Dict[str, Any]:
        return guardrail_validate_sql(sql, use_case, workspace_id, tenant_id, self.manifest_path)

    def build_safe_context(
        self,
        use_case:    str = "",
        workspace_id: str = "",
        tenant_id:   str = "",
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        names_str = json.dumps(model_names) if model_names else ""
        return guardrail_build_safe_context(use_case, workspace_id, tenant_id, names_str, self.manifest_path)

    def refresh(self) -> Dict[str, Any]:
        """Force refresh of the policy catalog."""
        self._compiler = None
        return guardrail_refresh_catalog(self.manifest_path)

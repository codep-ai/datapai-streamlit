"""
guardrail.context_filter
========================

Pre-generation context filtering: removes unsafe, restricted, or policy-denied
tables and columns from the AI context before any LLM call.

This is the "left-shift" guardrail — it prevents sensitive schema or content
from ever reaching the LLM prompt, rather than just blocking the generated output.

Key functions:
  filter_context()              — Filter a schema context dict in-place
  build_safe_schema_context()   — Build a policy-filtered schema context from catalog
  get_allowed_assets_for_use_case() — Return eligible model policies
  get_allowed_fields_for_asset()    — Return eligible column policies for a model

Context format (input / output):
  {
    "tables": {
      "customers": {
        "description": "...",
        "columns": {
          "customer_id": { "type": "integer", "description": "..." },
          "email":       { "type": "varchar", "description": "..." },
          ...
        }
      },
      ...
    }
  }
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from .metadata_schema import (
    AiPolicyCatalog,
    AnswerMode,
    ColumnAiPolicy,
    ModelAiPolicy,
    MaskingRule,
    PiiClass,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Context filtering
# ──────────────────────────────────────────────────────────────────────────────

def filter_context(
    schema_context: Dict[str, Any],
    catalog:        AiPolicyCatalog,
    use_case:       str = "",
    workspace_id:   str = "",
    tenant_id:      str = "",
) -> Dict[str, Any]:
    """
    Filter a schema context dict so only AI-eligible tables and columns remain.

    Parameters
    ----------
    schema_context : dict
        The raw schema context, typically {"tables": {"model_name": {...}, ...}}.
    catalog : AiPolicyCatalog
        The compiled AI policy catalog.
    use_case : str
        The AI use case (e.g. "text2sql", "rag", "summarization").
    workspace_id, tenant_id : str
        Current user context for workspace/tenant-specific allow rules.

    Returns
    -------
    dict
        A filtered copy of schema_context with:
        - Denied/ineligible tables removed
        - Ineligible/PII/PHI columns removed or replaced with placeholder
        - Descriptions replaced by safe_description where available
        - Columns annotated with masking notes for AI context
    """
    filtered_tables: Dict[str, Any] = {}

    raw_tables = schema_context.get("tables") or schema_context

    for table_name, table_info in raw_tables.items():
        model = catalog.get_model(table_name)

        if model is None:
            # Unknown model — include with a warning annotation but no column detail
            logger.debug("context_filter: model '%s' not in catalog, including with warning", table_name)
            filtered_tables[table_name] = {
                "description": table_info.get("description", ""),
                "ai_note":     "[Not in AI policy catalog — warehouse access controls apply]",
                "columns":     {},
            }
            continue

        if not model.is_ai_eligible():
            logger.info(
                "context_filter: excluding model '%s' (not AI-eligible)", table_name
            )
            continue

        if workspace_id and not model.is_approved_for_workspace(workspace_id):
            logger.info(
                "context_filter: excluding model '%s' (not approved for workspace '%s')",
                table_name, workspace_id,
            )
            continue

        if tenant_id and not model.is_approved_for_tenant(tenant_id):
            logger.info(
                "context_filter: excluding model '%s' (not approved for tenant '%s')",
                table_name, tenant_id,
            )
            continue

        if use_case and use_case in model.blocked_ai_use_cases:
            logger.info(
                "context_filter: excluding model '%s' (use case '%s' blocked)",
                table_name, use_case,
            )
            continue

        # Model is eligible — filter its columns
        safe_columns = _filter_columns(
            table_name   = table_name,
            table_info   = table_info,
            catalog      = catalog,
            use_case     = use_case,
            answer_mode  = model.default_answer_mode,
        )

        # Use safe_description in the prompt context
        description = (
            model.safe_description or model.description
            or table_info.get("description", "")
        )

        entry: Dict[str, Any] = {
            "description": description,
            "columns":     safe_columns,
        }

        # Annotate with governance metadata for the LLM context
        if model.default_answer_mode == AnswerMode.AGGREGATE_ONLY:
            entry["ai_note"] = "[Aggregate-only: do not generate row-level output from this table]"
        elif model.contains_pii:
            entry["ai_note"] = "[Contains PII: sensitive fields have been masked or excluded]"

        filtered_tables[table_name] = entry

    return {"tables": filtered_tables}


def _filter_columns(
    table_name:  str,
    table_info:  Dict[str, Any],
    catalog:     AiPolicyCatalog,
    use_case:    str,
    answer_mode: AnswerMode,
) -> Dict[str, Any]:
    """
    Filter columns for a single model.  Returns safe column metadata dict.
    """
    raw_columns: Dict[str, Any] = table_info.get("columns") or {}
    safe_cols:   Dict[str, Any] = {}

    for col_name, col_info in raw_columns.items():
        col = catalog.get_column(table_name, col_name)

        if col is None:
            # Unknown column — include with note; warehouse enforces final access
            safe_cols[col_name] = {
                "type":        col_info.get("type", ""),
                "description": col_info.get("description", ""),
                "ai_note":     "[Not in column policy catalog]",
            }
            continue

        if not col.ai_exposed:
            logger.debug(
                "context_filter: excluding column '%s.%s' (ai_exposed=False)",
                table_name, col_name,
            )
            continue  # Completely remove from context

        # Build the safe column entry
        col_entry: Dict[str, Any] = {
            "type":        col_info.get("type", ""),
            "description": col.description or col_info.get("description", ""),
        }

        # Annotate PII/masking for LLM awareness
        if col.pii != PiiClass.NONE:
            col_entry["pii"] = col.pii.value
            if col.masking_rule != MaskingRule.NONE:
                col_entry["ai_note"] = (
                    f"[PII field — masking rule: {col.masking_rule.value}. "
                    f"Do not expose raw values in output]"
                )
            else:
                col_entry["ai_note"] = (
                    f"[PII field — use in filters/grouping only, do not select for output]"
                )

        if col.phi:
            col_entry["ai_note"] = "[PHI field — must not appear in output]"

        # Business term / semantic alias for better LLM context
        if col.business_term:
            col_entry["business_term"] = col.business_term
        if col.semantic_aliases:
            col_entry["aliases"] = col.semantic_aliases
        if col.notes_for_ai:
            col_entry["note"] = col.notes_for_ai

        # Use-case-specific note
        if use_case in ("text2sql", "sql_generation"):
            if not col.allowed_in_output:
                col_entry["ai_note"] = col_entry.get(
                    "ai_note",
                    f"[Allowed in WHERE/GROUP BY only — do not SELECT '{col_name}' in output]",
                )

        safe_cols[col_name] = col_entry

    # Aggregate-only annotation: add a directive for the LLM
    if answer_mode == AnswerMode.AGGREGATE_ONLY and safe_cols:
        safe_cols["__aggregate_only__"] = {
            "type":        "directive",
            "description": (
                "This table may only be queried with aggregate functions "
                "(COUNT, SUM, AVG, MIN, MAX). Do not generate row-level SELECT queries."
            ),
        }

    return safe_cols


# ──────────────────────────────────────────────────────────────────────────────
# Schema context builder
# ──────────────────────────────────────────────────────────────────────────────

def build_safe_schema_context(
    catalog:      AiPolicyCatalog,
    use_case:     str = "",
    workspace_id: str = "",
    tenant_id:    str = "",
    model_names:  Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build a policy-filtered schema context directly from the AI policy catalog.

    This is used when the context hasn't been pre-built from a live schema
    (e.g. for RAG or explanation use cases that only need the governed model list).

    Parameters
    ----------
    model_names : list of str, optional
        If provided, limit context to these models.  Otherwise include all eligible.
    """
    eligible = catalog.eligible_models(use_case, workspace_id, tenant_id)
    if model_names:
        eligible = [m for m in eligible if m.model_name in model_names]

    tables: Dict[str, Any] = {}
    for model in eligible:
        cols_policy = catalog.get_columns_for_model(model.model_name)
        safe_cols: Dict[str, Any] = {}

        for col_name, col in cols_policy.items():
            if not col.ai_exposed:
                continue
            col_entry: Dict[str, Any] = {"description": col.description}
            if col.business_term:
                col_entry["business_term"] = col.business_term
            if col.semantic_aliases:
                col_entry["aliases"] = col.semantic_aliases
            if col.pii != PiiClass.NONE:
                col_entry["pii"]     = col.pii.value
                col_entry["ai_note"] = f"[PII: {col.pii.value}]"
            if col.notes_for_ai:
                col_entry["note"] = col.notes_for_ai
            safe_cols[col_name] = col_entry

        if model.default_answer_mode == AnswerMode.AGGREGATE_ONLY:
            safe_cols["__aggregate_only__"] = {
                "type": "directive",
                "description": (
                    "Aggregate queries only. Do not select individual rows."
                ),
            }

        description = model.safe_description or model.description
        entry: Dict[str, Any] = {
            "description": description,
            "columns":     safe_cols,
        }
        if model.is_certified():
            entry["certified_for_ai"] = True
        tables[model.model_name] = entry

    return {"tables": tables}


# ──────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_allowed_assets_for_use_case(
    catalog:      AiPolicyCatalog,
    use_case:     str = "",
    workspace_id: str = "",
    tenant_id:    str = "",
) -> List[ModelAiPolicy]:
    """Return eligible model policies for a given use case and context."""
    return catalog.eligible_models(use_case, workspace_id, tenant_id)


def get_allowed_fields_for_asset(
    catalog:     AiPolicyCatalog,
    model_name:  str,
    use_case:    str = "",
    output_only: bool = False,
) -> List[ColumnAiPolicy]:
    """
    Return AI-eligible column policies for a model.

    Parameters
    ----------
    output_only : bool
        If True, return only columns safe for AI output (is_output_safe() == True).
    """
    cols = catalog.eligible_columns_for_model(model_name, use_case)
    if output_only:
        cols = [c for c in cols if c.is_output_safe()]
    return cols


def summarize_filtered_context(filtered_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a concise summary of what was included/excluded.
    Useful for the Streamlit governance panel.
    """
    tables      = filtered_context.get("tables") or {}
    model_count = len(tables)
    col_count   = sum(
        len([c for c in t.get("columns", {}) if not c.startswith("__")])
        for t in tables.values()
    )
    certified = [
        name for name, t in tables.items() if t.get("certified_for_ai")
    ]
    aggregate_only = [
        name for name, t in tables.items()
        if "__aggregate_only__" in t.get("columns", {})
    ]
    pii_annotated = [
        f"{tname}.{cname}"
        for tname, t in tables.items()
        for cname, c in t.get("columns", {}).items()
        if isinstance(c, dict) and c.get("pii")
    ]

    return {
        "included_models":       list(tables.keys()),
        "model_count":           model_count,
        "column_count":          col_count,
        "certified_models":      certified,
        "aggregate_only_models": aggregate_only,
        "pii_annotated_columns": pii_annotated,
    }

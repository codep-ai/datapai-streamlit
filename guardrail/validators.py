"""
guardrail.validators
====================

Pre- and post-generation validators for all AI use cases.

These validators enforce the AI policy catalog at runtime:
  - validate_sql_against_policy()      — SQL queries
  - validate_summary_against_policy()  — Summarization outputs
  - validate_retrieval_against_policy()— RAG retrieval results
  - validate_tool_action_against_policy() — Agent tool calls
  - validate_ai_action_against_policy()   — General AI action (any use case)

Each validator returns a GuardrailResult with:
  allowed         : bool — proceed or block
  violations      : list of policy violations found
  blocked_objects : models/fields that caused a block
  answer_mode     : effective mode after enforcement
  suggestion      : safer alternative if available
  policy_version  : catalog version used for audit

Rule precedence (from spec Section 18):
  1. Hard deny by runtime policy or metadata
  2. Hard deny by model access=private / ai_access_level=deny
  3. Hard deny on restricted PII/PHI output
  4. aggregate_only enforcement
  5. masked-only enforcement
  6. metadata-only enforcement
  7. workspace/tenant/persona-specific allow rules
  8. General AI-certified allow
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .metadata_schema import (
    AiAccessLevel,
    AiPolicyCatalog,
    AnswerMode,
    ColumnAiPolicy,
    ModelAiPolicy,
    PiiClass,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PolicyViolation:
    code:    str    # Short machine-readable code, e.g. "MODEL_DENIED"
    message: str    # Human-readable explanation
    object:  str = ""  # The table/column/asset involved


@dataclass
class GuardrailResult:
    """
    Returned by every validator.

    Fields:
      allowed          : True → proceed;  False → block
      violations       : list of PolicyViolation describing what was blocked
      blocked_models   : list of model names that caused a block
      blocked_columns  : list of "model.column" keys that caused a block
      answer_mode      : effective answer mode after enforcement
      suggestion       : natural-language suggestion for a safer alternative
      policy_version   : catalog version used (for audit)
      metadata         : any additional context (use-case, validated_text, etc.)
    """
    allowed:         bool               = True
    violations:      List[PolicyViolation] = field(default_factory=list)
    blocked_models:  List[str]          = field(default_factory=list)
    blocked_columns: List[str]          = field(default_factory=list)
    answer_mode:     AnswerMode         = AnswerMode.FULL
    suggestion:      str                = ""
    policy_version:  str                = ""
    metadata:        Dict[str, Any]     = field(default_factory=dict)

    def block(self, code: str, message: str, obj: str = "") -> "GuardrailResult":
        """Convenience: add a violation and set allowed=False."""
        self.allowed = False
        self.violations.append(PolicyViolation(code=code, message=message, object=obj))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed":         self.allowed,
            "answer_mode":     self.answer_mode.value,
            "violations":      [
                {"code": v.code, "message": v.message, "object": v.object}
                for v in self.violations
            ],
            "blocked_models":  self.blocked_models,
            "blocked_columns": self.blocked_columns,
            "suggestion":      self.suggestion,
            "policy_version":  self.policy_version,
            "metadata":        self.metadata,
        }

    def user_message(self) -> str:
        """Return a user-friendly message suitable for display."""
        if self.allowed:
            if self.answer_mode == AnswerMode.AGGREGATE_ONLY:
                return "I can answer this, but only in aggregate form due to data governance policy."
            if self.answer_mode == AnswerMode.MASKED:
                return "I can answer this, but some sensitive fields will be masked."
            return "Request is allowed by data governance policy."

        if not self.violations:
            return "This request was blocked by data governance policy."

        reasons = "; ".join(v.message for v in self.violations[:3])
        base    = f"This request was blocked by data governance policy. Reason: {reasons}"
        if self.suggestion:
            base += f" Suggestion: {self.suggestion}"
        return base


# ──────────────────────────────────────────────────────────────────────────────
# SQL object extractor
# ──────────────────────────────────────────────────────────────────────────────

# Very lightweight SQL parser — only needs to find table/column references,
# not fully parse SQL. A proper AST parser would be preferable for production
# but introduces no new dependency here.

_DANGEROUS_PATTERNS = re.compile(
    r"\b(DROP|TRUNCATE|DELETE|UPDATE|INSERT|MERGE|CREATE|ALTER|GRANT|REVOKE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)

_FROM_PATTERN = re.compile(
    r"\bFROM\b\s+([\w.`\[\]\"]+)", re.IGNORECASE
)
_JOIN_PATTERN = re.compile(
    r"\bJOIN\b\s+([\w.`\[\]\"]+)", re.IGNORECASE
)
_INTO_PATTERN = re.compile(
    r"\bINTO\b\s+([\w.`\[\]\"]+)", re.IGNORECASE
)
_SELECT_COLS  = re.compile(
    r"\bSELECT\b(.*?)\bFROM\b", re.IGNORECASE | re.DOTALL
)


def extract_referenced_tables(sql: str) -> Set[str]:
    """Return set of table/model names referenced in the SQL."""
    tables: Set[str] = set()
    for pattern in (_FROM_PATTERN, _JOIN_PATTERN, _INTO_PATTERN):
        for match in pattern.finditer(sql):
            raw = match.group(1).strip().strip("`\"[]")
            # Take only the last part (e.g. schema.table → table)
            tables.add(raw.split(".")[-1].lower())
    return tables


def extract_selected_columns(sql: str) -> Set[str]:
    """
    Best-effort extraction of column names from SELECT clause.
    Handles "SELECT *", aliases (col AS alias), and expressions.
    """
    m = _SELECT_COLS.search(sql)
    if not m:
        return set()
    select_text = m.group(1)

    # Detect bare wildcard SELECT * (not COUNT(*), SUM(*) etc.)
    # A bare wildcard looks like: "SELECT *", "SELECT t.*", or ", *" — i.e.
    # the * is not immediately preceded by an open parenthesis.
    if re.search(r"(?<!\()\*(?!\))", select_text):
        return {"*"}  # Wildcard — must check model policy

    cols: Set[str] = set()
    for part in select_text.split(","):
        part = part.strip()
        # Handle "expr AS alias" — take the alias as final name
        if re.search(r"\bAS\b", part, re.IGNORECASE):
            alias = re.split(r"\bAS\b", part, flags=re.IGNORECASE)[-1].strip()
            cols.add(alias.strip("`\"[] \n"))
        else:
            # Could be "table.column" or just "column"
            raw = part.strip().strip("`\"[] \n").split(".")[-1]
            if raw:
                cols.add(raw.lower())
    return cols


def check_query_risk(sql: str) -> Tuple[bool, List[str]]:
    """
    Check for dangerous SQL patterns.
    Returns (is_risky, list_of_reasons).
    """
    reasons: List[str] = []
    for m in _DANGEROUS_PATTERNS.finditer(sql):
        reasons.append(f"Dangerous SQL keyword detected: {m.group(0).upper()}")
    return bool(reasons), reasons


# ──────────────────────────────────────────────────────────────────────────────
# Core validators
# ──────────────────────────────────────────────────────────────────────────────

def check_model_access_rules(
    model: ModelAiPolicy,
    use_case: str,
    workspace_id: str,
    tenant_id: str,
    result: GuardrailResult,
) -> bool:
    """
    Apply model-level access rules.  Mutates result in-place.
    Returns True if model is blocked.
    """
    if not model.is_ai_eligible():
        result.block(
            "MODEL_DENIED",
            f"Model '{model.model_name}' is not AI-eligible "
            f"(ai_enabled={model.ai_enabled}, ai_access_level={model.ai_access_level.value}).",
            model.model_name,
        )
        result.blocked_models.append(model.model_name)
        return True

    if workspace_id and not model.is_approved_for_workspace(workspace_id):
        result.block(
            "WORKSPACE_DENIED",
            f"Model '{model.model_name}' is not approved for workspace '{workspace_id}'.",
            model.model_name,
        )
        result.blocked_models.append(model.model_name)
        return True

    if tenant_id and not model.is_approved_for_tenant(tenant_id):
        result.block(
            "TENANT_DENIED",
            f"Model '{model.model_name}' is not approved for tenant '{tenant_id}'.",
            model.model_name,
        )
        result.blocked_models.append(model.model_name)
        return True

    if use_case and use_case in model.blocked_ai_use_cases:
        result.block(
            "USE_CASE_DENIED",
            f"Use case '{use_case}' is blocked for model '{model.model_name}'.",
            model.model_name,
        )
        result.blocked_models.append(model.model_name)
        return True

    return False


def check_column_output_rules(
    model_name: str,
    col: ColumnAiPolicy,
    result: GuardrailResult,
) -> bool:
    """
    Apply column output rules.  Mutates result.  Returns True if column is blocked.
    """
    col_key = f"{model_name}.{col.column_name}"

    if col.phi:
        result.block(
            "PHI_OUTPUT_DENIED",
            f"Column '{col_key}' contains PHI and cannot appear in output.",
            col_key,
        )
        result.blocked_columns.append(col_key)
        return True

    if col.pii == PiiClass.DIRECT and not col.is_output_safe():
        result.block(
            "PII_OUTPUT_DENIED",
            f"Column '{col_key}' contains direct PII. Output requires masking.",
            col_key,
        )
        result.blocked_columns.append(col_key)
        return True

    if not col.allowed_in_output:
        result.block(
            "COLUMN_OUTPUT_DENIED",
            f"Column '{col_key}' is not allowed in AI output (allowed_in_output=false).",
            col_key,
        )
        result.blocked_columns.append(col_key)
        return True

    return False


def check_export_rules(
    model: ModelAiPolicy,
    result: GuardrailResult,
) -> bool:
    from .metadata_schema import ExportPolicy
    if model.export_policy == ExportPolicy.DENY:
        result.block(
            "EXPORT_DENIED",
            f"Export is denied for model '{model.model_name}' by policy.",
            model.model_name,
        )
        result.blocked_models.append(model.model_name)
        return True
    return False


def build_policy_violation_report(result: GuardrailResult) -> Dict[str, Any]:
    """Convert a GuardrailResult into a structured policy violation report."""
    return {
        "allowed":          result.allowed,
        "violation_count":  len(result.violations),
        "violations":       [
            {"code": v.code, "message": v.message, "object": v.object}
            for v in result.violations
        ],
        "blocked_models":   result.blocked_models,
        "blocked_columns":  result.blocked_columns,
        "answer_mode":      result.answer_mode.value,
        "suggestion":       result.suggestion,
        "policy_version":   result.policy_version,
        "user_message":     result.user_message(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public validator functions
# ──────────────────────────────────────────────────────────────────────────────

def validate_sql_against_policy(
    sql:          str,
    catalog:      AiPolicyCatalog,
    use_case:     str = "text2sql",
    workspace_id: str = "",
    tenant_id:    str = "",
) -> GuardrailResult:
    """
    Validate an AI-generated SQL query against the AI policy catalog.

    Checks:
    - Dangerous SQL keywords (DROP, DELETE, etc.)
    - Each referenced table is AI-eligible
    - Workspace/tenant access rules
    - Output columns against column-level policy
    - aggregate_only enforcement (wildcard SELECT on restricted models)
    - Export rules if EXPORT intent detected
    """
    result = GuardrailResult(policy_version=catalog.version)

    # 1. Dangerous keyword check (hard deny)
    is_risky, risk_reasons = check_query_risk(sql)
    if is_risky:
        for r in risk_reasons:
            result.block("DANGEROUS_SQL", r)
        result.suggestion = "Use SELECT queries only. Write/delete operations are not permitted."
        return result

    # 2. Extract referenced tables
    tables = extract_referenced_tables(sql)
    if not tables:
        logger.debug("validate_sql: no tables extracted from SQL")

    # 3. Check each table
    aggregate_only_models: List[str] = []
    for table_name in tables:
        model = catalog.get_model(table_name)
        if model is None:
            # Unknown model — warn but don't hard-block (warehouse is true enforcer)
            result.violations.append(PolicyViolation(
                code    = "UNKNOWN_MODEL",
                message = f"Model '{table_name}' not found in AI policy catalog.",
                object  = table_name,
            ))
            continue

        blocked = check_model_access_rules(model, use_case, workspace_id, tenant_id, result)
        if blocked:
            continue

        # Aggregate-only enforcement
        if model.default_answer_mode == AnswerMode.AGGREGATE_ONLY:
            aggregate_only_models.append(table_name)

    if result.blocked_models:
        result.suggestion = (
            "Use only AI-approved models. "
            "Check the governance panel for certified assets available in this workspace."
        )
        return result

    # 4. Column-level output check
    selected_cols = extract_selected_columns(sql)
    is_wildcard   = "*" in selected_cols

    for table_name in tables:
        model = catalog.get_model(table_name)
        if not model or not model.is_ai_eligible():
            continue

        if is_wildcard and table_name in aggregate_only_models:
            result.block(
                "AGGREGATE_ONLY_VIOLATION",
                f"Model '{table_name}' is aggregate_only. SELECT * is not permitted.",
                table_name,
            )
            result.suggestion = (
                f"Use GROUP BY with aggregate functions (COUNT, SUM, AVG) instead of SELECT *."
            )

        if not is_wildcard:
            for col_name in selected_cols:
                col = catalog.get_column(table_name, col_name)
                if col is None:
                    continue  # Unknown column — warehouse enforces
                check_column_output_rules(table_name, col, result)

    # 5. Set effective answer mode
    if not result.allowed:
        result.answer_mode = AnswerMode.DENY
    elif aggregate_only_models:
        result.answer_mode = AnswerMode.AGGREGATE_ONLY
    else:
        result.answer_mode = AnswerMode.FULL

    return result


def validate_summary_against_policy(
    summary_text:  str,
    model_names:   List[str],
    catalog:       AiPolicyCatalog,
    use_case:      str = "summarization",
    workspace_id:  str = "",
    tenant_id:     str = "",
) -> GuardrailResult:
    """
    Validate a generated summary against the AI policy catalog.

    Checks whether the models referenced allow summarization,
    and whether the summary mode is appropriate.
    """
    from .metadata_schema import SummarizationPolicy

    result = GuardrailResult(policy_version=catalog.version)

    for model_name in model_names:
        model = catalog.get_model(model_name)
        if model is None:
            continue

        blocked = check_model_access_rules(model, use_case, workspace_id, tenant_id, result)
        if blocked:
            continue

        if model.summarization_policy == SummarizationPolicy.DENY:
            result.block(
                "SUMMARIZATION_DENIED",
                f"Summarization is not allowed for model '{model_name}'.",
                model_name,
            )
            result.suggestion = "Request metadata-only information about this dataset."

        elif model.summarization_policy == SummarizationPolicy.AGGREGATE_ONLY:
            result.answer_mode = AnswerMode.AGGREGATE_ONLY

        elif model.summarization_policy == SummarizationPolicy.MASKED_ONLY:
            if result.answer_mode != AnswerMode.AGGREGATE_ONLY:
                result.answer_mode = AnswerMode.MASKED

    if not result.allowed:
        result.answer_mode = AnswerMode.DENY

    result.metadata["summary_length"] = len(summary_text)
    result.metadata["models_checked"] = model_names
    return result


def validate_retrieval_against_policy(
    retrieved_chunks: List[str],
    source_model:     str,
    catalog:          AiPolicyCatalog,
    use_case:         str = "rag",
    workspace_id:     str = "",
    tenant_id:        str = "",
) -> GuardrailResult:
    """
    Validate RAG retrieval results against the AI policy catalog.

    Checks whether the source model allows retrieval/RAG exposure.
    """
    from .metadata_schema import RagExposure, RetrievalPolicy

    result = GuardrailResult(policy_version=catalog.version)
    model  = catalog.get_model(source_model)

    if model is None:
        result.violations.append(PolicyViolation(
            code    = "UNKNOWN_SOURCE",
            message = f"Source model '{source_model}' not found in AI policy catalog.",
            object  = source_model,
        ))
        return result

    blocked = check_model_access_rules(model, use_case, workspace_id, tenant_id, result)
    if blocked:
        result.answer_mode = AnswerMode.DENY
        return result

    if model.rag_exposure == RagExposure.DENY:
        result.block(
            "RAG_DENIED",
            f"RAG retrieval is not allowed for model '{source_model}'.",
            source_model,
        )
        result.answer_mode = AnswerMode.DENY
        result.suggestion  = "Use metadata-only mode or request a summary without specific records."
        return result

    if model.rag_exposure == RagExposure.METADATA_ONLY:
        result.answer_mode = AnswerMode.METADATA_ONLY
        result.metadata["mode_reason"] = "rag_exposure=metadata_only"

    if model.retrieval_policy == RetrievalPolicy.DENY:
        result.block(
            "RETRIEVAL_DENIED",
            f"Retrieval is not allowed for model '{source_model}'.",
            source_model,
        )
        result.answer_mode = AnswerMode.DENY
        return result

    result.metadata["chunk_count"]   = len(retrieved_chunks)
    result.metadata["source_model"]  = source_model
    return result


def validate_tool_action_against_policy(
    tool_name:    str,
    action_args:  Dict[str, Any],
    catalog:      AiPolicyCatalog,
    workspace_id: str = "",
    tenant_id:    str = "",
) -> GuardrailResult:
    """
    Validate an agent tool call against the AI policy catalog.

    Checks:
    - Any referenced model in action_args is AI-eligible
    - Agent action policy for that model
    - Export-related tool calls
    """
    from .metadata_schema import AgentActionPolicy

    result = GuardrailResult(policy_version=catalog.version)
    use_case = "agent_tool"

    # Extract any model/table names from action args
    referenced = set()
    for v in action_args.values():
        if isinstance(v, str):
            # Heuristic: look for model names that appear in the catalog
            for model_name in catalog.models:
                if model_name.lower() in v.lower():
                    referenced.add(model_name)

    for model_name in referenced:
        model = catalog.get_model(model_name)
        if not model:
            continue

        blocked = check_model_access_rules(model, use_case, workspace_id, tenant_id, result)
        if blocked:
            continue

        if model.agent_action_policy == AgentActionPolicy.DENY:
            result.block(
                "AGENT_ACTION_DENIED",
                f"Agent tool actions are denied for model '{model_name}'.",
                model_name,
            )
            result.suggestion = "This operation requires manual execution or explicit approval."

        elif model.agent_action_policy == AgentActionPolicy.APPROVAL_REQUIRED:
            result.answer_mode = AnswerMode.DENY  # Block until approved
            result.block(
                "AGENT_ACTION_APPROVAL_REQUIRED",
                f"Agent tool action on '{model_name}' requires explicit approval.",
                model_name,
            )
            result.suggestion = "Request approval from a data steward before proceeding."

        # Export check
        if any(kw in tool_name.lower() for kw in ("export", "download", "write", "send")):
            check_export_rules(model, result)

    result.metadata["tool_name"]   = tool_name
    result.metadata["models_referenced"] = list(referenced)
    return result


def validate_ai_action_against_policy(
    use_case:     str,
    target_models: List[str],
    catalog:      AiPolicyCatalog,
    workspace_id: str = "",
    tenant_id:    str = "",
    action_meta:  Optional[Dict[str, Any]] = None,
) -> GuardrailResult:
    """
    General-purpose AI action validator for any use case.

    Routes to use-case-specific checks based on use_case string.
    """
    result = GuardrailResult(policy_version=catalog.version)
    action_meta = action_meta or {}

    for model_name in target_models:
        model = catalog.get_model(model_name)
        if not model:
            continue

        blocked = check_model_access_rules(model, use_case, workspace_id, tenant_id, result)
        if blocked:
            continue

        # Use-case-specific checks
        if use_case in ("text2sql", "sql_generation"):
            pass  # Delegated to validate_sql_against_policy for actual SQL
        elif use_case in ("summarization", "summary"):
            from .metadata_schema import SummarizationPolicy
            if model.summarization_policy == SummarizationPolicy.DENY:
                result.block(
                    "SUMMARIZATION_DENIED",
                    f"Summarization is denied for '{model_name}'.",
                    model_name,
                )
        elif use_case in ("export",):
            check_export_rules(model, result)
        elif use_case in ("rag", "retrieval"):
            from .metadata_schema import RetrievalPolicy
            if model.retrieval_policy == RetrievalPolicy.DENY:
                result.block(
                    "RETRIEVAL_DENIED",
                    f"Retrieval is denied for '{model_name}'.",
                    model_name,
                )
        elif use_case in ("explanation", "explain"):
            from .metadata_schema import ExplanationPolicy
            if model.explanation_policy == ExplanationPolicy.DENY:
                result.block(
                    "EXPLANATION_DENIED",
                    f"Explanation is denied for '{model_name}'.",
                    model_name,
                )

    # Set final answer mode
    if not result.allowed:
        result.answer_mode = AnswerMode.DENY

    result.metadata["use_case"]       = use_case
    result.metadata["target_models"]  = target_models
    result.metadata["action_meta"]    = action_meta
    return result

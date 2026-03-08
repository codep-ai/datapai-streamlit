"""
guardrail — Datap.ai dbt-driven AI Guardrail Framework
=======================================================

This package provides a runtime AI guardrail system driven by dbt metadata.

Architecture:
  dbt metadata (manifest.json) → policy_compiler → runtime enforcement

Modules:
  metadata_schema   — Python dataclasses for model-level and column-level AI governance metadata
  policy_compiler   — Reads dbt manifest, normalises metadata, emits structured policy catalog
  validators        — Pre- and post-generation validators (SQL, RAG, summarization, tool actions)
  context_filter    — Pre-generation context filtering (exclude denied/unsafe assets and fields)
  governed_action   — Governed AI Action Framework (13-step lifecycle)
  trace_helpers     — Thin wrappers that emit policy decision events into the Trace Ledger

Public surface:
  from guardrail import PolicyCompiler, GuardrailResult, filter_context, validate_sql_against_policy
"""

from .metadata_schema import (
    AiAccessLevel,
    SensitivityLevel,
    AnswerMode,
    ExportPolicy,
    RetrievalPolicy,
    SummarizationPolicy,
    ExplanationPolicy,
    RagExposure,
    AgentActionPolicy,
    RiskTier,
    PiiClass,
    SecurityClass,
    MaskingRule,
    ModelAiPolicy,
    ColumnAiPolicy,
    AiPolicyCatalog,
)

from .policy_compiler import PolicyCompiler

from .validators import (
    GuardrailResult,
    validate_sql_against_policy,
    validate_summary_against_policy,
    validate_retrieval_against_policy,
    validate_tool_action_against_policy,
    validate_ai_action_against_policy,
)

from .context_filter import (
    filter_context,
    build_safe_schema_context,
    get_allowed_assets_for_use_case,
    get_allowed_fields_for_asset,
)

from .governed_action import GovernedAction, AiUseCase

__all__ = [
    # Schema
    "AiAccessLevel", "SensitivityLevel", "AnswerMode", "ExportPolicy",
    "RetrievalPolicy", "SummarizationPolicy", "ExplanationPolicy",
    "RagExposure", "AgentActionPolicy", "RiskTier", "PiiClass",
    "SecurityClass", "MaskingRule", "ModelAiPolicy", "ColumnAiPolicy",
    "AiPolicyCatalog",
    # Compiler
    "PolicyCompiler",
    # Validators
    "GuardrailResult",
    "validate_sql_against_policy",
    "validate_summary_against_policy",
    "validate_retrieval_against_policy",
    "validate_tool_action_against_policy",
    "validate_ai_action_against_policy",
    # Context filter
    "filter_context",
    "build_safe_schema_context",
    "get_allowed_assets_for_use_case",
    "get_allowed_fields_for_asset",
    # Governed action
    "GovernedAction",
    "AiUseCase",
]

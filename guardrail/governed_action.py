"""
guardrail.governed_action
=========================

Governed AI Action Framework — 13-step lifecycle for any AI use case.

This module provides a wrapper that any AI capability (Text2SQL, RAG, summarization,
explanation, tool call, etc.) can use to run through the full AI governance lifecycle:

  1.  receive_request
  2.  identify_context (tenant/workspace/user/session)
  3.  classify_use_case
  4.  load_policy_catalog
  5.  resolve_eligible_assets
  6.  build_safe_context
  7.  invoke_model_or_tool
  8.  validate_generated_action
  9.  enforce_block_mask_aggregate
  10. execute_if_allowed
  11. trace_all_decisions
  12. return_response
  13. store_feedback (optional)

Usage:
    from guardrail.governed_action import GovernedAction, AiUseCase

    ga = GovernedAction(
        catalog=compiler.compile(),
        ledger=trace_ledger,   # optional
    )
    result = ga.run(
        use_case=AiUseCase.TEXT2SQL,
        request="Show me total orders by customer",
        identity={"tenant_id": "acme", "workspace_id": "analytics", "user_id": "u1"},
        invoke_fn=my_sql_generator,
    )
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .metadata_schema import AiPolicyCatalog, AnswerMode
from .validators import GuardrailResult, validate_ai_action_against_policy
from .context_filter import (
    filter_context,
    build_safe_schema_context,
    summarize_filtered_context,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Use case enum
# ──────────────────────────────────────────────────────────────────────────────

class AiUseCase(str, Enum):
    TEXT2SQL             = "text2sql"
    SQL_EXPLANATION      = "sql_explanation"
    SQL_EXECUTION        = "sql_execution"
    BI_METRIC_EXPLANATION = "bi_metric_explanation"
    DASHBOARD_EXPLANATION = "dashboard_explanation"
    DBT_MODEL_EXPLANATION = "dbt_model_explanation"
    RAG_RETRIEVAL        = "rag"
    SUMMARIZATION        = "summarization"
    NARRATIVE_INSIGHT    = "narrative_insight"
    FIELD_EXPLANATION    = "field_explanation"
    DOCUMENT_EXTRACTION  = "document_extraction"
    EXPORT               = "export"
    AIRBYTE_TRIGGER      = "airbyte_trigger"
    AIRFLOW_TRIGGER      = "airflow_trigger"
    WORKFLOW_TOOL        = "workflow_tool"
    GENERIC              = "generic"


# ──────────────────────────────────────────────────────────────────────────────
# Governed action request / response
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GovernedRequest:
    """Input to the GovernedAction.run() method."""
    use_case:     AiUseCase
    request:      str
    identity:     Dict[str, str]       # tenant_id, workspace_id, user_id, session_id
    schema_context: Optional[Dict[str, Any]] = None   # Raw schema context (optional)
    target_models:  Optional[List[str]]      = None   # Specific models to check
    invoke_kwargs:  Dict[str, Any]           = field(default_factory=dict)
    request_id:     Optional[str]            = None

    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())[:8]


@dataclass
class GovernedResponse:
    """Output from the GovernedAction.run() method."""
    allowed:           bool
    request_id:        str
    use_case:          str
    answer_mode:       AnswerMode
    result:            Optional[Any]      = None
    raw_generated:     Optional[Any]      = None   # Pre-validation LLM output
    guardrail_result:  Optional[GuardrailResult] = None
    safe_context_summary: Dict[str, Any]  = field(default_factory=dict)
    policy_version:    str                = ""
    trace_id:          Optional[str]      = None
    duration_ms:       int                = 0
    user_message:      str                = ""
    governance_panel:  Dict[str, Any]     = field(default_factory=dict)
    error:             Optional[str]      = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed":            self.allowed,
            "request_id":         self.request_id,
            "use_case":           self.use_case,
            "answer_mode":        self.answer_mode.value,
            "policy_version":     self.policy_version,
            "trace_id":           self.trace_id,
            "duration_ms":        self.duration_ms,
            "user_message":       self.user_message,
            "governance_panel":   self.governance_panel,
            "error":              self.error,
            "guardrail_result":   self.guardrail_result.to_dict() if self.guardrail_result else None,
            "safe_context_summary": self.safe_context_summary,
        }


# ──────────────────────────────────────────────────────────────────────────────
# GovernedAction framework
# ──────────────────────────────────────────────────────────────────────────────

class GovernedAction:
    """
    Wraps any AI capability in the full 13-step AI governance lifecycle.

    Parameters
    ----------
    catalog : AiPolicyCatalog
        The compiled AI policy catalog.
    ledger : optional
        A TraceLedger instance for tracing policy decisions.
        If None, tracing is skipped.
    """

    def __init__(
        self,
        catalog: AiPolicyCatalog,
        ledger:  Optional[Any] = None,
    ):
        self.catalog = catalog
        self.ledger  = ledger

        # Import trace helpers lazily to avoid hard dependency
        try:
            from .trace_helpers import GuardrailTracer
            self._tracer = GuardrailTracer(ledger) if ledger else None
        except ImportError:
            self._tracer = None

    def run(
        self,
        request:   GovernedRequest,
        invoke_fn: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    ) -> GovernedResponse:
        """
        Execute the 13-step governed AI action lifecycle.

        Parameters
        ----------
        request : GovernedRequest
            The incoming AI action request.
        invoke_fn : callable(request_text, safe_context) → raw_output, optional
            The actual AI model/tool invocation.  If None, returns the safe
            context only (useful for context inspection without LLM call).

        Returns
        -------
        GovernedResponse
        """
        start = time.time()
        request_id = request.request_id or str(uuid.uuid4())[:8]

        identity     = request.identity or {}
        tenant_id    = identity.get("tenant_id", "")
        workspace_id = identity.get("workspace_id", "")
        user_id      = identity.get("user_id", "")
        session_id   = identity.get("session_id", "")
        use_case_str = request.use_case.value

        # ── Step 1-2: Receive + identify context ──────────────────────────────
        logger.info("[GovernedAction] %s request_id=%s user=%s", use_case_str, request_id, user_id)

        # ── Step 3: Classify use case ─────────────────────────────────────────
        use_case = request.use_case

        # ── Step 4: Load policy catalog ───────────────────────────────────────
        catalog = self.catalog

        # ── Step 5: Resolve eligible assets ───────────────────────────────────
        target_models = request.target_models or []
        eligible      = catalog.eligible_models(use_case_str, workspace_id, tenant_id)
        eligible_names = [m.model_name for m in eligible]

        if target_models:
            # Check specified targets first
            pre_validation = validate_ai_action_against_policy(
                use_case     = use_case_str,
                target_models= target_models,
                catalog      = catalog,
                workspace_id = workspace_id,
                tenant_id    = tenant_id,
            )
            if not pre_validation.allowed:
                return self._build_blocked_response(
                    request_id   = request_id,
                    use_case_str = use_case_str,
                    validation   = pre_validation,
                    start        = start,
                    trace_id     = None,
                )

        # ── Step 6: Build safe context ────────────────────────────────────────
        if request.schema_context:
            safe_ctx = filter_context(
                schema_context = request.schema_context,
                catalog        = catalog,
                use_case       = use_case_str,
                workspace_id   = workspace_id,
                tenant_id      = tenant_id,
            )
        else:
            safe_ctx = build_safe_schema_context(
                catalog      = catalog,
                use_case     = use_case_str,
                workspace_id = workspace_id,
                tenant_id    = tenant_id,
                model_names  = target_models or None,
            )

        ctx_summary = summarize_filtered_context(safe_ctx)

        if self._tracer:
            self._tracer.trace_context_filtering(
                request_id   = request_id,
                use_case     = use_case_str,
                included     = ctx_summary.get("included_models", []),
                identity     = identity,
            )

        # ── Step 7: Invoke model or tool ──────────────────────────────────────
        raw_output = None
        if invoke_fn is not None:
            try:
                raw_output = invoke_fn(request.request, safe_ctx, **request.invoke_kwargs)
            except Exception as exc:
                logger.error("[GovernedAction] invoke_fn raised: %s", exc, exc_info=True)
                return GovernedResponse(
                    allowed    = False,
                    request_id = request_id,
                    use_case   = use_case_str,
                    answer_mode = AnswerMode.DENY,
                    error      = f"Invocation error: {exc}",
                    duration_ms = int((time.time() - start) * 1000),
                    user_message = "The AI action failed due to an internal error.",
                )

        # ── Step 8: Validate generated action / response ──────────────────────
        post_validation = self._validate_output(
            use_case     = use_case_str,
            raw_output   = raw_output,
            catalog      = catalog,
            target_models= eligible_names,
            workspace_id = workspace_id,
            tenant_id    = tenant_id,
        )

        if self._tracer:
            self._tracer.trace_action_validation(
                request_id = request_id,
                use_case   = use_case_str,
                result     = post_validation,
                identity   = identity,
            )

        # ── Step 9: Enforce block / mask / aggregate rules ────────────────────
        if not post_validation.allowed:
            if self._tracer:
                self._tracer.trace_blocked_action(
                    request_id = request_id,
                    use_case   = use_case_str,
                    result     = post_validation,
                    identity   = identity,
                )
            return self._build_blocked_response(
                request_id   = request_id,
                use_case_str = use_case_str,
                validation   = post_validation,
                start        = start,
                trace_id     = None,
                ctx_summary  = ctx_summary,
            )

        # ── Step 10: Execute if allowed ────────────────────────────────────────
        # (Actual execution happens in the caller's invoke_fn; GovernedAction
        #  governs the context and validates the output — execution is external.)

        # ── Steps 11-12: Trace + return ───────────────────────────────────────
        duration_ms   = int((time.time() - start) * 1000)
        answer_mode   = post_validation.answer_mode
        policy_ver    = catalog.version

        governance_panel = self._build_governance_panel(
            use_case        = use_case_str,
            answer_mode     = answer_mode,
            ctx_summary     = ctx_summary,
            validation      = post_validation,
            policy_version  = policy_ver,
        )

        user_msg = _answer_mode_user_message(answer_mode, post_validation)

        return GovernedResponse(
            allowed           = True,
            request_id        = request_id,
            use_case          = use_case_str,
            answer_mode       = answer_mode,
            result            = raw_output,
            raw_generated     = raw_output,
            guardrail_result  = post_validation,
            safe_context_summary = ctx_summary,
            policy_version    = policy_ver,
            duration_ms       = duration_ms,
            user_message      = user_msg,
            governance_panel  = governance_panel,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _validate_output(
        self,
        use_case:     str,
        raw_output:   Any,
        catalog:      AiPolicyCatalog,
        target_models: List[str],
        workspace_id: str,
        tenant_id:    str,
    ) -> GuardrailResult:
        """Route output validation to the appropriate validator."""
        if raw_output is None:
            # No output yet — just validate model eligibility
            return validate_ai_action_against_policy(
                use_case      = use_case,
                target_models = target_models,
                catalog       = catalog,
                workspace_id  = workspace_id,
                tenant_id     = tenant_id,
            )

        if use_case in (AiUseCase.TEXT2SQL.value, AiUseCase.SQL_EXECUTION.value):
            from .validators import validate_sql_against_policy
            sql_text = str(raw_output)
            return validate_sql_against_policy(
                sql          = sql_text,
                catalog      = catalog,
                use_case     = use_case,
                workspace_id = workspace_id,
                tenant_id    = tenant_id,
            )

        if use_case in (AiUseCase.SUMMARIZATION.value, AiUseCase.NARRATIVE_INSIGHT.value):
            from .validators import validate_summary_against_policy
            return validate_summary_against_policy(
                summary_text  = str(raw_output),
                model_names   = target_models,
                catalog       = catalog,
                use_case      = use_case,
                workspace_id  = workspace_id,
                tenant_id     = tenant_id,
            )

        if use_case in (AiUseCase.RAG_RETRIEVAL.value,):
            # For RAG, raw_output is typically a list of chunks
            from .validators import validate_retrieval_against_policy
            chunks = raw_output if isinstance(raw_output, list) else [str(raw_output)]
            # Validate against first target model as source
            source = target_models[0] if target_models else ""
            return validate_retrieval_against_policy(
                retrieved_chunks = chunks,
                source_model     = source,
                catalog          = catalog,
                use_case         = use_case,
                workspace_id     = workspace_id,
                tenant_id        = tenant_id,
            )

        # Generic fallback
        return validate_ai_action_against_policy(
            use_case      = use_case,
            target_models = target_models,
            catalog       = catalog,
            workspace_id  = workspace_id,
            tenant_id     = tenant_id,
        )

    def _build_blocked_response(
        self,
        request_id:   str,
        use_case_str: str,
        validation:   GuardrailResult,
        start:        float,
        trace_id:     Optional[str],
        ctx_summary:  Dict[str, Any] = None,
    ) -> GovernedResponse:
        duration_ms = int((time.time() - start) * 1000)
        panel = self._build_governance_panel(
            use_case       = use_case_str,
            answer_mode    = AnswerMode.DENY,
            ctx_summary    = ctx_summary or {},
            validation     = validation,
            policy_version = self.catalog.version,
        )
        return GovernedResponse(
            allowed           = False,
            request_id        = request_id,
            use_case          = use_case_str,
            answer_mode       = AnswerMode.DENY,
            guardrail_result  = validation,
            safe_context_summary = ctx_summary or {},
            policy_version    = self.catalog.version,
            trace_id          = trace_id,
            duration_ms       = duration_ms,
            user_message      = validation.user_message(),
            governance_panel  = panel,
        )

    def _build_governance_panel(
        self,
        use_case:       str,
        answer_mode:    AnswerMode,
        ctx_summary:    Dict[str, Any],
        validation:     GuardrailResult,
        policy_version: str,
    ) -> Dict[str, Any]:
        """Build the governance panel dict for Streamlit / OpenWebUI display."""
        return {
            "use_case":              use_case,
            "answer_mode":           answer_mode.value,
            "policy_version":        policy_version,
            "certified_assets_used": ctx_summary.get("certified_models", []),
            "included_models":       ctx_summary.get("included_models", []),
            "aggregate_only_models": ctx_summary.get("aggregate_only_models", []),
            "pii_annotated_columns": ctx_summary.get("pii_annotated_columns", []),
            "blocked_models":        validation.blocked_models,
            "blocked_columns":       validation.blocked_columns,
            "violations":            [
                {"code": v.code, "message": v.message}
                for v in validation.violations
            ],
            "suggestion":            validation.suggestion,
        }


# ──────────────────────────────────────────────────────────────────────────────
# User-message helper
# ──────────────────────────────────────────────────────────────────────────────

def _answer_mode_user_message(
    answer_mode: AnswerMode,
    result:      GuardrailResult,
) -> str:
    if not result.allowed:
        return result.user_message()
    if answer_mode == AnswerMode.AGGREGATE_ONLY:
        return (
            "I can answer this, but only in aggregate form because this "
            "dataset is governed for aggregate use only."
        )
    if answer_mode == AnswerMode.MASKED:
        return (
            "I can answer this, but some sensitive fields have been masked "
            "in accordance with data governance policy."
        )
    if answer_mode == AnswerMode.METADATA_ONLY:
        return (
            "I can provide metadata about this asset, but I cannot expose "
            "the underlying record data due to governance policy."
        )
    return ""

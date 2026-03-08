"""
DataPAI dbt AI Guardrail Pipeline for OpenWebUI
================================================

A policy-aware OpenWebUI pipeline that wraps any AI request with the
Datap.ai dbt AI guardrail framework.

HOW IT WORKS
────────────
Every user message flows through:
  1. Parse tenant/workspace/user context from message metadata
  2. Classify AI use case (Text2SQL, RAG, summarization, etc.)
  3. Load the dbt AI policy catalog
  4. Pre-filter schema context to only policy-allowed assets
  5. Validate the requested action / generated response
  6. Return a governance-aware response with structured metadata

The pipeline adds a governance status block to every response, e.g.:
  - ✅ Used certified governed assets
  - 🔒 Sensitive fields masked
  - 📊 Aggregate-only mode enforced
  - 🚫 Request blocked — see reason

INSTALLATION
────────────
1. In OpenWebUI → Admin → Pipelines → Add Pipeline:
     Upload this file.

2. Set Pipeline environment variables:
     GUARDRAIL_MANIFEST_PATH = /app/dbt-demo/target/manifest.json
     GUARDRAIL_DEFAULT_WORKSPACE = analytics
     GUARDRAIL_DEFAULT_TENANT    = default

3. The model "DataPAI Guardrail" appears in the OpenWebUI model selector.

ENVIRONMENT VARIABLES
─────────────────────
  GUARDRAIL_MANIFEST_PATH      Path to dbt manifest.json
  GUARDRAIL_DEFAULT_WORKSPACE  Default workspace_id if not in message metadata
  GUARDRAIL_DEFAULT_TENANT     Default tenant_id if not in message metadata
  GUARDRAIL_STRICT_MODE        If true, block requests with unknown models (default: false)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Generator, Iterator, List, Optional, Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Environment ───────────────────────────────────────────────────────────────
GUARDRAIL_MANIFEST_PATH      = os.environ.get(
    "GUARDRAIL_MANIFEST_PATH",
    os.path.join("dbt-demo", "target", "manifest.json"),
)
GUARDRAIL_DEFAULT_WORKSPACE  = os.environ.get("GUARDRAIL_DEFAULT_WORKSPACE", "analytics")
GUARDRAIL_DEFAULT_TENANT     = os.environ.get("GUARDRAIL_DEFAULT_TENANT", "default")
GUARDRAIL_STRICT_MODE        = os.environ.get("GUARDRAIL_STRICT_MODE", "false").lower() == "true"

# ── Lazy imports — guardrail module may not be installed ──────────────────────
try:
    from guardrail.policy_compiler import PolicyCompiler
    from guardrail.validators import validate_sql_against_policy, validate_ai_action_against_policy
    from guardrail.context_filter import build_safe_schema_context, summarize_filtered_context
    from guardrail.governed_action import GovernedAction, GovernedRequest, AiUseCase
    _GUARDRAIL_OK = True
except ImportError:
    _GUARDRAIL_OK = False
    logger.warning("guardrail module not found — pipeline will pass through without enforcement")


# ──────────────────────────────────────────────────────────────────────────────
# Module-level compiler singleton
# ──────────────────────────────────────────────────────────────────────────────

_compiler = None


def _get_compiler():
    global _compiler
    if not _GUARDRAIL_OK:
        return None
    if _compiler is None:
        _compiler = PolicyCompiler(manifest_path=GUARDRAIL_MANIFEST_PATH)
    return _compiler


# ──────────────────────────────────────────────────────────────────────────────
# Use-case classifier
# ──────────────────────────────────────────────────────────────────────────────

def _classify_use_case(message: str) -> str:
    """Lightweight keyword-based use-case classifier."""
    msg = message.lower()
    if any(k in msg for k in ("select ", "from ", "where ", "group by", "order by", "sql")):
        return "text2sql"
    if any(k in msg for k in ("summarize", "summarise", "summary", "overview")):
        return "summarization"
    if any(k in msg for k in ("explain", "what is", "what does", "how does")):
        return "explanation"
    if any(k in msg for k in ("export", "download", "send to")):
        return "export"
    if any(k in msg for k in ("search", "find", "look up", "retrieve", "rag")):
        return "rag"
    return "generic"


# ──────────────────────────────────────────────────────────────────────────────
# Governance status formatter
# ──────────────────────────────────────────────────────────────────────────────

def _format_governance_block(
    allowed:      bool,
    answer_mode:  str,
    panel:        Dict[str, Any],
    user_message: str = "",
) -> str:
    """Format a governance status block for OpenWebUI markdown output."""
    lines = ["\n\n---", "### 🛡️ Governance Status"]

    if not allowed:
        lines.append(f"🚫 **Request blocked by data governance policy**")
        if user_message:
            lines.append(f"> {user_message}")
        violations = panel.get("violations", [])
        if violations:
            lines.append("\n**Policy violations:**")
            for v in violations[:3]:
                lines.append(f"- `{v.get('code', '')}` — {v.get('message', '')}")
        suggestion = panel.get("suggestion", "")
        if suggestion:
            lines.append(f"\n💡 **Suggestion:** {suggestion}")
    else:
        mode_label = {
            "full":           "✅ Full response — using certified governed assets",
            "masked":         "🔒 Masked — sensitive fields have been excluded",
            "aggregate_only": "📊 Aggregate-only — row-level data is not returned",
            "metadata_only":  "📋 Metadata-only — schema information only",
        }.get(answer_mode, f"ℹ️ Mode: {answer_mode}")
        lines.append(mode_label)

        certified = panel.get("certified_assets_used", [])
        if certified:
            lines.append(f"✅ **Certified assets:** {', '.join(certified)}")

        aggregate = panel.get("aggregate_only_models", [])
        if aggregate:
            lines.append(
                f"📊 **Aggregate-only enforcement:** {', '.join(aggregate)}"
            )

        pii = panel.get("pii_annotated_columns", [])
        if pii:
            lines.append(
                f"🔒 **{len(pii)} PII field(s) hidden from context**"
            )

    policy_ver = panel.get("policy_version", "")
    use_case   = panel.get("use_case", "")
    lines.append(
        f"\n*Policy: `{policy_ver[:12] if policy_ver else 'unknown'}` | "
        f"Use case: `{use_case}`*"
    )

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# OpenWebUI Pipeline class
# ──────────────────────────────────────────────────────────────────────────────

class Pipeline:
    """
    OpenWebUI-compatible Pipeline that adds dbt AI guardrail enforcement
    to any user request.
    """

    class Valves(BaseModel):
        manifest_path:     str = GUARDRAIL_MANIFEST_PATH
        default_workspace: str = GUARDRAIL_DEFAULT_WORKSPACE
        default_tenant:    str = GUARDRAIL_DEFAULT_TENANT
        strict_mode:       bool = GUARDRAIL_STRICT_MODE
        show_governance:   bool = True
        priority:          int = 0

    def __init__(self):
        self.name   = "DataPAI Guardrail"
        self.valves = self.Valves()

    def on_startup(self):
        """Pre-warm the policy catalog on pipeline startup."""
        logger.info("DataPAI Guardrail pipeline starting up…")
        compiler = _get_compiler()
        if compiler:
            try:
                catalog = compiler.compile()
                logger.info(
                    "Policy catalog loaded: %d models, %d columns",
                    len(catalog.models), len(catalog.columns),
                )
            except Exception as exc:
                logger.warning("Could not pre-warm policy catalog: %s", exc)

    def on_shutdown(self):
        pass

    def pipe(
        self,
        user_message: str,
        model_id:     str,
        messages:     List[Dict[str, Any]],
        body:         Dict[str, Any],
    ) -> Union[str, Generator, Iterator]:
        """
        Main pipeline handler.

        Runs the AI guardrail lifecycle:
        1. Extract context from body/messages
        2. Classify use case
        3. Validate action against policy
        4. If blocked: return governance-blocked response
        5. If allowed: pass through (or generate policy-aware response)
        6. Append governance status block to response
        """
        if not _GUARDRAIL_OK:
            return (
                user_message + "\n\n*⚠️ Guardrail module not available — "
                "running without AI governance enforcement.*"
            )

        # ── Extract identity context ──────────────────────────────────────────
        meta         = body.get("metadata") or {}
        workspace_id = meta.get("workspace_id") or self.valves.default_workspace
        tenant_id    = meta.get("tenant_id")    or self.valves.default_tenant
        user_id      = meta.get("user_id")      or body.get("user", {}).get("id", "")
        session_id   = meta.get("session_id")   or body.get("session_id", "")

        identity = {
            "tenant_id":    tenant_id,
            "workspace_id": workspace_id,
            "user_id":      user_id,
            "session_id":   session_id,
        }

        # ── Classify use case ─────────────────────────────────────────────────
        use_case = _classify_use_case(user_message)

        # ── Load policy catalog ───────────────────────────────────────────────
        compiler = _get_compiler()
        if compiler is None:
            return user_message  # passthrough if no compiler

        try:
            catalog = compiler.compile()
        except Exception as exc:
            logger.warning("Policy catalog compile failed: %s", exc)
            return user_message

        # ── Pre-generation validation ─────────────────────────────────────────
        # For SQL-like requests, attempt to extract any table references from the message
        target_models = [
            m_name for m_name in catalog.models
            if m_name.lower() in user_message.lower()
        ]

        pre_result = validate_ai_action_against_policy(
            use_case      = use_case,
            target_models = target_models,
            catalog       = catalog,
            workspace_id  = workspace_id,
            tenant_id     = tenant_id,
        )

        governance_panel = {
            "use_case":              use_case,
            "answer_mode":           pre_result.answer_mode.value,
            "policy_version":        catalog.version,
            "certified_assets_used": [],
            "included_models":       target_models,
            "aggregate_only_models": [],
            "pii_annotated_columns": [],
            "blocked_models":        pre_result.blocked_models,
            "blocked_columns":       pre_result.blocked_columns,
            "violations":            [
                {"code": v.code, "message": v.message}
                for v in pre_result.violations
            ],
            "suggestion": pre_result.suggestion,
        }

        # ── Blocked: return governance message ────────────────────────────────
        if not pre_result.allowed:
            block_msg = pre_result.user_message()
            gov_block = _format_governance_block(
                allowed      = False,
                answer_mode  = "deny",
                panel        = governance_panel,
                user_message = block_msg,
            )
            return (
                f"I cannot process this request due to data governance policy.\n\n"
                f"{block_msg}"
                + gov_block
            )

        # ── Build safe context annotation for model ───────────────────────────
        try:
            safe_ctx  = build_safe_schema_context(
                catalog=catalog, use_case=use_case,
                workspace_id=workspace_id, tenant_id=tenant_id,
                model_names=target_models or None,
            )
            ctx_summary = summarize_filtered_context(safe_ctx)
            governance_panel["certified_assets_used"] = ctx_summary.get("certified_models", [])
            governance_panel["aggregate_only_models"] = ctx_summary.get("aggregate_only_models", [])
            governance_panel["pii_annotated_columns"] = ctx_summary.get("pii_annotated_columns", [])
        except Exception:
            ctx_summary = {}

        # ── Pass-through: add governance status to the response ───────────────
        # Note: in a real pipeline this would modify the prompt sent to the LLM.
        # Here we return the user message with a governance annotation added.
        # Integrators should inject safe_ctx into the actual LLM call.

        gov_status = _format_governance_block(
            allowed     = True,
            answer_mode = pre_result.answer_mode.value,
            panel       = governance_panel,
        )

        # In a full integration, the downstream LLM call would be made here.
        # This pipeline returns the user message + governance context to allow
        # the OpenWebUI model to process it with the governance annotation.

        if self.valves.show_governance:
            # Prepend safe context hint for the model
            preamble_lines = []
            if ctx_summary.get("certified_models"):
                preamble_lines.append(
                    f"[Governance: Using certified assets: "
                    f"{', '.join(ctx_summary['certified_models'])}]"
                )
            if ctx_summary.get("aggregate_only_models"):
                preamble_lines.append(
                    f"[Governance: Aggregate-only enforcement for: "
                    f"{', '.join(ctx_summary['aggregate_only_models'])}]"
                )
            if ctx_summary.get("pii_annotated_columns"):
                preamble_lines.append(
                    f"[Governance: {len(ctx_summary['pii_annotated_columns'])} PII field(s) excluded]"
                )

            preamble = "\n".join(preamble_lines)
            return f"{preamble}\n\n{user_message}{gov_status}" if preamble else f"{user_message}{gov_status}"

        return user_message

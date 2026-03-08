"""
guardrail.policy_compiler
=========================

Reads dbt manifest.json and compiles a structured AiPolicyCatalog for runtime
AI guardrail enforcement.

Usage:
    from guardrail.policy_compiler import PolicyCompiler

    compiler = PolicyCompiler("dbt-demo/target/manifest.json")
    catalog  = compiler.compile()

    model_policy  = catalog.get_model("customers")
    column_policy = catalog.get_column("customers", "email")

Design notes:
  - The compiler reads dbt-native fields (description, access, tags, group)
    PLUS Datap.ai custom fields inside model/column `meta` blocks.
  - Custom fields live under meta.datapai or meta.ai_* keys.
  - Missing fields fall back to safe conservative defaults (see metadata_schema.py).
  - The catalog is versioned by an MD5 hash of the manifest file.
  - Compiled catalogs are cached in-process; call refresh() to force rebuild.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .metadata_schema import (
    AgentActionPolicy,
    AiAccessLevel,
    AiPolicyCatalog,
    AnswerMode,
    ColumnAiPolicy,
    ExplanationPolicy,
    ExportPolicy,
    MaskingRule,
    ModelAiPolicy,
    PiiClass,
    RagExposure,
    RetrievalPolicy,
    RiskTier,
    SecurityClass,
    SensitivityLevel,
    SummarizationPolicy,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Enum coercers — safely parse string values from YAML/JSON meta
# ──────────────────────────────────────────────────────────────────────────────

def _ai_access(v: Any, default: AiAccessLevel = AiAccessLevel.DENY) -> AiAccessLevel:
    try:
        return AiAccessLevel(str(v).lower()) if v else default
    except ValueError:
        return default

def _sensitivity(v: Any, default: SensitivityLevel = SensitivityLevel.CONFIDENTIAL) -> SensitivityLevel:
    try:
        return SensitivityLevel(str(v).lower()) if v else default
    except ValueError:
        return default

def _answer_mode(v: Any, default: AnswerMode = AnswerMode.DENY) -> AnswerMode:
    try:
        return AnswerMode(str(v).lower()) if v else default
    except ValueError:
        return default

def _export_policy(v: Any, default: ExportPolicy = ExportPolicy.DENY) -> ExportPolicy:
    try:
        return ExportPolicy(str(v).lower()) if v else default
    except ValueError:
        return default

def _retrieval_policy(v: Any, default: RetrievalPolicy = RetrievalPolicy.DENY) -> RetrievalPolicy:
    try:
        return RetrievalPolicy(str(v).lower()) if v else default
    except ValueError:
        return default

def _summarization(v: Any, default: SummarizationPolicy = SummarizationPolicy.DENY) -> SummarizationPolicy:
    try:
        return SummarizationPolicy(str(v).lower()) if v else default
    except ValueError:
        return default

def _explanation(v: Any, default: ExplanationPolicy = ExplanationPolicy.SAFE_ONLY) -> ExplanationPolicy:
    try:
        return ExplanationPolicy(str(v).lower()) if v else default
    except ValueError:
        return default

def _rag_exposure(v: Any, default: RagExposure = RagExposure.DENY) -> RagExposure:
    try:
        return RagExposure(str(v).lower()) if v else default
    except ValueError:
        return default

def _agent_action(v: Any, default: AgentActionPolicy = AgentActionPolicy.DENY) -> AgentActionPolicy:
    try:
        return AgentActionPolicy(str(v).lower()) if v else default
    except ValueError:
        return default

def _risk_tier(v: Any, default: RiskTier = RiskTier.HIGH) -> RiskTier:
    try:
        return RiskTier(str(v).lower()) if v else default
    except ValueError:
        return default

def _pii_class(v: Any, default: PiiClass = PiiClass.NONE) -> PiiClass:
    try:
        return PiiClass(str(v).lower()) if v else default
    except ValueError:
        return default

def _security_class(v: Any, default: SecurityClass = SecurityClass.INTERNAL) -> SecurityClass:
    try:
        return SecurityClass(str(v).lower()) if v else default
    except ValueError:
        return default

def _masking_rule(v: Any, default: MaskingRule = MaskingRule.NONE) -> MaskingRule:
    try:
        return MaskingRule(str(v).lower()) if v else default
    except ValueError:
        return default

def _bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "yes", "1")

def _list(v: Any) -> List[str]:
    if not v:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]

def _str(v: Any, default: str = "") -> str:
    return str(v).strip() if v else default


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers — extract meta blocks
# ──────────────────────────────────────────────────────────────────────────────

def _model_meta(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the Datap.ai AI-governance meta for a model node.

    Supports both:
      meta.datapai.*  (namespaced block — preferred)
      meta.ai_*       (flat key convention — backwards compat)
    """
    raw_meta: Dict[str, Any] = node.get("meta") or {}

    # Prefer the namespaced block if present
    if "datapai" in raw_meta:
        dp = dict(raw_meta["datapai"])
    else:
        dp = {}

    # Also pick up any top-level ai_* keys as a fallback/override
    for k, v in raw_meta.items():
        if k.startswith("ai_") or k in (
            "certified_for_ai", "compliance_domain", "sensitivity_level",
            "contains_pii", "contains_phi", "default_answer_mode",
            "allowed_actions", "disallowed_actions", "owner_team",
            "business_owner", "steward", "risk_tier", "safe_description",
            "approved_personas", "approved_workspaces", "approved_tenants",
            "allowed_join_domains", "blocked_join_domains",
            "allowed_datasource_modes", "retention_class",
            "export_policy", "retrieval_policy", "summarization_policy",
            "explanation_policy", "rag_exposure", "agent_action_policy",
            "tool_use_policy", "approved_ai_use_cases", "blocked_ai_use_cases",
        ):
            dp.setdefault(k, v)

    return dp


def _col_meta(col_info: Dict[str, Any]) -> Dict[str, Any]:
    """Return the Datap.ai AI-governance meta for a column."""
    raw_meta: Dict[str, Any] = col_info.get("meta") or {}

    if "datapai" in raw_meta:
        dp = dict(raw_meta["datapai"])
    else:
        dp = {}

    for k, v in raw_meta.items():
        if k.startswith("ai_") or k in (
            "pii", "phi", "security_class", "masking_rule", "answer_mode",
            "allowed_in_output", "allowed_in_where", "allowed_in_group_by",
            "allowed_in_order_by", "allowed_in_retrieval", "allowed_in_summary",
            "allowed_in_explanation", "allowed_in_export",
            "semantic_aliases", "business_term", "join_sensitivity",
            "export_allowed", "notes_for_ai",
        ):
            dp.setdefault(k, v)

    return dp


def _pii_from_description(description: str) -> PiiClass:
    """
    Heuristic: detect PII class from column description text when explicit
    meta is absent.  Used only when column has no explicit pii meta key.
    """
    desc_lower = description.lower()
    direct_signals = ["pii", "personally identifiable", "email", "ssn", "passport",
                      "phone", "credit card", "social security", "date of birth"]
    indirect_signals = ["dob", "postcode", "zip", "ip address", "location"]

    for s in direct_signals:
        if s in desc_lower:
            return PiiClass.DIRECT
    for s in indirect_signals:
        if s in desc_lower:
            return PiiClass.INDIRECT
    return PiiClass.NONE


# ──────────────────────────────────────────────────────────────────────────────
# PolicyCompiler
# ──────────────────────────────────────────────────────────────────────────────

class PolicyCompiler:
    """
    Reads a dbt manifest.json and produces a runtime AiPolicyCatalog.

    Parameters
    ----------
    manifest_path : str
        Path to the dbt manifest.json.  Defaults to the dbt-demo target path.
    policy_version : str, optional
        Override the version tag.  Defaults to MD5 hash of the manifest file.
    """

    DEFAULT_MANIFEST = os.path.join("dbt-demo", "target", "manifest.json")

    def __init__(
        self,
        manifest_path: Optional[str] = None,
        policy_version: Optional[str] = None,
    ):
        self.manifest_path = manifest_path or self.DEFAULT_MANIFEST
        self._policy_version = policy_version
        self._catalog: Optional[AiPolicyCatalog] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def compile(self, force_refresh: bool = False) -> AiPolicyCatalog:
        """
        Compile (or return cached) the AiPolicyCatalog.

        Parameters
        ----------
        force_refresh : bool
            If True, reparse the manifest even if a cached catalog exists.
        """
        if self._catalog and not force_refresh:
            return self._catalog
        self._catalog = self._build_catalog()
        return self._catalog

    def refresh(self) -> AiPolicyCatalog:
        """Force a re-compile from the manifest file."""
        return self.compile(force_refresh=True)

    def get_policy_for_model(self, model_name: str) -> Optional[ModelAiPolicy]:
        return self.compile().get_model(model_name)

    def get_policy_for_column(
        self, model_name: str, column_name: str
    ) -> Optional[ColumnAiPolicy]:
        return self.compile().get_column(model_name, column_name)

    def get_allowed_assets_for_use_case(
        self,
        use_case:     str,
        workspace_id: str = "",
        tenant_id:    str = "",
    ) -> List[ModelAiPolicy]:
        return self.compile().eligible_models(use_case, workspace_id, tenant_id)

    def get_allowed_fields_for_use_case(
        self,
        model_name: str,
        use_case:   str = "",
    ) -> List[ColumnAiPolicy]:
        return self.compile().eligible_columns_for_model(model_name, use_case)

    def explain_policy_decision(
        self,
        model_name:  str,
        column_name: Optional[str] = None,
        use_case:    str = "",
    ) -> Dict[str, Any]:
        """
        Return a structured explanation of why a model/column is allowed,
        restricted, masked, or denied.
        """
        catalog = self.compile()

        model = catalog.get_model(model_name)
        if not model:
            return {
                "allowed": False,
                "reason":  f"Model '{model_name}' not found in policy catalog.",
                "model_name": model_name,
            }

        if column_name:
            col = catalog.get_column(model_name, column_name)
            return self._explain_column(model, col, column_name, use_case)
        else:
            return self._explain_model(model, use_case)

    def get_allowed_actions_for_asset(self, model_name: str) -> List[str]:
        model = self.get_policy_for_model(model_name)
        if not model or not model.is_ai_eligible():
            return []
        return [a for a in model.allowed_actions if a not in model.disallowed_actions]

    def catalog_summary(self) -> Dict[str, Any]:
        """Return a high-level summary of the compiled catalog."""
        catalog = self.compile()
        total_models  = len(catalog.models)
        ai_eligible   = sum(1 for m in catalog.models.values() if m.is_ai_eligible())
        certified     = sum(1 for m in catalog.models.values() if m.is_certified())
        total_cols    = len(catalog.columns)
        pii_cols      = sum(1 for c in catalog.columns.values() if c.is_pii_or_phi())
        output_safe   = sum(1 for c in catalog.columns.values() if c.is_output_safe())

        return {
            "catalog_version":      catalog.version,
            "compiled_at":          catalog.metadata.get("compiled_at"),
            "manifest_path":        self.manifest_path,
            "total_models":         total_models,
            "ai_eligible_models":   ai_eligible,
            "certified_models":     certified,
            "denied_models":        total_models - ai_eligible,
            "total_columns":        total_cols,
            "pii_or_phi_columns":   pii_cols,
            "output_safe_columns":  output_safe,
            "blocked_columns":      total_cols - output_safe,
        }

    # ── Internal build ────────────────────────────────────────────────────────

    def _build_catalog(self) -> AiPolicyCatalog:
        manifest  = self._load_manifest()
        version   = self._policy_version or self._manifest_hash()
        models:   Dict[str, ModelAiPolicy]  = {}
        columns:  Dict[str, ColumnAiPolicy] = {}

        nodes = dict(manifest.get("nodes", {}))
        # Also include sources
        for src_key, src_val in manifest.get("sources", {}).items():
            nodes[src_key] = src_val

        for node_id, node in nodes.items():
            rtype = node.get("resource_type", "")
            if rtype not in ("model", "source", "seed", "snapshot"):
                continue

            model_policy = self._extract_model_policy(node, version)
            models[model_policy.model_name] = model_policy

            for col_name, col_info in node.get("columns", {}).items():
                col_policy = self._extract_column_policy(
                    model_policy, col_name, col_info
                )
                columns[col_policy.key] = col_policy

        return AiPolicyCatalog(
            models=models,
            columns=columns,
            version=version,
            metadata={
                "manifest_path": self.manifest_path,
                "compiled_at":   datetime.now(timezone.utc).isoformat(),
                "node_count":    len(nodes),
            },
        )

    def _extract_model_policy(
        self, node: Dict[str, Any], version: str
    ) -> ModelAiPolicy:
        meta = _model_meta(node)

        name        = node.get("name", "")
        description = _str(node.get("description"))
        access      = _str(node.get("access"))
        tags        = _list(node.get("tags"))
        group       = _str(node.get("group"))
        schema      = _str(node.get("schema"))
        database    = _str(node.get("database"))
        rtype       = _str(node.get("resource_type"), "model")
        unique_id   = _str(node.get("unique_id"))

        # ai_enabled: explicit OR inferred from ai_access_level not being deny
        ai_enabled = _bool(meta.get("ai_enabled"), default=False)
        ai_access_raw = meta.get("ai_access_level", "")
        ai_access = _ai_access(ai_access_raw)

        # If ai_access_level is explicitly set and not deny, treat as ai_enabled
        if ai_access_raw and ai_access != AiAccessLevel.DENY:
            ai_enabled = True

        # dbt native: access=public → allow retrieval by default
        # access=private → hard deny
        if access == "private":
            ai_access = AiAccessLevel.DENY
            ai_enabled = False

        # Contains PII/PHI: infer from description if not explicit
        contains_pii = _bool(
            meta.get("contains_pii"),
            default="pii" in description.lower(),
        )
        contains_phi = _bool(
            meta.get("contains_phi"),
            default="phi" in description.lower() or "protected health" in description.lower(),
        )

        # Safe description: prefer explicit safe_description, else use dbt description
        safe_desc = _str(meta.get("safe_description")) or description

        # Determine default answer mode
        # If model is not AI-enabled → deny; if aggregate_only → aggregate_only; else use meta
        if not ai_enabled or ai_access == AiAccessLevel.DENY:
            default_answer_mode = AnswerMode.DENY
        else:
            default_answer_mode = _answer_mode(
                meta.get("default_answer_mode"),
                default=AnswerMode.FULL if not contains_pii else AnswerMode.MASKED,
            )

        # Per-use-case policies — defaults depend on sensitivity
        is_sensitive = contains_pii or contains_phi
        default_retrieval = RetrievalPolicy.DENY if not ai_enabled else (
            RetrievalPolicy.LIMITED if is_sensitive else RetrievalPolicy.ALLOW
        )
        default_summarization = SummarizationPolicy.DENY if not ai_enabled else (
            SummarizationPolicy.MASKED_ONLY if is_sensitive else SummarizationPolicy.ALLOW
        )

        return ModelAiPolicy(
            model_name        = name,
            unique_id         = unique_id,
            schema            = schema,
            database          = database,
            resource_type     = rtype,
            ai_enabled        = ai_enabled,
            certified_for_ai  = _bool(meta.get("certified_for_ai")),
            ai_access_level   = ai_access,
            compliance_domain = _str(meta.get("compliance_domain")),
            sensitivity_level = _sensitivity(meta.get("sensitivity_level")),
            contains_pii      = contains_pii,
            contains_phi      = contains_phi,
            risk_tier         = _risk_tier(meta.get("risk_tier")),
            default_answer_mode  = default_answer_mode,
            allowed_actions      = _list(meta.get("allowed_actions")),
            disallowed_actions   = _list(meta.get("disallowed_actions")),
            owner_team           = _str(meta.get("owner_team")),
            business_owner       = _str(meta.get("business_owner")),
            steward              = _str(meta.get("steward")),
            safe_description     = safe_desc,
            description          = description,
            approved_personas    = _list(meta.get("approved_personas")),
            approved_workspaces  = _list(meta.get("approved_workspaces")),
            approved_tenants     = _list(meta.get("approved_tenants")),
            allowed_join_domains = _list(meta.get("allowed_join_domains")),
            blocked_join_domains = _list(meta.get("blocked_join_domains")),
            allowed_datasource_modes = _list(meta.get("allowed_datasource_modes")),
            retention_class      = _str(meta.get("retention_class")),
            export_policy        = _export_policy(meta.get("export_policy")),
            retrieval_policy     = _retrieval_policy(
                meta.get("retrieval_policy"), default=default_retrieval
            ),
            summarization_policy = _summarization(
                meta.get("summarization_policy"), default=default_summarization
            ),
            explanation_policy   = _explanation(meta.get("explanation_policy")),
            rag_exposure         = _rag_exposure(meta.get("rag_exposure")),
            agent_action_policy  = _agent_action(meta.get("agent_action_policy")),
            tool_use_policy      = _str(meta.get("tool_use_policy"), default="deny"),
            approved_ai_use_cases = _list(meta.get("approved_ai_use_cases")),
            blocked_ai_use_cases  = _list(meta.get("blocked_ai_use_cases")),
            access               = access,
            tags                 = tags,
            group                = group,
            policy_version       = version,
        )

    def _extract_column_policy(
        self,
        model: ModelAiPolicy,
        col_name: str,
        col_info: Dict[str, Any],
    ) -> ColumnAiPolicy:
        meta = _col_meta(col_info)
        description = _str(col_info.get("description"))

        # If the model itself is deny, all columns are deny
        if not model.is_ai_eligible():
            return ColumnAiPolicy(
                model_name      = model.model_name,
                column_name     = col_name,
                ai_exposed      = False,
                ai_selectable   = False,
                allowed_in_output = False,
                answer_mode     = AnswerMode.DENY,
                description     = description,
            )

        # PII: explicit meta wins; fall back to description heuristic
        pii_raw = meta.get("pii")
        if pii_raw is not None:
            pii = _pii_class(pii_raw)
        else:
            pii = _pii_from_description(description)

        phi = _bool(meta.get("phi"))

        # Security class — default from model sensitivity
        sec_class_default = SecurityClass.CONFIDENTIAL
        if model.sensitivity_level == SensitivityLevel.PUBLIC:
            sec_class_default = SecurityClass.PUBLIC
        elif model.sensitivity_level == SensitivityLevel.INTERNAL:
            sec_class_default = SecurityClass.INTERNAL

        security_class = _security_class(meta.get("security_class"), default=sec_class_default)

        # ai_exposed: default to False (safe default) UNLESS model is approved
        # and column has no explicit restriction
        default_exposed = (
            model.ai_access_level == AiAccessLevel.APPROVED
            and pii == PiiClass.NONE
            and not phi
        )
        ai_exposed = _bool(meta.get("ai_exposed"), default=default_exposed)

        # ai_selectable: default to ai_exposed value
        ai_selectable = _bool(meta.get("ai_selectable"), default=ai_exposed)

        # Masking rule: if PII/PHI and no explicit rule, default to redact
        masking_default = MaskingRule.NONE
        if pii in (PiiClass.DIRECT, PiiClass.INDIRECT) or phi:
            masking_default = MaskingRule.REDACT
        masking_rule = _masking_rule(meta.get("masking_rule"), default=masking_default)

        # Answer mode for column
        if phi or pii == PiiClass.DIRECT:
            answer_mode_default = (
                AnswerMode.MASKED if masking_rule != MaskingRule.NONE else AnswerMode.DENY
            )
        elif not ai_exposed:
            answer_mode_default = AnswerMode.DENY
        else:
            answer_mode_default = AnswerMode.FULL
        answer_mode = _answer_mode(meta.get("answer_mode"), default=answer_mode_default)

        # allowed_in_output: default True only for non-sensitive exposed columns
        output_default = (
            ai_exposed and ai_selectable
            and pii == PiiClass.NONE
            and not phi
        )
        allowed_in_output = _bool(meta.get("allowed_in_output"), default=output_default)

        return ColumnAiPolicy(
            model_name        = model.model_name,
            column_name       = col_name,
            ai_exposed        = ai_exposed,
            ai_selectable     = ai_selectable,
            ai_filterable     = _bool(meta.get("ai_filterable"), default=True),
            ai_groupable      = _bool(meta.get("ai_groupable"), default=True),
            ai_sortable       = _bool(meta.get("ai_sortable"), default=True),
            pii               = pii,
            phi               = phi,
            security_class    = security_class,
            masking_rule      = masking_rule,
            answer_mode       = answer_mode,
            allowed_in_output      = allowed_in_output,
            allowed_in_where       = _bool(meta.get("allowed_in_where"),    default=True),
            allowed_in_group_by    = _bool(meta.get("allowed_in_group_by"), default=True),
            allowed_in_order_by    = _bool(meta.get("allowed_in_order_by"), default=True),
            allowed_in_retrieval   = _bool(meta.get("allowed_in_retrieval"), default=ai_exposed),
            allowed_in_summary     = _bool(
                meta.get("allowed_in_summary"), default=(ai_exposed and pii == PiiClass.NONE)
            ),
            allowed_in_explanation = _bool(meta.get("allowed_in_explanation"), default=True),
            allowed_in_export      = _bool(meta.get("allowed_in_export"), default=False),
            semantic_aliases  = _list(meta.get("semantic_aliases")),
            business_term     = _str(meta.get("business_term")),
            notes_for_ai      = _str(meta.get("notes_for_ai")),
            description       = description,
            join_sensitivity  = _str(meta.get("join_sensitivity"), default="medium"),
            export_allowed    = _bool(meta.get("export_allowed"), default=False),
        )

    # ── Manifest I/O ──────────────────────────────────────────────────────────

    def _load_manifest(self) -> Dict[str, Any]:
        if not os.path.exists(self.manifest_path):
            logger.warning(
                "dbt manifest not found at %s — returning empty catalog.",
                self.manifest_path,
            )
            return {}
        with open(self.manifest_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _manifest_hash(self) -> str:
        if not os.path.exists(self.manifest_path):
            return "no_manifest"
        with open(self.manifest_path, "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()[:12]

    # ── Explanation helpers ───────────────────────────────────────────────────

    def _explain_model(self, model: ModelAiPolicy, use_case: str) -> Dict[str, Any]:
        allowed  = model.is_ai_eligible()
        reasons  = []
        mode     = model.effective_answer_mode(use_case)

        if not model.ai_enabled:
            reasons.append("ai_enabled is false — model excluded from AI context")
        if model.ai_access_level == AiAccessLevel.DENY:
            reasons.append("ai_access_level is deny")
        if model.access == "private":
            reasons.append("dbt access is private — hard deny")
        if use_case and use_case in model.blocked_ai_use_cases:
            reasons.append(f"use case '{use_case}' is in blocked_ai_use_cases")
        if not reasons and allowed:
            reasons.append(f"Model is AI-eligible (access_level={model.ai_access_level.value})")

        return {
            "allowed":          allowed,
            "answer_mode":      mode.value,
            "certified":        model.is_certified(),
            "model_name":       model.model_name,
            "ai_access_level":  model.ai_access_level.value,
            "sensitivity":      model.sensitivity_level.value,
            "contains_pii":     model.contains_pii,
            "contains_phi":     model.contains_phi,
            "approved_workspaces": model.approved_workspaces,
            "reasons":          reasons,
            "retrieval_policy": model.retrieval_policy.value,
            "export_policy":    model.export_policy.value,
        }

    def _explain_column(
        self,
        model:       ModelAiPolicy,
        col:         Optional[ColumnAiPolicy],
        column_name: str,
        use_case:    str,
    ) -> Dict[str, Any]:
        if not col:
            return {
                "allowed":      False,
                "reason":       f"Column '{column_name}' not in policy catalog for model '{model.model_name}'",
                "model_name":   model.model_name,
                "column_name":  column_name,
            }

        output_safe = col.is_output_safe()
        mode        = col.effective_answer_mode()
        reasons     = []

        if not model.is_ai_eligible():
            reasons.append("Parent model is not AI-eligible")
        if col.phi:
            reasons.append("Column contains PHI — restricted by default")
        if col.pii != PiiClass.NONE:
            reasons.append(f"Column PII class: {col.pii.value}")
        if not col.ai_exposed:
            reasons.append("ai_exposed is false")
        if not col.allowed_in_output:
            reasons.append("allowed_in_output is false")
        if col.masking_rule != MaskingRule.NONE:
            reasons.append(f"Masking rule applied: {col.masking_rule.value}")
        if not reasons:
            reasons.append("Column is AI-output-safe")

        return {
            "allowed":            output_safe,
            "answer_mode":        mode.value,
            "model_name":         model.model_name,
            "column_name":        column_name,
            "pii":                col.pii.value,
            "phi":                col.phi,
            "masking_rule":       col.masking_rule.value,
            "security_class":     col.security_class.value,
            "allowed_in_output":  col.allowed_in_output,
            "allowed_in_where":   col.allowed_in_where,
            "allowed_in_group_by": col.allowed_in_group_by,
            "allowed_in_retrieval": col.allowed_in_retrieval,
            "allowed_in_summary": col.allowed_in_summary,
            "allowed_in_export":  col.allowed_in_export,
            "reasons":            reasons,
        }

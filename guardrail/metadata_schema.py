"""
guardrail.metadata_schema
=========================

Python dataclasses that represent the Datap.ai dbt AI governance metadata standard.

This schema covers both:
  - Model/table/asset-level AI metadata   (ModelAiPolicy)
  - Column/field-level AI metadata        (ColumnAiPolicy)

The compiled runtime representation is an AiPolicyCatalog that maps
  model_name  → ModelAiPolicy
  column_key  → ColumnAiPolicy  (key = "model_name.column_name")

Safe defaults
-------------
When dbt metadata is absent or incomplete, conservative defaults are applied:
  - models are NOT AI-eligible unless ai_enabled=true
  - columns are NOT AI-output-eligible unless ai_exposed=true
  - PII/PHI columns default to deny/masked
  - export and agent actions default to deny
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Enum definitions — model level
# ──────────────────────────────────────────────────────────────────────────────

class AiAccessLevel(str, Enum):
    """Coarse AI access level for a model/asset."""
    DENY       = "deny"        # Hard deny — never expose to AI
    INTERNAL   = "internal"    # Internal tools only; not public-facing
    RESTRICTED = "restricted"  # Conditional access (persona/workspace rules apply)
    APPROVED   = "approved"    # Explicitly AI-approved


class SensitivityLevel(str, Enum):
    PUBLIC       = "public"
    INTERNAL     = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED   = "restricted"


class AnswerMode(str, Enum):
    """How AI may answer/return data from this asset."""
    FULL           = "full"           # Full row-level output allowed
    MASKED         = "masked"         # Output but with sensitive fields masked
    AGGREGATE_ONLY = "aggregate_only" # Only aggregate/summary; no row-level
    DENY           = "deny"           # No answer at all
    METADATA_ONLY  = "metadata_only"  # Only schema/description, not actual data


class ExportPolicy(str, Enum):
    ALLOW            = "allow"
    DENY             = "deny"
    APPROVAL_REQUIRED = "approval_required"


class RetrievalPolicy(str, Enum):
    ALLOW   = "allow"
    LIMITED = "limited"
    DENY    = "deny"


class SummarizationPolicy(str, Enum):
    ALLOW          = "allow"
    MASKED_ONLY    = "masked_only"
    AGGREGATE_ONLY = "aggregate_only"
    DENY           = "deny"


class ExplanationPolicy(str, Enum):
    ALLOW     = "allow"
    SAFE_ONLY = "safe_only"
    DENY      = "deny"


class RagExposure(str, Enum):
    ALLOW         = "allow"
    METADATA_ONLY = "metadata_only"
    DENY          = "deny"


class AgentActionPolicy(str, Enum):
    ALLOW            = "allow"
    APPROVAL_REQUIRED = "approval_required"
    DENY             = "deny"


class RiskTier(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# ──────────────────────────────────────────────────────────────────────────────
# Enum definitions — column level
# ──────────────────────────────────────────────────────────────────────────────

class PiiClass(str, Enum):
    NONE             = "none"
    DIRECT           = "direct"           # Name, email, SSN, etc.
    INDIRECT         = "indirect"         # DOB, postcode, etc.
    QUASI_IDENTIFIER = "quasi_identifier"  # Combination risk


class SecurityClass(str, Enum):
    PUBLIC       = "public"
    INTERNAL     = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED   = "restricted"


class MaskingRule(str, Enum):
    NONE         = "none"
    REDACT       = "redact"        # Replace with [REDACTED]
    PARTIAL_MASK = "partial_mask"  # e.g., john.***@example.com
    HASH         = "hash"          # One-way hash
    TOKENISE     = "tokenise"      # Reversible tokenization


# ──────────────────────────────────────────────────────────────────────────────
# Model-level policy
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelAiPolicy:
    """
    Runtime AI policy for a single dbt model/asset.

    Populated by the PolicyCompiler from dbt manifest.json meta sections.
    All fields have conservative safe defaults.
    """

    # Identity
    model_name:    str = ""
    unique_id:     str = ""
    schema:        str = ""
    database:      str = ""
    resource_type: str = "model"  # model, source, seed, snapshot

    # Core eligibility
    ai_enabled:       bool = False   # If False, exclude entirely from AI context
    certified_for_ai: bool = False   # Data steward-certified for AI use

    # Access level (safe default = deny)
    ai_access_level: AiAccessLevel = AiAccessLevel.DENY

    # Compliance / sensitivity
    compliance_domain: str              = ""
    sensitivity_level: SensitivityLevel = SensitivityLevel.CONFIDENTIAL
    contains_pii:      bool             = False
    contains_phi:      bool             = False
    risk_tier:         RiskTier         = RiskTier.HIGH

    # Answer mode (safe default = deny)
    default_answer_mode: AnswerMode = AnswerMode.DENY

    # Allowed and disallowed actions
    allowed_actions:    List[str] = field(default_factory=list)
    disallowed_actions: List[str] = field(default_factory=list)

    # Ownership / stewardship
    owner_team:     str = ""
    business_owner: str = ""
    steward:        str = ""

    # Descriptions for AI context (safe, curated)
    safe_description: str = ""
    description:      str = ""  # Raw dbt description

    # Persona / workspace / tenant constraints
    approved_personas:   List[str] = field(default_factory=list)
    approved_workspaces: List[str] = field(default_factory=list)
    approved_tenants:    List[str] = field(default_factory=list)

    # Join domain constraints
    allowed_join_domains:  List[str] = field(default_factory=list)
    blocked_join_domains:  List[str] = field(default_factory=list)

    # Datasource modes allowed
    allowed_datasource_modes: List[str] = field(default_factory=list)

    # Retention
    retention_class: str = ""

    # Per-use-case policies
    export_policy:        ExportPolicy        = ExportPolicy.DENY
    retrieval_policy:     RetrievalPolicy     = RetrievalPolicy.DENY
    summarization_policy: SummarizationPolicy = SummarizationPolicy.DENY
    explanation_policy:   ExplanationPolicy   = ExplanationPolicy.SAFE_ONLY
    rag_exposure:         RagExposure         = RagExposure.DENY
    agent_action_policy:  AgentActionPolicy   = AgentActionPolicy.DENY
    tool_use_policy:      str                 = "deny"

    # Use-case allow/block lists
    approved_ai_use_cases: List[str] = field(default_factory=list)
    blocked_ai_use_cases:  List[str] = field(default_factory=list)

    # dbt-native governance
    access: str       = ""  # dbt access: public / protected / private
    tags:   List[str] = field(default_factory=list)
    group:  str       = ""

    # Policy version (set by compiler)
    policy_version: str = ""

    def is_ai_eligible(self) -> bool:
        """Return True only if this model may be used in any AI context."""
        return (
            self.ai_enabled
            and self.ai_access_level != AiAccessLevel.DENY
            and self.access != "private"
        )

    def is_certified(self) -> bool:
        return self.certified_for_ai and self.is_ai_eligible()

    def is_approved_for_workspace(self, workspace_id: str) -> bool:
        if not self.approved_workspaces:
            return True  # No restriction → allowed
        return workspace_id in self.approved_workspaces

    def is_approved_for_tenant(self, tenant_id: str) -> bool:
        if not self.approved_tenants:
            return True
        return tenant_id in self.approved_tenants

    def effective_answer_mode(self, use_case: str = "") -> AnswerMode:
        """
        Return effective answer mode, respecting use-case blocks.
        """
        if not self.is_ai_eligible():
            return AnswerMode.DENY
        if use_case and use_case in self.blocked_ai_use_cases:
            return AnswerMode.DENY
        return self.default_answer_mode

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name":          self.model_name,
            "unique_id":           self.unique_id,
            "ai_enabled":          self.ai_enabled,
            "certified_for_ai":    self.certified_for_ai,
            "ai_access_level":     self.ai_access_level.value,
            "sensitivity_level":   self.sensitivity_level.value,
            "contains_pii":        self.contains_pii,
            "contains_phi":        self.contains_phi,
            "risk_tier":           self.risk_tier.value,
            "default_answer_mode": self.default_answer_mode.value,
            "allowed_actions":     self.allowed_actions,
            "disallowed_actions":  self.disallowed_actions,
            "owner_team":          self.owner_team,
            "safe_description":    self.safe_description or self.description,
            "approved_workspaces": self.approved_workspaces,
            "approved_tenants":    self.approved_tenants,
            "export_policy":       self.export_policy.value,
            "retrieval_policy":    self.retrieval_policy.value,
            "summarization_policy": self.summarization_policy.value,
            "explanation_policy":  self.explanation_policy.value,
            "rag_exposure":        self.rag_exposure.value,
            "agent_action_policy": self.agent_action_policy.value,
            "approved_ai_use_cases": self.approved_ai_use_cases,
            "blocked_ai_use_cases":  self.blocked_ai_use_cases,
            "access":              self.access,
            "tags":                self.tags,
            "policy_version":      self.policy_version,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Column-level policy
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ColumnAiPolicy:
    """
    Runtime AI policy for a single column/field within a dbt model.

    key format: "<model_name>.<column_name>"
    """

    # Identity
    model_name:  str = ""
    column_name: str = ""

    # Core exposure flags (safe default = not exposed)
    ai_exposed:    bool = False  # May appear in any AI context
    ai_selectable: bool = False  # May appear in SELECT output
    ai_filterable: bool = True   # May be used in WHERE clause
    ai_groupable:  bool = True   # May be used in GROUP BY
    ai_sortable:   bool = True   # May be used in ORDER BY

    # PII / PHI
    pii: PiiClass = PiiClass.NONE
    phi: bool     = False

    # Security classification
    security_class: SecurityClass = SecurityClass.INTERNAL

    # Masking
    masking_rule: MaskingRule = MaskingRule.NONE

    # Per-usage answer mode
    answer_mode: AnswerMode = AnswerMode.DENY

    # Fine-grained usage permissions (safe defaults = deny output, allow filter/group)
    allowed_in_output:      bool = False
    allowed_in_where:       bool = True
    allowed_in_group_by:    bool = True
    allowed_in_order_by:    bool = True
    allowed_in_retrieval:   bool = False
    allowed_in_summary:     bool = False
    allowed_in_explanation: bool = True
    allowed_in_export:      bool = False

    # Semantic helpers
    semantic_aliases: List[str] = field(default_factory=list)
    business_term:    str       = ""
    notes_for_ai:     str       = ""
    description:      str       = ""

    # Join sensitivity
    join_sensitivity: str = "medium"  # low / medium / high

    # Export
    export_allowed: bool = False

    @property
    def key(self) -> str:
        return f"{self.model_name}.{self.column_name}"

    def is_output_safe(self) -> bool:
        """Return True if column may appear in AI response output."""
        if self.phi:
            return False
        if self.pii in (PiiClass.DIRECT, PiiClass.INDIRECT):
            # PII column is output-safe only when masking is applied AND output is explicitly allowed
            return self.masking_rule != MaskingRule.NONE and self.allowed_in_output
        return self.ai_exposed and self.ai_selectable and self.allowed_in_output

    def is_pii_or_phi(self) -> bool:
        return self.pii != PiiClass.NONE or self.phi

    def effective_answer_mode(self) -> AnswerMode:
        if self.phi or self.pii == PiiClass.DIRECT:
            if self.masking_rule == MaskingRule.NONE:
                return AnswerMode.DENY
        if not self.ai_exposed or not self.ai_selectable:
            return AnswerMode.DENY
        return self.answer_mode

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key":                  self.key,
            "model_name":           self.model_name,
            "column_name":          self.column_name,
            "ai_exposed":           self.ai_exposed,
            "ai_selectable":        self.ai_selectable,
            "ai_filterable":        self.ai_filterable,
            "ai_groupable":         self.ai_groupable,
            "pii":                  self.pii.value,
            "phi":                  self.phi,
            "security_class":       self.security_class.value,
            "masking_rule":         self.masking_rule.value,
            "answer_mode":          self.answer_mode.value,
            "allowed_in_output":    self.allowed_in_output,
            "allowed_in_where":     self.allowed_in_where,
            "allowed_in_group_by":  self.allowed_in_group_by,
            "allowed_in_retrieval": self.allowed_in_retrieval,
            "allowed_in_summary":   self.allowed_in_summary,
            "allowed_in_export":    self.allowed_in_export,
            "business_term":        self.business_term,
            "notes_for_ai":         self.notes_for_ai,
            "export_allowed":       self.export_allowed,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Compiled policy catalog
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AiPolicyCatalog:
    """
    The compiled AI policy catalog — the in-memory runtime representation
    produced by the PolicyCompiler.

    Holds:
      models  : model_name → ModelAiPolicy
      columns : "model_name.column_name" → ColumnAiPolicy
      version : opaque hash of the source manifest
      metadata: any extra catalog metadata (manifest path, compile time, etc.)
    """

    models:   Dict[str, ModelAiPolicy]  = field(default_factory=dict)
    columns:  Dict[str, ColumnAiPolicy] = field(default_factory=dict)
    version:  str                       = ""
    metadata: Dict[str, Any]            = field(default_factory=dict)

    # ── Lookup helpers ────────────────────────────────────────────────────────

    def get_model(self, model_name: str) -> Optional[ModelAiPolicy]:
        return self.models.get(model_name)

    def get_column(self, model_name: str, column_name: str) -> Optional[ColumnAiPolicy]:
        return self.columns.get(f"{model_name}.{column_name}")

    def get_columns_for_model(self, model_name: str) -> Dict[str, ColumnAiPolicy]:
        prefix = f"{model_name}."
        return {
            k.removeprefix(prefix): v
            for k, v in self.columns.items()
            if k.startswith(prefix)
        }

    # ── Eligibility helpers ───────────────────────────────────────────────────

    def eligible_models(
        self,
        use_case:     str = "",
        workspace_id: str = "",
        tenant_id:    str = "",
    ) -> List[ModelAiPolicy]:
        """Return all models eligible for AI use under the given context."""
        out = []
        for m in self.models.values():
            if not m.is_ai_eligible():
                continue
            if workspace_id and not m.is_approved_for_workspace(workspace_id):
                continue
            if tenant_id and not m.is_approved_for_tenant(tenant_id):
                continue
            if use_case and use_case in m.blocked_ai_use_cases:
                continue
            out.append(m)
        return out

    def eligible_columns_for_model(
        self,
        model_name: str,
        use_case:   str = "",
    ) -> List[ColumnAiPolicy]:
        """Return columns eligible for AI use within a specific model."""
        model = self.get_model(model_name)
        if not model or not model.is_ai_eligible():
            return []
        return [
            col for col in self.get_columns_for_model(model_name).values()
            if col.ai_exposed
        ]

    def output_safe_columns(self, model_name: str) -> List[ColumnAiPolicy]:
        """Return columns that are safe to include in AI-generated output."""
        return [
            col for col in self.get_columns_for_model(model_name).values()
            if col.is_output_safe()
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version":  self.version,
            "metadata": self.metadata,
            "models":   {k: v.to_dict() for k, v in self.models.items()},
            "columns":  {k: v.to_dict() for k, v in self.columns.items()},
        }

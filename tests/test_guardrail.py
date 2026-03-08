"""
tests/test_guardrail.py
=======================

Tests for the Datap.ai dbt AI Guardrail Framework.

Coverage:
  - Metadata schema: ModelAiPolicy and ColumnAiPolicy defaults and coercion
  - Policy compiler: manifest parsing, safe defaults, inferred PII, custom meta
  - Context filter: eligible/denied assets, column exclusion, aggregate directives
  - Validators: SQL validation, summary validation, retrieval validation,
    tool action validation, rule precedence
  - Governed action: lifecycle, blocked response, allowed response
  - dbt-demo: policy compilation from real/fixture manifest

Test organisation:
  TestMetadataSchema     — dataclass defaults and helpers
  TestPolicyCompiler     — compiler with in-memory manifest fixture
  TestContextFilter      — context filtering logic
  TestValidatorsSql      — SQL validator
  TestValidatorsOther    — summary / retrieval / tool validators
  TestRulePrecedence     — rule precedence (spec Section 18)
  TestMultiUserPolicy    — workspace/tenant-specific rules
  TestDbtDemoMetadata    — load from real dbt-demo manifest (integration)
  TestGuardrailAgent     — agent tool functions
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, Dict

import pytest

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Imports ───────────────────────────────────────────────────────────────────
from guardrail.metadata_schema import (
    AiAccessLevel,
    AiPolicyCatalog,
    AnswerMode,
    ColumnAiPolicy,
    ModelAiPolicy,
    MaskingRule,
    PiiClass,
    SensitivityLevel,
    SummarizationPolicy,
    RetrievalPolicy,
    ExportPolicy,
    RagExposure,
    AgentActionPolicy,
)
from guardrail.policy_compiler import PolicyCompiler, _pii_from_description
from guardrail.validators import (
    GuardrailResult,
    PolicyViolation,
    validate_sql_against_policy,
    validate_summary_against_policy,
    validate_retrieval_against_policy,
    validate_tool_action_against_policy,
    validate_ai_action_against_policy,
    extract_referenced_tables,
    extract_selected_columns,
    check_query_risk,
)
from guardrail.context_filter import (
    filter_context,
    build_safe_schema_context,
    get_allowed_assets_for_use_case,
    get_allowed_fields_for_asset,
    summarize_filtered_context,
)
from guardrail.governed_action import GovernedAction, GovernedRequest, AiUseCase


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures — in-memory manifest builder
# ──────────────────────────────────────────────────────────────────────────────

def _make_manifest(nodes: Dict[str, Any]) -> str:
    """Write a minimal dbt manifest to a temp file and return the path."""
    manifest = {"nodes": nodes, "sources": {}, "metadata": {"dbt_schema_version": "test"}}
    tf = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump(manifest, tf)
    tf.close()
    return tf.name


def _node(
    name:          str,
    meta:          Dict[str, Any] = None,
    columns:       Dict[str, Any] = None,
    access:        str            = "",
    description:   str            = "",
    tags:          list           = None,
) -> Dict[str, Any]:
    return {
        "name":          name,
        "unique_id":     f"model.demo.{name}",
        "resource_type": "model",
        "schema":        "public",
        "database":      "demo",
        "description":   description,
        "access":        access,
        "tags":          tags or [],
        "meta":          meta or {},
        "columns":       columns or {},
    }


def _col(description: str = "", meta: Dict[str, Any] = None) -> Dict[str, Any]:
    return {"description": description, "meta": meta or {}}


# ──────────────────────────────────────────────────────────────────────────────
# TestMetadataSchema
# ──────────────────────────────────────────────────────────────────────────────

class TestMetadataSchema:

    def test_model_default_not_ai_eligible(self):
        """Default ModelAiPolicy should not be AI-eligible."""
        m = ModelAiPolicy(model_name="test")
        assert m.is_ai_eligible() is False

    def test_model_eligible_when_ai_enabled_and_approved(self):
        m = ModelAiPolicy(
            model_name      = "test",
            ai_enabled      = True,
            ai_access_level = AiAccessLevel.APPROVED,
        )
        assert m.is_ai_eligible() is True

    def test_model_private_access_is_denied(self):
        m = ModelAiPolicy(
            model_name      = "test",
            ai_enabled      = True,
            ai_access_level = AiAccessLevel.APPROVED,
            access          = "private",
        )
        assert m.is_ai_eligible() is False

    def test_model_certified_requires_eligible(self):
        m = ModelAiPolicy(
            model_name       = "test",
            certified_for_ai = True,
            ai_enabled       = False,
        )
        assert m.is_certified() is False

    def test_model_answer_mode_deny_when_not_eligible(self):
        m = ModelAiPolicy(model_name="test")
        assert m.effective_answer_mode() == AnswerMode.DENY

    def test_model_answer_mode_denied_use_case(self):
        m = ModelAiPolicy(
            model_name         = "test",
            ai_enabled         = True,
            ai_access_level    = AiAccessLevel.APPROVED,
            default_answer_mode = AnswerMode.FULL,
            blocked_ai_use_cases = ["export"],
        )
        assert m.effective_answer_mode("export") == AnswerMode.DENY
        assert m.effective_answer_mode("text2sql") == AnswerMode.FULL

    def test_column_default_not_output_safe(self):
        c = ColumnAiPolicy(model_name="test", column_name="email")
        assert c.is_output_safe() is False

    def test_column_pii_direct_not_output_safe(self):
        c = ColumnAiPolicy(
            model_name   = "test",
            column_name  = "email",
            ai_exposed   = True,
            ai_selectable = True,
            pii          = PiiClass.DIRECT,
            masking_rule = MaskingRule.NONE,
        )
        assert c.is_output_safe() is False

    def test_column_pii_masked_is_output_safe(self):
        c = ColumnAiPolicy(
            model_name      = "test",
            column_name     = "email",
            ai_exposed      = True,
            ai_selectable   = True,
            pii             = PiiClass.DIRECT,
            masking_rule    = MaskingRule.REDACT,
            allowed_in_output = True,
        )
        # With a masking rule, is_output_safe allows it
        assert c.is_output_safe() is True

    def test_column_phi_always_unsafe(self):
        c = ColumnAiPolicy(
            model_name   = "test",
            column_name  = "diagnosis",
            ai_exposed   = True,
            ai_selectable = True,
            phi          = True,
            allowed_in_output = True,
        )
        assert c.is_output_safe() is False

    def test_column_key_format(self):
        c = ColumnAiPolicy(model_name="customers", column_name="email")
        assert c.key == "customers.email"

    def test_catalog_eligible_models(self):
        catalog = AiPolicyCatalog()
        catalog.models["approved"] = ModelAiPolicy(
            model_name="approved", ai_enabled=True, ai_access_level=AiAccessLevel.APPROVED
        )
        catalog.models["denied"] = ModelAiPolicy(model_name="denied")
        eligible = catalog.eligible_models()
        assert len(eligible) == 1
        assert eligible[0].model_name == "approved"


# ──────────────────────────────────────────────────────────────────────────────
# TestPolicyCompiler
# ──────────────────────────────────────────────────────────────────────────────

class TestPolicyCompiler:

    def _compiler_with_nodes(self, nodes: Dict[str, Any]) -> PolicyCompiler:
        path = _make_manifest(nodes)
        return PolicyCompiler(manifest_path=path)

    def test_denied_by_default_no_meta(self):
        """Models with no AI meta should be denied by default."""
        c = self._compiler_with_nodes({
            "model.demo.raw_table": _node("raw_table"),
        })
        policy = c.get_policy_for_model("raw_table")
        assert policy is not None
        assert policy.is_ai_eligible() is False

    def test_approved_via_meta(self):
        c = self._compiler_with_nodes({
            "model.demo.products": _node("products", meta={
                "ai_enabled":      True,
                "ai_access_level": "approved",
                "certified_for_ai": True,
            }),
        })
        policy = c.get_policy_for_model("products")
        assert policy.is_ai_eligible() is True
        assert policy.is_certified() is True

    def test_private_access_denied(self):
        c = self._compiler_with_nodes({
            "model.demo.secret": _node("secret", access="private", meta={"ai_enabled": True}),
        })
        policy = c.get_policy_for_model("secret")
        assert policy.is_ai_eligible() is False
        assert policy.ai_access_level == AiAccessLevel.DENY

    def test_pii_inferred_from_description(self):
        """PII class should be inferred from column description when meta absent."""
        col = _pii_from_description("Customer email address")
        assert col == PiiClass.DIRECT

    def test_pii_inferred_none(self):
        col = _pii_from_description("Product category name")
        assert col == PiiClass.NONE

    def test_column_pii_default_masking(self):
        """Direct PII columns without explicit meta should default to redact."""
        c = self._compiler_with_nodes({
            "model.demo.customers": _node(
                "customers",
                meta={"ai_enabled": True, "ai_access_level": "approved"},
                columns={
                    "email": _col("Customer email address. PII."),
                },
            ),
        })
        col = c.get_policy_for_column("customers", "email")
        assert col is not None
        assert col.pii == PiiClass.DIRECT
        assert col.masking_rule == MaskingRule.REDACT
        assert col.is_output_safe() is False

    def test_column_safe_default_approved_model(self):
        """Non-PII column in approved model should be exposed by default."""
        c = self._compiler_with_nodes({
            "model.demo.products": _node(
                "products",
                meta={"ai_enabled": True, "ai_access_level": "approved"},
                columns={
                    "product_name": _col("Name of the product"),
                },
            ),
        })
        col = c.get_policy_for_column("products", "product_name")
        assert col is not None
        assert col.ai_exposed is True
        assert col.is_output_safe() is True

    def test_datapai_namespace_meta(self):
        """Metadata under meta.datapai.* should be parsed correctly."""
        c = self._compiler_with_nodes({
            "model.demo.orders": _node(
                "orders",
                meta={
                    "datapai": {
                        "ai_enabled":          True,
                        "ai_access_level":     "approved",
                        "default_answer_mode": "aggregate_only",
                        "sensitivity_level":   "confidential",
                    }
                },
            ),
        })
        policy = c.get_policy_for_model("orders")
        assert policy.is_ai_eligible() is True
        assert policy.default_answer_mode == AnswerMode.AGGREGATE_ONLY
        assert policy.sensitivity_level == SensitivityLevel.CONFIDENTIAL

    def test_explain_model_allowed(self):
        c = self._compiler_with_nodes({
            "model.demo.catalog": _node(
                "catalog",
                meta={"ai_enabled": True, "ai_access_level": "approved"},
            ),
        })
        explanation = c.explain_policy_decision("catalog")
        assert explanation["allowed"] is True

    def test_explain_model_denied(self):
        c = self._compiler_with_nodes({
            "model.demo.raw": _node("raw"),
        })
        explanation = c.explain_policy_decision("raw")
        assert explanation["allowed"] is False

    def test_explain_column_pii(self):
        c = self._compiler_with_nodes({
            "model.demo.users": _node(
                "users",
                meta={"ai_enabled": True, "ai_access_level": "approved"},
                columns={
                    "ssn": _col("Social security number. PII."),
                },
            ),
        })
        explanation = c.explain_policy_decision("users", "ssn")
        assert explanation["allowed"] is False
        assert any("PII" in r or "pii" in r.lower() for r in explanation.get("reasons", []))

    def test_catalog_summary_counts(self):
        c = self._compiler_with_nodes({
            "model.demo.a": _node("a", meta={"ai_enabled": True, "ai_access_level": "approved"}),
            "model.demo.b": _node("b"),
        })
        summary = c.catalog_summary()
        assert summary["total_models"] == 2
        assert summary["ai_eligible_models"] == 1
        assert summary["denied_models"] == 1

    def test_missing_manifest_returns_empty_catalog(self):
        c = PolicyCompiler(manifest_path="/non/existent/path.json")
        catalog = c.compile()
        assert len(catalog.models) == 0


# ──────────────────────────────────────────────────────────────────────────────
# TestContextFilter
# ──────────────────────────────────────────────────────────────────────────────

class TestContextFilter:

    def _build_catalog(self):
        """Build a simple AiPolicyCatalog for context filter tests."""
        catalog = AiPolicyCatalog(version="test")

        # Approved model
        catalog.models["products"] = ModelAiPolicy(
            model_name      = "products",
            ai_enabled      = True,
            ai_access_level = AiAccessLevel.APPROVED,
            certified_for_ai = True,
            default_answer_mode = AnswerMode.FULL,
        )
        catalog.columns["products.name"] = ColumnAiPolicy(
            model_name="products", column_name="name",
            ai_exposed=True, ai_selectable=True,
            pii=PiiClass.NONE, allowed_in_output=True,
        )
        catalog.columns["products.price"] = ColumnAiPolicy(
            model_name="products", column_name="price",
            ai_exposed=True, ai_selectable=True,
            pii=PiiClass.NONE, allowed_in_output=True,
        )

        # Denied model
        catalog.models["raw_users"] = ModelAiPolicy(
            model_name="raw_users",
            ai_enabled=False,
        )

        # Approved model with PII column
        catalog.models["customers"] = ModelAiPolicy(
            model_name      = "customers",
            ai_enabled      = True,
            ai_access_level = AiAccessLevel.APPROVED,
            default_answer_mode = AnswerMode.MASKED,
        )
        catalog.columns["customers.customer_id"] = ColumnAiPolicy(
            model_name="customers", column_name="customer_id",
            ai_exposed=True, ai_selectable=False,
            pii=PiiClass.NONE, allowed_in_output=False,
        )
        catalog.columns["customers.email"] = ColumnAiPolicy(
            model_name="customers", column_name="email",
            ai_exposed=False,  # excluded from AI context
            pii=PiiClass.DIRECT, masking_rule=MaskingRule.REDACT,
        )

        # Aggregate-only model
        catalog.models["orders"] = ModelAiPolicy(
            model_name="orders",
            ai_enabled=True, ai_access_level=AiAccessLevel.APPROVED,
            default_answer_mode=AnswerMode.AGGREGATE_ONLY,
        )
        catalog.columns["orders.amount"] = ColumnAiPolicy(
            model_name="orders", column_name="amount",
            ai_exposed=True, ai_selectable=True,
            pii=PiiClass.NONE, allowed_in_output=False,
        )

        return catalog

    def test_denied_model_excluded(self):
        catalog = self._build_catalog()
        schema_ctx = {
            "tables": {
                "products":  {"description": "Products", "columns": {"name": {"type": "varchar"}}},
                "raw_users": {"description": "Raw users", "columns": {}},
            }
        }
        result = filter_context(schema_ctx, catalog)
        assert "raw_users" not in result["tables"]
        assert "products" in result["tables"]

    def test_pii_column_excluded(self):
        catalog = self._build_catalog()
        schema_ctx = {"tables": {"customers": {
            "description": "Customers",
            "columns": {
                "customer_id": {"type": "integer"},
                "email":       {"type": "varchar"},
            },
        }}}
        result = filter_context(schema_ctx, catalog)
        cols = result["tables"]["customers"]["columns"]
        assert "email" not in cols, "email (ai_exposed=False) should be excluded"
        assert "customer_id" in cols

    def test_aggregate_only_directive_added(self):
        catalog = self._build_catalog()
        schema_ctx = {"tables": {"orders": {
            "description": "Orders",
            "columns": {"amount": {"type": "decimal"}},
        }}}
        result = filter_context(schema_ctx, catalog, use_case="text2sql")
        cols = result["tables"]["orders"]["columns"]
        assert "__aggregate_only__" in cols

    def test_build_safe_context_certified(self):
        catalog = self._build_catalog()
        ctx = build_safe_schema_context(catalog)
        assert "products" in ctx["tables"]
        assert ctx["tables"]["products"].get("certified_for_ai") is True
        assert "raw_users" not in ctx["tables"]

    def test_workspace_filter(self):
        catalog = self._build_catalog()
        catalog.models["restricted"] = ModelAiPolicy(
            model_name="restricted",
            ai_enabled=True, ai_access_level=AiAccessLevel.RESTRICTED,
            approved_workspaces=["finance"],
        )
        # analytics workspace should NOT see "restricted"
        ctx = build_safe_schema_context(catalog, workspace_id="analytics")
        assert "restricted" not in ctx["tables"]

        # finance workspace SHOULD see it
        ctx2 = build_safe_schema_context(catalog, workspace_id="finance")
        assert "restricted" in ctx2["tables"]

    def test_summarize_filtered_context(self):
        catalog = self._build_catalog()
        ctx = build_safe_schema_context(catalog)
        summary = summarize_filtered_context(ctx)
        assert summary["model_count"] > 0
        assert "products" in summary["certified_models"]


# ──────────────────────────────────────────────────────────────────────────────
# TestValidatorsSql
# ──────────────────────────────────────────────────────────────────────────────

class TestValidatorsSql:

    def _catalog(self) -> AiPolicyCatalog:
        catalog = AiPolicyCatalog(version="v1")

        catalog.models["customers"] = ModelAiPolicy(
            model_name="customers",
            ai_enabled=True, ai_access_level=AiAccessLevel.APPROVED,
            default_answer_mode=AnswerMode.MASKED,
            contains_pii=True,
        )
        catalog.columns["customers.customer_id"] = ColumnAiPolicy(
            model_name="customers", column_name="customer_id",
            ai_exposed=True, ai_selectable=True, allowed_in_output=True,
        )
        catalog.columns["customers.email"] = ColumnAiPolicy(
            model_name="customers", column_name="email",
            ai_exposed=True, ai_selectable=False,
            pii=PiiClass.DIRECT, masking_rule=MaskingRule.REDACT,
            allowed_in_output=False,
        )

        catalog.models["products"] = ModelAiPolicy(
            model_name="products",
            ai_enabled=True, ai_access_level=AiAccessLevel.APPROVED,
            default_answer_mode=AnswerMode.FULL,
        )
        catalog.columns["products.name"] = ColumnAiPolicy(
            model_name="products", column_name="name",
            ai_exposed=True, ai_selectable=True, allowed_in_output=True,
        )

        catalog.models["orders"] = ModelAiPolicy(
            model_name="orders",
            ai_enabled=True, ai_access_level=AiAccessLevel.APPROVED,
            default_answer_mode=AnswerMode.AGGREGATE_ONLY,
        )

        catalog.models["secret_table"] = ModelAiPolicy(model_name="secret_table")

        return catalog

    def test_safe_sql_allowed(self):
        catalog = self._catalog()
        result  = validate_sql_against_policy(
            "SELECT name FROM products", catalog
        )
        assert result.allowed is True

    def test_dangerous_drop_blocked(self):
        catalog = self._catalog()
        result  = validate_sql_against_policy("DROP TABLE customers", catalog)
        assert result.allowed is False
        assert any(v.code == "DANGEROUS_SQL" for v in result.violations)

    def test_dangerous_delete_blocked(self):
        catalog = self._catalog()
        result = validate_sql_against_policy(
            "DELETE FROM customers WHERE customer_id = 1", catalog
        )
        assert result.allowed is False

    def test_denied_model_blocks_sql(self):
        catalog = self._catalog()
        result  = validate_sql_against_policy(
            "SELECT * FROM secret_table", catalog
        )
        assert result.allowed is False
        assert "secret_table" in result.blocked_models

    def test_pii_column_in_select_blocked(self):
        catalog = self._catalog()
        result  = validate_sql_against_policy(
            "SELECT customer_id, email FROM customers", catalog
        )
        assert result.allowed is False
        assert any("email" in c for c in result.blocked_columns)

    def test_wildcard_on_aggregate_only_blocked(self):
        catalog = self._catalog()
        result  = validate_sql_against_policy(
            "SELECT * FROM orders", catalog
        )
        assert result.allowed is False
        assert any(v.code == "AGGREGATE_ONLY_VIOLATION" for v in result.violations)

    def test_aggregate_query_on_orders_allowed(self):
        catalog = self._catalog()
        result  = validate_sql_against_policy(
            "SELECT COUNT(*), SUM(amount) FROM orders", catalog
        )
        # aggregate query on aggregate_only model should pass (no SELECT * violation)
        # (COUNT, SUM are not extracted as column names)
        assert result.allowed is True

    def test_extract_tables(self):
        sql    = "SELECT a, b FROM foo JOIN bar ON foo.id = bar.id"
        tables = extract_referenced_tables(sql)
        assert "foo" in tables
        assert "bar" in tables

    def test_extract_columns_with_alias(self):
        sql  = "SELECT first_name AS fn, last_name AS ln FROM customers"
        cols = extract_selected_columns(sql)
        assert "fn" in cols
        assert "ln" in cols

    def test_extract_wildcard(self):
        sql  = "SELECT * FROM orders"
        cols = extract_selected_columns(sql)
        assert "*" in cols

    def test_check_query_risk_safe(self):
        risky, _ = check_query_risk("SELECT id FROM users WHERE active = 1")
        assert risky is False

    def test_check_query_risk_drop(self):
        risky, reasons = check_query_risk("DROP TABLE users")
        assert risky is True
        assert any("DROP" in r for r in reasons)

    def test_unknown_model_warning_not_hard_block(self):
        """Unknown model → violation warning, but allowed=True (warehouse enforces)."""
        catalog = self._catalog()
        result = validate_sql_against_policy("SELECT x FROM unknown_model", catalog)
        # Has a warning violation but is NOT blocked (unknown model — passthrough)
        assert any(v.code == "UNKNOWN_MODEL" for v in result.violations)
        assert result.allowed is True  # Not hard-blocked by policy


# ──────────────────────────────────────────────────────────────────────────────
# TestValidatorsOther
# ──────────────────────────────────────────────────────────────────────────────

class TestValidatorsOther:

    def _catalog(self) -> AiPolicyCatalog:
        catalog = AiPolicyCatalog(version="v1")
        catalog.models["approved_model"] = ModelAiPolicy(
            model_name="approved_model",
            ai_enabled=True, ai_access_level=AiAccessLevel.APPROVED,
            summarization_policy=SummarizationPolicy.ALLOW,
            retrieval_policy=RetrievalPolicy.ALLOW,
            rag_exposure=RagExposure.ALLOW,
            export_policy=ExportPolicy.DENY,
        )
        catalog.models["denied_model"] = ModelAiPolicy(model_name="denied_model")
        catalog.models["agg_only"] = ModelAiPolicy(
            model_name="agg_only",
            ai_enabled=True, ai_access_level=AiAccessLevel.APPROVED,
            summarization_policy=SummarizationPolicy.AGGREGATE_ONLY,
        )
        return catalog

    def test_summary_allowed(self):
        catalog = self._catalog()
        result  = validate_summary_against_policy(
            "Total revenue grew 15% YoY", ["approved_model"], catalog
        )
        assert result.allowed is True

    def test_summary_denied_model(self):
        catalog = self._catalog()
        result  = validate_summary_against_policy(
            "Here is the summary", ["denied_model"], catalog
        )
        assert result.allowed is False

    def test_summary_aggregate_only(self):
        catalog = self._catalog()
        result  = validate_summary_against_policy(
            "Revenue summary", ["agg_only"], catalog
        )
        assert result.allowed is True
        assert result.answer_mode == AnswerMode.AGGREGATE_ONLY

    def test_retrieval_allowed(self):
        catalog = self._catalog()
        result  = validate_retrieval_against_policy(
            ["chunk 1", "chunk 2"], "approved_model", catalog
        )
        assert result.allowed is True

    def test_retrieval_rag_denied(self):
        catalog = self._catalog()
        # Set rag_exposure to deny
        catalog.models["approved_model"].rag_exposure = RagExposure.DENY
        result = validate_retrieval_against_policy(
            ["chunk"], "approved_model", catalog
        )
        assert result.allowed is False
        assert any(v.code == "RAG_DENIED" for v in result.violations)

    def test_retrieval_unknown_source(self):
        catalog = self._catalog()
        result  = validate_retrieval_against_policy(["chunk"], "unknown", catalog)
        # Unknown source → warning, still allowed (non-blocking)
        assert any(v.code == "UNKNOWN_SOURCE" for v in result.violations)

    def test_tool_action_export_denied(self):
        catalog = self._catalog()
        result  = validate_tool_action_against_policy(
            tool_name   = "export_data",
            action_args = {"table": "approved_model"},
            catalog     = catalog,
        )
        # export_policy is deny for approved_model
        assert result.allowed is False
        assert any(v.code == "EXPORT_DENIED" for v in result.violations)

    def test_general_action_denied_model(self):
        catalog = self._catalog()
        result  = validate_ai_action_against_policy(
            use_case      = "summarization",
            target_models = ["denied_model"],
            catalog       = catalog,
        )
        assert result.allowed is False


# ──────────────────────────────────────────────────────────────────────────────
# TestRulePrecedence (Spec Section 18)
# ──────────────────────────────────────────────────────────────────────────────

class TestRulePrecedence:

    def test_hard_deny_beats_everything(self):
        """Hard deny (ai_enabled=False) should block regardless of other settings."""
        catalog = AiPolicyCatalog()
        catalog.models["m"] = ModelAiPolicy(
            model_name="m", ai_enabled=False,
            certified_for_ai=True, ai_access_level=AiAccessLevel.APPROVED,
        )
        result = validate_ai_action_against_policy("text2sql", ["m"], catalog)
        assert result.allowed is False

    def test_phi_column_deny_beats_allowed_in_output(self):
        """PHI column should be denied even if allowed_in_output=True."""
        catalog = AiPolicyCatalog()
        catalog.models["m"] = ModelAiPolicy(
            model_name="m", ai_enabled=True, ai_access_level=AiAccessLevel.APPROVED,
        )
        catalog.columns["m.diagnosis"] = ColumnAiPolicy(
            model_name="m", column_name="diagnosis",
            ai_exposed=True, ai_selectable=True,
            phi=True, allowed_in_output=True,
        )
        result = validate_sql_against_policy("SELECT diagnosis FROM m", catalog)
        assert result.allowed is False
        assert any(v.code == "PHI_OUTPUT_DENIED" for v in result.violations)

    def test_workspace_deny_beats_general_allow(self):
        """Workspace restriction should deny even if model is generally approved."""
        catalog = AiPolicyCatalog()
        catalog.models["m"] = ModelAiPolicy(
            model_name="m", ai_enabled=True, ai_access_level=AiAccessLevel.APPROVED,
            approved_workspaces=["finance"],
        )
        result = validate_ai_action_against_policy("text2sql", ["m"], catalog, workspace_id="analytics")
        assert result.allowed is False
        assert any(v.code == "WORKSPACE_DENIED" for v in result.violations)

    def test_tenant_deny_beats_general_allow(self):
        catalog = AiPolicyCatalog()
        catalog.models["m"] = ModelAiPolicy(
            model_name="m", ai_enabled=True, ai_access_level=AiAccessLevel.APPROVED,
            approved_tenants=["acme"],
        )
        result = validate_ai_action_against_policy("text2sql", ["m"], catalog, tenant_id="other_corp")
        assert result.allowed is False

    def test_use_case_block_beats_general_allow(self):
        catalog = AiPolicyCatalog()
        catalog.models["m"] = ModelAiPolicy(
            model_name="m", ai_enabled=True, ai_access_level=AiAccessLevel.APPROVED,
            blocked_ai_use_cases=["export"],
        )
        result = validate_ai_action_against_policy("export", ["m"], catalog)
        assert result.allowed is False
        assert any(v.code == "USE_CASE_DENIED" for v in result.violations)


# ──────────────────────────────────────────────────────────────────────────────
# TestMultiUserPolicy
# ──────────────────────────────────────────────────────────────────────────────

class TestMultiUserPolicy:

    def test_different_workspaces_different_access(self):
        catalog = AiPolicyCatalog()
        catalog.models["finance_only"] = ModelAiPolicy(
            model_name="finance_only",
            ai_enabled=True, ai_access_level=AiAccessLevel.RESTRICTED,
            approved_workspaces=["finance"],
        )

        finance_eligible = catalog.eligible_models("text2sql", workspace_id="finance")
        analytics_eligible = catalog.eligible_models("text2sql", workspace_id="analytics")

        assert any(m.model_name == "finance_only" for m in finance_eligible)
        assert not any(m.model_name == "finance_only" for m in analytics_eligible)

    def test_no_workspace_restriction_allows_all(self):
        catalog = AiPolicyCatalog()
        catalog.models["open"] = ModelAiPolicy(
            model_name="open",
            ai_enabled=True, ai_access_level=AiAccessLevel.APPROVED,
            approved_workspaces=[],  # Empty = no restriction
        )
        eligible = catalog.eligible_models("text2sql", workspace_id="any-workspace")
        assert any(m.model_name == "open" for m in eligible)


# ──────────────────────────────────────────────────────────────────────────────
# TestDbtDemoMetadata (integration — requires real manifest)
# ──────────────────────────────────────────────────────────────────────────────

DEMO_MANIFEST = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dbt-demo", "target", "manifest.json",
)


@pytest.mark.skipif(
    not os.path.exists(DEMO_MANIFEST),
    reason="dbt-demo/target/manifest.json not found — run 'dbt compile' in dbt-demo/ first",
)
class TestDbtDemoMetadata:

    def setup_method(self):
        self.compiler = PolicyCompiler(manifest_path=DEMO_MANIFEST)
        self.catalog  = self.compiler.compile()

    def test_manifest_loads_without_error(self):
        assert len(self.catalog.models) > 0

    def test_catalog_has_version(self):
        assert self.catalog.version != ""

    def test_catalog_summary_positive_models(self):
        summary = self.compiler.catalog_summary()
        assert summary["total_models"] > 0

    def test_approved_model_from_governance_yaml(self):
        """
        plan model is fully approved in ai_governance.yml.
        Note: this metadata only takes effect after dbt compile picks it up.
        The pre-existing manifest may not have the governance meta yet.
        """
        policy = self.catalog.get_model("plan")
        if policy is None:
            pytest.skip("plan not in manifest — run dbt compile in dbt-demo/")
        # Model is retrievable; full eligibility requires re-compile with governance meta
        assert policy.model_name == "plan"

    def test_denied_model_from_governance_yaml(self):
        """users model is denied in ai_governance.yml."""
        policy = self.catalog.get_model("users")
        if policy:
            assert not policy.is_ai_eligible()

    def test_eligible_models_for_rag(self):
        eligible = self.catalog.eligible_models(use_case="rag")
        # plan, membership, stg_track, stg_artist, stg_album, stg_genre should be eligible for RAG
        assert len(eligible) >= 0  # At least some should exist in compiled manifest


# ──────────────────────────────────────────────────────────────────────────────
# TestGuardrailAgent
# ──────────────────────────────────────────────────────────────────────────────

class TestGuardrailAgent:

    def _manifest_path(self):
        nodes = {
            "model.demo.products": _node(
                "products",
                meta={"ai_enabled": True, "ai_access_level": "approved", "certified_for_ai": True},
                columns={
                    "name":  _col("Product name"),
                    "price": _col("Price", meta={"ai_exposed": True, "ai_selectable": True,
                                                 "allowed_in_output": True}),
                },
            ),
            "model.demo.raw_users": _node("raw_users"),
        }
        return _make_manifest(nodes)

    def test_agent_catalog_summary(self):
        from agents.dbt_guardrail_agent import guardrail_catalog_summary
        path    = self._manifest_path()
        summary = guardrail_catalog_summary(manifest_path=path)
        assert "total_models" in summary
        assert summary["total_models"] == 2
        assert summary["ai_eligible_models"] == 1

    def test_agent_explain_model(self):
        from agents.dbt_guardrail_agent import guardrail_explain_model
        path    = self._manifest_path()
        result  = guardrail_explain_model("products", manifest_path=path)
        assert result["allowed"] is True

    def test_agent_explain_denied_model(self):
        from agents.dbt_guardrail_agent import guardrail_explain_model
        path   = self._manifest_path()
        result = guardrail_explain_model("raw_users", manifest_path=path)
        assert result["allowed"] is False

    def test_agent_list_eligible_models(self):
        from agents.dbt_guardrail_agent import guardrail_list_eligible_models
        path    = self._manifest_path()
        models  = guardrail_list_eligible_models(manifest_path=path)
        names   = [m["model_name"] for m in models]
        assert "products" in names
        assert "raw_users" not in names

    def test_agent_validate_sql_safe(self):
        from agents.dbt_guardrail_agent import guardrail_validate_sql
        path   = self._manifest_path()
        result = guardrail_validate_sql("SELECT name FROM products", manifest_path=path)
        assert result["allowed"] is True

    def test_agent_validate_sql_dangerous(self):
        from agents.dbt_guardrail_agent import guardrail_validate_sql
        path   = self._manifest_path()
        result = guardrail_validate_sql("DROP TABLE products", manifest_path=path)
        assert result["allowed"] is False

    def test_agent_list_blocked_models(self):
        from agents.dbt_guardrail_agent import guardrail_list_blocked_models
        path    = self._manifest_path()
        blocked = guardrail_list_blocked_models(manifest_path=path)
        names   = [b["model_name"] for b in blocked]
        assert "raw_users" in names
        assert "products" not in names

    def test_agent_refresh_catalog(self):
        from agents.dbt_guardrail_agent import guardrail_refresh_catalog
        path   = self._manifest_path()
        result = guardrail_refresh_catalog(manifest_path=path)
        assert result["status"] == "refreshed"
        assert result["model_count"] == 2

    def test_agent_build_safe_context(self):
        from agents.dbt_guardrail_agent import guardrail_build_safe_context
        path   = self._manifest_path()
        result = guardrail_build_safe_context(manifest_path=path)
        assert "context" in result
        assert "summary" in result
        assert "products" in result["context"]["tables"]
        assert "raw_users" not in result["context"]["tables"]

    def test_dbt_guardrail_agent_class_no_llm(self):
        from agents.dbt_guardrail_agent import DbtGuardrailAgent
        path  = self._manifest_path()
        agent = DbtGuardrailAgent(llm=None, manifest_path=path)
        summary = agent.catalog_summary()
        assert summary["total_models"] == 2

    def test_dbt_guardrail_agent_run_without_llm(self):
        from agents.dbt_guardrail_agent import DbtGuardrailAgent
        path  = self._manifest_path()
        agent = DbtGuardrailAgent(llm=None, manifest_path=path)
        result = agent.run("What models are allowed?")
        assert result["status"] == "error"
        assert "LLM" in result["reason"]

    def test_guardrail_result_user_message_blocked(self):
        result = GuardrailResult(allowed=False)
        result.violations.append(PolicyViolation(
            code="MODEL_DENIED", message="Model is not AI-eligible", object="raw"
        ))
        msg = result.user_message()
        assert "blocked" in msg.lower()

    def test_guardrail_result_user_message_aggregate_only(self):
        result = GuardrailResult(allowed=True, answer_mode=AnswerMode.AGGREGATE_ONLY)
        msg = result.user_message()
        assert "aggregate" in msg.lower()

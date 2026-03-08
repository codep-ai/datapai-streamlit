"""
guardrail.streamlit_ui
======================

Streamlit UI components for the dbt AI Guardrail Framework.

Provides reusable widgets and panels that can be embedded in any Streamlit
page to show users the governance status of an AI response.

Key components:
  render_governance_panel()   — compact governance summary panel
  render_policy_explorer()    — interactive policy catalog explorer
  render_blocked_response()   — user-friendly blocked action UI
  render_catalog_admin()      — admin/debug catalog inspection page

Usage (in any Streamlit tab or page):
    from guardrail.streamlit_ui import render_governance_panel

    render_governance_panel(governance_panel_dict)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _st_import():
    """Lazy import of streamlit to keep module importable without st installed."""
    import streamlit as st
    return st


# ──────────────────────────────────────────────────────────────────────────────
# Governance panel — compact summary for every AI response
# ──────────────────────────────────────────────────────────────────────────────

def render_governance_panel(
    panel:      Dict[str, Any],
    expanded:   bool = False,
    title:      str  = "🛡️ Governance",
) -> None:
    """
    Render a compact governance summary panel.

    Parameters
    ----------
    panel : dict
        Governance panel dict from GovernedResponse.governance_panel.
        Expected keys: use_case, answer_mode, policy_version,
        certified_assets_used, included_models, aggregate_only_models,
        pii_annotated_columns, blocked_models, blocked_columns,
        violations, suggestion.
    expanded : bool
        Whether the expander is open by default.
    title : str
        Expander label.
    """
    st = _st_import()

    answer_mode = panel.get("answer_mode", "")
    allowed     = len(panel.get("violations", [])) == 0
    blocked_m   = panel.get("blocked_models", [])
    blocked_c   = panel.get("blocked_columns", [])
    certified   = panel.get("certified_assets_used", [])
    included    = panel.get("included_models", [])
    aggregate   = panel.get("aggregate_only_models", [])
    pii_cols    = panel.get("pii_annotated_columns", [])
    suggestion  = panel.get("suggestion", "")
    policy_ver  = panel.get("policy_version", "")
    use_case    = panel.get("use_case", "")

    # Determine icon and colour based on answer_mode
    mode_badge = {
        "full":           "✅ Full",
        "masked":         "🔒 Masked",
        "aggregate_only": "📊 Aggregate-only",
        "deny":           "🚫 Blocked",
        "metadata_only":  "📋 Metadata-only",
    }.get(answer_mode, answer_mode)

    with st.expander(title, expanded=expanded):

        # ── Status row ────────────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Answer Mode", mode_badge)
        with col2:
            st.metric("Models Used", len(included))
        with col3:
            st.metric("Policy Version", policy_ver[:8] if policy_ver else "—")

        # ── Certified assets ──────────────────────────────────────────────────
        if certified:
            st.success(
                f"✅ **Certified assets used:** {', '.join(certified)}"
            )

        # ── Aggregate-only annotation ─────────────────────────────────────────
        if aggregate:
            st.info(
                f"📊 **Aggregate-only enforcement:** "
                f"{', '.join(aggregate)} — row-level output is not permitted for these models."
            )

        # ── PII-annotated columns ─────────────────────────────────────────────
        if pii_cols:
            st.warning(
                f"🔒 **PII-annotated fields hidden from context:** "
                f"{len(pii_cols)} field(s) — "
                + ", ".join(pii_cols[:6])
                + (" …" if len(pii_cols) > 6 else "")
            )

        # ── Blocked assets ────────────────────────────────────────────────────
        if blocked_m:
            st.error(
                f"🚫 **Blocked models:** {', '.join(blocked_m)}"
            )
        if blocked_c:
            st.error(
                f"🚫 **Blocked columns:** {', '.join(blocked_c[:8])}"
                + (" …" if len(blocked_c) > 8 else "")
            )

        # ── Violations detail ─────────────────────────────────────────────────
        violations = panel.get("violations", [])
        if violations:
            with st.expander("📋 Policy violations detail", expanded=False):
                for v in violations:
                    code = v.get("code", "")
                    msg  = v.get("message", "")
                    st.markdown(f"- `{code}` — {msg}")

        # ── Suggestion ────────────────────────────────────────────────────────
        if suggestion:
            st.info(f"💡 **Suggestion:** {suggestion}")

        # ── All included models ───────────────────────────────────────────────
        if included:
            with st.expander("📦 Assets used in this request", expanded=False):
                for m in included:
                    icon = "✅" if m in certified else "📄"
                    agg  = " *(aggregate-only)*" if m in aggregate else ""
                    st.markdown(f"- {icon} **{m}**{agg}")

        # ── Use case + policy version (footer) ───────────────────────────────
        st.caption(
            f"Use case: `{use_case}` | "
            f"Policy: `{policy_ver[:12] if policy_ver else 'unknown'}`"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Blocked response UI
# ──────────────────────────────────────────────────────────────────────────────

def render_blocked_response(
    user_message: str,
    panel:        Dict[str, Any],
    suggestion:   str = "",
) -> None:
    """
    Render a user-friendly message when an AI action was blocked by policy.

    Parameters
    ----------
    user_message : str
        The human-readable block reason.
    panel : dict
        Governance panel for detail display.
    suggestion : str
        Optional override suggestion.
    """
    st = _st_import()

    st.error(f"🚫 **Request blocked by data governance policy**\n\n{user_message}")

    if suggestion or panel.get("suggestion"):
        msg = suggestion or panel.get("suggestion", "")
        st.info(f"💡 **Suggestion:** {msg}")

    alternatives = [
        "Use aggregate functions (COUNT, SUM, AVG) instead of row-level SELECT",
        "Request a certified business-facing asset instead of a raw table",
        "Ask for metadata or a schema description instead of the data itself",
        "Contact your data steward to request access to this asset",
    ]
    with st.expander("💡 Safer alternatives", expanded=True):
        for alt in alternatives:
            st.markdown(f"- {alt}")

    render_governance_panel(panel, expanded=True, title="📋 Policy Details")


# ──────────────────────────────────────────────────────────────────────────────
# Policy catalog explorer — admin/debug page
# ──────────────────────────────────────────────────────────────────────────────

def render_catalog_admin(manifest_path: Optional[str] = None) -> None:
    """
    Render a full policy catalog admin/inspection page in Streamlit.

    Includes:
    - Catalog summary metrics
    - AI-eligible models table
    - Blocked/denied models table
    - Restricted columns table
    - Model detail expander (with per-column policy)
    - Live SQL validator

    Call this function inside a Streamlit tab or page.
    """
    st = _st_import()

    st.header("🛡️ dbt AI Guardrail — Policy Catalog")

    try:
        from guardrail.policy_compiler import PolicyCompiler
        from guardrail.validators import validate_sql_against_policy
    except ImportError:
        st.error("guardrail module not found. Ensure guardrail/ is in your Python path.")
        return

    # ── Compiler + catalog ────────────────────────────────────────────────────
    path     = manifest_path or "dbt-demo/target/manifest.json"
    compiler = PolicyCompiler(manifest_path=path)

    with st.spinner("Compiling AI policy catalog from dbt manifest…"):
        summary = compiler.catalog_summary()
        catalog = compiler.compile()

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.subheader("📊 Catalog Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Models",       summary.get("total_models", 0))
    c2.metric("AI-Eligible",        summary.get("ai_eligible_models", 0))
    c3.metric("Certified",          summary.get("certified_models", 0))
    c4.metric("Denied",             summary.get("denied_models", 0))
    c5.metric("PII/PHI Columns",    summary.get("pii_or_phi_columns", 0))
    c6.metric("Output-Safe Cols",   summary.get("output_safe_columns", 0))

    st.caption(
        f"Policy version: `{summary.get('catalog_version', 'n/a')}` | "
        f"Compiled: {summary.get('compiled_at', 'n/a')} | "
        f"Manifest: `{path}`"
    )

    if st.button("🔄 Refresh Catalog"):
        catalog = compiler.refresh()
        st.success("Catalog refreshed.")
        st.rerun()

    st.divider()

    # ── Filter controls ───────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    use_case_filter = col_a.selectbox(
        "Filter by use case",
        options=["(all)", "text2sql", "rag", "summarization",
                 "bi_metric_explanation", "export", "dbt_model_explanation"],
    )
    filter_uc = "" if use_case_filter == "(all)" else use_case_filter

    tab_eligible, tab_blocked, tab_columns, tab_validate = st.tabs([
        "✅ AI-Eligible Models",
        "🚫 Blocked / Denied Models",
        "🔒 Restricted Columns",
        "🧪 SQL Validator",
    ])

    # ── AI-eligible models ────────────────────────────────────────────────────
    with tab_eligible:
        eligible = catalog.eligible_models(use_case=filter_uc)
        st.markdown(f"**{len(eligible)} model(s) AI-eligible** for use case `{filter_uc or 'all'}`")

        for m in sorted(eligible, key=lambda x: x.model_name):
            cert_icon = "✅" if m.is_certified() else "📄"
            with st.expander(
                f"{cert_icon} **{m.model_name}** — {m.ai_access_level.value} | "
                f"{m.default_answer_mode.value} | {m.sensitivity_level.value}",
                expanded=False,
            ):
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"**Access level:** `{m.ai_access_level.value}`")
                col2.markdown(f"**Answer mode:** `{m.default_answer_mode.value}`")
                col3.markdown(f"**Risk tier:** `{m.risk_tier.value}`")

                if m.safe_description:
                    st.info(m.safe_description.strip())

                st.markdown(f"**Approved use cases:** {', '.join(m.approved_ai_use_cases) or 'all'}")
                if m.blocked_ai_use_cases:
                    st.warning(f"**Blocked use cases:** {', '.join(m.blocked_ai_use_cases)}")

                # Column breakdown
                cols_policy = catalog.get_columns_for_model(m.model_name)
                if cols_policy:
                    st.markdown("**Columns:**")
                    col_rows = []
                    for col_name, col in sorted(cols_policy.items()):
                        col_rows.append({
                            "Column":          col_name,
                            "Exposed":         "✅" if col.ai_exposed else "❌",
                            "Output-safe":     "✅" if col.is_output_safe() else "❌",
                            "PII":             col.pii.value,
                            "Masking":         col.masking_rule.value,
                            "Answer mode":     col.answer_mode.value,
                            "Business term":   col.business_term or "—",
                        })
                    st.dataframe(col_rows, use_container_width=True)

    # ── Blocked/denied models ─────────────────────────────────────────────────
    with tab_blocked:
        blocked = [m for m in catalog.models.values() if not m.is_ai_eligible()]
        st.markdown(f"**{len(blocked)} model(s) denied / not AI-eligible**")

        for m in sorted(blocked, key=lambda x: x.model_name):
            reason = (
                "access=private" if m.access == "private" else
                "ai_enabled=false" if not m.ai_enabled else
                f"ai_access_level={m.ai_access_level.value}"
            )
            with st.expander(f"🚫 **{m.model_name}** — {reason}", expanded=False):
                st.markdown(f"- **ai_enabled:** `{m.ai_enabled}`")
                st.markdown(f"- **ai_access_level:** `{m.ai_access_level.value}`")
                st.markdown(f"- **dbt access:** `{m.access or 'not set'}`")
                st.markdown(f"- **sensitivity:** `{m.sensitivity_level.value}`")
                st.markdown(f"- **risk_tier:** `{m.risk_tier.value}`")
                if m.safe_description:
                    st.caption(m.safe_description.strip())

    # ── Restricted columns ────────────────────────────────────────────────────
    with tab_columns:
        restricted = [
            c for c in catalog.columns.values()
            if c.is_pii_or_phi() or not c.is_output_safe()
        ]
        st.markdown(f"**{len(restricted)} restricted column(s)** (PII, PHI, or not output-safe)")

        if restricted:
            rows = [
                {
                    "Key":         c.key,
                    "Model":       c.model_name,
                    "Column":      c.column_name,
                    "PII":         c.pii.value,
                    "PHI":         "✅" if c.phi else "",
                    "Masking":     c.masking_rule.value,
                    "Answer mode": c.answer_mode.value,
                    "In output":   "❌" if not c.allowed_in_output else "✅",
                }
                for c in sorted(restricted, key=lambda x: x.key)
            ]
            st.dataframe(rows, use_container_width=True)

    # ── SQL validator ─────────────────────────────────────────────────────────
    with tab_validate:
        st.markdown("**Paste an AI-generated SQL query to validate against the policy catalog:**")
        sql_input = st.text_area(
            "SQL query",
            value="SELECT customer_id, first_name, email FROM customers LIMIT 10",
            height=150,
        )
        workspace_input = st.text_input("Workspace ID (optional)", value="")
        tenant_input    = st.text_input("Tenant ID (optional)", value="")

        if st.button("🛡️ Validate SQL"):
            result = validate_sql_against_policy(
                sql          = sql_input,
                catalog      = catalog,
                use_case     = "text2sql",
                workspace_id = workspace_input,
                tenant_id    = tenant_input,
            )
            if result.allowed:
                st.success(f"✅ SQL is allowed (answer_mode: {result.answer_mode.value})")
            else:
                st.error("🚫 SQL is blocked by policy")
                for v in result.violations:
                    st.markdown(f"- `{v.code}`: {v.message}")
                if result.blocked_models:
                    st.warning(f"Blocked models: {', '.join(result.blocked_models)}")
                if result.blocked_columns:
                    st.warning(f"Blocked columns: {', '.join(result.blocked_columns)}")
                if result.suggestion:
                    st.info(f"💡 {result.suggestion}")


# ──────────────────────────────────────────────────────────────────────────────
# Standalone Streamlit governance tab page
# ──────────────────────────────────────────────────────────────────────────────

def render_governance_tab(manifest_path: Optional[str] = None) -> None:
    """
    Render a full governance tab page, suitable for embedding in app_ai_agent.py.

    Usage:
        with tab_governance:
            from guardrail.streamlit_ui import render_governance_tab
            render_governance_tab()
    """
    render_catalog_admin(manifest_path=manifest_path)

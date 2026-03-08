# Datap.ai dbt AI Guardrail Framework

## Overview

The Datap.ai dbt AI Guardrail Framework is a runtime AI governance system where:

- **The data warehouse / lakehouse** remains the hard enforcement layer (row-level security, column masking, access controls)
- **dbt metadata** is the policy authoring layer (who/what/how AI may access governed data)
- **Datap.ai** is the runtime policy compiler and guardrail executor for any AI use case

This framework applies to **all AI capabilities** — not just Text2SQL. It covers RAG retrieval, summarization, explanation, narrative insight generation, document extraction, agent tool calls, workflow triggering, and any other AI capability that may interact with governed data.

---

## Architecture

```
dbt manifest.json
     │
     ▼
PolicyCompiler ──► AiPolicyCatalog
     │                    │
     │          ┌─────────┴──────────┐
     │          ▼                    ▼
     │    ModelAiPolicy        ColumnAiPolicy
     │    (per table)          (per column)
     │
     ▼
GovernedAction (13-step lifecycle)
     │
     ├── filter_context()     ← pre-generation context filtering
     │
     ├── invoke_fn()          ← your AI model/tool
     │
     └── validators           ← post-generation validation
           ├── validate_sql_against_policy()
           ├── validate_summary_against_policy()
           ├── validate_retrieval_against_policy()
           └── validate_tool_action_against_policy()
```

---

## dbt Metadata Standard

### Model-Level Metadata (`meta.datapai.*`)

Add these keys under `models[].meta.datapai` in your dbt YAML files:

| Key | Type | Values | Default | Description |
|-----|------|---------|---------|-------------|
| `ai_enabled` | bool | true/false | **false** | Model may be used in any AI context |
| `certified_for_ai` | bool | true/false | false | Data steward certified for AI use |
| `ai_access_level` | string | deny/internal/restricted/approved | **deny** | Coarse AI access tier |
| `compliance_domain` | string | any | "" | Governance domain (e.g. crm, finance, hr) |
| `sensitivity_level` | string | public/internal/confidential/restricted | confidential | Data sensitivity |
| `contains_pii` | bool | | false | Model contains PII |
| `contains_phi` | bool | | false | Model contains PHI |
| `default_answer_mode` | string | full/masked/aggregate_only/deny/metadata_only | **deny** | How AI may return data |
| `risk_tier` | string | low/medium/high/critical | high | Overall risk classification |
| `owner_team` | string | | "" | Owning team |
| `safe_description` | string | | "" | Curated description for AI context (preferred over raw description) |
| `export_policy` | string | allow/deny/approval_required | **deny** | Export/download policy |
| `retrieval_policy` | string | allow/limited/deny | **deny** | Retrieval/lookup policy |
| `summarization_policy` | string | allow/masked_only/aggregate_only/deny | **deny** | Summarization policy |
| `explanation_policy` | string | allow/safe_only/deny | safe_only | Explanation policy |
| `rag_exposure` | string | allow/metadata_only/deny | **deny** | RAG/vector retrieval policy |
| `agent_action_policy` | string | allow/approval_required/deny | **deny** | Agent tool action policy |
| `allowed_actions` | list | | [] | Explicitly allowed AI actions |
| `disallowed_actions` | list | | [] | Explicitly disallowed AI actions |
| `approved_ai_use_cases` | list | | [] | Use cases allowed (empty = all allowed) |
| `blocked_ai_use_cases` | list | | [] | Use cases explicitly blocked |
| `approved_workspaces` | list | | [] | Allowed workspace IDs (empty = unrestricted) |
| `approved_tenants` | list | | [] | Allowed tenant IDs (empty = unrestricted) |

### Column-Level Metadata (`columns[].meta.datapai.*`)

| Key | Type | Values | Default | Description |
|-----|------|---------|---------|-------------|
| `ai_exposed` | bool | | **false** | Column may appear in any AI context |
| `ai_selectable` | bool | | false | Column may appear in SELECT/output |
| `ai_filterable` | bool | | true | Column may be used in WHERE |
| `ai_groupable` | bool | | true | Column may be used in GROUP BY |
| `pii` | string | none/direct/indirect/quasi_identifier | none | PII classification |
| `phi` | bool | | false | Protected Health Information |
| `security_class` | string | public/internal/confidential/restricted | internal | Security classification |
| `masking_rule` | string | none/redact/partial_mask/hash/tokenise | none | Masking to apply in output |
| `answer_mode` | string | full/masked/aggregate_only/deny | deny | Output mode for this column |
| `allowed_in_output` | bool | | **false** | May appear in AI-generated output |
| `allowed_in_where` | bool | | true | May be used in WHERE clause |
| `allowed_in_group_by` | bool | | true | May be used in GROUP BY |
| `allowed_in_retrieval` | bool | | false | May appear in RAG retrieval results |
| `allowed_in_summary` | bool | | false | May appear in generated summaries |
| `allowed_in_export` | bool | | false | May be exported/downloaded |
| `business_term` | string | | "" | Business-friendly name for AI context |
| `semantic_aliases` | list | | [] | Alternative names recognized by AI |
| `notes_for_ai` | string | | "" | Guidance note embedded in AI context |

### Example dbt YAML

```yaml
models:
  - name: customers
    meta:
      datapai:
        ai_enabled:             true
        certified_for_ai:       true
        ai_access_level:        approved
        compliance_domain:      crm
        sensitivity_level:      confidential
        contains_pii:           true
        default_answer_mode:    masked
        retrieval_policy:       limited
        summarization_policy:   masked_only
        rag_exposure:           metadata_only
        export_policy:          deny
        approved_ai_use_cases:
          - text2sql
          - bi_metric_explanation
        blocked_ai_use_cases:
          - export
        safe_description: >
          Customer dimension table. Contains PII fields that are masked
          in AI output. Aggregate analytics only.
    columns:
      - name: email
        meta:
          datapai:
            ai_exposed:        true
            ai_selectable:     false
            pii:               direct
            masking_rule:      redact
            answer_mode:       masked
            allowed_in_output: false
            allowed_in_where:  true
            notes_for_ai:      Direct PII — never expose in output.
```

---

## Safe Defaults

When dbt metadata is absent or incomplete, the framework applies conservative defaults:

- Models are **NOT AI-eligible** unless `ai_enabled=true`
- Columns are **NOT output-eligible** unless `ai_exposed=true` AND `allowed_in_output=true`
- PII fields default to `masking_rule=redact` and `allowed_in_output=false`
- PHI fields are hard-denied in output regardless of other settings
- Export and agent actions default to `deny`
- RAG exposure defaults to `deny`
- dbt `access=private` → hard deny regardless of any AI meta

---

## Rule Precedence (Section 18 of spec)

When rules conflict, the framework applies them in this order:

1. **Hard deny** — `ai_enabled=false`, `ai_access_level=deny`, `access=private`
2. **Hard deny** — disallowed action or blocked use case
3. **Hard deny** — PHI output; direct PII output without masking
4. **Aggregate-only** enforcement — `default_answer_mode=aggregate_only`
5. **Masked-only** enforcement — `default_answer_mode=masked`
6. **Metadata-only** enforcement — `rag_exposure=metadata_only`
7. **Workspace/tenant/persona-specific** allow rules
8. **General AI-certified allow**

---

## Runtime Modules

### `guardrail.policy_compiler.PolicyCompiler`

Reads `dbt manifest.json` and produces a runtime `AiPolicyCatalog`.

```python
from guardrail.policy_compiler import PolicyCompiler

compiler = PolicyCompiler("dbt-demo/target/manifest.json")
catalog  = compiler.compile()

# Explain a policy decision
compiler.explain_policy_decision("customers")
compiler.explain_policy_decision("customers", "email")

# List eligible models
compiler.get_allowed_assets_for_use_case("text2sql", workspace_id="analytics")

# Validate SQL
from guardrail.validators import validate_sql_against_policy
result = validate_sql_against_policy(sql, catalog)
```

### `guardrail.context_filter`

Filter schema context before any AI call:

```python
from guardrail.context_filter import filter_context, build_safe_schema_context

# Filter a raw schema context dict
safe_ctx = filter_context(
    schema_context = raw_schema,
    catalog        = catalog,
    use_case       = "text2sql",
    workspace_id   = "analytics",
    tenant_id      = "acme",
)

# Or build from catalog directly
safe_ctx = build_safe_schema_context(catalog, use_case="rag", workspace_id="analytics")
```

### `guardrail.validators`

Validate AI-generated outputs:

```python
from guardrail.validators import (
    validate_sql_against_policy,
    validate_summary_against_policy,
    validate_retrieval_against_policy,
    validate_tool_action_against_policy,
    validate_ai_action_against_policy,
)

result = validate_sql_against_policy(sql, catalog)
if not result.allowed:
    print(result.user_message())
    # result.violations, result.blocked_models, result.blocked_columns
```

### `guardrail.governed_action.GovernedAction`

Full 13-step governed AI action lifecycle:

```python
from guardrail.governed_action import GovernedAction, GovernedRequest, AiUseCase

ga       = GovernedAction(catalog=catalog, ledger=trace_ledger)
request  = GovernedRequest(
    use_case    = AiUseCase.TEXT2SQL,
    request     = "Show me total orders by country",
    identity    = {"tenant_id": "acme", "workspace_id": "analytics",
                   "user_id": "u1", "session_id": "s1"},
)
response = ga.run(request, invoke_fn=my_sql_generator)

if response.allowed:
    print(response.result)
    print(response.governance_panel)
else:
    print(response.user_message)
```

---

## dbt AI Guardrail Agent

A dedicated LLM agent that explains and inspects dbt AI governance:

```python
from agents.dbt_guardrail_agent import DbtGuardrailAgent
from llm_client import RouterChatClient

agent = DbtGuardrailAgent(llm=RouterChatClient())
result = agent.run(
    "Which models are AI-eligible for text2sql in the analytics workspace?"
)

# Without LLM — deterministic tool calls only
agent = DbtGuardrailAgent(llm=None)
summary     = agent.catalog_summary()
explanation = agent.explain("customers", "email")
validation  = agent.validate_sql("SELECT email FROM customers")
safe_ctx    = agent.build_safe_context(use_case="text2sql")
```

Available `@tool` functions registered globally:
- `guardrail_catalog_summary`
- `guardrail_explain_model` / `guardrail_explain_column`
- `guardrail_list_eligible_models` / `guardrail_list_eligible_columns`
- `guardrail_validate_sql` / `guardrail_validate_action`
- `guardrail_build_safe_context`
- `guardrail_refresh_catalog`
- `guardrail_list_blocked_models` / `guardrail_list_restricted_columns`

---

## Streamlit UI

```python
from guardrail.streamlit_ui import render_governance_panel, render_blocked_response

# Compact governance panel (embed in any tab)
render_governance_panel(response.governance_panel, expanded=False)

# Full blocked response UI
render_blocked_response(response.user_message, response.governance_panel)

# Full admin/debug page
from guardrail.streamlit_ui import render_governance_tab
with tab_governance:
    render_governance_tab()
```

---

## OpenWebUI Middleware

Upload `agents/openwebui_guardrail_pipeline.py` as an OpenWebUI pipeline.

Environment variables:
```
GUARDRAIL_MANIFEST_PATH      = /app/dbt-demo/target/manifest.json
GUARDRAIL_DEFAULT_WORKSPACE  = analytics
GUARDRAIL_DEFAULT_TENANT     = default
GUARDRAIL_STRICT_MODE        = false
```

Every user request will flow through the guardrail lifecycle, and governance
status will be appended to the response.

---

## dbt-demo Governance Metadata

Governance metadata has been added to the following dbt-demo models:

### full-jaffle-shop (`models/full-jaffle-shop/ai_governance.yml`)

| Model | Access | Answer Mode | Notes |
|-------|--------|-------------|-------|
| `customers` | approved | masked | PII: name, age masked |
| `orders` | approved | aggregate_only | Financial — no row-level output |
| `payments` | restricted | aggregate_only | PCI-adjacent — finance workspace only |
| `membership` | approved | full | Safe for all use cases |
| `events` | internal | aggregate_only | Behavioural data — no RAG |
| `users` | **deny** | deny | Identity table — hard deny |
| `plan` | approved | full | Product catalog — fully safe |

### chinook (`models/chinook/ai_governance.yml`)

| Model | Access | Notes |
|-------|--------|-------|
| `stg_customer` | restricted | Direct PII — name/email/phone excluded |
| `stg_employee` | restricted | PII + salary — critically restricted |
| `stg_invoice` | approved | Financial — aggregate_only |
| `stg_track` | approved | Public catalog — fully approved |
| `stg_artist/album/genre` | approved | Public reference — fully approved |

---

## What the Warehouse Still Enforces

dbt metadata shapes AI behavior — it does **not replace** warehouse security:

- **Row-level security** (Snowflake RLS, BigQuery row access policies) — still enforced
- **Column-level masking** (Dynamic data masking, column encryption) — still enforced
- **Role-based access control** (GRANT/REVOKE) — still enforced
- **Network policies and IP restrictions** — still enforced

Datap.ai guardrails operate **before** warehouse execution: filtering prompts,
blocking generations, and explaining policy decisions. The warehouse provides
the final security boundary even if a guardrail is misconfigured.

---

## Traceability Integration

All policy decisions are traced through the existing `traceability.TraceLedger`:

Event types emitted:
- `policy_catalog_loaded`
- `context_filtered`
- `policy_check_passed`
- `policy_check_failed`
- `action_blocked`
- `action_modified_for_safety`

These flow into the `fct_ai_policy_checks` and `fct_ai_tool_calls` mart tables
for Lightdash dashboards and compliance audit.

---

## Extending the Policy Catalog

1. Add metadata to dbt YAML files under `meta.datapai.*`
2. Run `dbt compile` (or `dbt build`) to update `manifest.json`
3. Call `compiler.refresh()` or restart the service to pick up changes
4. The policy catalog is versioned by MD5 hash of the manifest file

---

## Known Limitations / Next Steps

- SQL column extraction is heuristic (regex-based); a proper SQL AST parser (e.g. `sqlglot`) would be more accurate for complex queries
- The `filter_context()` function uses model names for matching; aliased CTEs or subqueries are not currently resolved
- Approval-required agent actions currently block the request; an approval workflow (Slack/email notification) is a future extension
- OpenWebUI pipeline currently annotates requests but does not modify the LLM system prompt — a deeper integration hook would pass the safe context to the model
- `ai_governance.yml` metadata requires a `dbt compile` to be picked up by the manifest; real-time metadata updates without recompile would need a separate metadata store

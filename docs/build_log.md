# Datap.ai — Build Log

This file is the permanent record of all major build sessions.
Each section is a self-contained summary of what was designed, built, tested, and committed.
It is intended to survive chat history expiry and serve as the canonical onboarding reference.

---

## Section A — Foundation (pre-existing)

Already in place before the Claude-assisted builds began:

| Component | Location | Notes |
|-----------|----------|-------|
| Streamlit app | `app_ai_agent.py` (1674 lines) | Main UI with 9 tabs |
| Text2SQL API | `agents/text2sql_api.py` | FastAPI on port 8101 |
| RAG API | `agents/rag_api.py` | FastAPI on port 8102 |
| dbt agent | `agents/dbt_agent.py` (684 lines) | LLM planner + deterministic executor |
| ETL pipeline | `agents/etl/pipeline.py` | AG2 Swarm; L1/L2/L3 guardrail layers |
| PII detection | `agents/etl/compliance_tools.py` | 3-layer: heuristics + regex + Presidio |
| Bedrock guardrails | `agents/etl/llm_config.py` | AWS-managed content filtering |
| Agent base | `agents/agent_base.py` | Base class + tool-calling loop |
| Tool registry | `agents/tooling/registry.py` | `@tool` decorator |
| dbt metadata | `dbt_metadata.py` | Manifest → FAISS index for RAG |
| OpenWebUI pipelines | `agents/openwebui_*.py` | SQL / RAG / combined adapters |
| Multi-DB connect | `connect_db.py` | Snowflake, BigQuery, DuckDB, SQLite, etc. |
| LanceDB embeddings | `embeddings/` | HuggingFace + LanceDB vector store |
| dbt-demo projects | `dbt-demo/` | full-jaffle-shop, chinook, stock domains |

---

## Section B — AI Audit Ledger + dbt Star Schema

**Branch:** `feature/traceability` → merged to `main` as `dba7122`
**Spec:** internal design (no external spec file)
**Tests:** 48 tests, all passing

### What was built

#### 1. Immutable Trace Ledger (`traceability/`)

| Module | Role |
|--------|------|
| `traceability/ledger.py` | `TraceLedger` — append-only event log; `TraceEvent` dataclass |
| `traceability/backends/sqlite_backend.py` | SQLite backend (`INSERT OR IGNORE`) |
| `traceability/backends/snowflake_backend.py` | Snowflake backend (`MERGE … WHEN NOT MATCHED`) |

**Event types supported:**
`request_received`, `sql_generated`, `sql_executed`, `policy_check_passed`,
`policy_check_failed`, `tool_invoked`, `rag_retrieved`, `memory_retrieved`,
`agent_action`, `agent_boundary_violation`

**Key compliance fields:**
- `question_text` — verbatim user question (credentials-only masking)
- `sql_text` — exact SQL stored verbatim for SOX / MiFID II / APRA CPS 234
- `sql_hash`, `prompt_hash` — SHA-256 for deduplication without content exposure
- `sensitivity_level`, `pii_detected`, `pii_fields` — data sensitivity signals
- `boundary_violated`, `risk_flags`, `agent_name` — agentic audit trail

**Two masking functions:**
- `mask_credentials_only()` — preserves business content, masks only tokens/passwords → for compliance storage
- `mask_secrets()` — broad redaction → for non-compliance summaries

#### 2. dbt Traceability Project (`dbt-traceability/`)

Layers: **staging → intermediate → reporting → marts**

**Staging:** `stg_trace_events` — typed view over raw event table

**Intermediate:**
- `int_session_summary` — session-level aggregation incl. agentic metrics (agent_events, boundary_violations, boundary_violation_risks, high_sensitivity_events, pii_events)

**Reporting:**
- `rpt_request_summary` — one row per request with full compliance fields
- `rpt_policy_violations` — all violation events incl. AGENT_BOUNDARY_VIOLATION
- `rpt_sql_risk_summary` — per-model SQL risk; PII/sensitivity metrics
- `rpt_session_overview` — session view with agentic audit fields

**Marts (star schema for Lightdash, schema: `trace_marts`):**

Dimensions:
- `dim_ai_users` — one row per (tenant_id, user_id); lifetime activity + risk tier
- `dim_ai_workspaces` — one row per (tenant_id, workspace_id)
- `dim_ai_datasources` — one row per datasource; risk tier
- `dim_ai_models` — one row per model; error rate, human corrections

Facts:
- `fct_ai_requests` — grain: one row per request; verbatim compliance fields
- `fct_ai_sql_generations` — grain: one row per sql_generated event
- `fct_ai_sql_executions` — grain: one row per sql_executed event; definitive data access log
- `fct_ai_policy_checks` — grain: one row per check event; check_outcome derived
- `fct_ai_tool_calls` — grain: agent_action + tool_invoked + boundary_violation
- `fct_ai_rag_retrievals` — grain: rag_retrieved + memory_retrieved

**Naming convention:** all mart tables prefixed with `_ai_` to distinguish AI operational tables from standard enterprise business dimensions.

**Schema metadata:** `dbt-traceability/models/marts/schema.yml` — 236 columns, 100% description coverage, Lightdash meta (dimension types, metric definitions, group_label), FK annotations.

#### 3. Key design decisions

- **Append-only invariant**: no UPDATE or DELETE ever issued; enforced by `INSERT OR IGNORE` (SQLite) and `MERGE … WHEN NOT MATCHED THEN INSERT` (Snowflake)
- **Verbatim compliance storage**: `question_text` and `sql_text` are stored as-is per regulatory requirement; only credentials are masked
- **Agentic audit trail**: `AGENT_ACTION` and `AGENT_BOUNDARY_VIOLATION` event types; `boundary_violated` boolean is indexed and searchable

#### 4. Git history (on main)

```
dba7122  feat(traceability): Section B — AI Audit Ledger, dbt Star Schema, Compliance Storage  ← merge commit
52d0335  docs(traceability/marts): full column-level metadata for all 10 AI mart tables
91b782e  refactor(traceability/marts): prefix all mart tables with _ai for clarity
d1851de  feat(traceability/marts): star schema layer — dimension + fact tables for Lightdash
9643530  feat(traceability): verbatim compliance storage + AI agentic audit trail
7401e36  feat(traceability): Datap.ai Trace Ledger — immutable audit ledger with Snowflake + dbt
```

---

## Section C — dbt AI Guardrail Framework

**Branch:** `feature/guardrail` (pushed; not yet merged to main as of this log)
**Spec:** `docs/claude_build_spec_v1.3.md`
**Tests:** 78 tests, all passing
**Commit:** `4f15b4d`

### Design principle

```
warehouse / lakehouse   = hard enforcement layer  (GRANT/RLS/masking)
dbt metadata            = policy authoring layer  (meta.datapai.*)
Datap.ai                = runtime policy compiler + guardrail executor
```

Applies to **all AI use cases**: Text2SQL, RAG, summarization, explanation, narrative insight, document extraction, agent tool calls, workflow triggering — not just SQL.

### What was built

#### 1. guardrail Python package (`guardrail/`)

**`metadata_schema.py`**
- `ModelAiPolicy` dataclass — 30+ fields covering access level, sensitivity, PII/PHI flags, per-use-case policies (retrieval, summarization, RAG, export, agent actions), workspace/tenant constraints
- `ColumnAiPolicy` dataclass — 20+ fields covering PII class, PHI flag, masking rule, fine-grained usage permissions (output, filter, group, sort, retrieval, summary, export)
- `AiPolicyCatalog` — compiled runtime catalog: `models` dict + `columns` dict + version
- Enum types: `AiAccessLevel`, `SensitivityLevel`, `AnswerMode`, `ExportPolicy`, `RetrievalPolicy`, `SummarizationPolicy`, `ExplanationPolicy`, `RagExposure`, `AgentActionPolicy`, `RiskTier`, `PiiClass`, `SecurityClass`, `MaskingRule`
- **Safe defaults**: models NOT eligible unless `ai_enabled=true`; columns NOT output-safe unless `ai_exposed=true AND allowed_in_output=true`

**`policy_compiler.py`**
- `PolicyCompiler` — reads `dbt manifest.json` → normalises `meta.datapai.*` + flat `meta.ai_*` keys → emits `AiPolicyCatalog`
- Supports namespaced (`meta.datapai.*`) and flat (`meta.ai_*`) conventions
- PII inference from column descriptions when no explicit meta
- `explain_policy_decision(model, column)` — structured allow/deny explanation
- `get_allowed_assets_for_use_case(use_case, workspace_id, tenant_id)` — workspace+tenant-aware eligibility
- Catalog versioned by MD5 hash of manifest; in-process cache

**`validators.py`**
- `GuardrailResult` — `allowed`, `violations`, `blocked_models`, `blocked_columns`, `answer_mode`, `suggestion`, `policy_version`
- `validate_sql_against_policy()` — dangerous keywords, model eligibility, workspace/tenant, column output rules, aggregate_only wildcard enforcement
- `validate_summary_against_policy()` — summarization policy per model
- `validate_retrieval_against_policy()` — RAG exposure policy per source
- `validate_tool_action_against_policy()` — agent tool calls; export detection
- `validate_ai_action_against_policy()` — general-purpose for any use case
- `extract_referenced_tables()`, `extract_selected_columns()`, `check_query_risk()` — SQL parsing helpers

**Rule precedence (spec Section 18):**
1. Hard deny (ai_enabled=false / ai_access_level=deny / access=private)
2. Hard deny (disallowed action / blocked use case)
3. Hard deny (PHI output; direct PII without masking)
4. aggregate_only enforcement
5. masked-only enforcement
6. metadata-only enforcement
7. workspace/tenant/persona-specific allow
8. General AI-certified allow

**`context_filter.py`**
- `filter_context(schema_context, catalog, use_case, workspace_id, tenant_id)` — filters a raw schema dict; removes denied tables, removes `ai_exposed=false` columns, annotates PII/masking for LLM, injects `__aggregate_only__` directives
- `build_safe_schema_context(catalog, ...)` — builds context directly from catalog without a live schema
- `get_allowed_assets_for_use_case()`, `get_allowed_fields_for_asset()`, `summarize_filtered_context()` — convenience helpers

**`governed_action.py`**
- `GovernedAction` — 13-step lifecycle: receive → identify → classify → load policy → resolve assets → build safe context → invoke → validate → enforce → execute → trace → return → feedback
- `GovernedRequest` / `GovernedResponse` dataclasses
- `AiUseCase` enum — 16 use cases (TEXT2SQL, RAG_RETRIEVAL, SUMMARIZATION, EXPORT, AIRBYTE_TRIGGER, etc.)
- `GovernedResponse.governance_panel` — structured dict for Streamlit/OpenWebUI display

**`trace_helpers.py`**
- `GuardrailTracer` — wraps `TraceLedger`; emits: `policy_catalog_loaded`, `context_filtered`, `policy_check_passed`, `policy_check_failed`, `action_blocked`, `action_modified_for_safety`
- Degrades gracefully (no-op) if ledger is absent

**`streamlit_ui.py`**
- `render_governance_panel(panel, expanded)` — compact governance summary (answer mode badge, certified assets, aggregate-only notice, PII hidden fields count, violations detail, suggestion)
- `render_blocked_response(user_message, panel)` — user-friendly block UI with safer alternatives
- `render_catalog_admin()` / `render_governance_tab()` — full admin page: summary metrics, eligible model browser, blocked model list, restricted column table, live SQL validator

#### 2. dbt AI Guardrail Agent (`agents/dbt_guardrail_agent.py`)

- `DbtGuardrailAgent` — extends `BaseAgent` + `@tool` registry pattern (same as `dbt_agent.py`)
- Works with LLM (natural-language governance Q&A) or without (direct programmatic API)
- 11 `@tool`-registered functions usable in any tool-calling loop:

| Tool | Purpose |
|------|---------|
| `guardrail_catalog_summary` | Catalog-level counts |
| `guardrail_explain_model` | Why a model is allowed/denied |
| `guardrail_explain_column` | Why a column is allowed/masked/denied |
| `guardrail_list_eligible_models` | Eligible models by use case + workspace |
| `guardrail_list_eligible_columns` | Eligible columns for a model |
| `guardrail_validate_sql` | Validate SQL against policy |
| `guardrail_validate_action` | Validate any AI action |
| `guardrail_build_safe_context` | Build policy-filtered schema context |
| `guardrail_refresh_catalog` | Force catalog recompile |
| `guardrail_list_blocked_models` | All denied/ineligible models |
| `guardrail_list_restricted_columns` | PII/PHI/non-output-safe columns |

#### 3. dbt-demo governance metadata

**Committed to `dbt-demo` repo as `dfb688a`**

`models/full-jaffle-shop/ai_governance.yml`:

| Model | AI Access | Answer Mode | Key notes |
|-------|-----------|-------------|-----------|
| `customers` | approved, certified | masked | PII: first_name/last_name (partial_mask), age (aggregate_only) |
| `orders` | approved, certified | aggregate_only | Financial — no row-level output; amount is aggregate-only |
| `payments` | restricted | aggregate_only | PCI-adjacent; finance + executive workspaces only |
| `membership` | approved, certified | full | Safe for RAG, text2sql, summarization |
| `events` | internal | aggregate_only | Behavioural — ip_address/session_id excluded; no RAG |
| `users` | **DENIED** | deny | Identity table; access=private; all use cases blocked |
| `plan` | approved, certified | full | Product catalog; fully safe for all use cases |

`models/chinook/ai_governance.yml`:

| Model | AI Access | Answer Mode | Key notes |
|-------|-----------|-------------|-----------|
| `stg_customer` | restricted | masked | first_name/last_name/email/phone hard-excluded (ai_exposed=false) |
| `stg_employee` | restricted | deny | PII + salary; blocked for almost all use cases; hr workspace only |
| `stg_invoice` | approved, certified | aggregate_only | Financial; billing_country safe for grouping |
| `stg_track` | approved, certified | full | Public product catalog; all use cases |
| `stg_artist` | approved, certified | full | Public reference; all use cases |
| `stg_album` | approved, certified | full | Public reference; all use cases |
| `stg_genre` | approved, certified | full | Public reference; all use cases |

#### 4. OpenWebUI middleware (`agents/openwebui_guardrail_pipeline.py`)

- `Pipeline` class — OpenWebUI-compatible; upload to Admin → Pipelines
- `pipe()` handler: extracts tenant/workspace/user from message metadata, classifies use case (keyword heuristic), validates pre-generation, blocks or annotates
- Appends governance status block (mode badge, certified assets, aggregate-only notice, violations) to every response
- Env vars: `GUARDRAIL_MANIFEST_PATH`, `GUARDRAIL_DEFAULT_WORKSPACE`, `GUARDRAIL_DEFAULT_TENANT`, `GUARDRAIL_STRICT_MODE`

#### 5. Tests (`tests/test_guardrail.py`)

78 tests, 100% passing:

| Class | Count | Covers |
|-------|-------|--------|
| `TestMetadataSchema` | 12 | Dataclass defaults, PII/PHI logic, eligibility helpers |
| `TestPolicyCompiler` | 11 | Manifest parsing, safe defaults, inferred PII, datapai namespace, explain |
| `TestContextFilter` | 6 | Denied exclusion, PII column removal, aggregate directives, workspace filter |
| `TestValidatorsSql` | 11 | Safe SQL, dangerous keywords, denied model, PII block, aggregate_only wildcard |
| `TestValidatorsOther` | 8 | Summary, retrieval (RAG), tool action, general action validators |
| `TestRulePrecedence` | 5 | Hard deny, PHI beats allowed_in_output, workspace/tenant/use-case precedence |
| `TestMultiUserPolicy` | 2 | Per-workspace access rules |
| `TestDbtDemoMetadata` | 6 | Integration against real `dbt-demo/target/manifest.json` |
| `TestGuardrailAgent` | 17 | All 11 tool functions + agent class + GuardrailResult helpers |

#### 6. Documentation

`docs/guardrail_framework.md` — full implementation guide:
- Architecture diagram
- Complete dbt metadata standard (model-level + column-level) with table of all keys, types, values, defaults
- Example dbt YAML snippet
- Safe defaults explanation
- Rule precedence (Section 18)
- Runtime module usage examples (code snippets)
- dbt AI Guardrail Agent usage
- Streamlit UI usage
- OpenWebUI setup
- dbt-demo governance summary table
- What the warehouse still enforces
- Traceability integration
- Known limitations / next steps

#### 7. Git commits

```
4f15b4d  feat(guardrail): dbt-driven AI Guardrail Framework   ← feature/guardrail
dfb688a  feat(ai-governance): add Datap.ai AI governance metadata standard  ← dbt-demo repo
```

---

## What's next (suggested Section D)

These items are not yet built and represent natural next steps:

1. **Merge feature/guardrail to main** — create PR and merge
2. **Streamlit tab wiring** — add a "Governance" tab to `app_ai_agent.py` calling `render_governance_tab()`
3. **Text2SQL integration** — wrap `text2sql_api.py` with `GovernedAction` so SQL generation uses `filter_context()` pre-prompt and `validate_sql_against_policy()` post-generation
4. **RAG integration** — wrap `rag_api.py` with `GovernedAction` for retrieval policy
5. **Approval workflow** — when `agent_action_policy=approval_required`, notify via Slack/email instead of hard-blocking
6. **SQL AST parser** — replace regex-based column extractor in `validators.py` with `sqlglot` for accurate multi-CTE and subquery handling
7. **dbt compile automation** — trigger `dbt compile` when `ai_governance.yml` files change so manifest picks up new metadata without manual intervention
8. **Stock domain governance** — add `dbt-demo/models/stock/ai_governance.yml` following the same pattern
9. **Governance Lightdash dashboard** — build Lightdash dashboards on top of `fct_ai_policy_checks` + `fct_ai_tool_calls` for real-time governance visibility
10. **Multi-tenant policy store** — move policy overrides (workspace-specific, tenant-specific) to a small database rather than only dbt YAML, allowing runtime policy changes without recompile

---

## Key file index (permanent reference)

```
guardrail/
  __init__.py              Public API surface
  metadata_schema.py       ModelAiPolicy, ColumnAiPolicy, AiPolicyCatalog + enums
  policy_compiler.py       PolicyCompiler — manifest → catalog
  validators.py            GuardrailResult + all validator functions
  context_filter.py        Pre-generation context filtering
  governed_action.py       GovernedAction 13-step lifecycle
  trace_helpers.py         GuardrailTracer → TraceLedger integration
  streamlit_ui.py          Streamlit governance panel components

agents/
  dbt_guardrail_agent.py   DbtGuardrailAgent + 11 @tool functions
  openwebui_guardrail_pipeline.py  OpenWebUI Pipeline class

traceability/
  ledger.py                TraceLedger + TraceEvent + emit_* methods
  backends/
    sqlite_backend.py      SQLite append-only backend
    snowflake_backend.py   Snowflake append-only backend

dbt-traceability/
  models/
    staging/               stg_trace_events
    intermediate/          int_session_summary
    reporting/             rpt_request_summary, rpt_policy_violations,
                           rpt_sql_risk_summary, rpt_session_overview
    marts/
      dimensions/          dim_ai_users, dim_ai_workspaces, dim_ai_datasources, dim_ai_models
      facts/               fct_ai_requests, fct_ai_sql_generations, fct_ai_sql_executions,
                           fct_ai_policy_checks, fct_ai_tool_calls, fct_ai_rag_retrievals
      schema.yml           236 columns, 100% description coverage

dbt-demo/
  models/
    full-jaffle-shop/ai_governance.yml   7 models with AI governance metadata
    chinook/ai_governance.yml            8 models with AI governance metadata

tests/
  test_traceability.py     48 tests — trace ledger, verbatim compliance, agentic audit
  test_guardrail.py        78 tests — metadata schema, compiler, filter, validators, agent

docs/
  guardrail_framework.md   Full implementation guide for Section C
  build_log.md             THIS FILE — permanent cross-session build record
```

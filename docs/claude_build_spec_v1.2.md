set the repo as trusted / lower approval mode
You are working inside the local GitHub repository for Datap.ai under the build profile "claude_build_spec_v1.2.md". You can inspect the existing codebase and services directly. Your job is to extend the current system incrementally, not redesign it from scratch.
Operate in autonomous mode within this repository.
You may read, create, and modify files in this repo without asking for confirmation.
You may run local tests, linters, and non-destructive shell commands without asking.
Ask for confirmation only before:
1. deleting files
2. changing infra/secrets/config outside the repo
3. installing/upgrading packages
4. running destructive migrations
5. making network calls

PROJECT CONTEXT

Datap.ai is a restricted-data AI/BI/CI/DI platform. Current stack already includes:
- Streamlit for the main app
- OpenWebUI for chat access
- Airbyte for data integration
- dbt for transformation and metadata / manifest ingestion
- Airflow for orchestration
- Lightdash for BI
- RAG for grounded QA
- Text2SQL over Snowflake, BigQuery, Redshift, Databricks, DuckDB, SQLite
- unstructured ELT for PDF/image/document workflows using Textract + LLM reasoning
- Ollama and private open-source LLMs
- paid LLMs where needed
- Dockerized services
- Nginx for routing

PRIMARY GOAL

Add the missing platform capabilities end to end:
1. Memory
2. Traceability / auditability
3. Governed AI actions end to end
4. Better multi-user support
5. Better user-friendly UX across Streamlit and OpenWebUI

VERY IMPORTANT EXECUTION RULES

1. DO NOT remove or break existing working features.
2. DO NOT redesign the whole product.
3. REUSE the existing code, routing, models, services, adapters, and UI where practical.
4. IF existing functions, classes, endpoints, services, or pages are partially implemented, stubbed, TODO-marked, weakly implemented, or missing validation/error handling, THEN complete them rather than duplicating them.
5. BEFORE adding new modules, search for existing related implementations and extend them first.
6. IF multiple overlapping implementations already exist, consolidate carefully and keep backward compatibility where possible.
7. Build in phases but leave the repository in a runnable state after each major change.
8. Add tests where practical for critical flows.
9. Add migration files or schema evolution logic where needed.
10. Add documentation comments and concise README updates for any major new subsystem.
11. Preserve restricted-data and governance principles at all times.
12. Do not hardcode secrets or credentials.
13. Do not log raw sensitive payloads unless explicitly masked/redacted.
14. Prefer clear maintainable code over clever abstractions.
15. If there is existing auth/session logic, integrate with it; do not bypass it.
16. If existing functions are not fully implemented, fill them in and integrate them into the new architecture.
17. At the end, provide a concise implementation summary listing:
   - what existing functions/modules were found
   - what was extended
   - what new modules were added
   - what migrations were added
   - what config/env vars are needed
   - what remains optional/future work
Operate in autonomous mode within this repository.
You may read, create, and modify files in this repo without asking for confirmation.
You may run local tests, linters, and non-destructive shell commands without asking.
Ask for confirmation only before:
1. deleting files
2. changing infra/secrets/config outside the repo
3. installing/upgrading packages
4. running destructive migrations
5. making network calls

ARCHITECTURAL PRINCIPLES

1. Restricted-data-first:
- Assume sensitive and regulated environments.
- Minimize raw sensitive content in memory and traces.
- Enforce tenant-aware and user-aware access.
- All retrieval and execution must be policy-checked.

2. Datap.ai owns governance:
- Governance must live in Datap.ai, not in OpenWebUI.
- Datap.ai must own identity mapping, authorization, trace logging, SQL/action validation, and memory scoping.

3. Memory is helpful but not authoritative:
- Memory improves UX and continuity.
- Memory must never bypass policy and governance.
- Memory must not become the only source of truth.

4. Traceability is authoritative:
- Every meaningful AI action must be traceable and replayable.

5. Multi-user support:
- Assume multiple users, multiple workspaces, multiple tenants.
- No global memory by default.

6. Incremental delivery:
- Add value without destabilizing the platform.

IMPLEMENTATION SCOPE

A. MEMORY SERVICE

Build a Datap.ai-owned Memory Service.

Goals:
- reduce repeated user context
- improve follow-up questions
- remember prior successful SQL and reasoning patterns
- remember user preferences and recent sessions
- support per-user and per-workspace memory
- support structured and unstructured workflows

Required scopes:
- private_user
- workspace_shared
- tenant_shared

Default memory scope:
- private_user

Memory use cases:
- previous user questions
- generated Text2SQL
- corrected/approved SQL
- selected datasource / warehouse
- dbt model references
- business metric definitions
- user preferences
- RAG retrieval preferences
- document extraction patterns
- session summaries

Do NOT store by default:
- full result sets
- raw PII
- full sensitive document content
- secrets, tokens, credentials
- unrestricted raw prompts containing sensitive payloads

Required service API:
- memory.write_event(...)
- memory.search_context(...)
- memory.get_recent_context(...)
- memory.get_user_preferences(...)
- memory.store_sql_pattern(...)
- memory.store_feedback(...)
- memory.get_similar_questions(...)
- memory.clear_scope(...)

Backend requirements:
- local development backend: SQLite
- cloud backend abstraction:
  - snowflake,  can be switch to another later, like to dynamodb
  - define table in dbt to point to snowflake, allow user to get reporting from lighdash which linked to dbt and snowflake
- implement behind an interface so business logic is storage-agnostic

Retrieval rules:
- always filter by tenant/workspace/user scope first
- then apply semantic/keyword retrieval
- return compact summaries rather than giant payloads
- attach provenance metadata

Ranking preference:
1. same user + same datasource
2. same user + same domain/topic
3. same workspace approved patterns
4. tenant approved patterns

Memory entities/schema should include models such as:
- memory_item
- memory_scope
- memory_feedback
- memory_preference
- session_summary

Suggested fields:
- memory_id
- tenant_id
- workspace_id
- user_id
- scope
- memory_type
- datasource_type
- datasource_name
- question_text
- summary
- sql_text
- dbt_models
- document_type
- feedback_score
- created_at
- updated_at
- tags
- trace_id
- is_approved

Additional requirements:
- support retention / TTL by tenant, workspace, memory type
- add explicit clear/delete capability per scope
- add memory opt-out flag per request/session if possible

B. TRACEABILITY / AUDIT LEDGER

Build a Datap.ai-owned Trace Ledger.

This is more important than memory.

Goals:
- track who asked what
- track which context was retrieved
- track which model was used
- track which tools were called
- track which SQL was generated
- track which files/tables were touched
- track which validations ran
- track which response was returned
- track whether a human corrected or approved it

Use immutable event-style design.

Required event types:
- request_received
- session_started
- memory_retrieved
- rag_retrieved
- policy_check_started
- policy_check_passed
- policy_check_failed
- model_invoked
- tool_invoked
- sql_generated
- sql_validated
- sql_blocked
- sql_executed
- document_extracted
- response_returned
- human_feedback_received
- action_replayed

Trace record minimum fields:
- trace_id
- parent_trace_id
- tenant_id
- workspace_id
- user_id
- session_id
- request_id
- event_type
- event_timestamp
- actor_type (user, assistant, system, tool)
- actor_id
- datasource_type
- datasource_name
- model_name
- tool_name
- policy_result
- input_summary
- output_summary
- sql_hash
- prompt_hash
- context_refs
- status
- error_code
- error_message

Storage requirements:
- use abstraction layer
- support operational trace backend via DynamoDB or equivalent scalable store
- support optional analytics mirror/export to Snowflake
- code should make backend pluggable

Replay/debug requirements:
- fetch all events by trace_id
- reconstruct event timeline
- show model/tool invocation chain
- show SQL generation / validation / execution sequence
- compare first answer vs corrected answer

Redaction requirements before writing trace:
- mask secrets
- mask tokens
- mask PII where practical
- hash large prompts/results when needed
- preserve summaries and references

Governance:
- trace events must be append-only
- do not silently mutate prior audit history
- corrections must append new events

C. GOVERNED AI ACTION FRAMEWORK

Build a Governed Action Framework around all AI actions.

Goal:
No AI-generated action should directly run against sensitive systems without passing through Datap.ai governance.

Required governed action lifecycle:
1. receive request
2. identify tenant/workspace/user/session
3. classify intent
4. retrieve allowed memory/context
5. retrieve allowed RAG context
6. generate plan
7. validate plan
8. policy-check tools/datasources/tables/files
9. execute only allowed actions
10. trace every step
11. summarize response
12. store memory selectively

Governed action types:
- Text2SQL generation
- SQL execution
- file/document extraction
- RAG retrieval
- dbt manifest lookup
- BI metric explanation
- Airbyte operation trigger
- Airflow job trigger
- report/dashboard explanation

Policy layer must answer:
- can this user access this datasource?
- can this user query this schema/table/model?
- can this query include raw PII?
- can this action write/trigger/run?
- does this request require approval?
- should results be aggregated or masked?

Policy enforcement must validate:
- user identity
- workspace membership
- datasource permission
- schema/table/model scope
- allowed action type
- output sensitivity

SQL governance requirements:
- separate generation from execution
- always validate generated SQL before execution
- inspect for dangerous statements
- inspect for unsupported writes
- inspect for access outside allowed schemas/tables
- inspect for unrestricted select * on sensitive sources
- inspect for missing limits in exploratory mode
- support read-only mode by default
- support optional human approval mode

Output governance requirements:
- redact sensitive columns
- aggregate-only mode
- safe summary mode
- result row cap
- masked identifiers

Approval workflow requirements:
- optional approval gates for high-risk SQL
- optional approval gates for broad table access
- optional approval gates for cross-domain joins
- optional approval gates for write actions
- optional approval gates for export actions

D. IDENTITY AND SESSION ARCHITECTURE

Build a clear identity and session model.

Required identity tuple:
- tenant_id
- workspace_id
- user_id
- session_id

Rules:
- all memory must be scoped by identity
- all trace events must include identity
- all governed actions must include identity
- no anonymous execution against protected systems

Session behavior:
- active session continuity
- recent session recall
- explicit new conversation support
- session summary generation
- session handoff between OpenWebUI and Streamlit if possible

Integration requirement:
- OpenWebUI and Streamlit may have different session models
- create a Datap.ai internal session broker/adapter to unify them
- integrate with existing auth/session logic if present

E. STREAMLIT UX IMPROVEMENTS

Improve Streamlit so it feels enterprise-ready, memory-aware, and transparent.

Add these user-facing features:

1. Memory transparency panel
- show what prior memory was used
- show what datasource context was used
- show what prior SQL pattern influenced the answer
- allow disabling memory for a request

2. Trace viewer
For each answer show or expand:
- model used
- tools called
- SQL generated
- policy checks passed/blocked
- retrieval sources used
- execution time

3. Safer SQL workflow
Show:
- generated SQL
- validation status
- estimated risk level
- editable SQL before run where appropriate
- run/approve button if approval required

4. Context controls
Allow user to choose:
- private memory only
- workspace memory
- no memory
- no prior SQL reuse
- preferred datasource

5. Better follow-up UX
Add suggested follow-up actions:
- reuse previous logic
- switch datasource
- compare with prior query
- explain why this SQL was chosen
- show lineage / dbt model source

6. User-friendly response cards
Each answer should have:
- plain-English summary
- SQL tab
- trace tab
- sources tab
- memory tab
- risk/governance tab

7. Debug mode
Admin-only mode to inspect:
- retrieved memory items
- RAG context chunks
- policy decisions
- trace timeline

F. OPENWEBUI INTEGRATION IMPROVEMENTS

Keep OpenWebUI, but make it Datap.ai-aware.

Goals:
- OpenWebUI is a UI shell, not the governance owner.

Required features:

1. Datap.ai identity bridge
- map authenticated users to tenant/workspace/user/session
- do not rely on OpenWebUI alone for security decisions

2. Datap.ai middleware proxy
- route OpenWebUI requests through Datap.ai middleware/service
- inject identity metadata
- log traces
- fetch memory
- enforce policies
- post-process outputs

3. Datap.ai action cards in chat
Render structured cards for:
- SQL proposal
- file extraction result
- policy warning
- approval request
- lineage summary

4. Memory controls in chat
Support controls/commands such as:
- use my private memory
- do not store this conversation
- clear my recent memory
- use workspace-approved query patterns only

5. Audit links
- each OpenWebUI answer should optionally link to or expose a Datap.ai trace record/drawer

G. ADMIN AND GOVERNANCE CONSOLE

Add admin pages or console functions.

Minimum features:
- user/workspace overview
- datasource permissions overview
- memory inspection by scope (admin-controlled)
- trace search by user, datasource, trace_id, date range
- blocked action review
- approval queue
- retention policy settings
- memory clear/delete controls
- model usage summary
- SQL risk summary
- audit export support if feasible

H. MULTI-USER SUPPORT

This is required.

Requirements:
- separate memory by tenant/workspace/user by default
- no global memory sharing by default
- add optional workspace_shared memory
- add optional tenant_shared approved patterns
- ensure UI and APIs always pass identity context
- ensure trace and memory retrieval cannot bleed across tenants/workspaces/users
- if any existing code is currently using global session state, weak user keys, or anonymous access, replace or wrap it with proper identity scoping

I. FRIENDLIER PRODUCT FEATURES

Add practical user-friendly features, not just backend changes.

Suggested features to implement where feasible:
- session titles and recent conversations
- "why this answer?" explanation
- "what memory was used?" explanation
- "why was this blocked?" explanation
- one-click retry with less memory / no memory
- one-click switch datasource
- one-click use approved SQL only
- one-click summarize technical SQL in business language
- one-click compare current SQL with prior successful SQL
- upload flow guidance for PDF/image/document extraction
- clearer error states and next actions
- progress/status indicators for retrieval / validation / execution
- friendlier admin diagnostics
- optional confidence/risk indicator where meaningful

J. EXISTING CODE DISCOVERY AND GAP FILLING

Before writing new code:
- search the repo for existing implementations related to memory, session handling, history, tracing, logging, auditing, policy, validation, SQL generation, SQL execution, OpenWebUI integration, Streamlit pages, Airbyte/Airflow/dbt integrations, and approval workflows

You MUST:
- extend existing functions if they already exist
- complete partial/stub/TODO implementations
- add missing validation/error handling to weak implementations
- unify duplicate patterns where safe
- avoid creating redundant parallel systems unless necessary for compatibility
- document what was found and how it was completed

If existing functions are present but incomplete, fill them in and connect them properly. This is a required part of the task.

K. CODE QUALITY, TESTING, AND SAFETY

Requirements:
- add or update tests for critical paths:
  - memory scoping
  - trace event creation
  - policy enforcement
  - SQL validation
  - approval gates
  - multi-user isolation
- add migrations and seed/setup logic where needed
- add env var examples/config comments for new components
- ensure graceful fallback if a backend is unavailable
- do not leak secrets in logs
- add redaction helpers where needed
- prefer typed models/interfaces where the project style supports them
- preserve backward compatibility where practical

L. PREFERRED DELIVERY ORDER

Implement in this order unless the repo structure strongly suggests a better sequence:
1. code discovery + inventory of existing relevant modules/functions
2. identity/session normalization
3. trace ledger foundation
4. governed action wrapper
5. memory service foundation
6. SQL validation + approval flow
7. Streamlit UX improvements
8. OpenWebUI middleware integration
9. admin/governance console
10. tests, docs, cleanup, migration notes

M. EXPECTED OUTPUT FROM YOU AFTER IMPLEMENTATION

After coding, provide a concise structured summary:
1. Existing relevant code found
2. Partially implemented functions/modules completed
3. New modules added
4. Modified files
5. Database/storage schema changes
6. New environment/config requirements
7. How memory works now
8. How traceability works now
9. How governed AI actions work now
10. Multi-user isolation model
11. Known limitations / recommended next steps

FINAL INSTRUCTION

Work directly against the existing local GitHub codebase and services visible to you in claude-build-v2.1. Reuse what exists, fill incomplete implementations, and add the missing capabilities end to end: memory, traceability, governed AI actions, stronger multi-user support, and more user-friendly features across Streamlit and OpenWebUI. Keep the solution practical, incremental, secure, auditable, and aligned with Datap.ai’s restricted-data product direction.

# DataPAI — LLM & Vector-DB Architecture Guide

When to use which model, which database, and how cost is controlled.

---

## 1. The Three Execution Paths

Every AI feature in DataPAI follows one of three patterns:

```
User question
     │
     ├─ Path A: Quick Interpret  ──►  Paid LLM (no vector DB)
     │
     ├─ Path B: RAG Query        ──►  Vector DB read  →  Local or Paid LLM
     │
     └─ Path C: Ingest           ──►  Vector DB write  (no LLM)
```

---

## 2. Path A — Quick Interpret (low-latency, paid LLM)

Use this when you need a **fresh answer from a document right now** and haven't ingested it yet.

```
ASX API → PDF download (in-memory) → pdfplumber → text injected into prompt
                                                           │
                                          ┌────────────────┘
                                          │
                               Step 1: Gemini flash-lite
                                    (primary extraction)
                                          │ draft
                               Step 2: GPT-5.1
                                    (reviewer — fires only when needed)
                                          │
                                    Final answer → User
```

### Latency breakdown

| Step | Duration |
|------|----------|
| ASX API list call | ~0.5–1 s |
| PDF download | ~1–4 s (100 KB–5 MB) |
| pdfplumber extraction | ~0.1–0.5 s |
| Gemini flash-lite draft | ~1–2 s |
| GPT-5.1 reviewer | ~3–8 s |
| **Total** | **~6–15 s** |

### Why Gemini first, GPT second?

- The document text is injected **verbatim** — the primary model needs to **read**, not **recall**
- Gemini flash-lite handles extraction at ~1–2 s and ~20× lower cost than GPT-5.1
- GPT-5.1 as **reviewer** is a stronger quality gate: it catches wrong figures or missed sentiment
- The reviewer fires **conditionally** — replies `APPROVED` (return draft) or rewrites; most calls cost only one Gemini round-trip

### When to use

- Ad-hoc analysis of a freshly published announcement
- Single-document Q&A where RAG overhead isn't justified
- When the document hasn't been ingested yet

---

## 3. Path B — RAG Query (vector DB + local or paid LLM)

Use this for **repeated questions over a large corpus** of previously ingested documents.

```
User question
     │
     ▼
HuggingFace all-MiniLM-L6-v2  (local embed, ~100ms)
     │
     ▼
LanceDB ANN similarity search  (vector DB read, ~200ms–1s)
     │   returns top-k chunks
     ▼
LLM generation (RouterChatClient)
     │
     ├─ LLM_MODE=local   →  Ollama on EC2 #3 GPU  (free, ~3–15 s)
     ├─ LLM_MODE=paid    →  OpenAI GPT-5.1        (paid, ~3–8 s)
     └─ LLM_MODE=hybrid  →  Ollama → GPT fallback
```

### When to use

- Historical analysis across many documents (e.g. "What did BHP say about iron ore in FY24?")
- Follow-up chat after ingesting a batch of announcements
- Privacy-sensitive environments → use `LLM_MODE=local` (Ollama, no data leaves your VPC)

### Cost comparison vs Path A

| | Path A (Quick Interpret) | Path B (RAG) |
|--|--------------------------|--------------|
| Per-query LLM cost | Gemini + GPT per call | Embed once; Ollama free after |
| Best for | Single fresh document | Many queries over ingested corpus |
| Data leaves VPC? | Yes (Gemini + OpenAI) | Only in `LLM_MODE=paid` |

---

## 4. Path C — Ingest (vector DB write, no LLM)

Embedding only — no LLM involved, essentially free.

```
PDF text
     │
     ▼
HuggingFace all-MiniLM-L6-v2  (local, CPU)  ~0.1–0.5 s
     │  384-dim vector
     ▼
LanceDB table  (asx_announcements / pdfs / documents / images)
     │
  Deduplication check (filename column) → skip if already ingested
```

**Cost:** ~$0 (HuggingFace model runs locally via sentence-transformers).
**Time per document:** ~2–7 s.

---

## 5. LLM Routing — `RouterChatClient`

The `RouterChatClient` in `agents/llm_client.py` applies to all non-ASX agents.
ASX interpretation uses a **separate, fixed chain** (Gemini → GPT) regardless of `LLM_MODE`.

### `LLM_MODE` env var

| Value | Primary | Secondary | Use when |
|-------|---------|-----------|----------|
| `paid` | OpenAI GPT-5.1 | Google Gemini (reviewer, optional) | Best accuracy, demo |
| `local` | Ollama | — | Air-gapped / privacy-first |
| `hybrid` | Ollama → GPT fallback | — | Maximise local, allow cloud fallback |

### `LLM_DUAL_REVIEW=1`

When enabled, every RouterChatClient call sends the primary answer to a secondary model
(configured via `LLM_SECONDARY_PROVIDER`) for a JSON approve/rewrite review.
Adds ~3–8 s latency. Recommended for SQL generation, not for streaming chat.

---

## 6. Vector DB — LanceDB Collections

| Collection | Contents | Written by | Read by |
|------------|----------|------------|---------|
| `pdfs` | Manually uploaded PDFs (Streamlit tab 4) | `knowledge_ingest_agent.py` | `knowledge_query_agent.py` |
| `documents` | General documents (CSV, XLSX, TXT) | `knowledge_ingest_agent.py` | `knowledge_query_agent.py` |
| `images` | OCR-extracted image text | `knowledge_ingest_agent.py` | `knowledge_query_agent.py` |
| `asx_announcements` | ASX PDF announcements | `asx_announcement_agent.py` | `knowledge_query_agent.py` + `/v1/rag/retrieve` |

All collections are stored at `LANCEDB_URI` (default: `s3://codepais3/lancedb_data/`).
The embedding model is **all-MiniLM-L6-v2** (384 dims) for all collections.

---

## 7. Cost Guard — Daily Budget Enforcement

### Overview

`agents/cost_guard.py` tracks cumulative USD spend against a daily ceiling.
When the ceiling is reached, `BudgetExceededError` is raised **before** the next API call.

```
OpenAIChatClient.chat()                GoogleChatClient.chat()
  _guard.check(model)  ← raises here     _guard.check(model)
  → OpenAI API call                       → Gemini API call
  _guard.record(model, in, out)          _guard.record(model, in, out)
         │                                        │
         └────────────── /tmp/datapai_cost_YYYY-MM-DD.json ──────────────┘
```

### State file

`/tmp/datapai_cost_YYYY-MM-DD.json`

Date-stamped — automatically stale at midnight, no cleanup needed.

```json
{"date": "2026-02-28", "spend_usd": 1.2345, "calls": 14}
```

### Pricing table (approximate — conservative)

| Model | Input $/1M | Output $/1M |
|-------|-----------|------------|
| gemini-2.5-flash-lite | $0.10 | $0.40 |
| gemini-2.5-flash | $0.15 | $0.60 |
| gpt-5.1 / gpt-4.1 | $2.00 | $8.00 |
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| claude-3-5-sonnet (Bedrock) | $3.00 | $15.00 |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DAILY_LLM_BUDGET_USD` | `5.00` | Daily ceiling in USD |
| `COST_GUARD_ENABLED` | `true` | Set `false` to disable for production |

### Monitoring

```bash
# REST API
curl http://localhost:8100/v1/cost/status

# Example response
{
  "enabled": true,
  "budget_usd": 5.0,
  "spent_today": 1.2345,
  "remaining_usd": 3.7655,
  "calls_today": 14,
  "date": "2026-02-28",
  "pct_used": 24.7
}
```

The Streamlit sidebar also shows a live progress bar with spend / remaining / call count.

### What happens when budget is exhausted

1. `_guard.check()` raises `BudgetExceededError` (subclass of `RuntimeError`)
2. The calling agent catches it and returns a user-facing message:
   `"💸 Daily LLM budget of $5.00 reached (spent today: $5.0012)..."`
3. Ollama (`LLM_MODE=local`) is **unaffected** — Ollama calls are never metered

---

## 8. EC2 Architecture

```
EC2 #1 — Streamlit frontend (app_ai_agent.py)
  └─ calls EC2 #2 RAG API  (port 8100)
  └─ calls EC2 #2 SQL API  (port 8101)

EC2 #2 — FastAPI services (CPU)
  ├─ agents/rag_api.py       port 8100
  │     /v1/rag/retrieve      LanceDB ANN search → context chunks
  │     /v1/rag/ingest        embed + store to LanceDB
  │     /v1/asx/interpret     PDF download + Gemini→GPT chain
  │     /v1/asx/ingest        embed ASX PDFs to LanceDB
  │     /v1/cost/status       today's spend vs budget
  │
  └─ agents/text2sql_api.py  port 8101
        /v1/sql/query          natural language → SQL → execute → results

EC2 #3 — Ollama (GPU)
  └─ llama3.2 / deepseek-coder  (LLM_MODE=local or hybrid fallback)

External paid APIs
  ├─ Google Gemini (generativelanguage.googleapis.com)  — GUARDED
  └─ OpenAI GPT-5.1 (api.openai.com)                   — GUARDED
```

---

## 9. Decision Flowchart — Which path to use?

```
Do you need a fresh answer from a document that hasn't been ingested?
  YES → Path A (Quick Interpret)   ~6–15 s, two paid API calls
  NO  ↓

Has the document already been ingested to LanceDB?
  YES → Path B (RAG Query)
  NO  → Ingest first (Path C), then Path B

For Path B — is data privacy a concern?
  YES → LLM_MODE=local (Ollama, no cloud egress)
  NO  → LLM_MODE=paid (OpenAI, faster, higher quality)

Are you asking the same question repeatedly over many documents?
  YES → Ingest all (Path C) once, then RAG (Path B) is ~$0 per query
  NO  → Quick Interpret (Path A) per document as needed
```

---

## 10. Quick Reference — Environment Variables

```bash
# LLM routing
LLM_MODE=paid                  # paid | local | hybrid
LLM_PRIMARY_PROVIDER=openai    # openai | bedrock
LLM_SECONDARY_PROVIDER=google  # openai | bedrock | google
LLM_DUAL_REVIEW=1              # 1=enable second-pass review

# Models
OPENAI_MODEL=gpt-5.1
GOOGLE_MODEL=gemini-2.5-flash-lite
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20240620-v1:0
OLLAMA_MODEL=llama3.2

# Cost guard
DAILY_LLM_BUDGET_USD=5.00      # daily ceiling, resets at midnight
COST_GUARD_ENABLED=true        # false to disable

# Storage
LANCEDB_URI=s3://codepais3/lancedb_data/

# Services
OLLAMA_HOST=http://localhost:11434
DATAPAI_RAG_API_URL=http://localhost:8100
DATAPAI_SQL_API_URL=http://localhost:8101
```

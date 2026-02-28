# ASX Market Announcements Integration

Real-time fetching and AI interpretation of ASX market announcements — no manual PDF uploads required.

---

## Overview

| Before | After |
|--------|-------|
| User manually downloads PDF from ASX website | Agent fetches PDF directly from ASX API |
| User uploads PDF to Streamlit | Pipeline downloads PDF in-memory |
| PDF ingested to LanceDB → RAG query | **Two paths**: Quick Interpret (direct LLM) or Ingest → RAG |
| One LLM call (Ollama) | **Two-step chain**: Gemini flash-lite drafts → GPT-5.1 reviews |

---

## Architecture

```
User Input (Ticker + Intent)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                 ASX Public API                            │
│  GET /asx/1/company/{TICKER}/announcements                │
│  → Returns JSON list: headline, date, PDF URL, type       │
└───────────────────┬───────────────────────────────────────┘
                    │  PDF URL
                    ▼
         ┌──────────────────────┐
         │  Download PDF bytes  │  (in-memory, no disk write)
         │  pdfplumber extract  │
         └──────────┬───────────┘
                    │  Extracted text
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
  ┌──────────────┐   ┌────────────────────┐
  │  Quick Path  │   │  Knowledge Base    │
  │ (low-latency)│   │  Path (RAG)        │
  │              │   │                    │
  │ Gemini lite  │   │ HuggingFace        │
  │   ↓ draft    │   │ all-MiniLM-L6-v2   │
  │ GPT-5.1      │   │   ↓ embed          │
  │   ↓ review   │   │ LanceDB store      │
  │              │   │   (asx_announce-   │
  │ → Answer     │   │    ments table)    │
  └──────────────┘   └────────────────────┘
                              │
                              ▼
                     RAG queries via
                     Knowledge Ingest
                     Agent tab or
                     OpenWebUI RAG
                     pipeline
```

### Why Gemini → GPT (not GPT → Gemini)?

For **document-grounded extraction** tasks like ASX announcement analysis:

- The document text is injected verbatim → the model needs to **read**, not **know**
- Gemini flash-lite handles extraction at ~1–2s and ~20× lower cost than GPT-5.1
- GPT-5.1 as the **reviewer** is a stronger quality gate: it catches missed figures, wrong sentiment, or hallucinated numbers before the user sees them
- The reviewer fires conditionally — it only rewrites when genuinely needed, so most responses cost only one Gemini call

---

## New Files

| File | Purpose |
|------|---------|
| `agents/asx_announcement_agent.py` | Core logic: fetch, download, extract, interpret, ingest |
| `agents/openwebui_asx_pipeline.py` | Standalone OpenWebUI pipeline ("DataPAI ASX Announcements") |
| `docs/asx_announcements.md` | This document |

## Modified Files

| File | Change |
|------|--------|
| `app_ai_agent.py` | Added **Tab 8: ASX Announcements** |
| `agents/rag_api.py` | Added `/v1/asx/announcements`, `/v1/asx/interpret`, `/v1/asx/ingest` |
| `agents/knowledge_query_agent.py` | Added `asx_announcements` to default search collections |
| `agents/openwebui_combined_pipeline.py` | Added `asx` as a fourth route |
| `requirements.txt` | Added `pdfplumber` explicitly |

---

## Setup

### Prerequisites

```bash
pip install pdfplumber requests lancedb
```

### Required Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `GOOGLE_API_KEY` | Gemini flash-lite (primary LLM) | `AIzaSy...` |
| `OPENAI_API_KEY` | GPT-5.1 reviewer | `sk-proj-...` |
| `OPENAI_MODEL` | GPT model name | `gpt-5.1` |
| `GOOGLE_MODEL` | Gemini model name | `gemini-2.5-flash-lite` |
| `LANCEDB_URI` | Where to store embeddings | `s3://codepais3/lancedb_data/` |

> **Note:** All env vars above are already configured in `.env`. No new credentials needed.

### Start the RAG API (includes ASX endpoints)

```bash
uvicorn agents.rag_api:app --host 0.0.0.0 --port 8100 --reload
```

The three new ASX endpoints are automatically available at:
- `POST http://localhost:8100/v1/asx/announcements`
- `POST http://localhost:8100/v1/asx/interpret`
- `POST http://localhost:8100/v1/asx/ingest`

---

## Usage

### 1. Streamlit UI — Tab 8: ASX Announcements

Navigate to `app_ai_agent.py` → **ASX Announcements** tab.

| Control | Description |
|---------|-------------|
| ASX Ticker(s) | Comma-separated symbols, e.g. `BHP, CBA, RIO` |
| Number of announcements | Slider: 5–50 per ticker |
| Market-sensitive only | Filter for price-sensitive announcements |
| 🔍 Fetch Announcements | Load the announcement table |
| ⚡ Quick Interpret | Download PDF → Gemini draft → GPT review → instant result |
| 📥 Ingest to Knowledge Base | Embed + store in LanceDB for future RAG queries |
| 📦 Ingest ALL | Bulk ingest all fetched announcements with progress bar |
| 💬 Follow-up chat | Ask questions about the interpreted announcement |

**Example flow:**

```
1. Enter "BHP" → Fetch Announcements
2. Select "Full Year Results 2024" from the dropdown
3. Click ⚡ Quick Interpret → AI analysis appears in ~5s
4. Ask in chat: "What is the dividend per share?"
5. Click 📥 Ingest → BHP results are now in the knowledge base
6. Switch to Knowledge Ingest Agent tab → ask "What did BHP say about iron ore?"
```

---

### 2. OpenWebUI — Standalone ASX Pipeline

Upload `agents/openwebui_asx_pipeline.py` to OpenWebUI → Admin → Pipelines.

The pipeline appears as **"DataPAI ASX Announcements"** in the model selector.

**Supported commands (natural language):**

```
# List announcements
"Show me BHP's recent announcements"
"Fetch last 10 ASX announcements for CBA"

# Interpret (default)
"What did ANZ announce about their results?"
"Analyse RIO's latest quarterly report"
"Interpret WBC earnings"
ASX:MQG interpret

# Ingest to knowledge base
"Ingest CBA announcements to knowledge base"
"Save RIO announcements"
```

**Pipeline Valves (configurable in OpenWebUI):**

| Valve | Default | Description |
|-------|---------|-------------|
| `DATAPAI_RAG_API_URL` | `http://localhost:8100` | DataPAI RAG API URL (EC2 #2) |
| `DATAPAI_RAG_API_KEY` | *(empty)* | Bearer token if `RAG_API_KEY` is set |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama for RAG fallback only |
| `ASX_DEFAULT_COUNT` | `20` | Default announcements per ticker |
| `ASX_MARKET_SENSITIVE` | `false` | Filter for market-sensitive only |

---

### 3. OpenWebUI — Smart Router Pipeline

`agents/openwebui_combined_pipeline.py` now routes ASX-related questions automatically.

The classifier detects:
- Explicit tags: `ASX:BHP`, `[asx:cba]`
- Keywords: `announce`, `quarterly result`, `half-year`, `asx listed`, `market sensitive`
- LLM classifier: updated to return `asx` as a fourth intent label

```
User: "What did CSL announce about their earnings?"
  → Classifier: "asx"
  → Calls /v1/asx/interpret {ticker: "CSL"}
  → Returns structured analysis
```

---

### 4. REST API

All ASX endpoints are on the RAG API (port 8100) and share the same bearer token auth.

#### List announcements
```bash
curl -X POST http://localhost:8100/v1/asx/announcements \
  -H "Content-Type: application/json" \
  -d '{"ticker": "BHP", "count": 10, "market_sensitive_only": false}'
```

Response:
```json
{
  "ticker": "BHP",
  "count": 10,
  "announcements": [
    {
      "id": "...",
      "ticker": "BHP",
      "document_date": "2024-08-20T10:30:00+10:00",
      "headline": "Full Year Results FY2024",
      "url": "https://www.asx.com.au/asxpdf/...",
      "market_sensitive": true,
      "number_of_pages": 42,
      "size_kb": 1820.4,
      "doc_type": "Results - Full Year"
    }
  ]
}
```

#### Interpret an announcement
```bash
curl -X POST http://localhost:8100/v1/asx/interpret \
  -H "Content-Type: application/json" \
  -d '{"ticker": "BHP", "question": "What is the dividend per share?"}'
```

Response:
```json
{
  "ticker": "BHP",
  "headline": "Full Year Results FY2024",
  "date": "2024-08-20",
  "source_url": "https://www.asx.com.au/asxpdf/...",
  "interpretation": "## Executive Summary\n...",
  "reviewed": true,
  "question": "What is the dividend per share?"
}
```

#### Ingest to knowledge base
```bash
curl -X POST http://localhost:8100/v1/asx/ingest \
  -H "Content-Type: application/json" \
  -d '{"ticker": "BHP", "count": 20}'
```

Response:
```json
{
  "ticker": "BHP",
  "ingested": 18,
  "skipped": 2,
  "errors": 0,
  "db_uri": "s3://codepais3/lancedb_data/"
}
```

---

## LLM Chain Detail

```
interpret_announcement()
│
├─ Step 1: GoogleChatClient (Gemini flash-lite)
│    system: Senior financial analyst — structured 5-section report
│    user:   ASX announcement text (up to 8000 chars)
│    → draft (1–2s, low cost)
│
└─ Step 2: OpenAIChatClient (GPT-5.1)
     system: Senior analyst reviewing a junior's draft
     user:   Source document + draft to review
     → "APPROVED" (return draft as-is)
        OR rewrite (return corrected analysis)

Fallbacks:
  • Gemini fails → GPT runs primary directly (no error shown)
  • GPT reviewer fails → Gemini draft returned (logged as warning)
```

### Structured output format (no question)

```
1. Executive Summary      — 2-3 sentence overview
2. Key Financial Figures  — revenue, EBITDA, NPAT, EPS, DPS, guidance
3. Material Events        — acquisitions, capital raises, leadership changes
4. Market Sentiment       — bullish / neutral / bearish + rationale
5. Key Risks / Concerns   — caveats, uncertainties, negative signals
```

---

## Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Image-only PDFs | No text extracted | Falls back to `(No readable text)` notice; consider enabling AWS Textract |
| ASX API rate limits | Heavy bulk ingestion may throttle | Add `time.sleep(0.5)` between requests for large batches |
| Context window (8000 chars) | Very long PDFs are truncated | Increase `max_doc_chars` for documents that need full-text analysis |
| ASX API availability | If ASX website is down, fetch fails | Catch `requests.HTTPError` — UI shows a clear error message |
| Ticker detection accuracy | Common English words may be misdetected | Use explicit `ASX:TICKER` tag for reliability |

---

## Data Flow Summary

```
ASX Website (public API, no auth)
  └─ JSON: announcement list + PDF URLs
       └─ PDF bytes (in-memory via requests)
            └─ pdfplumber → plain text
                 ├─ Quick Interpret:  Gemini → GPT → user
                 └─ Ingest path:     HuggingFace embed → LanceDB (asx_announcements table)
                                          └─ RAG queries via /v1/rag/retrieve
```

"""
DataPAI ASX Announcements Pipeline for OpenWebUI
=================================================
Brings live ASX market announcements directly into the OpenWebUI chat interface.
No manual PDF uploads — the pipeline fetches, extracts, and interprets in real-time.

HOW IT WORKS
────────────
  1. Detects ASX ticker symbols and intent from the user's message.
  2. Routes to one of three actions:
       fetch      → List recent announcements for the ticker
       interpret  → Download the PDF, extract text, run Gemini→GPT analysis
       ingest     → Ingest announcements into LanceDB for future RAG queries
  3. Returns structured markdown — or falls through to the RAG pipeline for
     questions about previously ingested announcements.

COMMANDS (natural language or explicit)
──────────────────────────────────────
  "What did BHP announce recently?"
  "Interpret CBA's latest results"          → latest announcement
  "Analyse ANZ quarterly report"
  "Fetch last 10 ASX announcements for RIO"
  "Ingest WBC announcements to knowledge base"
  "ASX:MQG interpret"                       → explicit ticker tag

LLM CHAIN FOR INTERPRETATION
──────────────────────────────
  Step 1 → DataPAI RAG API /v1/asx/interpret  (Gemini flash-lite extraction)
  Step 2 → GPT-5.1 quality review             (corrects errors, approves draft)

  Both cloud API calls are made server-side on EC2 #2; no API keys needed
  in the OpenWebUI Valves.

INSTALLATION
────────────
  1. Ensure the DataPAI RAG API is running on EC2 #2 with the ASX endpoints:
       uvicorn agents.rag_api:app --host 0.0.0.0 --port 8100

  2. OpenWebUI → Admin → Pipelines → Add Pipeline → paste this file.

  3. Set Valves (or env vars):
       DATAPAI_RAG_API_URL   http://<ec2-2-ip>:8100   (default: localhost:8100)
       DATAPAI_RAG_API_KEY   <bearer token>            (if RAG_API_KEY is set)
       OLLAMA_HOST           http://<ec2-3-ip>:11434   (RAG fallback generation)
       ASX_DEFAULT_COUNT     20                        (announcements per fetch)

ENVIRONMENT VARIABLES
─────────────────────
  DATAPAI_RAG_API_URL   DataPAI RAG/ASX API base URL
  DATAPAI_RAG_API_KEY   Bearer token for the API (empty = no auth)
  OLLAMA_HOST           Ollama endpoint (used as RAG fallback only)
  ASX_DEFAULT_COUNT     Default number of announcements to fetch (default: 20)
  ASX_MARKET_SENSITIVE  Fetch only market-sensitive announcements (default: false)
"""

from __future__ import annotations

import json
import os
import re
from typing import Generator, List, Optional, Union

import requests
from pydantic import BaseModel


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class Pipeline:
    """
    DataPAI ASX Announcements — live market announcement analysis in OpenWebUI.

    Appears as "DataPAI ASX Announcements" in the OpenWebUI model selector.
    """

    class Valves(BaseModel):
        DATAPAI_RAG_API_URL:    str  = os.getenv("DATAPAI_RAG_API_URL",    "http://localhost:8100")
        DATAPAI_RAG_API_KEY:    str  = os.getenv("DATAPAI_RAG_API_KEY",    "")
        OLLAMA_HOST:            str  = os.getenv("OLLAMA_HOST",             "http://localhost:11434")
        ASX_DEFAULT_COUNT:      int  = int(os.getenv("ASX_DEFAULT_COUNT",   "20"))
        ASX_MARKET_SENSITIVE:   bool = os.getenv("ASX_MARKET_SENSITIVE",    "false").lower() == "true"
        ASX_SIGNAL_TIMEOUT:     int  = int(os.getenv("ASX_SIGNAL_TIMEOUT",  "180"))  # signal = 2 LLM calls + yfinance

    def __init__(self):
        self.name   = "DataPAI ASX Announcements"
        self.valves = self.Valves()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def on_startup(self):
        try:
            r = requests.get(
                f"{self.valves.DATAPAI_RAG_API_URL}/health", timeout=5
            )
            r.raise_for_status()
            print(f"[DataPAI ASX] RAG API reachable at {self.valves.DATAPAI_RAG_API_URL}")
        except Exception as exc:
            print(f"[DataPAI ASX] ⚠ RAG API not reachable: {exc}")

    async def on_shutdown(self):
        print("[DataPAI ASX] Pipeline shutdown.")

    async def on_valves_updated(self):
        print(f"[DataPAI ASX] Config updated — RAG API: {self.valves.DATAPAI_RAG_API_URL}")

    # ── Main entrypoint ────────────────────────────────────────────────────

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Parse user intent, route to fetch / interpret / ingest, return markdown.
        """
        intent, ticker, count, question = self._parse_message(user_message)

        if not ticker:
            return (
                "👋 **DataPAI ASX Announcements**\n\n"
                "I couldn't detect an ASX ticker in your message. Try:\n"
                "- `Interpret BHP latest announcement`\n"
                "- `Fetch 10 announcements for CBA`\n"
                "- `What did ANZ announce about their results?`\n"
                "- `Ingest RIO announcements to knowledge base`\n"
                "- `Trading signal for BHP` — AI signal with buy/sell/hold + price targets\n\n"
                "You can also use the explicit tag: `ASX:TICKER interpret` or `ASX:TICKER signal`\n\n"
                "> ⚠️ Trading signals are AI-generated for informational purposes only. "
                "NOT financial advice."
            )

        if intent == "fetch":
            return self._handle_fetch(ticker, count)
        elif intent == "ingest":
            return self._handle_ingest(ticker, count)
        elif intent == "signal":
            return self._handle_signal(ticker)
        else:
            # Default to interpret (also handles natural language questions)
            return self._handle_interpret(ticker, question)

    # ── Intent + ticker parsing ────────────────────────────────────────────

    # Regex: explicit ASX tag — "ASX:BHP", "asx:cba", "[asx: rio]", "ASX bhp"
    _EXPLICIT_TAG = re.compile(
        r"(?:ASX[:\s]+|\[asx:\s*)([A-Za-z]{2,5})", re.IGNORECASE
    )

    # Words to EXCLUDE from ticker candidates — case-insensitive comparison.
    # Covers common English words, financial acronyms, and intent words that
    # would otherwise look like 2-5 letter tickers.
    _NOT_TICKERS: set = {
        # Articles / prepositions / conjunctions
        "A", "AN", "THE", "AND", "OR", "BUT", "FOR", "OF", "TO", "IN",
        "ON", "AT", "BY", "AS", "IS", "IT", "BE", "DO", "GO", "MY", "NO",
        "SO", "UP", "US", "WE", "IF", "HE", "ME", "AM",
        # Pronouns / question words
        "YOU", "WHO", "WHY", "HOW", "WHAT", "WHEN", "WHERE", "WHICH",
        "THIS", "THAT", "THEY", "THEM", "THEIR", "THESE", "THOSE",
        "HIM", "HER", "ITS", "OUR", "YOUR",
        # Common verbs (3-5 chars)
        "ARE", "WAS", "HAS", "HAD", "HAVE", "BEEN", "WERE",
        "GET", "GOT", "LET", "PUT", "SAY", "SET", "DID", "CAN", "MAY",
        "WILL", "DOES", "DONE", "WENT", "COME", "CAME",
        "TELL", "SHOW", "GIVE", "TAKE", "MAKE", "LIKE", "KNOW", "LOOK",
        "FIND", "WANT", "NEED", "SEEM", "FEEL", "KEEP", "MADE", "SAID",
        "USED", "CALL", "WORK", "JUST", "ALSO", "THAN", "THEN", "EVEN",
        "WELL", "BACK", "MUCH", "HELP", "HOLD", "TURN", "OPEN", "MOVE",
        # Common adjectives / adverbs
        "GOOD", "BEST", "MOST", "MANY", "MUCH", "SOME", "ONLY", "VERY",
        "BOTH", "EACH", "SUCH", "FULL", "FREE", "REAL", "SAME", "LONG",
        "LAST", "NEXT", "LESS", "MORE", "LATE", "EARLY", "HIGH", "JUST",
        "OVER", "INTO", "FROM", "WITH", "ALSO", "EACH", "FEW", "NEW",
        # Common nouns
        "TIME", "YEAR", "WEEK", "DAYS", "DATE", "TEXT", "DATA", "NEWS",
        "PART", "AREA", "CASE", "FORM", "TERM", "FACT", "RATE", "HALF",
        "ONCE", "THEN", "THEM", "THAN", "THAN", "THUS", "ELSE",
        # Intent / command words (to avoid self-matching)
        "FETCH", "LIST", "SHOW", "FIND", "SAVE", "INGEST", "STORE",
        "EMBED", "INDEX", "HELP", "INFO", "TELL", "GIVE",
        # Financial acronyms that aren't ASX tickers
        "ASX", "PDF", "CEO", "CFO", "COO", "CTO", "AGM", "EGM",
        "NTA", "DPS", "EPS", "IPO", "FY", "HY", "PY",
        "Q1", "Q2", "Q3", "Q4", "H1", "H2",
        "YOY", "MOM", "WOW", "GDP", "CPI", "RBA", "ATO", "ABS",
        "ASIC", "APRA", "ACCC", "AUSTRAC",
        "ESG", "ROI", "ROE", "ROA", "EBIT", "NPAT", "FCF", "CAPEX",
        "OPEX", "EBITDA", "CAGR", "IRR", "NAV", "NTA", "PNL",
        # Common 2-char words already handled but listed for completeness
        "OF", "TO", "IN", "IS", "IT", "BE", "AS", "AT", "BY", "OR",
    }

    # Keywords signalling list/fetch intent
    _FETCH_WORDS = re.compile(
        r"\b(fetch|list|show|recent|latest|history|last \d+|get announcements?)\b",
        re.IGNORECASE,
    )

    # Keywords signalling ingest intent
    _INGEST_WORDS = re.compile(
        r"\b(ingest|save|store|add to|knowledge base|index|embed)\b",
        re.IGNORECASE,
    )

    # Keywords signalling interpret/analyse intent (also used as "financial context")
    _INTERPRET_WORDS = re.compile(
        r"\b(interpret|analyse|analyze|summary|summarize|explain|what did|tell me"
        r"|results?|earnings?|report|quarterly|half.year|annual|guidance|dividend"
        r"|acquisition|capital raise|profit|revenue|outlook|material|announce)\b",
        re.IGNORECASE,
    )

    # Keywords signalling trading signal intent
    _SIGNAL_WORDS = re.compile(
        r"\b(signal|trade|trading signal|buy|sell|strong buy|strong sell"
        r"|entry|stop.?loss|take.?profit|price target|technical analysis"
        r"|rsi|macd|bollinger|ema|moving average|ohlcv|candlestick|chart"
        r"|should i (buy|sell)|when to (buy|sell)|is it a buy|is it a sell"
        r"|bullish|bearish|momentum|breakout|support|resistance)\b",
        re.IGNORECASE,
    )

    def _extract_ticker(self, message: str) -> Optional[str]:
        """
        Robustly extract an ASX ticker from any message — any case.

        Priority:
          1. Explicit tag: "ASX:BHP", "asx:bhp", "ASX bhp"  → always trusted
          2. Uppercase token in original message: "BHP" → high confidence
          3. Any 2-5 letter token (any case) with financial context nearby
             e.g. "bhp results", "What did cba announce?"  → medium confidence
          4. Any 2-5 letter token (any case) if no match yet
             (last resort; ASX API will validate and return 404 if wrong)

        All candidates are normalised to uppercase and checked against
        _NOT_TICKERS before being returned.
        """
        # 1. Explicit ASX tag (case insensitive)
        explicit = self._EXPLICIT_TAG.search(message)
        if explicit:
            return explicit.group(1).upper()

        has_financial = bool(self._INTERPRET_WORDS.search(message) or self._FETCH_WORDS.search(message))

        # 2. Uppercase-only tokens in original message (e.g. "BHP", "CBA")
        for word in re.findall(r"\b([A-Z]{2,5})\b", message):
            if word not in self._NOT_TICKERS:
                return word

        # 3. Any case — but only when financial context words are present
        #    (avoids "What is the API rate?" → "API" being treated as a ticker)
        if has_financial:
            for word in re.findall(r"\b([A-Za-z]{2,5})\b", message):
                upper = word.upper()
                if upper not in self._NOT_TICKERS:
                    return upper

        # 4. Last resort: any case, no context requirement
        #    (user selected this pipeline explicitly, so likely intentional)
        for word in re.findall(r"\b([A-Za-z]{2,5})\b", message):
            upper = word.upper()
            if upper not in self._NOT_TICKERS:
                return upper

        return None

    def _parse_message(
        self, message: str
    ) -> tuple[str, Optional[str], int, Optional[str]]:
        """
        Parse user message and return (intent, ticker, count, question).

        intent:   "fetch" | "interpret" | "ingest"
        ticker:   ASX ticker symbol (any case input, always returned uppercase) or None
        count:    number of announcements requested
        question: specific question to answer (for QA mode)
        """
        ticker = self._extract_ticker(message)

        # Detect count ("fetch last 10 for BHP", "show 5 announcements")
        count_match = re.search(r"\b(\d+)\b", message)
        count = int(count_match.group(1)) if count_match else self.valves.ASX_DEFAULT_COUNT
        count = max(1, min(count, 50))

        # Detect intent — signal takes priority over interpret
        if self._INGEST_WORDS.search(message):
            intent = "ingest"
        elif self._SIGNAL_WORDS.search(message):
            intent = "signal"
        elif self._FETCH_WORDS.search(message) and not self._INTERPRET_WORDS.search(message):
            intent = "fetch"
        else:
            intent = "interpret"

        # Extract user's specific question (strip ticker and intent words)
        question: Optional[str] = None
        if intent == "interpret" and ticker:
            clean = re.sub(
                r"\b" + re.escape(ticker) + r"\b",
                "",
                message,
                flags=re.IGNORECASE,
            ).strip()
            clean = re.sub(r"\s+", " ", clean).strip()
            # Only pass through as a question if it's substantive
            if len(clean) > 10 and not re.fullmatch(
                r"(interpret|analyse|analyze|summary|summarize|latest|report|results?|announce(ment)?)\s*",
                clean.lower().strip(),
            ):
                question = clean

        return intent, ticker, count, question

    # ── Handlers ───────────────────────────────────────────────────────────

    def _api_headers(self) -> dict:
        h: dict = {"Content-Type": "application/json"}
        if self.valves.DATAPAI_RAG_API_KEY:
            h["Authorization"] = f"Bearer {self.valves.DATAPAI_RAG_API_KEY}"
        return h

    def _handle_fetch(self, ticker: str, count: int) -> str:
        """Fetch and display a list of recent announcements for a ticker."""
        try:
            resp = requests.post(
                f"{self.valves.DATAPAI_RAG_API_URL}/v1/asx/announcements",
                json={
                    "ticker":                ticker,
                    "count":                 count,
                    "market_sensitive_only": self.valves.ASX_MARKET_SENSITIVE,
                },
                headers=self._api_headers(),
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            return (
                f"⚠️ DataPAI RAG API not reachable at `{self.valves.DATAPAI_RAG_API_URL}`.\n"
                f"Please ensure the service is running on EC2 #2."
            )
        except Exception as exc:
            return f"⚠️ Error fetching ASX announcements for **{ticker}**: `{exc}`"

        anns = data.get("announcements", [])
        if not anns:
            return f"No announcements found for **{ticker}**. Check the ticker symbol."

        lines = [
            f"## 📈 {ticker} — Recent Announcements",
            f"*{len(anns)} announcement(s) from the ASX*\n",
            "| # | Date | Headline | Type | Pages | 🔴 |",
            "|---|------|----------|------|-------|----|",
        ]
        for i, a in enumerate(anns, 1):
            date      = (a.get("document_date") or "")[:10]
            headline  = (a.get("headline") or "—")[:70]
            doc_type  = a.get("doc_type", "—")
            pages     = a.get("number_of_pages", "—")
            sensitive = "🔴" if a.get("market_sensitive") else ""
            lines.append(f"| {i} | {date} | {headline} | {doc_type} | {pages} | {sensitive} |")

        lines.append(
            f"\n💡 **Tip:** Ask me to `interpret {ticker}` to get an AI analysis of the latest announcement."
        )
        return "\n".join(lines)

    def _handle_interpret(
        self, ticker: str, question: Optional[str]
    ) -> str:
        """Fetch and interpret the latest ASX announcement via Gemini→GPT."""
        q_note = f" — answering: *\"{question}\"*" if question else ""
        try:
            resp = requests.post(
                f"{self.valves.DATAPAI_RAG_API_URL}/v1/asx/interpret",
                json={
                    "ticker":         ticker,
                    "question":       question,
                    "max_doc_chars":  8000,
                },
                headers=self._api_headers(),
                timeout=120,   # Gemini + GPT review can take up to 30s
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            return (
                f"⚠️ DataPAI RAG API not reachable at `{self.valves.DATAPAI_RAG_API_URL}`.\n"
                f"Please ensure the service is running on EC2 #2."
            )
        except Exception as exc:
            return f"⚠️ Error interpreting **{ticker}**: `{exc}`"

        headline  = data.get("headline", "—")
        date      = (data.get("date") or "")[:10]
        source    = data.get("source_url", "")
        interp    = data.get("interpretation", "No interpretation returned.")
        reviewed  = data.get("reviewed", False)

        reviewer_note = " *(Gemini → GPT reviewed)*" if reviewed else " *(Gemini draft)*"

        lines = [
            f"## 📊 {ticker} — {date}{q_note}",
            f"**{headline}**\n",
            interp,
            "\n---",
            f"📎 **Source:** [{headline[:60]}]({source})" if source else "",
            f"🤖 **LLM chain:** Gemini flash-lite → GPT-5.1 reviewer{reviewer_note}",
            f"\n💡 Ask me a follow-up: `{ticker} what is the dividend guidance?`",
        ]
        return "\n".join(l for l in lines if l)

    def _handle_ingest(self, ticker: str, count: int) -> str:
        """Ingest ASX announcements into LanceDB via the RAG API."""
        try:
            resp = requests.post(
                f"{self.valves.DATAPAI_RAG_API_URL}/v1/asx/ingest",
                json={
                    "ticker":                ticker,
                    "count":                 count,
                    "market_sensitive_only": self.valves.ASX_MARKET_SENSITIVE,
                },
                headers=self._api_headers(),
                timeout=300,   # bulk ingest can take time
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            return (
                f"⚠️ DataPAI RAG API not reachable at `{self.valves.DATAPAI_RAG_API_URL}`.\n"
                f"Please ensure the service is running on EC2 #2."
            )
        except Exception as exc:
            return f"⚠️ Ingest failed for **{ticker}**: `{exc}`"

        ingested = data.get("ingested", 0)
        skipped  = data.get("skipped",  0)
        errors   = data.get("errors",   0)

        lines = [
            f"## 📥 Ingested {ticker} Announcements",
            "",
            f"| Status   | Count |",
            f"|----------|-------|",
            f"| ✅ Ingested | {ingested} |",
            f"| ⏭ Skipped  | {skipped}  |",
            f"| ❌ Errors   | {errors}   |",
            "",
            f"✅ **{ingested} announcement(s)** are now in the knowledge base.",
            "",
            f"💡 You can now ask the **DataPAI RAG** pipeline: "
            f"`What did {ticker} say about their outlook?`",
        ]
        return "\n".join(lines)

    def _handle_signal(self, ticker: str) -> str:
        """
        Generate an AI trading signal for the latest ASX announcement.

        Calls POST /v1/asx/signal on the RAG API backend, which:
          1. Fetches the latest announcement PDF for the ticker
          2. Retrieves live multi-timeframe OHLCV data via yfinance
          3. Calculates RSI, MACD, Bollinger Bands, EMAs (pure pandas)
          4. Runs Gemini flash-lite → GPT-5.1 reviewer LLM chain
          5. Returns a structured signal with entry/target/stop-loss per timeframe

        ⚠️ Output is AI-generated for informational purposes ONLY.
           NOT financial advice. Prominent disclaimers are embedded in the signal.
        """
        try:
            resp = requests.post(
                f"{self.valves.DATAPAI_RAG_API_URL}/v1/asx/signal",
                json={"ticker": ticker, "max_doc_chars": 6000},
                headers=self._api_headers(),
                timeout=self.valves.ASX_SIGNAL_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.ConnectionError:
            return (
                f"⚠️ DataPAI RAG API not reachable at `{self.valves.DATAPAI_RAG_API_URL}`.\n"
                f"Please ensure the service is running on EC2 #2."
            )
        except Exception as exc:
            return f"⚠️ Signal generation failed for **{ticker}**: `{exc}`"

        signal_md = data.get("signal_markdown", "No signal returned.")
        headline  = data.get("headline", "")[:70]
        date      = data.get("date", "")
        tfs       = data.get("timeframes_available", [])
        mkt_sens  = "🔴 Market Sensitive" if data.get("market_sensitive") else ""

        header = [
            f"## 🎯 AI Trading Signal — {ticker} ({date})",
            f"**{headline}** {mkt_sens}",
            "",
            "> ⚠️ **NOT FINANCIAL ADVICE** — AI-generated for informational purposes only.",
            "",
        ]
        footer = [
            "",
            "---",
            f"📊 Price data loaded for: {', '.join(tfs) if tfs else 'none (signal from announcement text only)'}",
            "🤖 LLM chain: Gemini flash-lite → GPT-5.1 reviewer",
            f"💡 Try: `interpret {ticker}` for a pure announcement analysis.",
        ]
        return "\n".join(header) + signal_md + "\n".join(footer)

# agents/sec_filing_agent.py
"""
SEC EDGAR Filing Agent
======================
Fetches, interprets, and generates trading signals from US SEC filings.

Parallel to asx_announcement_agent.py — same pattern, US markets.

Architecture
------------
  get_cik(ticker)                      →  int (SEC CIK number)
  fetch_sec_filings(ticker, ...)       →  List[Dict] of filing metadata
  download_filing_text(cik, acc, doc)  →  str (cleaned plain text)
  interpret_filing(filing, text)       →  Markdown analysis string
  answer_filing_question(...)          →  str (follow-up Q&A)
  generate_us_trading_signal(...)      →  Markdown signal string

Key SEC form types
------------------
  8-K    — Current report: material events, earnings, M&A, mgmt changes.
            Most market-sensitive; equivalent to ASX market-sensitive announcements.
  10-Q   — Quarterly financial report (unaudited)
  10-K   — Annual report (audited, comprehensive)
  DEF 14A— Proxy statement (shareholder meeting / director elections)
  S-1    — IPO registration

API
---
  Uses SEC EDGAR REST API (free, no key required).
  SEC policy requires a descriptive User-Agent with contact email.
  Set SEC_CONTACT_EMAIL env var, or "datapai@example.com" is used as fallback.

Dependencies
------------
  requests, beautifulsoup4, lxml  (already in requirements.txt)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

from agents.technical_analysis import (
    fetch_all_timeframes,
    build_technical_context,
    _format_grounding_sources,
    DEFAULT_TIMEFRAMES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SEC EDGAR API constants
# ---------------------------------------------------------------------------

_SEC_CONTACT = os.environ.get("SEC_CONTACT_EMAIL", "datapai@example.com")

_SEC_HEADERS = {
    # SEC policy: User-Agent must identify the application and a contact email
    "User-Agent": f"DataPAI/1.0 ({_SEC_CONTACT})",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json, text/html, */*",
}

_TICKERS_URL      = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL  = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_ARCHIVES_URL     = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"

# ---------------------------------------------------------------------------
# 8-K item number → human-readable description
# ---------------------------------------------------------------------------

_8K_ITEMS: Dict[str, str] = {
    "1.01": "Entry into Material Agreement",
    "1.02": "Termination of Material Agreement",
    "1.03": "Bankruptcy or Receivership",
    "1.04": "Mine Safety Reporting",
    "2.01": "Completion of Acquisition or Disposition",
    "2.02": "Results of Operations / Financial Condition",   # EARNINGS
    "2.03": "Creation of Direct Financial Obligation",
    "2.04": "Events Triggering Debt Acceleration",
    "2.05": "Costs Associated with Exit / Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modifications to Rights of Security Holders",
    "4.01": "Change in Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financial Statements",  # RESTATEMENT RISK
    "5.01": "Change in Control of Registrant",
    "5.02": "Departure / Election of Directors or Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.04": "Temporary Suspension of Trading Under Employee Benefit Plans",
    "5.05": "Amendments to Code of Ethics",
    "5.06": "Change in Shell Company Status",
    "5.07": "Submission of Matters to Vote of Security Holders",
    "5.08": "Shareholder Director Nominations",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}

# Items that make an 8-K market-sensitive (material to price)
_SENSITIVE_8K_ITEMS = {
    "1.01",  # Material agreement (M&A, JV, major contract)
    "1.02",  # Termination of material agreement
    "2.01",  # Acquisition / disposition
    "2.02",  # Earnings / results of operations
    "2.03",  # New debt obligation
    "2.04",  # Debt acceleration
    "2.05",  # Exit costs / restructuring
    "2.06",  # Material impairments / write-downs
    "4.01",  # Auditor change
    "4.02",  # Financial restatement risk
    "5.01",  # Change of control / takeover
}

# Default form types to fetch
DEFAULT_FORM_TYPES = ("8-K", "10-Q", "10-K")

# ---------------------------------------------------------------------------
# CIK lookup  (in-memory cache; loaded from SEC on first call)
# ---------------------------------------------------------------------------

_CIK_MAP: Dict[str, int] = {}   # ticker → CIK (upper-cased)


def _load_cik_map() -> None:
    """
    Populate _CIK_MAP from the SEC bulk company_tickers.json file.
    Called lazily on the first get_cik() call; subsequent calls are no-ops.
    The file is ~2 MB but only fetched once per process lifetime.
    """
    global _CIK_MAP
    if _CIK_MAP:
        return
    try:
        resp = requests.get(_TICKERS_URL, headers=_SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        for entry in resp.json().values():
            ticker = (entry.get("ticker") or "").upper()
            cik    = entry.get("cik_str")
            if ticker and cik:
                _CIK_MAP[ticker] = int(cik)
        logger.info("SEC ticker map loaded: %d entries", len(_CIK_MAP))
    except Exception as exc:
        logger.error("Failed to load SEC ticker map: %s", exc)


def get_cik(ticker: str) -> Optional[int]:
    """
    Look up the SEC CIK (Central Index Key) for a US ticker symbol.

    Parameters
    ----------
    ticker : NYSE/NASDAQ ticker symbol (e.g. "AAPL", "MSFT", "NVDA")

    Returns
    -------
    int CIK, or None if not found (e.g. non-US listed stock).
    """
    _load_cik_map()
    return _CIK_MAP.get(ticker.upper().strip())


# ---------------------------------------------------------------------------
# Headline and sensitivity helpers
# ---------------------------------------------------------------------------

def _build_8k_headline(items: List[str]) -> str:
    """Convert 8-K item numbers into a human-readable headline string."""
    # Exclude the boilerplate "9.01 Financial Statements and Exhibits"
    meaningful = [i for i in items if i != "9.01"]
    if not meaningful:
        return "Current Report (8-K)"
    descs = [_8K_ITEMS.get(i, f"Item {i}") for i in meaningful[:3]]
    return " / ".join(descs)


def _is_market_sensitive(form_type: str, items: List[str]) -> bool:
    """Heuristic: is this filing likely to move the stock price?"""
    if form_type in ("10-K", "10-Q"):
        return True
    if form_type == "8-K":
        return any(item in _SENSITIVE_8K_ITEMS for item in items)
    return False


def _form_headline(form_type: str, items: List[str]) -> str:
    """Build a filing headline from form type + items."""
    if form_type == "8-K":
        return _build_8k_headline(items)
    elif form_type == "10-Q":
        return "Quarterly Report (10-Q)"
    elif form_type == "10-K":
        return "Annual Report (10-K)"
    elif form_type == "DEF 14A":
        return "Proxy Statement — Annual Meeting (DEF 14A)"
    elif form_type == "S-1":
        return "IPO Registration Statement (S-1)"
    return form_type


# ---------------------------------------------------------------------------
# Filing fetch
# ---------------------------------------------------------------------------

def fetch_sec_filings(
    ticker: str,
    count: int = 20,
    form_types: tuple = DEFAULT_FORM_TYPES,
) -> List[Dict[str, Any]]:
    """
    Fetch recent SEC filings for a US-listed ticker via EDGAR.

    Parameters
    ----------
    ticker     : US stock ticker (e.g. "AAPL", "NVDA")
    count      : max filings to return (across all form types)
    form_types : which form types to include (default: 8-K, 10-Q, 10-K)

    Returns
    -------
    List of filing metadata dicts, sorted newest first, with keys:
      ticker, cik, accession, form_type, filed_date, report_date,
      headline, items, market_sensitive, primary_doc, url

    Raises
    ------
    ValueError : if the ticker has no SEC CIK (not a US-listed stock)
    requests.HTTPError : on EDGAR API errors
    """
    ticker = ticker.upper().strip()
    cik    = get_cik(ticker)
    if cik is None:
        raise ValueError(
            f"No SEC CIK found for '{ticker}'. "
            "This ticker may not be listed on a US exchange, or the SEC "
            "ticker database may need refreshing."
        )

    url  = _SUBMISSIONS_URL.format(cik=cik)
    resp = requests.get(url, headers=_SEC_HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # EDGAR submissions.recent is a dict of parallel arrays
    recent      = data.get("filings", {}).get("recent", {})
    accessions  = recent.get("accessionNumber", [])
    forms       = recent.get("form", [])
    filed_dates = recent.get("filingDate", [])
    report_dates= recent.get("reportDate", [])
    docs        = recent.get("primaryDocument", [])
    items_raw   = recent.get("items", [])

    results: List[Dict[str, Any]] = []

    for acc, form, filed, report, doc, items_str in zip(
        accessions, forms, filed_dates, report_dates, docs, items_raw
    ):
        if form not in form_types:
            continue

        items       = [i.strip() for i in str(items_str).split(",") if i.strip()]
        acc_nodash  = acc.replace("-", "")
        filing_url  = _ARCHIVES_URL.format(cik=cik, acc_nodash=acc_nodash, doc=doc)
        headline    = _form_headline(form, items)
        sensitive   = _is_market_sensitive(form, items)

        # Enrich 8-K with item descriptions as tags
        item_labels = [_8K_ITEMS.get(i, i) for i in items if i != "9.01"]

        results.append({
            "ticker":           ticker,
            "cik":              cik,
            "accession":        acc,
            "form_type":        form,
            "filed_date":       filed,
            "report_date":      report or filed,
            "headline":         headline,
            "items":            items,
            "item_labels":      item_labels,
            "market_sensitive": sensitive,
            "primary_doc":      doc,
            "url":              filing_url,
        })

        if len(results) >= count:
            break

    return results


# ---------------------------------------------------------------------------
# Filing text download
# ---------------------------------------------------------------------------

def download_filing_text(
    cik: int,
    accession: str,
    primary_doc: str,
    max_chars: int = 25000,
) -> str:
    """
    Download and extract plain text from an SEC EDGAR filing document.

    Handles both HTML (.htm/.html) and plain text (.txt) documents.
    Strips XBRL tags, scripts, and style blocks for clean LLM input.

    Parameters
    ----------
    cik          : SEC CIK number (int)
    accession    : accession number with dashes (e.g. "0000320193-24-000123")
    primary_doc  : filename of the primary document (e.g. "aapl-20240130.htm")
    max_chars    : truncation limit (default 25 000 chars ≈ ~6 000 tokens)

    Returns
    -------
    Cleaned plain text, truncated to max_chars.
    """
    acc_nodash = accession.replace("-", "")
    url = _ARCHIVES_URL.format(cik=cik, acc_nodash=acc_nodash, doc=primary_doc)

    try:
        resp = requests.get(url, headers=_SEC_HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("SEC filing download failed for %s: %s", url, exc)
        return f"(Filing text unavailable — download failed: {exc})"

    content_type = resp.headers.get("Content-Type", "")

    if "html" in content_type or primary_doc.lower().endswith((".htm", ".html")):
        soup = BeautifulSoup(resp.text, "lxml")
        # Remove non-content tags
        for tag in soup(["script", "style", "ix:header", "ix:nonfraction",
                          "ix:nonnumeric", "head"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
    else:
        text = resp.text

    # Normalise whitespace
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text  = "\n".join(lines)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... filing text truncated to fit context window ...]"

    return text or "(No readable text extracted from this filing)"


# ---------------------------------------------------------------------------
# LLM prompts — US market filing analysis
# ---------------------------------------------------------------------------

_SYSTEM_INTERPRET_US = """\
You are a principal sell-side analyst covering US-listed equities.
Analyse the provided SEC filing and return a structured report.

REQUIRED SECTIONS (all mandatory):

1. **Executive Summary** — 2-3 sentences: what happened and why it matters
2. **Key Financial Figures** — revenue, operating income/EBITDA, net income/EPS (GAAP
   and non-GAAP where reported), free cash flow, balance sheet highlights, guidance
3. **Material Events** — what was disclosed (acquisition, earnings, management change, etc.)
4. **Market Impact Assessment** — likely stock price reaction and rationale
5. **Regulatory / Legal Flags** — restatement risk, going-concern language, SEC inquiries
6. **Forward Outlook** — company guidance vs. analyst consensus (if mentioned)
7. **Risk Factors** — key risks cited or implied

Formatting rules:
- Report all figures in USD
- Distinguish GAAP vs non-GAAP clearly
- For 10-Q/10-K: compare YoY and QoQ where data is present
- For 8-K item 4.02 (non-reliance): flag "⚠️ RESTATEMENT RISK" prominently
- For 8-K item 5.01 (change of control): flag "⚠️ CHANGE OF CONTROL"
- Always end with: "⚠️ AI-generated analysis — NOT investment advice."
"""

_SYSTEM_QA_US = """\
You are a financial analyst assistant answering questions about a specific SEC filing.
You have the filing text and metadata available.
Answer concisely and precisely, citing specific figures from the filing where possible.
Use USD for all monetary values.
Always end your answer with: "⚠️ AI analysis — NOT investment advice."
"""

_SYSTEM_SIGNAL_US = """\
You are a quantitative analyst generating a structured US equity trading signal.
You have an SEC filing AND live multi-timeframe technical indicators.
Synthesise both into a structured signal.

MANDATORY DISCLAIMER — must appear verbatim at TOP and BOTTOM:

> ⚠️ **AI-GENERATED ANALYSIS — NOT FINANCIAL ADVICE**
> This signal is produced by an AI model for informational and educational
> purposes only. It does NOT constitute financial advice, a recommendation to
> buy or sell, or an offer of any financial product. Past price behaviour does
> not predict future results. Trading involves significant risk of capital loss.
> Always conduct your own research and consult a licensed financial adviser
> before making any investment decision.

Rules:
- Never use "I recommend" or definitive action language
- Use hedged language: "suggests", "indicates", "may", "could"
- All prices in USD
- For "DATA UNAVAILABLE" timeframes: write N/A, do not fabricate price levels

Signal definitions:
  STRONG BUY   — Positive/beat filing + ≥ 2 bullish indicators (RSI < 50, MACD bullish, price > EMA21)
  BUY          — Positive filing OR bullish technicals (≥ 2 of 3 above)
  HOLD/NEUTRAL — Mixed signals
  SELL         — Negative filing OR ≥ 2 bearish indicators
  STRONG SELL  — Negative filing + ≥ 2 bearish indicators
  NOT MATERIAL — Filing is administrative (proxy, routine 8-K item 9.01 only)

Stop-loss sizing:
  5min  : 0.1–0.3% from entry
  30min : 0.3–0.7% from entry
  1h    : 0.5–1.5% from entry
  Daily : 1.0–3.0% from entry

Respond with EXACTLY this structure:

> ⚠️ **AI-GENERATED ANALYSIS — NOT FINANCIAL ADVICE**
> ... (full disclaimer block above)

---

## 🎯 US Equity Signal: {TICKER} ({EXCHANGE})

**Filing**: {FORM_TYPE} — {HEADLINE}
**Filed**: {DATE}
**Market Sensitive**: [Yes / No]

**Signal**: [STRONG BUY / BUY / HOLD/NEUTRAL / SELL / STRONG SELL / NOT MATERIAL]
**Conviction**: [High / Medium / Low]
**Primary Driver**: [Filing / Technicals / Both / Neither]

**Signal Rationale** *(2–3 sentences)*: ...

---

### Per-Timeframe Trade Levels (USD)

| Timeframe | Trend | Entry Zone | Price Target | Stop-Loss | Risk/Reward | Notes |
|-----------|-------|------------|--------------|-----------|-------------|-------|
| 5-Minute  | ...   | $X.XX–$X.XX | $X.XX      | $X.XX    | 1:X.X       | ...   |
| 30-Minute | ...   | $X.XX–$X.XX | $X.XX      | $X.XX    | 1:X.X       | ...   |
| 1-Hour    | ...   | $X.XX–$X.XX | $X.XX      | $X.XX    | 1:X.X       | ...   |
| Daily     | ...   | $X.XX–$X.XX | $X.XX      | $X.XX    | 1:X.X       | ...   |

*All prices in USD. Risk/Reward = (Target − Entry) / (Entry − Stop-Loss)*

---

### Technical Summary
- **5min**: [RSI / MACD / trend — one line]
- **30min**: [...]
- **1h**: [...]
- **Daily**: [...]

---

### Filing Impact Assessment
[2–3 sentences on how the filing content affects near-term price outlook]

---

### Key Risks to This Signal
- [Risk 1]
- [Risk 2]
- [Always include: "AI model errors or hallucinations in indicator interpretation"]

---

> ⚠️ **DISCLAIMER**: This AI-generated signal is for informational and educational
> purposes only. DataPAI and its operators accept NO responsibility for any trading
> decisions based on this output. NOT financial advice. Consult a licensed financial
> adviser before trading.
"""

_SYSTEM_REVIEWER_US = """\
Review this AI-generated US equity trading signal for correctness and compliance.

Check:
1. Disclaimer at BOTH top AND bottom.
2. For BUY signals: Entry Zone < Target, Stop-Loss < Entry.
   For SELL signals: Entry Zone > Target, Stop-Loss > Entry.
3. Risk/Reward ≥ 1:1.5 (reward at least 1.5× the risk).
4. No "DATA UNAVAILABLE" timeframe has fabricated price levels.
5. No definitive instruction language ("I recommend", "you should buy").
6. All prices are in USD, not AUD.

Reply EXACTLY "APPROVED" if all checks pass, otherwise reply with
the fully corrected signal (complete markdown, no preamble).
"""


# ---------------------------------------------------------------------------
# Filing interpretation (LLM)
# ---------------------------------------------------------------------------

def interpret_filing(
    filing: Dict[str, Any],
    text: str,
    question: Optional[str] = None,
    use_grounding: bool = True,
) -> str:
    """
    Generate a structured LLM analysis of an SEC filing.

    Parameters
    ----------
    filing       : filing metadata dict from fetch_sec_filings()
    text         : extracted filing text from download_filing_text()
    question     : optional specific question to focus on
    use_grounding: enable Gemini Google Search for real-time analyst context

    Returns
    -------
    Markdown analysis string. Never raises.
    """
    from agents.llm_client import GoogleChatClient, OpenAIChatClient

    ticker    = filing.get("ticker", "UNKNOWN")
    form_type = filing.get("form_type", "")
    headline  = filing.get("headline", "")
    filed     = filing.get("filed_date", "")
    sensitive = filing.get("market_sensitive", False)

    user_content = (
        f"Company: {ticker}\n"
        f"Form: {form_type}  |  Filed: {filed}\n"
        f"Headline: {headline}\n"
        f"Market Sensitive: {'YES' if sensitive else 'NO'}\n\n"
        f"{'─' * 60}\n\n"
        f"{text}"
    )
    if question:
        user_content += f"\n\n{'─' * 60}\nSpecific question: {question}"

    messages = [
        {"role": "system", "content": _SYSTEM_INTERPRET_US},
        {"role": "user",   "content": user_content},
    ]

    try:
        gemini = GoogleChatClient()
        resp   = gemini.chat(messages, temperature=0.1, grounding=use_grounding)
        result = resp.get("content", "")

        sources = resp.get("grounding_sources", [])
        queries = resp.get("web_search_queries", [])
        if sources:
            result += _format_grounding_sources(sources, queries)
            logger.info("Gemini grounding: %d sources for %s filing", len(sources), ticker)

        return result or "(Gemini returned empty — please retry)"
    except Exception as exc:
        logger.warning("Gemini interpretation failed, trying GPT: %s", exc)
        try:
            gpt = OpenAIChatClient()
            return gpt.chat(messages, temperature=0.1).get("content", "⚠️ Interpretation failed.")
        except Exception as exc2:
            return f"⚠️ Interpretation failed: Gemini: {exc}  |  GPT: {exc2}"


# ---------------------------------------------------------------------------
# Filing Q&A chat
# ---------------------------------------------------------------------------

def answer_filing_question(
    filing: Dict[str, Any],
    text: str,
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Answer a follow-up question about a specific SEC filing.

    Parameters
    ----------
    filing       : filing metadata dict
    text         : extracted filing text
    question     : user's question
    chat_history : prior messages in [{"role": ..., "content": ...}] format

    Returns
    -------
    Answer string. Never raises.
    """
    from agents.llm_client import RouterChatClient

    ticker = filing.get("ticker", "UNKNOWN")
    form   = filing.get("form_type", "")

    context_msg = (
        f"SEC Filing context:\n"
        f"Ticker: {ticker}  Form: {form}  Filed: {filing.get('filed_date', '')}\n"
        f"Headline: {filing.get('headline', '')}\n\n"
        f"{'─' * 60}\n\n"
        f"{text[:8000]}"   # keep context window manageable
    )

    messages = [
        {"role": "system", "content": _SYSTEM_QA_US},
        {"role": "user",   "content": context_msg},
    ]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": question})

    try:
        client = RouterChatClient()
        return client.chat(messages, temperature=0.1).get("content", "⚠️ No response.")
    except Exception as exc:
        return f"⚠️ Q&A failed: {exc}"


# ---------------------------------------------------------------------------
# US trading signal
# ---------------------------------------------------------------------------

def generate_us_trading_signal(
    filing: Dict[str, Any],
    text: str,
    indicators_by_tf: Optional[Dict[str, Any]] = None,
    ticker_suffix: str = "",
    max_doc_chars: int = 6000,
    use_grounding: bool = True,
) -> str:
    """
    Generate a structured US equity trading signal via two-step LLM chain.

    Combines SEC filing content with live multi-timeframe technical indicators.
    For a standalone technical signal (no filing), use:
        agents.technical_analysis.generate_technical_signal()

    DataPAI hybrid approach
    -----------------------
    1. We compute exact RSI/MACD/BB/EMA indicators from live OHLCV (Yahoo/Polygon).
    2. We extract filing text from SEC EDGAR.
    3. Gemini interprets both + (with grounding) adds real-time analyst consensus.
    4. GPT reviews for compliance and arithmetic correctness.

    Parameters
    ----------
    filing          : filing metadata from fetch_sec_filings()
    text            : filing plain text
    indicators_by_tf: pre-computed indicators; if None, fetched via yfinance
    ticker_suffix   : exchange suffix for price data ("" for NYSE/NASDAQ)
    max_doc_chars   : truncation limit for filing text
    use_grounding   : enable Gemini Google Search grounding (default True)

    Returns
    -------
    Formatted Markdown signal with disclaimer at top and bottom.
    """
    from agents.llm_client import GoogleChatClient, OpenAIChatClient

    ticker    = filing.get("ticker", "UNKNOWN")
    form_type = filing.get("form_type", "")
    headline  = filing.get("headline", "")
    filed     = filing.get("filed_date", "")
    sensitive = filing.get("market_sensitive", False)

    # Fetch price data if not pre-supplied
    if indicators_by_tf is None:
        indicators_by_tf = fetch_all_timeframes(
            ticker, suffix=ticker_suffix, timeframes=DEFAULT_TIMEFRAMES
        )

    # Build technical context
    available = [tf for tf, v in (indicators_by_tf or {}).items() if v is not None]
    if available and indicators_by_tf:
        tech_context = build_technical_context(ticker, indicators_by_tf, suffix=ticker_suffix)
    else:
        tech_context = (
            f"PRICE DATA UNAVAILABLE: could not retrieve OHLCV data for {ticker}"
            f"{ticker_suffix} across any timeframe. "
            "Mark all timeframe rows as N/A in the signal."
        )

    # Truncate filing text
    text_body = text[:max_doc_chars]
    if len(text) > max_doc_chars:
        text_body += "\n\n[... filing text truncated to fit context window ...]"

    exchange_label = (
        "NYSE / NASDAQ" if ticker_suffix in ("", ".US") else
        f"Suffix: {ticker_suffix}"
    )

    user_msg = (
        f"US SEC Filing\n"
        f"Ticker: {ticker}  |  Exchange: {exchange_label}  |  Form: {form_type}\n"
        f"Filed: {filed}  |  Headline: {headline}\n"
        f"Market Sensitive: {'YES' if sensitive else 'NO'}\n\n"
        f"{'─' * 60}\n"
        f"{text_body}\n"
        f"{'─' * 60}\n\n"
        f"{tech_context}"
    )

    messages = [
        {"role": "system", "content": _SYSTEM_SIGNAL_US},
        {"role": "user",   "content": user_msg},
    ]

    # Step 1: Gemini primary (with grounding for live news / analyst data)
    try:
        gemini      = GoogleChatClient()
        gemini_resp = gemini.chat(messages, temperature=0.2, grounding=use_grounding)
        draft       = gemini_resp.get("content", "")

        sources = gemini_resp.get("grounding_sources", [])
        queries = gemini_resp.get("web_search_queries", [])
        if sources:
            draft += _format_grounding_sources(sources, queries)
            logger.info("Gemini grounding: %d sources for US signal %s", len(sources), ticker)

    except Exception as exc:
        logger.warning("Gemini US signal failed, falling back to GPT: %s", exc)
        try:
            gpt = OpenAIChatClient()
            return gpt.chat(messages, temperature=0.2).get(
                "content", "⚠️ Signal generation failed — no LLM response."
            )
        except Exception as exc2:
            return f"⚠️ US signal failed: Gemini: {exc}  |  GPT: {exc2}"

    if not draft.strip():
        draft = "(Gemini returned an empty signal — please retry)"

    # Step 2: GPT reviewer (compliance + arithmetic gate)
    try:
        reviewer_msgs = [
            {"role": "system", "content": _SYSTEM_REVIEWER_US},
            {
                "role": "user",
                "content": (
                    f"Source filing context:\n{user_msg[:3000]}\n\n"
                    f"{'═' * 60}\n\n"
                    f"Draft signal to review:\n{draft}"
                ),
            },
        ]
        gpt      = OpenAIChatClient()
        reviewed = gpt.chat(reviewer_msgs, temperature=0.1).get("content", "").strip()

        if reviewed.upper().startswith("APPROVED"):
            logger.debug("US signal reviewer: APPROVED")
            return draft

        logger.debug("US signal reviewer: REWRITTEN by GPT")
        return reviewed if reviewed else draft

    except Exception as exc:
        logger.warning("GPT reviewer failed for US signal, returning Gemini draft: %s", exc)
        return draft

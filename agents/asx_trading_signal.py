# agents/asx_trading_signal.py
"""
ASX Trading Signal Agent
========================
Combines an ASX market announcement with live multi-timeframe technical
analysis to produce a structured AI trading signal.

⚠️  IMPORTANT: All output from this module is for INFORMATIONAL AND
    EDUCATIONAL PURPOSES ONLY.  It is NOT financial advice.  Always consult
    a licensed financial adviser before making investment decisions.

Architecture
------------
1. Price / indicator data comes from agents.technical_analysis — the
   dedicated, exchange-agnostic pricing module (yfinance-backed).
   ASX tickers are automatically suffixed with ".AX" when calling that module.
2. This module owns ONLY the announcement-aware LLM prompts and the
   generate_trading_signal() entry point.
3. For a pure price signal (no announcement), use
   agents.technical_analysis.generate_technical_signal() directly.

Dependencies
------------
  agents.technical_analysis   (OHLCV fetch, indicators, context formatter)
  agents.llm_client           (GoogleChatClient, OpenAIChatClient)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

# ── All price / indicator logic lives in technical_analysis ──────────────────
from agents.technical_analysis import (
    fetch_all_timeframes,          # re-exported for callers that import from here
    build_technical_context,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM prompts — announcement-aware signal
# ---------------------------------------------------------------------------

_SIGNAL_SYSTEM = """\
You are a quantitative analyst generating a structured ASX trading signal.
You have been given an ASX market announcement AND live multi-timeframe technical
indicators. Your job is to synthesise both into a structured signal report.

MANDATORY DISCLAIMER RULES (non-negotiable):
1. The disclaimer block MUST appear verbatim at the TOP of your response.
2. The disclaimer block MUST appear verbatim at the BOTTOM of your response.
3. Never use the phrase "I recommend" or imply a definitive course of action.
4. Always use hedged language: "suggests", "indicates", "may", "could".

Signal definitions (apply these consistently):
  STRONG BUY   — Very positive market-sensitive announcement + ≥2 bullish indicators
                 (RSI < 45, MACD bullish cross, price above EMA21)
  BUY          — Positive announcement OR bullish technicals (at least 2 of 3 above)
  HOLD/NEUTRAL — Neutral/administrative announcement OR mixed technical signals
  SELL         — Negative/cautious announcement OR ≥2 bearish indicators
                 (RSI > 60 at resistance, MACD bearish, price below EMA21)
  STRONG SELL  — Very negative market-sensitive announcement + ≥2 bearish indicators
  NOT RELATED  — Purely administrative announcement with no material price impact
                 (e.g. director interest notice, substantial holder notice, governance)

Stop-loss sizing guidelines (put these in the Notes column):
  5min  trades: stop-loss 0.1–0.3% from entry
  30min trades: stop-loss 0.3–0.7% from entry
  1h    trades: stop-loss 0.5–1.5% from entry
  Daily trades: stop-loss 1.0–3.0% from entry

For "DATA UNAVAILABLE" timeframes: write "N/A" in all price columns and
"Insufficient price data" in the Notes column. Do NOT fabricate price levels.

Respond with this EXACT markdown structure (no extra sections):

> ⚠️ **AI-GENERATED ANALYSIS — NOT FINANCIAL ADVICE**
> This signal is produced by an AI model for informational and educational
> purposes only. It does NOT constitute financial advice, a recommendation to
> buy or sell, or an offer of any financial product. Past price behaviour does
> not predict future results. Trading involves significant risk of capital loss.
> Always conduct your own research and consult a licensed financial adviser
> before making any investment decision.

---

## 🎯 Trading Signal: {TICKER}

**Price Sensitivity**: [Market Sensitive / Potentially Market Sensitive / Not Market Sensitive]
*Rationale: one concise sentence.*

**Signal**: [STRONG BUY / BUY / HOLD/NEUTRAL / SELL / STRONG SELL / NOT RELATED]
**Conviction**: [High / Medium / Low]
**Primary Driver**: [Announcement / Technicals / Both / Neither]

**Signal Rationale** *(2–3 sentences)*: ...

---

### Per-Timeframe Trade Levels

| Timeframe | Trend | Entry Zone | Price Target | Stop-Loss | Notes |
|-----------|-------|------------|--------------|-----------|-------|
| 5-Minute  | ...   | $X.XX–$X.XX | $X.XX      | $X.XX    | ...   |
| 30-Minute | ...   | $X.XX–$X.XX | $X.XX      | $X.XX    | ...   |
| 1-Hour    | ...   | $X.XX–$X.XX | $X.XX      | $X.XX    | ...   |
| Daily     | ...   | $X.XX–$X.XX | $X.XX      | $X.XX    | ...   |

*Entry Zone = the price range where a position could be considered.*
*Price Target = realistic near-term target based on technicals + announcement catalyst.*
*Stop-Loss = the level at which the thesis is invalidated. Strictly respect your stop.*
*All prices in AUD.*

---

### Technical Summary

- **5min**: [RSI / MACD / trend reading — one line]
- **30min**: [...]
- **1h**: [...]
- **Daily**: [...]

---

### Announcement Impact Assessment

[2–3 sentences on how this announcement's content, sentiment, and materiality
affects the near-term price outlook. Reference specific figures or events if present.]

---

### Key Risks to This Signal

- [Risk 1]
- [Risk 2]
- [Risk 3 — always include "AI model errors or hallucinations in indicator interpretation"]

---

> ⚠️ **DISCLAIMER**: This AI-generated signal is for informational and educational
> purposes only. DataPAI and its operators accept NO responsibility for any trading
> decisions made based on this output. It is NOT financial advice. Consult a licensed
> financial adviser before trading.
"""

_SIGNAL_REVIEWER_SYSTEM = """\
You are reviewing an AI-generated ASX trading signal for correctness and compliance.

Check ALL of the following:
1. Disclaimer appears at BOTH top AND bottom of the response.
2. Signal (STRONG BUY / BUY / HOLD/NEUTRAL / SELL / STRONG SELL / NOT RELATED)
   is logically consistent with the indicators and announcement sentiment.
3. For BUY signals: Entry Zone < Price Target, Stop-Loss < Entry Zone.
   For SELL signals: Entry Zone > Price Target, Stop-Loss > Entry Zone.
4. No timeframe with "DATA UNAVAILABLE" has fabricated price levels.
5. No definitive recommendation language ("I recommend", "you should buy") is used.
6. All prices are in AUD (no USD symbols).

If all checks pass, reply EXACTLY:
  APPROVED

If any check fails, reply with the fully corrected signal (complete markdown, no preamble).
"""


# ---------------------------------------------------------------------------
# Main entry point — announcement-aware signal generation
# ---------------------------------------------------------------------------

def generate_trading_signal(
    announcement: Dict[str, Any],
    pdf_text: str,
    indicators_by_tf: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
    max_doc_chars: int = 6000,
) -> str:
    """
    Generate a structured ASX trading signal via a two-step LLM chain.

    Combines an ASX announcement with live multi-timeframe technical indicators.
    For a standalone technical signal (no announcement), use:
        agents.technical_analysis.generate_technical_signal()

    Step 1 — Gemini flash-lite: fast structured signal generation (~2–4 s)
    Step 2 — GPT-5.1 reviewer:  checks consistency, disclaimer presence (~3–8 s)

    Parameters
    ----------
    announcement    : metadata dict from fetch_asx_announcements()
    pdf_text        : extracted PDF text
    indicators_by_tf: result of fetch_all_timeframes(), or None to skip price data.
                      If None is passed, signal is based on announcement text only.
    max_doc_chars   : truncation limit for the announcement text

    Returns
    -------
    Formatted Markdown string with disclaimer blocks at top and bottom.
    Returns a user-facing error string (never raises) if all LLMs fail.
    """
    from agents.llm_client import GoogleChatClient, OpenAIChatClient

    ticker   = announcement.get("ticker", "UNKNOWN")
    doc_date = (announcement.get("document_date") or "")[:10]
    headline = announcement.get("headline", "")
    doc_type = announcement.get("doc_type", "")
    mkt_sens = announcement.get("market_sensitive", False)

    # Truncate announcement text to fit context window
    text_body = (pdf_text or "")[:max_doc_chars]
    if len(pdf_text or "") > max_doc_chars:
        text_body += "\n\n[... announcement text truncated to fit context window ...]"
    if not text_body.strip():
        text_body = "(No readable text extracted — this may be an image-only PDF)"

    # Build technical context from the shared module
    if indicators_by_tf is None:
        tech_context = "(No price data requested — signal based on announcement text only)"
    else:
        available = [tf for tf, v in indicators_by_tf.items() if v is not None]
        if available:
            # Pass suffix=".AX" since this is the ASX-specific signal module
            tech_context = build_technical_context(ticker, indicators_by_tf, suffix=".AX")
        else:
            tech_context = (
                "PRICE DATA UNAVAILABLE: yfinance could not retrieve OHLCV data "
                f"for {ticker}.AX across any timeframe. This may be due to a network "
                "issue or a thinly-traded / recently-listed stock. Signal must be "
                "based on announcement text alone. Mark all timeframe rows as N/A."
            )

    # Assemble user message
    user_msg = (
        f"ASX Announcement\n"
        f"Ticker: {ticker}  |  Date: {doc_date}  |  Type: {doc_type}\n"
        f"Headline: {headline}\n"
        f"Market Sensitive: {'YES' if mkt_sens else 'NO'}\n\n"
        f"{'─' * 60}\n"
        f"{text_body}\n"
        f"{'─' * 60}\n\n"
        f"{tech_context}"
    )

    messages = [
        {"role": "system", "content": _SIGNAL_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]

    # ── Step 1: Gemini flash-lite (primary, fast) ────────────────────────────
    try:
        gemini = GoogleChatClient()
        draft  = gemini.chat(messages, temperature=0.2).get("content", "")
    except Exception as exc:
        logger.warning("Gemini signal generation failed, falling back to GPT: %s", exc)
        try:
            gpt = OpenAIChatClient()
            return gpt.chat(messages, temperature=0.2).get(
                "content", "⚠️ Signal generation failed — no LLM response."
            )
        except Exception as exc2:
            return (
                f"⚠️ Trading signal generation failed: both Gemini and GPT are unavailable.\n"
                f"Errors: Gemini: {exc}  |  GPT: {exc2}"
            )

    if not draft.strip():
        draft = "(Gemini returned an empty signal — please try again)"

    # ── Step 2: GPT-5.1 reviewer (quality + compliance gate) ─────────────────
    try:
        reviewer_msgs = [
            {"role": "system", "content": _SIGNAL_REVIEWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Source announcement:\n{user_msg[:3000]}\n\n"
                    f"{'═' * 60}\n\n"
                    f"Draft trading signal to review:\n{draft}"
                ),
            },
        ]
        gpt      = OpenAIChatClient()
        reviewed = gpt.chat(reviewer_msgs, temperature=0.1).get("content", "").strip()

        if reviewed.upper().startswith("APPROVED"):
            logger.debug("ASX signal reviewer: APPROVED — returning Gemini draft")
            return draft

        logger.debug("ASX signal reviewer: REWRITTEN by GPT")
        return reviewed if reviewed else draft

    except Exception as exc:
        logger.warning("GPT reviewer failed for signal, returning Gemini draft: %s", exc)
        return draft

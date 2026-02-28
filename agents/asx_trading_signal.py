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
1. Fetch OHLCV bars for 5m / 30m / 1h / daily from Yahoo Finance (yfinance).
   ASX tickers are automatically suffixed with ".AX".
2. Calculate RSI, MACD, Bollinger Bands, EMA(9/21/50/200) using pure pandas
   — no ta-lib compilation required.
3. Combine the announcement text + technical summary → Gemini flash-lite drafts
   a structured signal → GPT-5.1 reviews for consistency and completeness.
4. Return formatted Markdown with prominent disclaimer blocks.

yfinance data limits
--------------------
  5m  data: last 60 days only (Yahoo Finance constraint)
  30m data: last 60 days only
  1h  data: last 730 days
  1d  data: years of history

Dependencies
------------
  pip install yfinance  (added to requirements.txt)
  pandas                (already present)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeframe configuration
# ---------------------------------------------------------------------------

_TF_CONFIG: Dict[str, Dict[str, Any]] = {
    "5m":  {"interval": "5m",  "period": "5d",  "bars": 120, "label": "5-Minute"},
    "30m": {"interval": "30m", "period": "30d", "bars": 120, "label": "30-Minute"},
    "1h":  {"interval": "1h",  "period": "60d", "bars": 120, "label": "1-Hour"},
    "1d":  {"interval": "1d",  "period": "2y",  "bars": 200, "label": "Daily"},
}

# ---------------------------------------------------------------------------
# OHLCV fetch
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    ticker: str,
    timeframe: str = "1d",
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV bars for an ASX ticker via yfinance.

    The .AX suffix is added internally — pass bare tickers like "BHP", "CBA".
    Returns a DataFrame with columns [Open, High, Low, Close, Volume],
    or None if the fetch fails for any reason (network, unknown ticker, etc.)
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed.  Run: pip install yfinance")
        return None

    cfg   = _TF_CONFIG.get(timeframe, _TF_CONFIG["1d"])
    yf_ticker = f"{ticker.upper().strip()}.AX"

    try:
        df = yf.download(
            yf_ticker,
            interval   = cfg["interval"],
            period     = cfg["period"],
            auto_adjust= True,
            prepost    = False,     # exclude pre/post market for intraday
            progress   = False,
            threads    = False,
        )
    except Exception as exc:
        logger.warning("yfinance download failed for %s [%s]: %s", yf_ticker, timeframe, exc)
        return None

    if df is None or df.empty:
        logger.info("yfinance returned empty DataFrame for %s [%s]", yf_ticker, timeframe)
        return None

    # Flatten MultiIndex columns (yfinance >= 0.2 may return them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only OHLCV columns
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        logger.warning("yfinance response missing columns %s for %s", missing, yf_ticker)
        return None

    df = df[required].dropna(subset=["Close"])

    if len(df) < 15:
        logger.info("Insufficient bars (%d) for %s [%s]", len(df), yf_ticker, timeframe)
        return None

    return df.tail(cfg["bars"])

# ---------------------------------------------------------------------------
# Pure-pandas technical indicators
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast    = close.ewm(span=fast,   adjust=False).mean()
    ema_slow    = close.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    middle = close.rolling(period).mean()
    std    = close.rolling(period).std()
    upper  = middle + std_dev * std
    lower  = middle - std_dev * std
    return upper, middle, lower


def _safe_last(series: pd.Series) -> Optional[float]:
    """Return the last non-NaN value of a Series, or None."""
    valid = series.dropna()
    return float(valid.iloc[-1]) if not valid.empty else None


def calc_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute RSI, MACD, Bollinger Bands, and EMAs from an OHLCV DataFrame.
    Returns None for any indicator that cannot be calculated (insufficient bars).
    """
    close = df["Close"]
    n     = len(df)

    # Current and previous prices
    current_price = float(close.iloc[-1])
    prev_close    = float(close.iloc[-2]) if n >= 2 else None
    change_pct    = (
        round((current_price - prev_close) / prev_close * 100, 2)
        if prev_close else None
    )

    # RSI
    rsi_val = _safe_last(_rsi(close)) if n >= 14 else None
    rsi_label = (
        "OVERBOUGHT" if rsi_val and rsi_val > 70 else
        "OVERSOLD"   if rsi_val and rsi_val < 30 else
        "NEUTRAL"    if rsi_val else None
    )

    # MACD
    macd_line_s, macd_sig_s, macd_hist_s = _macd(close)
    macd_line = _safe_last(macd_line_s) if n >= 26 else None
    macd_sig  = _safe_last(macd_sig_s)  if n >= 26 else None
    macd_hist = _safe_last(macd_hist_s) if n >= 26 else None
    macd_label = (
        "BULLISH" if macd_line and macd_line > 0 else
        "BEARISH" if macd_line and macd_line < 0 else None
    )

    # Bollinger Bands
    bb_upper_s, bb_mid_s, bb_lower_s = _bollinger(close)
    bb_upper = _safe_last(bb_upper_s) if n >= 20 else None
    bb_mid   = _safe_last(bb_mid_s)   if n >= 20 else None
    bb_lower = _safe_last(bb_lower_s) if n >= 20 else None
    bb_pct   = None
    bb_label = None
    if bb_upper and bb_lower and bb_upper != bb_lower:
        bb_pct = round((current_price - bb_lower) / (bb_upper - bb_lower), 3)
        bb_label = (
            "NEAR UPPER BAND" if bb_pct > 0.80 else
            "NEAR LOWER BAND" if bb_pct < 0.20 else
            "MID BAND"
        )

    # EMAs
    ema_9   = _safe_last(close.ewm(span=9,   adjust=False).mean()) if n >= 9   else None
    ema_21  = _safe_last(close.ewm(span=21,  adjust=False).mean()) if n >= 21  else None
    ema_50  = _safe_last(close.ewm(span=50,  adjust=False).mean()) if n >= 50  else None
    ema_200 = _safe_last(close.ewm(span=200, adjust=False).mean()) if n >= 200 else None

    # Trend
    if ema_21 and ema_50:
        if current_price > ema_21 and ema_21 > ema_50:
            trend = "UPTREND"
        elif current_price < ema_21 and ema_21 < ema_50:
            trend = "DOWNTREND"
        else:
            trend = "SIDEWAYS"
    elif prev_close:
        trend = "UPTREND" if current_price > prev_close else "DOWNTREND"
    else:
        trend = "UNKNOWN"

    # Volume
    volume = float(df["Volume"].iloc[-1])
    avg_vol_20 = float(df["Volume"].tail(21).iloc[:-1].mean()) if n >= 21 else None
    vol_ratio  = round(volume / avg_vol_20, 2) if avg_vol_20 and avg_vol_20 > 0 else None

    return {
        "current_price": current_price,
        "prev_close":    prev_close,
        "change_pct":    change_pct,
        "rsi":           round(rsi_val,   2) if rsi_val  else None,
        "rsi_label":     rsi_label,
        "macd_line":     round(macd_line, 5) if macd_line else None,
        "macd_signal":   round(macd_sig,  5) if macd_sig  else None,
        "macd_hist":     round(macd_hist, 5) if macd_hist  else None,
        "macd_label":    macd_label,
        "bb_upper":      round(bb_upper, 4) if bb_upper else None,
        "bb_middle":     round(bb_mid,   4) if bb_mid   else None,
        "bb_lower":      round(bb_lower, 4) if bb_lower else None,
        "bb_pct":        bb_pct,
        "bb_label":      bb_label,
        "ema_9":         round(ema_9,   4) if ema_9   else None,
        "ema_21":        round(ema_21,  4) if ema_21  else None,
        "ema_50":        round(ema_50,  4) if ema_50  else None,
        "ema_200":       round(ema_200, 4) if ema_200 else None,
        "trend":         trend,
        "volume":        int(volume),
        "vol_ratio":     vol_ratio,
    }


# ---------------------------------------------------------------------------
# Multi-timeframe aggregation
# ---------------------------------------------------------------------------

def fetch_all_timeframes(
    ticker: str,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Fetch and calculate indicators for all four timeframes.

    Returns a dict keyed by timeframe code ("5m", "30m", "1h", "1d").
    Values are the `calc_indicators` result dict, or None if data fetch failed.
    Daily is the most reliable and is always attempted; intraday may fail on
    lightly-traded or recently-listed stocks.
    """
    result: Dict[str, Optional[Dict[str, Any]]] = {}
    for tf in ("5m", "30m", "1h", "1d"):
        df = fetch_ohlcv(ticker, tf)
        if df is not None:
            try:
                result[tf] = calc_indicators(df)
            except Exception as exc:
                logger.warning("Indicator calculation failed for %s [%s]: %s", ticker, tf, exc)
                result[tf] = None
        else:
            result[tf] = None
    return result


# ---------------------------------------------------------------------------
# Technical context formatter for LLM injection
# ---------------------------------------------------------------------------

def _fmt(val: Optional[float], prefix: str = "$", decimals: int = 2) -> str:
    if val is None:
        return "N/A"
    return f"{prefix}{val:.{decimals}f}" if prefix else f"{val:.{decimals}f}"


def build_technical_context(
    ticker: str,
    indicators_by_tf: Dict[str, Optional[Dict[str, Any]]],
) -> str:
    """
    Format multi-timeframe technical indicators as a compact string for LLM injection.
    Each timeframe block is ~5 lines; the full output is ~100-300 chars per timeframe.
    """
    lines: List[str] = [f"=== TECHNICAL INDICATORS: {ticker}.AX ==="]
    labels = {"5m": "5-Minute", "30m": "30-Minute", "1h": "1-Hour", "1d": "Daily"}

    for tf, label in labels.items():
        ind = indicators_by_tf.get(tf)
        if ind is None:
            lines.append(f"\n[{label}] DATA UNAVAILABLE (yfinance fetch failed or insufficient history)")
            continue

        p    = ind["current_price"]
        chg  = f"+{ind['change_pct']}%" if ind["change_pct"] and ind["change_pct"] >= 0 else (
               f"{ind['change_pct']}%" if ind["change_pct"] is not None else "N/A")
        lines.append(f"\n[{label}]")
        lines.append(
            f"  Price: ${p:.4f} | Change: {chg} | Trend: {ind['trend']}"
        )
        # RSI
        rsi_str = (
            f"RSI(14): {ind['rsi']} ({ind['rsi_label']})"
            if ind["rsi"] else "RSI(14): N/A (insufficient bars)"
        )
        # MACD
        macd_str = (
            f"MACD: {ind['macd_line']:+.5f} / Sig: {ind['macd_signal']:+.5f} / "
            f"Hist: {ind['macd_hist']:+.5f} ({ind['macd_label']})"
            if ind["macd_line"] else "MACD: N/A"
        )
        # Bollinger
        bb_str = (
            f"BB: Upper ${ind['bb_upper']:.4f} / Mid ${ind['bb_middle']:.4f} / "
            f"Lower ${ind['bb_lower']:.4f} | BB%: {ind['bb_pct']:.2f} ({ind['bb_label']})"
            if ind["bb_upper"] else "Bollinger Bands: N/A"
        )
        # EMAs
        ema_parts = []
        for span, key in [(9, "ema_9"), (21, "ema_21"), (50, "ema_50"), (200, "ema_200")]:
            if ind[key]:
                ema_parts.append(f"EMA{span}: ${ind[key]:.4f}")
        ema_str = " | ".join(ema_parts) if ema_parts else "EMAs: N/A"
        # Volume
        vol_str = (
            f"Vol: {ind['volume']:,} ({ind['vol_ratio']}× avg)"
            if ind["vol_ratio"] else f"Vol: {ind['volume']:,}"
        )

        lines.extend([
            f"  {rsi_str}",
            f"  {macd_str}",
            f"  {bb_str}",
            f"  {ema_str}",
            f"  {vol_str}",
        ])

    lines.append("\n=== END OF TECHNICAL DATA ===")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM prompts
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
# Main signal generation function
# ---------------------------------------------------------------------------

def generate_trading_signal(
    announcement: Dict[str, Any],
    pdf_text: str,
    indicators_by_tf: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
    max_doc_chars: int = 6000,
) -> str:
    """
    Generate a structured ASX trading signal via a two-step LLM chain.

    Step 1 — Gemini flash-lite: fast structured signal generation (~2–4 s)
    Step 2 — GPT-5.1 reviewer:  checks consistency, disclaimer presence (~3–8 s)

    Parameters
    ----------
    announcement    : metadata dict from fetch_asx_announcements()
    pdf_text        : extracted PDF text
    indicators_by_tf: result of fetch_all_timeframes(), or None to skip price data
    max_doc_chars   : truncation limit for the announcement text

    Returns
    -------
    Formatted Markdown string with disclaimer blocks at top and bottom.
    Returns a user-facing error string (not raises) if all LLMs fail.
    """
    from agents.llm_client import GoogleChatClient, OpenAIChatClient

    ticker   = announcement.get("ticker", "UNKNOWN")
    doc_date = (announcement.get("document_date") or "")[:10]
    headline = announcement.get("headline", "")
    doc_type = announcement.get("doc_type", "")
    mkt_sens = announcement.get("market_sensitive", False)

    # Truncate announcement text
    text_body = (pdf_text or "")[:max_doc_chars]
    if len(pdf_text or "") > max_doc_chars:
        text_body += "\n\n[... announcement text truncated to fit context window ...]"
    if not text_body.strip():
        text_body = "(No readable text extracted — this may be an image-only PDF)"

    # Build technical context
    if indicators_by_tf is None:
        tech_context = "(No price data requested)"
    else:
        available = [tf for tf, v in indicators_by_tf.items() if v is not None]
        if available:
            tech_context = build_technical_context(ticker, indicators_by_tf)
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

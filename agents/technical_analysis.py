# agents/technical_analysis.py
"""
Standalone Technical Analysis Module
=====================================
Exchange-agnostic OHLCV fetching, indicator calculation, and AI signal
generation — completely independent of ASX announcements or PDF processing.

Can be called:
  • As a library:  from agents.technical_analysis import fetch_all_timeframes
  • Via REST API:  POST /v1/technical/signal  {"ticker": "BHP", "suffix": ".AX"}
  • Directly:      python -m agents.technical_analysis BHP

Supported exchanges (via data provider suffix):
  .AX  — ASX (Australian Securities Exchange)       e.g. BHP.AX
  ""   — NYSE / NASDAQ                              e.g. AAPL, MSFT
  .L   — London Stock Exchange                      e.g. BP.L
  .TO  — Toronto Stock Exchange (TSX)               e.g. TD.TO
  .HK  — Hong Kong Stock Exchange (HKEX)            e.g. 0700.HK

Data sources (pluggable via `source` parameter):
  "yahoo"   — Yahoo Finance (yfinance).  Free, default.  Best for ASX/US.
  "eodhd"   — EOD Historical Data.  Paid.  Better ASX intraday depth.
  "polygon" — Polygon.io.  Paid.  US stocks only.

Gemini grounding:
  When use_grounding=True (default), Gemini searches Google for real-time news,
  analyst ratings, and market sentiment to supplement our computed indicators.
  This is the DataPAI edge: deterministic math (RSI/MACD/BB/EMAs we compute)
  + live qualitative context (news/sentiment Gemini retrieves).

Architecture
------------
  fetch_ohlcv(ticker, timeframe, suffix, source)   →  pd.DataFrame | None
  calc_indicators(df)                               →  dict of indicators
  fetch_all_timeframes(ticker, suffix, source)      →  {tf: indicator_dict | None}
  build_technical_context(ticker, data)             →  formatted str for LLM injection
  _format_grounding_sources(sources, queries)       →  markdown footnote block
  generate_technical_signal(ticker, ..., source,
                             use_grounding)         →  Markdown signal string

⚠️  All signals are AI-generated for INFORMATIONAL AND EDUCATIONAL PURPOSES
    ONLY.  They are NOT financial advice.

Dependencies
------------
  pip install yfinance    (already in requirements.txt)
  pandas                  (already present)
  agents.data_providers   (pluggable OHLCV provider registry)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeframe configuration — canonical metadata shared by all providers
# ---------------------------------------------------------------------------

# bars  : how many bars to keep after fetch (tail limit)
# label : human-readable label used in context formatter and UI
# interval/period : used by Yahoo provider internally; other providers translate
_TF_CONFIG: Dict[str, Dict[str, Any]] = {
    "5m":  {"interval": "5m",  "period": "5d",  "bars": 120, "label": "5-Minute"},
    "30m": {"interval": "30m", "period": "30d", "bars": 120, "label": "30-Minute"},
    "1h":  {"interval": "1h",  "period": "60d", "bars": 120, "label": "1-Hour"},
    "1d":  {"interval": "1d",  "period": "2y",  "bars": 500, "label": "Daily"},
}

# Default timeframes to fetch when none are specified
DEFAULT_TIMEFRAMES: Tuple[str, ...] = ("5m", "30m", "1h", "1d")

# ---------------------------------------------------------------------------
# OHLCV fetch
# ---------------------------------------------------------------------------

def fetch_ohlcv(
    ticker: str,
    timeframe: str = "1d",
    suffix: str = ".AX",
    source: str = "yahoo",
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV bars for a ticker from the chosen data provider.

    Parameters
    ----------
    ticker    : bare ticker symbol (e.g. "BHP", "AAPL", "BP")
    timeframe : one of "5m" | "30m" | "1h" | "1d"
    suffix    : exchange suffix (e.g. ".AX" for ASX, "" for US)
    source    : data provider — "yahoo" (default) | "eodhd" | "polygon"
                Gracefully falls back to Yahoo if the provider has no API key.

    Returns
    -------
    pd.DataFrame with columns [Open, High, Low, Close, Volume],
    or None if the fetch fails for any reason (network, unknown ticker,
    insufficient history, etc.).  Never raises — all errors are logged.
    """
    from agents.data_providers import get_provider
    provider = get_provider(source)
    return provider.fetch(ticker, timeframe, suffix)

# ---------------------------------------------------------------------------
# Pure-pandas technical indicators
# (no ta-lib, no external dependency beyond pandas)
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
    Compute technical indicators from an OHLCV DataFrame.

    Indicators computed:
      RSI(14), MACD(12/26/9), Bollinger Bands(20/2),
      EMA(9/21/50/200), trend classification, volume ratio

    Returns None for any indicator that cannot be computed due to
    insufficient bars — callers should check each key before using.

    Returns
    -------
    dict with keys:
      current_price, prev_close, change_pct,
      rsi, rsi_label,
      macd_line, macd_signal, macd_hist, macd_label,
      bb_upper, bb_middle, bb_lower, bb_pct, bb_label,
      ema_9, ema_21, ema_50, ema_200,
      trend,
      volume, vol_ratio
    """
    close = df["Close"]
    n     = len(df)

    # Current / previous price
    current_price = float(close.iloc[-1])
    prev_close    = float(close.iloc[-2]) if n >= 2 else None
    change_pct    = (
        round((current_price - prev_close) / prev_close * 100, 2)
        if prev_close else None
    )

    # ── RSI ──────────────────────────────────────────────────────────────────
    rsi_val   = _safe_last(_rsi(close)) if n >= 14 else None
    rsi_label = (
        "OVERBOUGHT" if rsi_val and rsi_val > 70 else
        "OVERSOLD"   if rsi_val and rsi_val < 30 else
        "NEUTRAL"    if rsi_val else None
    )

    # ── MACD ─────────────────────────────────────────────────────────────────
    if n >= 26:
        ml_s, ms_s, mh_s = _macd(close)
        macd_line  = _safe_last(ml_s)
        macd_sig   = _safe_last(ms_s)
        macd_hist  = _safe_last(mh_s)
    else:
        macd_line = macd_sig = macd_hist = None

    macd_label = (
        "BULLISH" if macd_line and macd_line > 0 else
        "BEARISH" if macd_line and macd_line < 0 else None
    )

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    if n >= 20:
        bu_s, bm_s, bl_s = _bollinger(close)
        bb_upper = _safe_last(bu_s)
        bb_mid   = _safe_last(bm_s)
        bb_lower = _safe_last(bl_s)
    else:
        bb_upper = bb_mid = bb_lower = None

    bb_pct = bb_label = None
    if bb_upper and bb_lower and (bb_upper - bb_lower) > 0:
        bb_pct   = round((current_price - bb_lower) / (bb_upper - bb_lower), 3)
        bb_label = (
            "NEAR UPPER BAND" if bb_pct > 0.80 else
            "NEAR LOWER BAND" if bb_pct < 0.20 else
            "MID BAND"
        )

    # ── EMAs ─────────────────────────────────────────────────────────────────
    def _ema(span: int) -> Optional[float]:
        if n < span:
            return None
        return _safe_last(close.ewm(span=span, adjust=False).mean())

    ema_9   = _ema(9)
    ema_21  = _ema(21)
    ema_50  = _ema(50)
    ema_200 = _ema(200)

    # ── Trend ─────────────────────────────────────────────────────────────────
    if ema_21 and ema_50:
        trend = (
            "UPTREND"   if current_price > ema_21 and ema_21 > ema_50 else
            "DOWNTREND" if current_price < ema_21 and ema_21 < ema_50 else
            "SIDEWAYS"
        )
    elif prev_close:
        trend = "UPTREND" if current_price > prev_close else "DOWNTREND"
    else:
        trend = "UNKNOWN"

    # ── Volume ────────────────────────────────────────────────────────────────
    volume     = int(df["Volume"].iloc[-1])
    avg_vol_20 = float(df["Volume"].tail(21).iloc[:-1].mean()) if n >= 21 else None
    vol_ratio  = (
        round(volume / avg_vol_20, 2)
        if avg_vol_20 and avg_vol_20 > 0 else None
    )

    return {
        "current_price": current_price,
        "prev_close":    prev_close,
        "change_pct":    change_pct,
        # RSI
        "rsi":           round(rsi_val, 2) if rsi_val is not None else None,
        "rsi_label":     rsi_label,
        # MACD
        "macd_line":     round(macd_line, 5) if macd_line is not None else None,
        "macd_signal":   round(macd_sig,  5) if macd_sig  is not None else None,
        "macd_hist":     round(macd_hist, 5) if macd_hist is not None else None,
        "macd_label":    macd_label,
        # Bollinger
        "bb_upper":      round(bb_upper, 4) if bb_upper is not None else None,
        "bb_middle":     round(bb_mid,   4) if bb_mid   is not None else None,
        "bb_lower":      round(bb_lower, 4) if bb_lower is not None else None,
        "bb_pct":        bb_pct,
        "bb_label":      bb_label,
        # EMAs
        "ema_9":         round(ema_9,   4) if ema_9   is not None else None,
        "ema_21":        round(ema_21,  4) if ema_21  is not None else None,
        "ema_50":        round(ema_50,  4) if ema_50  is not None else None,
        "ema_200":       round(ema_200, 4) if ema_200 is not None else None,
        # Trend & volume
        "trend":         trend,
        "volume":        volume,
        "vol_ratio":     vol_ratio,
    }


# ---------------------------------------------------------------------------
# Multi-timeframe aggregation
# ---------------------------------------------------------------------------

def fetch_all_timeframes(
    ticker: str,
    suffix: str = ".AX",
    timeframes: Tuple[str, ...] = DEFAULT_TIMEFRAMES,
    source: str = "yahoo",
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Fetch OHLCV and calculate indicators for multiple timeframes.

    Parameters
    ----------
    ticker     : bare ticker (e.g. "BHP", "AAPL")
    suffix     : exchange suffix (e.g. ".AX", "", ".L")
    timeframes : tuple of timeframe codes to fetch
    source     : data provider — "yahoo" | "eodhd" | "polygon"
                 Falls back to Yahoo if the provider has no API key configured.

    Returns
    -------
    dict keyed by timeframe code — values are indicator dicts or None
    if data fetch / indicator calculation failed for that timeframe.
    """
    result: Dict[str, Optional[Dict[str, Any]]] = {}
    for tf in timeframes:
        df = fetch_ohlcv(ticker, tf, suffix=suffix, source=source)
        if df is not None:
            try:
                result[tf] = calc_indicators(df)
            except Exception as exc:
                logger.warning(
                    "Indicator calculation failed for %s%s [%s]: %s",
                    ticker, suffix, tf, exc
                )
                result[tf] = None
        else:
            result[tf] = None
    return result


# ---------------------------------------------------------------------------
# Technical context formatter for LLM injection
# ---------------------------------------------------------------------------

def _fmt_opt(val: Optional[float], decimals: int = 4) -> str:
    return f"{val:.{decimals}f}" if val is not None else "N/A"


def build_technical_context(
    ticker: str,
    indicators_by_tf: Dict[str, Optional[Dict[str, Any]]],
    suffix: str = ".AX",
) -> str:
    """
    Format multi-timeframe indicators as a compact plain-text block
    suitable for injection into an LLM system or user message.

    Each timeframe is ~8 lines; full output is ~300–500 chars per timeframe.
    Fields with None values are rendered as "N/A (insufficient bars)".
    """
    lines: List[str] = [f"=== TECHNICAL INDICATORS: {ticker}{suffix} ==="]
    labels = {"5m": "5-Minute", "30m": "30-Minute", "1h": "1-Hour", "1d": "Daily"}

    for tf in ("5m", "30m", "1h", "1d"):
        if tf not in indicators_by_tf:
            continue
        label = labels.get(tf, tf)
        ind   = indicators_by_tf[tf]

        if ind is None:
            lines.append(
                f"\n[{label}] DATA UNAVAILABLE "
                "(price fetch failed or insufficient history)"
            )
            continue

        p   = ind["current_price"]
        chg = (
            f"+{ind['change_pct']}%" if ind["change_pct"] and ind["change_pct"] >= 0
            else (f"{ind['change_pct']}%" if ind["change_pct"] is not None else "N/A")
        )

        lines.append(f"\n[{label}]")
        lines.append(f"  Price : ${p:.4f}  Change: {chg}  Trend: {ind['trend']}")

        # RSI
        rsi_str = (
            f"RSI(14): {ind['rsi']} — {ind['rsi_label']}"
            if ind["rsi"] else "RSI(14): N/A (< 14 bars)"
        )
        # MACD
        if ind["macd_line"] is not None:
            macd_str = (
                f"MACD: {ind['macd_line']:+.5f} | "
                f"Sig: {ind['macd_signal']:+.5f} | "
                f"Hist: {ind['macd_hist']:+.5f} — {ind['macd_label']}"
            )
        else:
            macd_str = "MACD: N/A (< 26 bars)"

        # Bollinger Bands
        if ind["bb_upper"] is not None:
            bb_str = (
                f"BB: Upper ${ind['bb_upper']:.4f} / "
                f"Mid ${ind['bb_middle']:.4f} / "
                f"Lower ${ind['bb_lower']:.4f}  "
                f"BB%: {ind['bb_pct']:.2f} — {ind['bb_label']}"
            )
        else:
            bb_str = "BB: N/A (< 20 bars)"

        # EMAs
        ema_parts = [
            f"EMA{span}: ${ind[key]:.4f}"
            for span, key in [(9, "ema_9"), (21, "ema_21"), (50, "ema_50"), (200, "ema_200")]
            if ind[key] is not None
        ]
        ema_str = " | ".join(ema_parts) if ema_parts else "EMAs: N/A"

        # Volume
        vol_str = (
            f"Vol: {ind['volume']:,} ({ind['vol_ratio']}× 20-bar avg)"
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
# LLM prompts — pure technical signal (no announcement context)
# ---------------------------------------------------------------------------

_TECH_SIGNAL_SYSTEM = """\
You are a quantitative technical analyst generating a structured trading signal
based solely on price action and technical indicators.

You have been given live multi-timeframe OHLCV indicators for a stock.
Your task: synthesise the indicators into a structured trading signal.

MANDATORY DISCLAIMER (both top and bottom — non-negotiable):
1. The disclaimer block MUST appear verbatim at the TOP.
2. The disclaimer block MUST appear verbatim at the BOTTOM.
3. Never use "I recommend", "you should buy/sell", or definitive instruction language.
4. Use hedged language throughout: "suggests", "indicates", "may", "could".

Signal definitions:
  STRONG BUY   — ≥ 3 bullish signals: RSI < 45, MACD bullish cross,
                 price above EMA21, price near BB lower (mean reversion), high volume
  BUY          — ≥ 2 bullish signals from the above set
  HOLD/NEUTRAL — mixed or conflicting signals
  SELL         — ≥ 2 bearish signals: RSI > 60 at resistance, MACD bearish,
                 price below EMA21, price near BB upper
  STRONG SELL  — ≥ 3 bearish signals

Stop-loss sizing by timeframe:
  5min  : 0.1–0.3% from entry
  30min : 0.3–0.7% from entry
  1h    : 0.5–1.5% from entry
  Daily : 1.0–3.0% from entry

For "DATA UNAVAILABLE" timeframes: use "N/A" in all price columns,
"Insufficient data" in Notes. Do NOT fabricate price levels.

Respond with EXACTLY this structure (no extra sections):

> ⚠️ **AI-GENERATED TECHNICAL ANALYSIS — NOT FINANCIAL ADVICE**
> This signal is produced by an AI model using publicly available price data.
> It is for informational and educational purposes only. It does NOT constitute
> financial advice, a recommendation to buy or sell, or an offer of any financial
> product. Past price behaviour does not predict future results. Trading involves
> significant risk of capital loss. Always conduct your own due diligence and
> consult a licensed financial adviser before making any investment decision.

---

## 📊 Technical Signal: {TICKER}

**Current Price**: $X.XX  |  **Change**: +/-X.XX%  |  **Exchange**: {EXCHANGE}

**Signal**: [STRONG BUY / BUY / HOLD/NEUTRAL / SELL / STRONG SELL]
**Conviction**: [High / Medium / Low]
**Primary Basis**: 1-2 sentence summary of what's driving the signal.

---

### Per-Timeframe Analysis

| Timeframe | Trend | RSI | MACD | BB Position | Signal |
|-----------|-------|-----|------|-------------|--------|
| 5-Minute  | ...   | ... | ...  | ...         | ...    |
| 30-Minute | ...   | ... | ...  | ...         | ...    |
| 1-Hour    | ...   | ... | ...  | ...         | ...    |
| Daily     | ...   | ... | ...  | ...         | ...    |

---

### Entry / Exit Levels

| Timeframe | Entry Zone | Price Target | Stop-Loss | Risk/Reward | Notes |
|-----------|------------|--------------|-----------|-------------|-------|
| 5-Minute  | $X.XX–$X.XX | $X.XX      | $X.XX    | 1:X.X       | ...   |
| 30-Minute | $X.XX–$X.XX | $X.XX      | $X.XX    | 1:X.X       | ...   |
| 1-Hour    | $X.XX–$X.XX | $X.XX      | $X.XX    | 1:X.X       | ...   |
| Daily     | $X.XX–$X.XX | $X.XX      | $X.XX    | 1:X.X       | ...   |

*All prices in the stock's local currency.*
*Risk/Reward = (Target − Entry) / (Entry − Stop-Loss)*

---

### Key Technical Observations

- **Trend confluence**: [are multiple timeframes aligned? describe]
- **Momentum**: [RSI + MACD reading across timeframes]
- **Volatility**: [Bollinger Band width and position — squeeze or expansion?]
- **Volume**: [confirms or contradicts the price move?]
- **Key levels**: [nearest support/resistance based on EMAs and BB bands]

---

### Risks to This Signal

- [Technical risk 1]
- [Technical risk 2]
- [Always include: "This signal is based on historical price data; past
   patterns do not guarantee future price behaviour"]

---

> ⚠️ **DISCLAIMER**: This AI-generated technical analysis is for informational
> and educational purposes only. DataPAI and its operators accept NO responsibility
> for any trading decisions based on this output. NOT financial advice. Consult a
> licensed financial adviser before trading.
"""

_TECH_REVIEWER_SYSTEM = """\
Review this AI-generated technical trading signal for correctness and compliance.

Check:
1. Disclaimer at BOTH top AND bottom.
2. For BUY signals: Entry Zone < Target, Stop-Loss < Entry.
   For SELL signals: Entry Zone > Target, Stop-Loss > Entry.
3. Risk/Reward ≥ 1:1.5 for all timeframes (reward at least 1.5× the risk).
4. No "DATA UNAVAILABLE" timeframe has fabricated price levels.
5. No definitive instruction language ("you should buy", "I recommend").
6. Risk/Reward calculation is arithmetically correct.

Reply EXACTLY "APPROVED" if all checks pass, otherwise reply with
the fully corrected signal (complete markdown, no preamble).
"""


# ---------------------------------------------------------------------------
# Grounding sources formatter
# ---------------------------------------------------------------------------

def _format_grounding_sources(
    sources: List[Dict[str, str]],
    queries: List[str],
) -> str:
    """
    Format Gemini Google Search grounding metadata into a markdown footnote.

    This block is appended to the AI signal so users can see exactly what
    real-time web sources Gemini consulted when supplementing our computed
    indicators with live news / analyst context.

    Parameters
    ----------
    sources : list of {"title": str, "uri": str} dicts from groundingChunks
    queries : list of Google Search queries Gemini issued

    Returns
    -------
    Markdown string (starts with newlines so it appends cleanly to signal).
    Empty string if no sources are present.
    """
    if not sources:
        return ""

    lines = [
        "\n\n---\n",
        "### 🌐 Real-Time Sources (Gemini Google Search Grounding)\n",
    ]
    if queries:
        q_str = " · ".join(f'"{q}"' for q in queries)
        lines.append(f"*Gemini searched for: {q_str}*\n")

    for i, s in enumerate(sources, 1):
        title = s.get("title") or "Source"
        uri   = s.get("uri", "")
        if uri:
            lines.append(f"{i}. [{title}]({uri})")
        else:
            lines.append(f"{i}. {title}")

    lines.append(
        "\n> *These real-time web results were retrieved by Gemini to supplement "
        "the deterministically computed technical indicators (RSI, MACD, Bollinger Bands, EMAs). "
        "Always verify sources independently. NOT financial advice.*"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone technical signal generation (no announcement required)
# ---------------------------------------------------------------------------

def generate_technical_signal(
    ticker: str,
    suffix: str = ".AX",
    question: Optional[str] = None,
    timeframes: Tuple[str, ...] = DEFAULT_TIMEFRAMES,
    indicators_by_tf: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
    source: str = "yahoo",
    use_grounding: bool = True,
    macro_context: str = "",
) -> str:
    """
    Generate a structured technical trading signal — no announcement required.

    DataPAI hybrid approach
    -----------------------
    1. We compute exact indicators (RSI, MACD, Bollinger, EMAs) from live OHLCV
       data via the chosen data provider — these are deterministic and auditable.
    2. Gemini interprets the indicators AND (when use_grounding=True) supplements
       with real-time news, analyst ratings, and market sentiment via Google Search.
    3. GPT reviews the draft for compliance and arithmetic correctness.

    This is strictly better than asking Gemini directly: Gemini's native search
    gives approximate/hallucinated indicator values; our computed values are exact.

    Parameters
    ----------
    ticker           : bare ticker symbol (e.g. "BHP", "AAPL")
    suffix           : exchange suffix (default ".AX" for ASX)
    question         : optional specific question (e.g. "Is there a breakout forming?")
    timeframes       : which timeframes to include
    indicators_by_tf : pre-computed indicator dict (skip fetch if already done)
    source           : data provider — "yahoo" | "eodhd" | "polygon"
    use_grounding    : enable Gemini Google Search grounding for real-time
                       news + analyst context (default True)
    macro_context    : optional sector/macro context string from
                       agents.market_context.fetch_sector_context() — injected
                       before technical indicators so the LLM sees sector ETF
                       performance, commodity prices, and FX rates alongside the
                       computed indicators

    Returns
    -------
    Formatted Markdown string with disclaimer at top and bottom.
    Returns a user-facing error string (never raises) if all LLMs fail.
    """
    from agents.llm_client import GoogleChatClient, OpenAIChatClient

    # Fetch price data if not pre-computed
    if indicators_by_tf is None:
        indicators_by_tf = fetch_all_timeframes(
            ticker, suffix=suffix, timeframes=timeframes, source=source
        )

    available = [tf for tf, v in indicators_by_tf.items() if v is not None]

    if not available:
        return (
            f"> ⚠️ **NOT FINANCIAL ADVICE** — AI-generated for informational purposes only.\n\n"
            f"## 📊 Technical Signal: {ticker}{suffix}\n\n"
            f"⚠️ **Price data unavailable** — could not retrieve OHLCV data for "
            f"`{ticker}{suffix}` via **{source}** across any timeframe.\n\n"
            f"Possible causes:\n"
            f"- Ticker not found on the exchange (verify the ticker symbol and suffix)\n"
            f"- Network issue reaching the data provider\n"
            f"- Lightly traded or recently listed stock with insufficient history\n"
            f"- Paid provider API key not set (try source=yahoo as a fallback)\n\n"
            f"Try with an explicit suffix: `BHP.AX`, `AAPL`, `BP.L`"
        )

    # Get current price from the most granular available timeframe
    daily_ind = indicators_by_tf.get("1d") or indicators_by_tf.get(available[-1])
    current_price = daily_ind["current_price"] if daily_ind else None

    # Build exchange label for display
    exchange_map = {
        ".AX": "ASX (Australian Securities Exchange)",
        ".L":  "LSE (London Stock Exchange)",
        ".TO": "TSX (Toronto Stock Exchange)",
        ".HK": "HKEX (Hong Kong Stock Exchange)",
        "":    "NYSE / NASDAQ",
    }
    exchange_label = exchange_map.get(suffix, f"Exchange suffix: {suffix}")

    # Build technical context for LLM
    tech_context = build_technical_context(ticker, indicators_by_tf, suffix=suffix)

    # Assemble user message
    user_msg_parts = [
        f"Ticker: {ticker}{suffix}",
        f"Exchange: {exchange_label}",
        f"Current Price: {'${:.4f}'.format(current_price) if current_price else 'N/A'}",
        f"Timeframes with data: {', '.join(available)}",
        "",
    ]

    # Inject sector/macro context (from market_context.py) BEFORE technical
    # indicators so the LLM frames the stock move in its macro backdrop first.
    if macro_context and macro_context.strip():
        user_msg_parts += [macro_context, ""]

    user_msg_parts.append(tech_context)

    if question:
        user_msg_parts += ["", f"Specific question: {question}"]

    user_msg = "\n".join(user_msg_parts)

    messages = [
        {"role": "system", "content": _TECH_SIGNAL_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]

    # ── Step 1: Gemini (primary) — indicators we computed + optional live news ──
    # DataPAI hybrid: we inject exact RSI/MACD/BB/EMA values, Gemini interprets
    # AND (with grounding=True) searches Google for real-time news + analyst data.
    try:
        gemini = GoogleChatClient()
        gemini_resp = gemini.chat(messages, temperature=0.2, grounding=use_grounding)
        draft       = gemini_resp.get("content", "")

        # Append grounding sources footnote if Gemini found live web results
        grounding_sources = gemini_resp.get("grounding_sources", [])
        grounding_queries = gemini_resp.get("web_search_queries", [])
        if grounding_sources:
            draft += _format_grounding_sources(grounding_sources, grounding_queries)
            logger.info(
                "Gemini grounding active for %s%s — %d source(s) retrieved",
                ticker, suffix, len(grounding_sources),
            )

    except Exception as exc:
        logger.warning(
            "Gemini technical signal failed, falling back to GPT: %s", exc
        )
        try:
            gpt = OpenAIChatClient()
            return gpt.chat(messages, temperature=0.2).get(
                "content", "⚠️ Signal generation failed — no LLM response."
            )
        except Exception as exc2:
            return (
                f"⚠️ Technical signal generation failed: both Gemini and GPT unavailable.\n"
                f"Gemini: {exc}  |  GPT: {exc2}"
            )

    if not draft.strip():
        draft = "(Gemini returned an empty response — please retry)"

    # ── Step 2: GPT-5.1 reviewer (quality + compliance gate) ──────────────────
    try:
        reviewer_msgs = [
            {"role": "system", "content": _TECH_REVIEWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Technical context (source data):\n{tech_context[:2000]}\n\n"
                    f"{'═' * 60}\n\n"
                    f"Draft signal to review:\n{draft}"
                ),
            },
        ]
        gpt      = OpenAIChatClient()
        reviewed = gpt.chat(reviewer_msgs, temperature=0.1).get("content", "").strip()

        if reviewed.upper().startswith("APPROVED"):
            logger.debug("Technical signal reviewer: APPROVED")
            return draft

        logger.debug("Technical signal reviewer: REWRITTEN by GPT")
        return reviewed if reviewed else draft

    except Exception as exc:
        logger.warning(
            "GPT reviewer failed for technical signal — returning Gemini draft: %s", exc
        )
        return draft


# ---------------------------------------------------------------------------
# CLI entry point — quick test without any web service
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "BHP"
    suffix_arg = sys.argv[2] if len(sys.argv) > 2 else ".AX"

    print(f"\nFetching technical data for {ticker_arg}{suffix_arg}...\n")
    ind_data = fetch_all_timeframes(ticker_arg, suffix=suffix_arg)
    print(build_technical_context(ticker_arg, ind_data, suffix=suffix_arg))
    print("\n" + "─" * 60)
    print("(To generate an LLM signal, call generate_technical_signal() "
          "with valid API keys set in environment.)")

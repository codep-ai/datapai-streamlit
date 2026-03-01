# agents/chart_vision.py
"""
Chart Vision Module
===================
Generates matplotlib technical analysis charts from OHLCV + indicator data,
then sends them to Gemini Vision (multimodal) for pattern recognition analysis.

Functions
---------
  render_chart(ticker, df, indicators, suffix, timeframe)  → bytes (PNG)
  analyse_chart_with_gemini(ticker, chart_bytes, indicators, suffix)  → str

Chart layout (3-panel stacked vertically):
  Panel 1 (60%): Closing price line + Bollinger Bands (shaded) + EMA21 + EMA50
  Panel 2 (20%): RSI(14) with 30/70 overbought/oversold bands
  Panel 3 (20%): MACD histogram (green/red) + MACD line + signal line

Why Vision analysis?
--------------------
Gemini Vision sees the CHART directly — it can identify:
  - Chart patterns (head & shoulders, double top/bottom, triangles, wedges)
  - Divergence between price and RSI/MACD
  - Bollinger Band squeeze / expansion
  - Volume profile implied by momentum
  - Support/resistance from visual price structure

This complements our deterministic computed indicators with visual pattern
recognition that would otherwise require complex geometric algorithms.

Dependencies
------------
  matplotlib >= 3.7   (add to requirements.txt)
  pandas              (already present)
  agents.llm_client   (GoogleChatClient.chat_vision)
  agents.technical_analysis  (fetch_ohlcv, _rsi, _macd, _bollinger)
"""
from __future__ import annotations

import io
import logging
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt for Gemini Vision chart analysis
# ---------------------------------------------------------------------------

_VISION_SYSTEM_PROMPT = """\
You are a quantitative technical analyst reviewing a stock chart image.
The chart contains three panels:
  1. Price with Bollinger Bands (blue shaded) and EMA21/EMA50 lines
  2. RSI(14) with 30/70 reference lines
  3. MACD histogram (green=bullish, red=bearish) with MACD and signal lines

Your task: provide a concise visual pattern analysis of the chart.

MANDATORY DISCLAIMER — include verbatim at top:
> ⚠️ AI-generated chart analysis — NOT financial advice. For informational
> and educational purposes only.

Then provide:

### 📊 Visual Pattern Recognition

**Pattern identified**: [name the dominant chart pattern if visible, or "No clear pattern"]
**Confidence**: [High / Medium / Low]

**Price structure**: [describe what you see — trend direction, consolidation, breakout, etc.]

**RSI reading**: [oversold / neutral / overbought — note any divergence with price]

**MACD reading**: [histogram expanding or contracting? Bullish/bearish crossover forming?]

**Bollinger Band status**: [squeeze building? price at upper/lower band? expanding?]

**Key visual observations**:
- [observation 1]
- [observation 2]
- [observation 3]

**Pattern implication**: [what does this pattern historically suggest — continuation, reversal, breakout?]

> ⚠️ DISCLAIMER: Chart pattern analysis is statistical in nature and does not
> guarantee future price behaviour. NOT financial advice. Past patterns do not
> predict future results. Always consult a licensed financial adviser.
"""


# ---------------------------------------------------------------------------
# Chart renderer
# ---------------------------------------------------------------------------

def render_chart(
    ticker: str,
    df: pd.DataFrame,
    indicators: Optional[Dict[str, Any]] = None,
    suffix: str = ".AX",
    timeframe: str = "1d",
    bars: int = 120,
) -> Optional[bytes]:
    """
    Render a 3-panel technical analysis chart and return PNG bytes.

    Parameters
    ----------
    ticker     : bare ticker symbol (e.g. "BHP", "AAPL")
    df         : OHLCV DataFrame indexed by datetime
    indicators : optional pre-computed indicator dict from calc_indicators()
    suffix     : exchange suffix for display
    timeframe  : timeframe label for chart title
    bars       : how many bars to show (tail)

    Returns
    -------
    PNG image bytes, or None if rendering fails.
    Never raises — all errors are logged.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — required for server use
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Patch
    except ImportError:
        logger.error("matplotlib is not installed. Run: pip install matplotlib")
        return None

    try:
        from agents.technical_analysis import _rsi, _macd, _bollinger
    except ImportError as exc:
        logger.error("Could not import technical_analysis helpers: %s", exc)
        return None

    # --- Data preparation ---
    if df is None or df.empty:
        return None

    # Flatten MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    df = df.tail(bars).copy()
    if len(df) < 10:
        return None

    close = df["Close"]
    dates = df.index

    # Compute indicators on the display slice
    rsi_series                  = _rsi(close)
    macd_line, sig_line, hist_s = _macd(close)
    bb_upper, bb_mid, bb_lower  = _bollinger(close)
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    # --- Layout ---
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor("#0e1117")
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="#cccccc", labelsize=8)
        ax.spines["bottom"].set_color("#333333")
        ax.spines["top"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        ax.spines["right"].set_color("#333333")
        ax.yaxis.label.set_color("#cccccc")
        ax.xaxis.label.set_color("#cccccc")
        ax.title.set_color("#ffffff")

    # ── Panel 1: Price + BB + EMAs ────────────────────────────────────────────
    ax1.set_title(
        f"{ticker}{suffix}  [{timeframe}]  —  {len(df)} bars",
        fontsize=11, color="#ffffff", pad=6,
    )

    # Bollinger Band fill
    ax1.fill_between(dates, bb_upper, bb_lower, alpha=0.15, color="#4a9eff", label="BB Bands")
    ax1.plot(dates, bb_upper, color="#4a9eff", linewidth=0.6, alpha=0.7)
    ax1.plot(dates, bb_lower, color="#4a9eff", linewidth=0.6, alpha=0.7)
    ax1.plot(dates, bb_mid,   color="#888888", linewidth=0.6, linestyle="--", alpha=0.6)

    # Price line
    ax1.plot(dates, close, color="#ffffff", linewidth=1.2, label="Close", zorder=3)

    # EMAs
    ax1.plot(dates, ema21, color="#f0a500", linewidth=1.0, label="EMA21", alpha=0.85)
    ax1.plot(dates, ema50, color="#e55c5c", linewidth=1.0, label="EMA50", alpha=0.85)

    ax1.legend(loc="upper left", fontsize=7, facecolor="#1a1a2e", labelcolor="#cccccc",
               framealpha=0.7, edgecolor="#333333")
    ax1.set_ylabel("Price", fontsize=8, color="#aaaaaa")

    # ── Panel 2: RSI ─────────────────────────────────────────────────────────
    ax2.plot(dates, rsi_series, color="#a0d8ef", linewidth=1.0)
    ax2.axhline(70, color="#e55c5c", linewidth=0.7, linestyle="--", alpha=0.7)
    ax2.axhline(30, color="#66cc88", linewidth=0.7, linestyle="--", alpha=0.7)
    ax2.fill_between(dates, rsi_series, 70,
                     where=(rsi_series >= 70), alpha=0.2, color="#e55c5c")
    ax2.fill_between(dates, rsi_series, 30,
                     where=(rsi_series <= 30), alpha=0.2, color="#66cc88")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("RSI(14)", fontsize=8, color="#aaaaaa")
    ax2.axhline(50, color="#555555", linewidth=0.5, linestyle=":")
    ax2.text(
        dates[-1], 72, "OB", fontsize=6, color="#e55c5c",
        va="bottom", ha="right"
    )
    ax2.text(
        dates[-1], 28, "OS", fontsize=6, color="#66cc88",
        va="top", ha="right"
    )

    # ── Panel 3: MACD ────────────────────────────────────────────────────────
    # Histogram — green when positive, red when negative
    colors = ["#66cc88" if v >= 0 else "#e55c5c" for v in hist_s]
    ax3.bar(dates, hist_s, color=colors, alpha=0.7, width=0.8, zorder=2)
    ax3.plot(dates, macd_line, color="#f0a500", linewidth=0.9, label="MACD", zorder=3)
    ax3.plot(dates, sig_line,  color="#a0d8ef", linewidth=0.9, label="Signal", zorder=3)
    ax3.axhline(0, color="#555555", linewidth=0.5)
    ax3.set_ylabel("MACD", fontsize=8, color="#aaaaaa")
    ax3.legend(loc="upper left", fontsize=7, facecolor="#1a1a2e",
               labelcolor="#cccccc", framealpha=0.7, edgecolor="#333333")

    # ── X-axis date formatting ────────────────────────────────────────────────
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=7)

    plt.tight_layout(pad=0.8)

    # Save to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Gemini Vision analysis
# ---------------------------------------------------------------------------

def analyse_chart_with_gemini(
    ticker: str,
    chart_bytes: bytes,
    indicators: Optional[Dict[str, Any]] = None,
    suffix: str = ".AX",
    timeframe: str = "1d",
) -> str:
    """
    Send a rendered chart to Gemini Vision and return a pattern analysis.

    Parameters
    ----------
    ticker      : bare ticker symbol
    chart_bytes : PNG bytes from render_chart()
    indicators  : optional indicator dict for supplemental context in the prompt
    suffix      : exchange suffix
    timeframe   : timeframe label (for the prompt)

    Returns
    -------
    Markdown string with pattern analysis and mandatory disclaimer.
    Returns an error message string on failure — never raises.
    """
    try:
        from agents.llm_client import GoogleChatClient
        client = GoogleChatClient()
    except Exception as exc:
        return f"⚠️ Could not initialise Gemini Vision client: {exc}"

    # Enrich the prompt with computed indicator values so Gemini can
    # cross-reference what it sees with exact numbers
    context_lines = [
        f"Stock: {ticker}{suffix}  |  Timeframe: {timeframe}",
    ]
    if indicators:
        p = indicators.get("current_price")
        if p:
            context_lines.append(f"Current price: ${p:.4f}")
        rsi = indicators.get("rsi")
        if rsi:
            context_lines.append(f"RSI(14): {rsi} ({indicators.get('rsi_label', '')})")
        ml = indicators.get("macd_line")
        if ml is not None:
            context_lines.append(
                f"MACD: {ml:+.5f}  Signal: {indicators.get('macd_signal', 0):+.5f}  "
                f"Hist: {indicators.get('macd_hist', 0):+.5f}"
            )
        bp = indicators.get("bb_pct")
        if bp is not None:
            context_lines.append(f"BB%: {bp:.2f} ({indicators.get('bb_label', '')})")
        trend = indicators.get("trend")
        if trend:
            context_lines.append(f"Trend: {trend}")

    computed_context = "\n".join(context_lines)

    prompt = (
        f"{_VISION_SYSTEM_PROMPT}\n\n"
        f"--- Computed indicator reference (cross-check with the chart) ---\n"
        f"{computed_context}\n"
        f"---\n\n"
        f"Please analyse the chart image above."
    )

    try:
        result = client.chat_vision(prompt, chart_bytes, mime_type="image/png")
        return result.get("content", "⚠️ Empty response from Gemini Vision.")
    except Exception as exc:
        logger.warning("Gemini Vision analysis failed for %s: %s", ticker, exc)
        return f"⚠️ Chart vision analysis failed: {exc}"

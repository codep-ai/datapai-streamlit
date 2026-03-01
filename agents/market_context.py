# agents/market_context.py
"""
Sector & Macro Context Enrichment
===================================
Fetches real-time sector ETF performance, commodity prices, and FX rates
relevant to a given stock — injects as structured context into the LLM
signal prompt.

This is the DataPAI compound advantage:
  Our code  → exact RSI / MACD / Bollinger / EMA (deterministic math)
  This module → sector performance, commodity prices, macro rates (real-time)
  Gemini     → interprets ALL of the above together + searches for news

Together these are strictly better than asking Gemini alone.

Architecture
------------
  CONTEXT_MAP                      — per-ticker sector / commodity / FX config
  fetch_sector_context(ticker)     — fetches ETF+commodity+FX via yfinance,
                                     returns a formatted plain-text block
  get_context_config(ticker)       — returns the config dict for a ticker
  _pct_change(df)                  — helper: 1d / 5d / 1m % change from OHLCV

Coverage
--------
  ASX: Materials (BHP, RIO, FMG, MIN, S32), Energy (WDS, STO, BPT),
       Financials (CBA, ANZ, NAB, WBC, MQG), Healthcare (CSL, RMD),
       Tech (WTC, XRO, SEK), Industrials (TCL, SYD, APA)
  US:  Semiconductors (NVDA, AMD, INTC, QCOM, AVGO, MU, AMAT),
       Mega-cap Tech (AAPL, MSFT, GOOGL, AMZN, META),
       Energy (XOM, CVX, COP), Financials (JPM, BAC, GS, MS),
       Biotech (MRNA, BNTX, GILD), EV/Auto (TSLA, F, GM, RIVN)

If a ticker is not in CONTEXT_MAP, a best-effort generic context is returned
using the stock's own 1d/5d/1m performance only.

Dependencies
------------
  yfinance  (already in requirements.txt)
  pandas    (already present)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-ticker context configuration
# ---------------------------------------------------------------------------
# Each entry maps:
#   sector_etf  : exchange-listed ETF representing the sector
#   commodity   : yfinance symbol for the key commodity driver (or None)
#   fx          : yfinance FX pair symbol (or None for USD-denominated stocks)
#   sector_label: human-readable sector name for the LLM prompt
#   commodity_label: human-readable commodity name
#   fx_label    : human-readable FX description

CONTEXT_MAP: Dict[str, Dict[str, Any]] = {
    # ── ASX Materials ─────────────────────────────────────────────────────
    "BHP":  {"sector_etf": "XMM.AX",  "commodity": "IRON30Y=EX", "commodity_label": "Iron Ore",   "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Materials"},
    "RIO":  {"sector_etf": "XMM.AX",  "commodity": "IRON30Y=EX", "commodity_label": "Iron Ore",   "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Materials"},
    "FMG":  {"sector_etf": "XMM.AX",  "commodity": "IRON30Y=EX", "commodity_label": "Iron Ore",   "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Materials"},
    "MIN":  {"sector_etf": "XMM.AX",  "commodity": "LIT",        "commodity_label": "Lithium ETF","fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Materials"},
    "S32":  {"sector_etf": "XMM.AX",  "commodity": "GC=F",       "commodity_label": "Gold",       "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Materials"},
    "NST":  {"sector_etf": "XGD.AX",  "commodity": "GC=F",       "commodity_label": "Gold",       "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Gold"},
    "NCM":  {"sector_etf": "XGD.AX",  "commodity": "GC=F",       "commodity_label": "Gold",       "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Gold"},
    "EVN":  {"sector_etf": "XGD.AX",  "commodity": "GC=F",       "commodity_label": "Gold",       "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Gold"},
    # ── ASX Energy ────────────────────────────────────────────────────────
    "WDS":  {"sector_etf": "XEJ.AX",  "commodity": "CL=F",       "commodity_label": "Crude Oil",  "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Energy"},
    "STO":  {"sector_etf": "XEJ.AX",  "commodity": "CL=F",       "commodity_label": "Crude Oil",  "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Energy"},
    "BPT":  {"sector_etf": "XEJ.AX",  "commodity": "CL=F",       "commodity_label": "Crude Oil",  "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Energy"},
    # ── ASX Financials ─────────────────────────────────────────────────────
    "CBA":  {"sector_etf": "XFJ.AX",  "commodity": None,          "commodity_label": None,         "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Financials"},
    "ANZ":  {"sector_etf": "XFJ.AX",  "commodity": None,          "commodity_label": None,         "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Financials"},
    "NAB":  {"sector_etf": "XFJ.AX",  "commodity": None,          "commodity_label": None,         "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Financials"},
    "WBC":  {"sector_etf": "XFJ.AX",  "commodity": None,          "commodity_label": None,         "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Financials"},
    "MQG":  {"sector_etf": "XFJ.AX",  "commodity": None,          "commodity_label": None,         "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Financials"},
    # ── ASX Healthcare ─────────────────────────────────────────────────────
    "CSL":  {"sector_etf": "XHJ.AX",  "commodity": None,          "commodity_label": None,         "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Healthcare"},
    "RMD":  {"sector_etf": "XHJ.AX",  "commodity": None,          "commodity_label": None,         "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Healthcare"},
    # ── ASX Tech ───────────────────────────────────────────────────────────
    "WTC":  {"sector_etf": "XTJ.AX",  "commodity": None,          "commodity_label": None,         "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Technology"},
    "XRO":  {"sector_etf": "XTJ.AX",  "commodity": None,          "commodity_label": None,         "fx": "AUDUSD=X", "fx_label": "AUD/USD", "sector_label": "ASX Technology"},
    # ── US Semiconductors ─────────────────────────────────────────────────
    "NVDA": {"sector_etf": "SOXX",    "commodity": None,          "commodity_label": None,         "fx": None,       "fx_label": None,      "sector_label": "US Semiconductors"},
    "AMD":  {"sector_etf": "SOXX",    "commodity": None,          "commodity_label": None,         "fx": None,       "fx_label": None,      "sector_label": "US Semiconductors"},
    "INTC": {"sector_etf": "SOXX",    "commodity": None,          "commodity_label": None,         "fx": None,       "fx_label": None,      "sector_label": "US Semiconductors"},
    "QCOM": {"sector_etf": "SOXX",    "commodity": None,          "commodity_label": None,         "fx": None,       "fx_label": None,      "sector_label": "US Semiconductors"},
    "AVGO": {"sector_etf": "SOXX",    "commodity": None,          "commodity_label": None,         "fx": None,       "fx_label": None,      "sector_label": "US Semiconductors"},
    "MU":   {"sector_etf": "SOXX",    "commodity": None,          "commodity_label": None,         "fx": None,       "fx_label": None,      "sector_label": "US Semiconductors"},
    "AMAT": {"sector_etf": "SOXX",    "commodity": None,          "commodity_label": None,         "fx": None,       "fx_label": None,      "sector_label": "US Semiconductors"},
    # ── US Mega-cap Tech ──────────────────────────────────────────────────
    "AAPL": {"sector_etf": "QQQ",     "commodity": None,          "commodity_label": None,         "fx": None,       "fx_label": None,      "sector_label": "US Mega-cap Tech"},
    "MSFT": {"sector_etf": "QQQ",     "commodity": None,          "commodity_label": None,         "fx": None,       "fx_label": None,      "sector_label": "US Mega-cap Tech"},
    "GOOGL":{"sector_etf": "QQQ",     "commodity": None,          "commodity_label": None,         "fx": None,       "fx_label": None,      "sector_label": "US Mega-cap Tech"},
    "AMZN": {"sector_etf": "QQQ",     "commodity": None,          "commodity_label": None,         "fx": None,       "fx_label": None,      "sector_label": "US Mega-cap Tech"},
    "META": {"sector_etf": "QQQ",     "commodity": None,          "commodity_label": None,         "fx": None,       "fx_label": None,      "sector_label": "US Mega-cap Tech"},
    "TSLA": {"sector_etf": "XLY",     "commodity": "LIT",         "commodity_label": "Lithium ETF","fx": None,       "fx_label": None,      "sector_label": "US Consumer Discretionary / EV"},
    # ── US Energy ─────────────────────────────────────────────────────────
    "XOM":  {"sector_etf": "XLE",     "commodity": "CL=F",        "commodity_label": "WTI Crude",  "fx": None,       "fx_label": None,      "sector_label": "US Energy"},
    "CVX":  {"sector_etf": "XLE",     "commodity": "CL=F",        "commodity_label": "WTI Crude",  "fx": None,       "fx_label": None,      "sector_label": "US Energy"},
    "COP":  {"sector_etf": "XLE",     "commodity": "CL=F",        "commodity_label": "WTI Crude",  "fx": None,       "fx_label": None,      "sector_label": "US Energy"},
    # ── US Financials ─────────────────────────────────────────────────────
    "JPM":  {"sector_etf": "XLF",     "commodity": None,          "commodity_label": None,         "fx": None,       "fx_label": None,      "sector_label": "US Financials"},
    "BAC":  {"sector_etf": "XLF",     "commodity": None,          "commodity_label": None,         "fx": None,       "fx_label": None,      "sector_label": "US Financials"},
    "GS":   {"sector_etf": "XLF",     "commodity": None,          "commodity_label": None,         "fx": None,       "fx_label": None,      "sector_label": "US Financials"},
    "MS":   {"sector_etf": "XLF",     "commodity": None,          "commodity_label": None,         "fx": None,       "fx_label": None,      "sector_label": "US Financials"},
}


def get_context_config(ticker: str) -> Optional[Dict[str, Any]]:
    """Return the CONTEXT_MAP entry for a ticker, or None if not mapped."""
    return CONTEXT_MAP.get(ticker.upper().strip())


# ---------------------------------------------------------------------------
# Price helper
# ---------------------------------------------------------------------------

def _pct_changes(symbol: str) -> Dict[str, Optional[float]]:
    """
    Fetch 1-month daily OHLCV for `symbol` via yfinance and compute
    1-day, 5-day, and 1-month percentage changes.

    Returns {"1d": float|None, "5d": float|None, "1m": float|None, "price": float|None}
    Falls back to all None on any error.
    """
    try:
        import yfinance as yf
        df = yf.download(symbol, period="1mo", interval="1d",
                         auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            return {"1d": None, "5d": None, "1m": None, "price": None}
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna()
        if len(close) < 2:
            return {"1d": None, "5d": None, "1m": None, "price": None}

        price = float(close.iloc[-1])

        def pct(n: int) -> Optional[float]:
            if len(close) <= n:
                return None
            return round((close.iloc[-1] - close.iloc[-1 - n]) / close.iloc[-1 - n] * 100, 2)

        return {"price": price, "1d": pct(1), "5d": pct(5), "1m": pct(max(1, len(close) - 1))}
    except Exception as exc:
        logger.debug("_pct_changes failed for %s: %s", symbol, exc)
        return {"1d": None, "5d": None, "1m": None, "price": None}


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.2f}%"


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def fetch_sector_context(ticker: str, suffix: str = ".AX") -> str:
    """
    Fetch real-time sector ETF, commodity, and FX context for a ticker.

    Parameters
    ----------
    ticker : bare ticker symbol (e.g. "BHP", "NVDA")
    suffix : exchange suffix (used for display only; price data always via Yahoo)

    Returns
    -------
    Formatted plain-text context block (empty string on total failure).
    The block is designed to be injected into the LLM user message BEFORE
    the technical indicator context, providing the analyst with macro backdrop.

    Never raises — all errors are caught and logged.
    """
    ticker  = ticker.upper().strip()
    cfg     = get_context_config(ticker)
    lines   = [f"=== SECTOR & MACRO CONTEXT: {ticker}{suffix} ==="]

    if cfg is None:
        lines.append(
            f"[No predefined sector/macro config for {ticker}. "
            "Using stock's own recent performance only.]"
        )
    else:
        lines.append(f"Sector: {cfg['sector_label']}")

    # ── Sector ETF ────────────────────────────────────────────────────────
    if cfg and cfg.get("sector_etf"):
        etf    = cfg["sector_etf"]
        etf_px = _pct_changes(etf)
        lines.append(
            f"Sector ETF ({etf}): "
            f"Price {etf_px['price']:.2f}  |  "
            f"1d {_fmt_pct(etf_px['1d'])}  |  "
            f"5d {_fmt_pct(etf_px['5d'])}  |  "
            f"1m {_fmt_pct(etf_px['1m'])}"
            if etf_px["price"] else f"Sector ETF ({etf}): unavailable"
        )

    # ── Commodity ─────────────────────────────────────────────────────────
    if cfg and cfg.get("commodity"):
        com     = cfg["commodity"]
        com_lbl = cfg.get("commodity_label", com)
        com_px  = _pct_changes(com)
        lines.append(
            f"{com_lbl} ({com}): "
            f"Price {com_px['price']:.2f}  |  "
            f"1d {_fmt_pct(com_px['1d'])}  |  "
            f"5d {_fmt_pct(com_px['5d'])}  |  "
            f"1m {_fmt_pct(com_px['1m'])}"
            if com_px["price"] else f"{com_lbl} ({com}): unavailable"
        )

    # ── FX rate ───────────────────────────────────────────────────────────
    if cfg and cfg.get("fx"):
        fx      = cfg["fx"]
        fx_lbl  = cfg.get("fx_label", fx)
        fx_px   = _pct_changes(fx)
        lines.append(
            f"{fx_lbl} ({fx}): "
            f"Rate {fx_px['price']:.4f}  |  "
            f"1d {_fmt_pct(fx_px['1d'])}  |  "
            f"5d {_fmt_pct(fx_px['5d'])}"
            if fx_px["price"] else f"{fx_lbl} ({fx}): unavailable"
        )

    # ── Broad market benchmarks ───────────────────────────────────────────
    benchmarks = (
        [("^AXJO", "ASX 200"), ("^AXMJ", "ASX Materials Index")]
        if suffix == ".AX" else
        [("^GSPC", "S&P 500"), ("^VIX", "VIX Fear Index"), ("^TNX", "US 10Y Yield")]
    )
    bench_parts = []
    for sym, lbl in benchmarks:
        bpx = _pct_changes(sym)
        if bpx["price"]:
            bench_parts.append(f"{lbl}: {bpx['price']:.2f} ({_fmt_pct(bpx['1d'])} 1d)")
    if bench_parts:
        lines.append("Broad market: " + "  |  ".join(bench_parts))

    lines.append("=== END SECTOR/MACRO CONTEXT ===")
    return "\n".join(lines)

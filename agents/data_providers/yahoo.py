# agents/data_providers/yahoo.py
"""
Yahoo Finance OHLCV provider — backed by yfinance.

Free, no API key, best ASX (.AX) coverage.
Intraday history capped at 60 days for 5m/30m by Yahoo Finance.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from agents.data_providers.base import OHLCVProvider

logger = logging.getLogger(__name__)

# Canonical timeframe code → yfinance download parameters
# This dict is the single source of truth for timeframe semantics;
# imported by technical_analysis.py for bars-limit and label strings.
_TF_CONFIG = {
    "5m":  {"interval": "5m",  "period": "5d",  "bars": 120, "label": "5-Minute"},
    "30m": {"interval": "30m", "period": "30d", "bars": 120, "label": "30-Minute"},
    "1h":  {"interval": "1h",  "period": "60d", "bars": 120, "label": "1-Hour"},
    "1d":  {"interval": "1d",  "period": "2y",  "bars": 500, "label": "Daily"},
}

_REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]
_MIN_BARS      = 15   # need at least this many bars for RSI(14)


class YahooProvider(OHLCVProvider):
    """
    OHLCV provider backed by yfinance (Yahoo Finance).

    Free, no API key required.
    Best suited for ASX (.AX), US (no suffix), LSE (.L), TSX (.TO), HK (.HK).
    """

    @property
    def source_name(self) -> str:
        return "Yahoo Finance (yfinance)"

    def fetch(
        self,
        ticker: str,
        timeframe: str,
        suffix: str,
    ) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            return None

        cfg    = _TF_CONFIG.get(timeframe, _TF_CONFIG["1d"])
        yf_sym = f"{ticker.upper().strip()}{suffix}"

        try:
            df = yf.download(
                yf_sym,
                interval    = cfg["interval"],
                period      = cfg["period"],
                auto_adjust = True,
                prepost     = False,
                progress    = False,
                threads     = False,
            )
        except Exception as exc:
            logger.warning("Yahoo: download failed %s [%s]: %s", yf_sym, timeframe, exc)
            return None

        if df is None or df.empty:
            logger.info("Yahoo: empty response for %s [%s]", yf_sym, timeframe)
            return None

        # Flatten MultiIndex columns (yfinance >= 0.2)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        missing = [c for c in _REQUIRED_COLS if c not in df.columns]
        if missing:
            logger.warning("Yahoo: missing columns %s for %s", missing, yf_sym)
            return None

        df = df[_REQUIRED_COLS].dropna(subset=["Close"])

        if len(df) < _MIN_BARS:
            logger.info(
                "Yahoo: insufficient bars (%d) for %s [%s] — min %d required",
                len(df), yf_sym, timeframe, _MIN_BARS,
            )
            return None

        return df.tail(cfg["bars"])

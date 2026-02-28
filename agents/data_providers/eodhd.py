# agents/data_providers/eodhd.py
"""
EOD Historical Data (eodhd.com) OHLCV provider.

Env:
  EODHD_API_KEY  — required

Why EODHD for ASX?
  - Better intraday history than Yahoo (not capped at 60 days)
  - More reliable corporate-action adjustment for ASX stocks
  - Stable commercial API — no scraping or rate-limit surprises
  - Covers 70+ exchanges worldwide

ASX note: EODHD uses ".AU" exchange code, not ".AX" (Yahoo's convention).
The suffix-to-exchange mapping below handles the translation automatically.
"""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Optional

import requests
import pandas as pd

from agents.data_providers.base import OHLCVProvider

logger = logging.getLogger(__name__)

# Canonical timeframe → EODHD interval/period notation
_EODHD_INTERVAL = {
    "5m":  "5m",
    "30m": "30m",
    "1h":  "1h",
    "1d":  "d",      # daily endpoint uses "d", not "1d"
}

_LOOKBACK_DAYS = {"5m": 30, "30m": 60, "1h": 90, "1d": 730}
_BARS_LIMIT    = {"5m": 120, "30m": 120, "1h": 120, "1d": 500}
_MIN_BARS      = 15

# Yahoo/internal exchange suffix → EODHD exchange code
# EODHD uses country codes, not market-specific codes like Yahoo
_SUFFIX_TO_EODHD = {
    ".AX": "AU",    # ASX (Australian Securities Exchange) — KEY difference from Yahoo
    "":    "US",    # NYSE / NASDAQ
    ".US": "US",
    ".L":  "LSE",   # London
    ".TO": "TO",    # Toronto
    ".HK": "HK",    # Hong Kong
    ".NZ": "NZ",    # New Zealand (NZX)
    ".SI": "SG",    # Singapore (SGX)
}

_EODHD_BASE_INTRADAY = "https://eodhd.com/api/intraday"
_EODHD_BASE_EOD      = "https://eodhd.com/api/eod"


class EODHDProvider(OHLCVProvider):
    """
    OHLCV provider backed by EOD Historical Data (eodhd.com).

    Recommended for ASX due to better intraday history depth and
    more reliable corporate-action adjustment than Yahoo Finance.
    Requires a paid EODHD API key.
    """

    @property
    def source_name(self) -> str:
        return "EOD Historical Data (eodhd.com)"

    def __init__(self) -> None:
        self.api_key = os.environ.get("EODHD_API_KEY", "")

    def fetch(
        self,
        ticker: str,
        timeframe: str,
        suffix: str,
    ) -> Optional[pd.DataFrame]:
        if not self.api_key:
            logger.warning("EODHD: EODHD_API_KEY not set")
            return None

        eodhd_interval = _EODHD_INTERVAL.get(timeframe)
        if eodhd_interval is None:
            logger.warning("EODHD: unsupported timeframe %s", timeframe)
            return None

        # Map internal suffix to EODHD exchange code
        exchange  = _SUFFIX_TO_EODHD.get(suffix, suffix.lstrip(".").upper())
        eodhd_sym = f"{ticker.upper().strip()}.{exchange}"

        lookback  = _LOOKBACK_DAYS.get(timeframe, 365)
        date_from = (date.today() - timedelta(days=lookback)).isoformat()
        date_to   = date.today().isoformat()

        # EODHD uses separate endpoints for intraday vs daily
        is_intraday = timeframe != "1d"
        url = _EODHD_BASE_INTRADAY if is_intraday else _EODHD_BASE_EOD

        params: dict = {
            "api_token": self.api_key,
            "from":      date_from,
            "to":        date_to,
            "fmt":       "json",
        }
        if is_intraday:
            params["interval"] = eodhd_interval

        url = f"{url}/{eodhd_sym}"

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("EODHD: request failed %s [%s]: %s", eodhd_sym, timeframe, exc)
            return None

        if not data or not isinstance(data, list):
            logger.info("EODHD: no data for %s [%s]", eodhd_sym, timeframe)
            return None

        df = pd.DataFrame(data)

        # EODHD intraday fields: datetime, open, high, low, close, volume
        # EODHD daily fields:    date,     open, high, low, close, adjusted_close, volume
        time_col = "datetime" if is_intraday else "date"
        df = df.rename(columns={
            time_col: "datetime",
            "open":   "Open",
            "high":   "High",
            "low":    "Low",
            "close":  "Close",       # unadjusted; use adjusted_close for daily if available
            "volume": "Volume",
        })

        # For daily data, prefer adjusted_close when available
        if not is_intraday and "adjusted_close" in df.columns:
            df["Close"] = df["adjusted_close"]

        if "datetime" not in df.columns:
            logger.warning("EODHD: unexpected response structure for %s", eodhd_sym)
            return None

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            logger.warning("EODHD: missing columns %s for %s", missing, eodhd_sym)
            return None

        df = df[required].dropna(subset=["Close"])

        if len(df) < _MIN_BARS:
            logger.info("EODHD: insufficient bars (%d) for %s [%s]", len(df), eodhd_sym, timeframe)
            return None

        return df.tail(_BARS_LIMIT.get(timeframe, 500))

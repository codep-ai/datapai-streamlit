# agents/data_providers/polygon.py
"""
Polygon.io OHLCV provider.

Env:
  POLYGON_API_KEY  — required (free tier has 15-min delay; paid = real-time)

Best suited for US-listed stocks (NYSE/NASDAQ).
ASX coverage is not available on Polygon — for ASX use Yahoo or EODHD.

Timeframe mapping (Polygon uses multiplier + timespan):
  "5m"  → multiplier=5,  timespan="minute"
  "30m" → multiplier=30, timespan="minute"
  "1h"  → multiplier=1,  timespan="hour"
  "1d"  → multiplier=1,  timespan="day"
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

_POLYGON_TF = {
    "5m":  {"multiplier": 5,  "timespan": "minute"},
    "30m": {"multiplier": 30, "timespan": "minute"},
    "1h":  {"multiplier": 1,  "timespan": "hour"},
    "1d":  {"multiplier": 1,  "timespan": "day"},
}

_LOOKBACK_DAYS = {"5m": 5, "30m": 30, "1h": 60, "1d": 730}
_BARS_LIMIT    = {"5m": 120, "30m": 120, "1h": 120, "1d": 500}
_MIN_BARS      = 15
_BASE_URL      = "https://api.polygon.io/v2/aggs/ticker"


class PolygonProvider(OHLCVProvider):
    """
    OHLCV provider backed by Polygon.io REST API v2.

    Free tier: 15-minute delayed data (real-time needs a paid plan).
    US stocks only — not suitable for ASX (.AX).
    """

    @property
    def source_name(self) -> str:
        return "Polygon.io"

    def __init__(self) -> None:
        self.api_key = os.environ.get("POLYGON_API_KEY", "")

    def fetch(
        self,
        ticker: str,
        timeframe: str,
        suffix: str,
    ) -> Optional[pd.DataFrame]:
        if not self.api_key:
            logger.warning("Polygon: POLYGON_API_KEY not set")
            return None

        if suffix not in ("", ".US"):
            logger.info(
                "Polygon: suffix '%s' is not supported (US only). "
                "Use Yahoo Finance or EODHD for non-US markets.", suffix
            )
            return None

        tf_cfg = _POLYGON_TF.get(timeframe)
        if tf_cfg is None:
            logger.warning("Polygon: unsupported timeframe %s", timeframe)
            return None

        poly_ticker = ticker.upper().strip()
        lookback    = _LOOKBACK_DAYS.get(timeframe, 365)
        date_to     = date.today().isoformat()
        date_from   = (date.today() - timedelta(days=lookback)).isoformat()

        url = (
            f"{_BASE_URL}/{poly_ticker}/range/"
            f"{tf_cfg['multiplier']}/{tf_cfg['timespan']}/"
            f"{date_from}/{date_to}"
        )
        params = {
            "adjusted": "true",
            "sort":     "asc",
            "limit":    50000,
            "apiKey":   self.api_key,
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Polygon: request failed %s [%s]: %s", poly_ticker, timeframe, exc)
            return None

        results = data.get("results")
        if not results:
            logger.info("Polygon: no results for %s [%s] (status=%s)", poly_ticker, timeframe, data.get("status"))
            return None

        df = pd.DataFrame(results)
        # Polygon fields: t=timestamp(ms), o=open, h=high, l=low, c=close, v=volume
        df["datetime"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.set_index("datetime")
        df = df.rename(columns={
            "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume",
        })

        required = ["Open", "High", "Low", "Close", "Volume"]
        missing  = [c for c in required if c not in df.columns]
        if missing:
            logger.warning("Polygon: missing columns %s for %s", missing, poly_ticker)
            return None

        df = df[required].dropna(subset=["Close"])

        if len(df) < _MIN_BARS:
            logger.info("Polygon: insufficient bars (%d) for %s [%s]", len(df), poly_ticker, timeframe)
            return None

        return df.tail(_BARS_LIMIT.get(timeframe, 500))

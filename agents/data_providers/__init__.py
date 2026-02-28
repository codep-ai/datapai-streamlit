# agents/data_providers/__init__.py
"""
OHLCV provider registry.

Usage
-----
    from agents.data_providers import get_provider
    provider = get_provider("eodhd")      # or "yahoo", "polygon"
    df = provider.fetch("BHP", "1d", ".AX")

Graceful degradation
--------------------
If the requested provider requires an API key that is not set in the
environment, get_provider() logs a warning and returns YahooProvider
instead.  Callers never need to handle "provider unavailable" — they
always get a working provider back.

Available providers
-------------------
  "yahoo"   — Yahoo Finance via yfinance.  Free, no key.
              Best for ASX, US, LSE, TSX, HK.  Intraday capped 60 days.

  "polygon" — Polygon.io REST API v2.  Needs POLYGON_API_KEY.
              US stocks only (NYSE/NASDAQ).  Free tier = 15-min delay.

  "eodhd"   — EOD Historical Data (eodhd.com).  Needs EODHD_API_KEY.
              Recommended for ASX — better intraday depth than Yahoo.
              Covers 70+ exchanges.
"""
from __future__ import annotations

import importlib
import logging
import os

from agents.data_providers.base import OHLCVProvider

logger = logging.getLogger(__name__)

# name → "module.ClassName"
_REGISTRY: dict[str, str] = {
    "yahoo":   "agents.data_providers.yahoo.YahooProvider",
    "polygon": "agents.data_providers.polygon.PolygonProvider",
    "eodhd":   "agents.data_providers.eodhd.EODHDProvider",
}

# Which env var is required for each provider ("" = no key needed)
_REQUIRED_KEY: dict[str, str] = {
    "yahoo":   "",
    "polygon": "POLYGON_API_KEY",
    "eodhd":   "EODHD_API_KEY",
}


def get_provider(name: str = "yahoo") -> OHLCVProvider:
    """
    Return an OHLCVProvider by name, with automatic Yahoo fallback.

    Parameters
    ----------
    name : "yahoo" | "polygon" | "eodhd"
           Case-insensitive. Unknown names fall back to Yahoo.

    Returns
    -------
    OHLCVProvider instance — always returns something usable.
    """
    name = (name or "yahoo").lower().strip()

    # Graceful degradation: missing API key → Yahoo
    required_key = _REQUIRED_KEY.get(name, "")
    if required_key and not os.environ.get(required_key):
        logger.warning(
            "Data provider '%s' requires %s which is not set in the environment. "
            "Falling back to Yahoo Finance.",
            name, required_key,
        )
        name = "yahoo"

    if name not in _REGISTRY:
        logger.warning(
            "Unknown data provider '%s'. Available: %s. Falling back to Yahoo.",
            name, list(_REGISTRY),
        )
        name = "yahoo"

    module_path, class_name = _REGISTRY[name].rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        cls    = getattr(module, class_name)
        return cls()
    except Exception as exc:
        logger.error(
            "Failed to load provider '%s' (%s): %s — falling back to Yahoo.",
            name, module_path, exc,
        )
        from agents.data_providers.yahoo import YahooProvider
        return YahooProvider()


__all__ = ["OHLCVProvider", "get_provider"]

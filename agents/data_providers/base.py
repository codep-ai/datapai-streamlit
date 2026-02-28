# agents/data_providers/base.py
"""
Abstract OHLCV provider interface.

All providers must translate four canonical timeframe codes:
  "5m"  | "30m" | "1h" | "1d"

into whatever the upstream API requires, then return a clean DataFrame
or None — never raise.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class OHLCVProvider(ABC):
    """
    Abstract base class for OHLCV data sources.

    Contract
    --------
    - Returns pd.DataFrame with columns [Open, High, Low, Close, Volume],
      indexed by datetime.
    - Returns None on ANY failure (network, auth, unknown ticker, etc.)
    - Never raises — all errors must be logged internally.
    - The `source` property is a human-readable name used in logs and UI.
    """

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        timeframe: str,
        suffix: str,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV bars.

        Parameters
        ----------
        ticker    : bare ticker symbol (e.g. "BHP", "AAPL")
        timeframe : canonical code — "5m" | "30m" | "1h" | "1d"
        suffix    : exchange suffix — ".AX" | "" | ".L" | ".TO" | ".HK"

        Returns
        -------
        pd.DataFrame[Open, High, Low, Close, Volume]  or  None
        """
        raise NotImplementedError

    @property
    def source_name(self) -> str:
        """Human-readable provider name shown in logs and UI."""
        return self.__class__.__name__

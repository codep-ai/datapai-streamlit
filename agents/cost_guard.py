# agents/cost_guard.py
"""
Daily LLM API cost guard for DataPAI demo / pre-revenue environments.

Tracks cumulative spending against a configurable daily budget (default $5).
When the budget ceiling is reached, BudgetExceededError is raised *before*
the next paid API call — zero surprise billing.

Integration
-----------
Transparent — integrated directly into GoogleChatClient and OpenAIChatClient
in llm_client.py.  No calling-code changes required.

Environment variables
---------------------
  DAILY_LLM_BUDGET_USD   float   Maximum daily spend in USD  (default: 5.00)
  COST_GUARD_ENABLED     bool    "false" / "0" to disable    (default: "true")

State file
----------
  /tmp/datapai_cost_YYYY-MM-DD.json
  Date-stamped → automatically becomes stale at midnight, no cleanup needed.
  Contents: {"date": "...", "spend_usd": 1.23, "calls": 7}

Pricing table
-------------
Conservative estimates — err on the high side so the guard fires before the
actual bill does.  Update _PRICING when vendors change published rates.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing table  (USD per 1 million tokens, conservative / rounded-up estimates)
# Prefix-matched against the model name (longest prefix wins).
# ---------------------------------------------------------------------------

_PRICING: dict[str, Tuple[float, float]] = {
    # model-name-prefix                (input $/1M,  output $/1M)
    # ── Google Gemini ────────────────────────────────────────────
    "gemini-2.5-flash-lite":           (0.10,        0.40),
    "gemini-2.5-flash":                (0.15,        0.60),
    "gemini-2.0-flash":                (0.10,        0.40),
    "gemini-1.5-flash":                (0.075,       0.30),
    "gemini-1.5-pro":                  (3.50,        10.50),
    # ── OpenAI ───────────────────────────────────────────────────
    "gpt-5.1":                         (2.00,        8.00),
    "gpt-4.1":                         (2.00,        8.00),
    "gpt-4o-mini":                     (0.15,        0.60),
    "gpt-4o":                          (2.50,        10.00),
    "gpt-3.5":                         (0.50,        1.50),
    "o3-mini":                         (1.10,        4.40),
    "o3":                              (10.00,       40.00),
    # ── Bedrock / Claude ─────────────────────────────────────────
    "anthropic.claude-3-5-sonnet":     (3.00,        15.00),
    "anthropic.claude-3-haiku":        (0.25,        1.25),
    "anthropic.claude":                (3.00,        15.00),
    "claude":                          (3.00,        15.00),
}

# Fallback for unrecognised models — deliberately expensive to be safe
_DEFAULT_PRICING: Tuple[float, float] = (10.00, 30.00)


def _get_pricing(model_name: str) -> Tuple[float, float]:
    """Return (input_$/1M, output_$/1M) for the closest matching model prefix."""
    name = (model_name or "").lower()
    # Longest-prefix match for specificity
    best_prefix, best_rates = "", _DEFAULT_PRICING
    for prefix, rates in _PRICING.items():
        if name.startswith(prefix) and len(prefix) > len(best_prefix):
            best_prefix, best_rates = prefix, rates
    return best_rates


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated USD cost for one API call given actual token counts."""
    in_rate, out_rate = _get_pricing(model)
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class BudgetExceededError(RuntimeError):
    """
    Raised *before* a paid LLM API call when the daily budget is exhausted.

    Attributes
    ----------
    spent_usd  : float  — today's accumulated spend
    budget_usd : float  — configured daily ceiling
    model      : str    — model that was about to be called
    """

    def __init__(self, spent_usd: float, budget_usd: float, model: str = ""):
        self.spent_usd  = spent_usd
        self.budget_usd = budget_usd
        self.model      = model
        super().__init__(
            f"💸 Daily LLM budget of ${budget_usd:.2f} reached "
            f"(spent today: ${spent_usd:.4f}). "
            f"Model requested: {model or 'unknown'}. "
            f"Increase DAILY_LLM_BUDGET_USD or wait until tomorrow."
        )


# ---------------------------------------------------------------------------
# Thread-safety: one file-level lock (sufficient for single-process servers)
# ---------------------------------------------------------------------------

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# CostGuard
# ---------------------------------------------------------------------------

class CostGuard:
    """
    Daily budget enforcer — singleton-friendly, thread-safe.

    Typical usage in an LLM client::

        _guard = CostGuard()          # reads env vars once at startup

        def chat(self, messages, ...):
            _guard.check(self.model)               # raises if over budget
            resp = _api_call(messages)
            _guard.record(self.model, in_tok, out_tok)
            return resp

    The guard is a *no-op* when COST_GUARD_ENABLED=false or when running
    in LLM_MODE=local (Ollama is free — no tracking needed).
    """

    def __init__(
        self,
        budget_usd: Optional[float] = None,
        state_dir: str = "/tmp",
    ):
        raw = os.environ.get("COST_GUARD_ENABLED", "true").lower()
        self.enabled    = raw not in ("false", "0", "no")
        self.budget_usd = budget_usd or float(
            os.environ.get("DAILY_LLM_BUDGET_USD", "5.00")
        )
        self.state_dir  = Path(state_dir)

    # ── State helpers ───────────────────────────────────────────────────────

    def _state_path(self) -> Path:
        return self.state_dir / f"datapai_cost_{date.today().isoformat()}.json"

    def _load(self) -> dict:
        p = self._state_path()
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
        return {"date": date.today().isoformat(), "spend_usd": 0.0, "calls": 0}

    def _save(self, state: dict) -> None:
        self._state_path().write_text(json.dumps(state, indent=2))

    # ── Public API ──────────────────────────────────────────────────────────

    def check(self, model: str = "") -> None:
        """
        Gate call — invoke BEFORE making a paid LLM request.

        Raises BudgetExceededError if today's spend >= daily ceiling.
        Silent no-op when disabled.
        """
        if not self.enabled:
            return
        with _lock:
            state = self._load()
            if state["spend_usd"] >= self.budget_usd:
                raise BudgetExceededError(
                    spent_usd  = state["spend_usd"],
                    budget_usd = self.budget_usd,
                    model      = model,
                )

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Accounting call — invoke AFTER a successful paid LLM call.

        Parameters
        ----------
        model         : model name string (matched against pricing table)
        input_tokens  : prompt tokens consumed
        output_tokens : completion tokens produced

        Returns the USD cost of this call.
        Emits a WARNING log when the budget ceiling is crossed.
        """
        cost = estimate_cost_usd(model, input_tokens, output_tokens)
        if not self.enabled:
            return cost

        with _lock:
            state = self._load()
            state["spend_usd"] = round(state["spend_usd"] + cost, 8)
            state["calls"]     += 1
            self._save(state)

            logger.debug(
                "CostGuard: +$%.6f  model=%s  in=%d  out=%d  "
                "today=$%.4f / $%.2f",
                cost, model, input_tokens, output_tokens,
                state["spend_usd"], self.budget_usd,
            )
            if state["spend_usd"] >= self.budget_usd:
                logger.warning(
                    "CostGuard: Daily budget $%.2f REACHED (spent $%.4f).",
                    self.budget_usd, state["spend_usd"],
                )
        return cost

    def status(self) -> dict:
        """
        Return a summary dict suitable for API responses or Streamlit display.

        Keys: enabled, budget_usd, spent_today, remaining_usd,
              calls_today, date, pct_used
        """
        state   = self._load()
        spent   = state.get("spend_usd", 0.0)
        calls   = state.get("calls", 0)
        remaining = max(0.0, self.budget_usd - spent)
        pct_used  = round(100.0 * spent / self.budget_usd, 1) if self.budget_usd else 0.0
        return {
            "enabled":       self.enabled,
            "budget_usd":    self.budget_usd,
            "spent_today":   round(spent, 6),
            "remaining_usd": round(remaining, 6),
            "calls_today":   calls,
            "date":          date.today().isoformat(),
            "pct_used":      pct_used,
        }

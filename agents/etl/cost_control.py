"""
Cost Control for ETL Agents and GenAI executions.

Three-level budget enforcement:
  1. Per-run cap    DATAPAI_BUDGET_PER_RUN_USD   (default $1.00)
  2. Per-day cap    DATAPAI_BUDGET_PER_DAY_USD   (default $10.00)
  3. Per-month cap  DATAPAI_BUDGET_PER_MONTH_USD (default $50.00)

Enforcement modes:
  WARN   log warning at DATAPAI_BUDGET_WARN_PCT  (default 80 %)
  STOP   raise CostBudgetExceeded at DATAPAI_BUDGET_STOP_PCT (default 100 %)

Model fallback (automatic cost reduction when approaching limits):
  claude-3-5-sonnet  → claude-3-haiku        (~12× cheaper)
  gpt-4o             → gpt-4o-mini           (~27× cheaper)
  bedrock-claude-3-5 → bedrock-claude-haiku  (~12× cheaper)
  any model          → ollama local          (free, if configured)

Pre-run estimation:
  estimate_run_cost(schema, row_count, provider) → cost range before the run starts

Monitoring:
  get_cost_dashboard(db)   live spend vs. budget across all horizons
  CostLedger               reads aggregated spend from etl_llm_calls (audit tables)

Environment variables
─────────────────────────────────────────────────────────────────────────────
  DATAPAI_BUDGET_PER_RUN_USD    float  default 1.00
  DATAPAI_BUDGET_PER_DAY_USD    float  default 10.00
  DATAPAI_BUDGET_PER_MONTH_USD  float  default 50.00
  DATAPAI_BUDGET_WARN_PCT       float  default 80    (warn when spend ≥ 80% of limit)
  DATAPAI_BUDGET_STOP_PCT       float  default 100   (hard stop when spend ≥ 100% of limit)
  DATAPAI_COST_FALLBACK         true|false  default true  (auto-downgrade model near limit)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

# ── Token pricing table (input_per_1M, output_per_1M) USD ────────────────────
_TOKEN_COSTS: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-3-5-sonnet-20241022":                (3.00, 15.00),
    "claude-3-5-haiku-20241022":                 (0.80,  4.00),
    "claude-3-haiku-20240307":                   (0.25,  1.25),
    "claude-3-opus-20240229":                    (15.0,  75.0),
    # Bedrock (same model, AWS pricing)
    "anthropic.claude-3-5-sonnet-20241022-v2:0": (3.00, 15.00),
    "anthropic.claude-3-haiku-20240307-v1:0":    (0.25,  1.25),
    # OpenAI
    "gpt-4o":                                    (5.00, 15.00),
    "gpt-4o-mini":                               (0.15,  0.60),
    "gpt-4-turbo":                               (10.0,  30.0),
    # Local — free
    "llama3.1":  (0.00, 0.00),
    "llama3":    (0.00, 0.00),
    "mistral":   (0.00, 0.00),
    "phi3":      (0.00, 0.00),
}

# Cheaper model for each expensive model (used when approaching budget limit)
_MODEL_FALLBACKS: dict[str, str] = {
    "claude-3-5-sonnet-20241022":                "claude-3-haiku-20240307",
    "claude-3-opus-20240229":                    "claude-3-5-sonnet-20241022",
    "anthropic.claude-3-5-sonnet-20241022-v2:0": "anthropic.claude-3-haiku-20240307-v1:0",
    "gpt-4o":                                    "gpt-4o-mini",
    "gpt-4-turbo":                               "gpt-4o-mini",
}

# Rough token counts for each pipeline stage (for pre-run estimation)
_STAGE_TOKEN_ESTIMATES = {
    "orchestrator_overhead":  1_500,   # system prompt + coordination messages
    "ingest_agent":           1_200,   # file path parsing + result summary
    "compliance_agent":       3_500,   # PII scan results + masking decisions (column-heavy)
    "quality_agent":          2_000,   # profiling results per column
    "transform_agent":        3_000,   # dbt SQL + YAML generation (scales with columns)
    "handoff_messages":       1_000,   # inter-agent transfer messages
}
_PER_COLUMN_TOKENS = 80   # additional tokens per column in schema
_PER_1K_ROWS_TOKENS = 20  # additional tokens per 1K rows (quality scan references)


# ═══════════════════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════════════════

class CostBudgetExceeded(Exception):
    """
    Raised when LLM spend reaches the configured hard-stop threshold.

    The pipeline catches this and terminates gracefully, writing the
    partial results and a cost report to the audit trail.
    """
    def __init__(
        self,
        limit_type: str,    # "per_run" | "per_day" | "per_month"
        spent: float,
        limit: float,
    ) -> None:
        self.limit_type = limit_type
        self.spent = spent
        self.limit = limit
        pct = spent / limit * 100 if limit > 0 else 0
        super().__init__(
            f"Cost budget exceeded [{limit_type}]: "
            f"spent ${spent:.4f} of ${limit:.4f} limit ({pct:.1f}%). "
            f"Pipeline terminated. Set DATAPAI_BUDGET_{limit_type.upper()}_USD "
            f"to raise the limit, or switch to a cheaper LLM provider."
        )


class CostBudgetWarning(UserWarning):
    """Emitted when spend approaches the warning threshold."""


# ═══════════════════════════════════════════════════════════════════════════════
# Budget configuration
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BudgetConfig:
    """
    Budget limits and thresholds.  All amounts in USD.

    Loaded from environment variables by default; override per-run
    by passing a BudgetConfig to CostController.
    """
    per_run_usd: float   = field(default_factory=lambda: float(os.getenv("DATAPAI_BUDGET_PER_RUN_USD",   "1.00")))
    per_day_usd: float   = field(default_factory=lambda: float(os.getenv("DATAPAI_BUDGET_PER_DAY_USD",  "10.00")))
    per_month_usd: float = field(default_factory=lambda: float(os.getenv("DATAPAI_BUDGET_PER_MONTH_USD","50.00")))
    warn_pct: float      = field(default_factory=lambda: float(os.getenv("DATAPAI_BUDGET_WARN_PCT",      "80")))
    stop_pct: float      = field(default_factory=lambda: float(os.getenv("DATAPAI_BUDGET_STOP_PCT",     "100")))
    enable_fallback: bool = field(default_factory=lambda: os.getenv("DATAPAI_COST_FALLBACK", "true").lower() != "false")

    def warn_threshold(self, limit: float) -> float:
        return limit * self.warn_pct / 100

    def stop_threshold(self, limit: float) -> float:
        return limit * self.stop_pct / 100

    def summary(self) -> str:
        return (
            f"Budget: run=${self.per_run_usd:.2f}  "
            f"day=${self.per_day_usd:.2f}  "
            f"month=${self.per_month_usd:.2f}  "
            f"warn={self.warn_pct}%  stop={self.stop_pct}%  "
            f"fallback={'on' if self.enable_fallback else 'off'}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CostLedger — reads aggregated spend from audit tables
# ═══════════════════════════════════════════════════════════════════════════════

class CostLedger:
    """Reads LLM spend from the etl_llm_calls audit table."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def _query_scalar(self, sql: str, params: Optional[list] = None) -> float:
        try:
            con = duckdb.connect(self.db_path)
            result = con.execute(sql, params or []).fetchone()
            con.close()
            return float(result[0]) if result and result[0] is not None else 0.0
        except Exception as exc:
            logger.debug("CostLedger query failed (table may not exist yet): %s", exc)
            return 0.0

    def get_run_spend(self, run_id: str) -> float:
        """Total cost for one pipeline run."""
        return self._query_scalar(
            "SELECT COALESCE(SUM(estimated_cost_usd), 0) "
            "FROM etl_llm_calls WHERE run_id = ?",
            [run_id],
        )

    def get_day_spend(self) -> float:
        """Total LLM cost for today (UTC)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self._query_scalar(
            "SELECT COALESCE(SUM(estimated_cost_usd), 0) "
            "FROM etl_llm_calls WHERE timestamp LIKE ?",
            [f"{today}%"],
        )

    def get_month_spend(self) -> float:
        """Total LLM cost for the current calendar month (UTC)."""
        month_prefix = datetime.now(timezone.utc).strftime("%Y-%m")
        return self._query_scalar(
            "SELECT COALESCE(SUM(estimated_cost_usd), 0) "
            "FROM etl_llm_calls WHERE timestamp LIKE ?",
            [f"{month_prefix}%"],
        )

    def get_run_token_count(self, run_id: str) -> int:
        """Total tokens used in a run."""
        return int(self._query_scalar(
            "SELECT COALESCE(SUM(total_tokens), 0) FROM etl_llm_calls WHERE run_id = ?",
            [run_id],
        ))

    def get_spend_by_agent(self, run_id: str) -> pd.DataFrame:
        """Cost breakdown by agent for one run."""
        try:
            con = duckdb.connect(self.db_path)
            df = con.execute(
                """
                SELECT   agent_name,
                         COUNT(*)                          AS llm_calls,
                         SUM(total_tokens)                 AS total_tokens,
                         ROUND(SUM(estimated_cost_usd), 6) AS cost_usd
                FROM     etl_llm_calls
                WHERE    run_id = ?
                GROUP BY agent_name
                ORDER BY cost_usd DESC
                """,
                [run_id],
            ).fetchdf()
            con.close()
            return df
        except Exception:
            return pd.DataFrame()

    def get_spend_summary(self) -> dict:
        """Return current spend at all time horizons."""
        return {
            "day_usd":   self.get_day_spend(),
            "month_usd": self.get_month_spend(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CostController — budget enforcement
# ═══════════════════════════════════════════════════════════════════════════════

class CostController:
    """
    Budget enforcement for a single ETL pipeline run.

    Enforces limits at three check points:
      pre_run_check()   — called before initiate_swarm_chat()
      check_mid_run()   — called inside audit_tool after each tool execution
      report()          — called after the pipeline to summarise spend

    Also provides:
      get_fallback_model()    — cheaper alternative when approaching limit
      assert_within_budget()  — raises CostBudgetExceeded if over stop threshold
    """

    def __init__(
        self,
        run_id: str,
        db_path: str,
        model: str = "unknown",
        budget: Optional[BudgetConfig] = None,
    ) -> None:
        self.run_id = run_id
        self.db_path = db_path
        self.model = model
        self.budget = budget or BudgetConfig()
        self.ledger = CostLedger(db_path)
        self._warned: set[str] = set()   # which limits have already triggered a warning

    # ── Model fallback ─────────────────────────────────────────────────────

    def get_fallback_model(self, budget_pct_used: float) -> Optional[str]:
        """
        Return a cheaper model when spend exceeds the warn threshold.

        Returns None if fallback is disabled, no cheaper model exists,
        or budget is still comfortable.
        """
        if not self.budget.enable_fallback:
            return None
        if budget_pct_used < self.budget.warn_pct:
            return None
        fallback = _MODEL_FALLBACKS.get(self.model)
        if fallback and fallback != self.model:
            logger.warning(
                "[COST] Budget at %.1f%% — suggesting model downgrade: %s → %s",
                budget_pct_used, self.model, fallback,
            )
            return fallback
        # Last resort: check if Ollama is available
        if os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_MODEL"):
            return os.getenv("OLLAMA_MODEL", "llama3.1")
        return None

    # ── Budget checks ──────────────────────────────────────────────────────

    def _check_limit(
        self,
        spent: float,
        limit: float,
        limit_type: str,
    ) -> Optional[str]:
        """
        Check one limit. Returns a warning string, raises on hard stop.
        Returns None if within comfortable range.
        """
        if limit <= 0:
            return None

        pct = spent / limit * 100

        # Hard stop
        if spent >= self.budget.stop_threshold(limit):
            raise CostBudgetExceeded(limit_type, spent, limit)

        # Warning (emit only once per limit per run)
        if spent >= self.budget.warn_threshold(limit) and limit_type not in self._warned:
            self._warned.add(limit_type)
            msg = (
                f"[COST WARNING] {limit_type} spend ${spent:.4f} "
                f"= {pct:.1f}% of ${limit:.2f} limit "
                f"(warn threshold: {self.budget.warn_pct}%)"
            )
            logger.warning(msg)
            return msg

        return None

    def pre_run_check(self) -> dict:
        """
        Check aggregate day/month budgets before starting the pipeline.

        Returns a status dict:
          {
            "can_proceed": bool,
            "day_spent": float,
            "month_spent": float,
            "day_remaining": float,
            "month_remaining": float,
            "warnings": list[str],
          }

        Raises CostBudgetExceeded if the aggregate limit is already breached.
        """
        summary = self.ledger.get_spend_summary()
        day_spent   = summary["day_usd"]
        month_spent = summary["month_usd"]

        warnings: list[str] = []

        w = self._check_limit(day_spent,   self.budget.per_day_usd,   "per_day")
        if w:
            warnings.append(w)
        w = self._check_limit(month_spent, self.budget.per_month_usd, "per_month")
        if w:
            warnings.append(w)

        result = {
            "can_proceed":      True,
            "day_spent_usd":    day_spent,
            "month_spent_usd":  month_spent,
            "day_remaining_usd":   max(0.0, self.budget.per_day_usd   - day_spent),
            "month_remaining_usd": max(0.0, self.budget.per_month_usd - month_spent),
            "per_run_limit_usd":   self.budget.per_run_usd,
            "warnings":         warnings,
        }

        logger.info(
            "[COST] Pre-run check  day=$%.4f/%s  month=$%.4f/%s  run_limit=$%.2f",
            day_spent, self.budget.per_day_usd,
            month_spent, self.budget.per_month_usd,
            self.budget.per_run_usd,
        )
        return result

    def check_mid_run(self) -> Optional[str]:
        """
        Check run-level and aggregate budgets mid-pipeline (called after each tool).

        Returns a warning string if approaching a limit, None if all clear.
        Raises CostBudgetExceeded on hard stop.
        """
        run_spent   = self.ledger.get_run_spend(self.run_id)
        day_spent   = self.ledger.get_day_spend()
        month_spent = self.ledger.get_month_spend()

        messages: list[str] = []
        for spent, limit, label in [
            (run_spent,   self.budget.per_run_usd,   "per_run"),
            (day_spent,   self.budget.per_day_usd,   "per_day"),
            (month_spent, self.budget.per_month_usd, "per_month"),
        ]:
            w = self._check_limit(spent, limit, label)
            if w:
                messages.append(w)

        return "\n".join(messages) if messages else None

    def report(self) -> str:
        """Generate a cost report for this run."""
        run_spent  = self.ledger.get_run_spend(self.run_id)
        run_tokens = self.ledger.get_run_token_count(self.run_id)
        day_spent  = self.ledger.get_day_spend()
        month_spent = self.ledger.get_month_spend()
        by_agent   = self.ledger.get_spend_by_agent(self.run_id)

        run_pct   = run_spent   / self.budget.per_run_usd   * 100 if self.budget.per_run_usd   else 0
        day_pct   = day_spent   / self.budget.per_day_usd   * 100 if self.budget.per_day_usd   else 0
        month_pct = month_spent / self.budget.per_month_usd * 100 if self.budget.per_month_usd else 0

        lines = [
            "── COST REPORT ──────────────────────────────────────────────────────",
            f"  Run   ${run_spent:>8.4f}  /  ${self.budget.per_run_usd:.2f}   ({run_pct:.1f}%)",
            f"  Day   ${day_spent:>8.4f}  /  ${self.budget.per_day_usd:.2f}  ({day_pct:.1f}%)",
            f"  Month ${month_spent:>8.4f}  /  ${self.budget.per_month_usd:.2f}  ({month_pct:.1f}%)",
            f"  Tokens this run: {run_tokens:,}",
        ]

        if not by_agent.empty:
            lines.append("  By agent:")
            for _, row in by_agent.iterrows():
                lines.append(
                    f"    {row['agent_name']:<22} "
                    f"{int(row['total_tokens']):>6,} tok  "
                    f"${row['cost_usd']:.4f}"
                )

        fallback = self.get_fallback_model(run_pct)
        if fallback:
            lines.append(
                f"  ⚠ Approaching run budget — consider switching to: {fallback}"
            )

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-run cost estimator
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_run_cost(
    schema: dict,
    row_count: int = 0,
    provider: str = "anthropic",
    model: Optional[str] = None,
) -> dict:
    """
    Estimate the LLM cost of an ETL pipeline run before it starts.

    Args:
        schema:     Column schema dict {col_name: dtype} from the source file.
        row_count:  Number of rows in the source file.
        provider:   LLM provider name (for pricing lookup).
        model:      Specific model name. Auto-selected from provider if None.

    Returns:
        {
          "model": str,
          "provider": str,
          "token_estimate": int,
          "cost_low_usd": float,
          "cost_high_usd": float,
          "cost_estimate_usd": float,
          "breakdown": { stage: tokens, … },
          "budget_config": BudgetConfig summary,
          "recommendation": str,
        }
    """
    # Resolve model for pricing
    _DEFAULT_MODELS = {
        "anthropic": "claude-3-5-sonnet-20241022",
        "openai":    "gpt-4o",
        "bedrock":   "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "ollama":    "llama3.1",
    }
    resolved_model = model or _DEFAULT_MODELS.get(provider, "gpt-4o")
    input_cost, output_cost = _TOKEN_COSTS.get(resolved_model, (5.0, 15.0))

    # Token estimation by stage
    col_count = len(schema)
    row_factor = row_count / 1_000  # per 1K rows

    breakdown: dict[str, int] = {
        stage: tokens for stage, tokens in _STAGE_TOKEN_ESTIMATES.items()
    }
    # Scale variable stages
    breakdown["compliance_agent"] += col_count * _PER_COLUMN_TOKENS
    breakdown["quality_agent"]    += col_count * _PER_COLUMN_TOKENS + int(row_factor * _PER_1K_ROWS_TOKENS)
    breakdown["transform_agent"]  += col_count * _PER_COLUMN_TOKENS

    total_tokens = sum(breakdown.values())

    # Rough input/output split: 60% input, 40% output
    input_tokens  = int(total_tokens * 0.60)
    output_tokens = int(total_tokens * 0.40)

    cost_estimate = (input_tokens * input_cost + output_tokens * output_cost) / 1_000_000

    # Confidence interval: ±40%
    cost_low  = cost_estimate * 0.60
    cost_high = cost_estimate * 1.40

    budget = BudgetConfig()
    run_pct = cost_estimate / budget.per_run_usd * 100 if budget.per_run_usd else 0

    # Recommendation
    if cost_estimate >= budget.per_run_usd:
        rec = (
            f"Estimated cost ${cost_estimate:.4f} exceeds run budget "
            f"${budget.per_run_usd:.2f}. "
            f"Consider switching to a cheaper model: "
            f"{_MODEL_FALLBACKS.get(resolved_model, 'ollama')}."
        )
    elif run_pct >= budget.warn_pct:
        rec = (
            f"Estimated cost uses {run_pct:.0f}% of run budget. "
            f"Watch spend during execution."
        )
    else:
        rec = f"Estimated cost is within budget ({run_pct:.0f}% of run limit)."

    return {
        "model":             resolved_model,
        "provider":          provider,
        "column_count":      col_count,
        "row_count":         row_count,
        "token_estimate":    total_tokens,
        "input_tokens":      input_tokens,
        "output_tokens":     output_tokens,
        "cost_low_usd":      round(cost_low, 6),
        "cost_estimate_usd": round(cost_estimate, 6),
        "cost_high_usd":     round(cost_high, 6),
        "breakdown_tokens":  breakdown,
        "budget_summary":    budget.summary(),
        "recommendation":    rec,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Cost dashboard — monitoring query
# ═══════════════════════════════════════════════════════════════════════════════

def get_cost_dashboard(db_path: str) -> dict:
    """
    Return a live cost monitoring snapshot across all time horizons.

    Suitable for Streamlit metrics, CLI health checks, or alerting.

    Returns:
        {
          "today_usd":          float,
          "month_usd":          float,
          "today_limit_usd":    float,
          "month_limit_usd":    float,
          "today_pct":          float,
          "month_pct":          float,
          "status":             "OK" | "WARNING" | "CRITICAL",
          "runs_today":         int,
          "avg_cost_per_run":   float,
          "top_cost_model":     str,
          "most_expensive_run": dict,
          "budget":             BudgetConfig summary,
        }
    """
    ledger = CostLedger(db_path)
    budget = BudgetConfig()

    day_spent   = ledger.get_day_spend()
    month_spent = ledger.get_month_spend()

    day_pct   = day_spent   / budget.per_day_usd   * 100 if budget.per_day_usd   else 0
    month_pct = month_spent / budget.per_month_usd * 100 if budget.per_month_usd else 0

    # Status
    if day_pct >= budget.stop_pct or month_pct >= budget.stop_pct:
        status = "EXCEEDED"
    elif day_pct >= budget.warn_pct or month_pct >= budget.warn_pct:
        status = "WARNING"
    elif day_pct >= budget.warn_pct * 0.6 or month_pct >= budget.warn_pct * 0.6:
        status = "WATCH"
    else:
        status = "OK"

    # Runs today
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    runs_today = 0
    top_model = "N/A"
    most_expensive_run: dict = {}

    try:
        con = duckdb.connect(db_path)
        row = con.execute(
            "SELECT COUNT(DISTINCT run_id) FROM etl_pipeline_runs WHERE started_at LIKE ?",
            [f"{today}%"],
        ).fetchone()
        if row:
            runs_today = int(row[0])

        row = con.execute(
            "SELECT model FROM etl_llm_calls GROUP BY model "
            "ORDER BY SUM(estimated_cost_usd) DESC LIMIT 1"
        ).fetchone()
        if row:
            top_model = row[0]

        row = con.execute(
            "SELECT run_id, table_name, ROUND(estimated_cost_usd, 6) AS cost_usd, started_at "
            "FROM etl_pipeline_runs ORDER BY estimated_cost_usd DESC LIMIT 1"
        ).fetchone()
        if row:
            most_expensive_run = {
                "run_id": row[0], "table": row[1],
                "cost_usd": row[2], "started_at": row[3],
            }
        con.close()
    except Exception as exc:
        logger.debug("cost_dashboard query failed: %s", exc)

    avg_cost = day_spent / runs_today if runs_today else 0.0

    return {
        "today_usd":          round(day_spent,   4),
        "month_usd":          round(month_spent, 4),
        "today_limit_usd":    budget.per_day_usd,
        "month_limit_usd":    budget.per_month_usd,
        "today_remaining_usd":  round(max(0, budget.per_day_usd   - day_spent),   4),
        "month_remaining_usd":  round(max(0, budget.per_month_usd - month_spent), 4),
        "today_pct":          round(day_pct,   1),
        "month_pct":          round(month_pct, 1),
        "status":             status,
        "runs_today":         runs_today,
        "avg_cost_per_run_usd": round(avg_cost, 4),
        "top_cost_model":     top_model,
        "most_expensive_run": most_expensive_run,
        "budget":             budget.summary(),
    }


def print_cost_dashboard(db_path: str) -> None:
    """Print a formatted cost dashboard to stdout."""
    d = get_cost_dashboard(db_path)
    status_icon = {"OK": "✓", "WATCH": "~", "WARNING": "⚠", "EXCEEDED": "✗"}.get(d["status"], "?")
    print(
        f"\n{'─'*54}\n"
        f"  COST DASHBOARD   {status_icon} {d['status']}\n"
        f"{'─'*54}\n"
        f"  Today   ${d['today_usd']:>8.4f} / ${d['today_limit_usd']:.2f}   "
        f"({d['today_pct']:.1f}%)   remaining ${d['today_remaining_usd']:.4f}\n"
        f"  Month   ${d['month_usd']:>8.4f} / ${d['month_limit_usd']:.2f}  "
        f"({d['month_pct']:.1f}%)   remaining ${d['month_remaining_usd']:.4f}\n"
        f"  Runs today: {d['runs_today']}   avg cost: ${d['avg_cost_per_run_usd']:.4f}/run\n"
        f"  Top cost model: {d['top_cost_model']}\n"
        f"  {d['budget']}\n"
        f"{'─'*54}\n"
    )

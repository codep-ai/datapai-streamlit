"""
guardrail.trace_helpers
=======================

Thin wrappers that emit policy decision trace events into the Datap.ai Trace Ledger.

Uses the existing traceability module (traceability.ledger.TraceLedger) from
the feature/traceability build.  If the ledger is not available, all helpers
degrade gracefully to no-ops.

Event types emitted:
  - policy_catalog_loaded
  - eligible_assets_resolved
  - context_filtered
  - action_validated
  - policy_check_passed
  - policy_check_failed
  - action_blocked
  - action_modified_for_safety

Usage (inside GovernedAction):
    from guardrail.trace_helpers import GuardrailTracer
    tracer = GuardrailTracer(ledger)
    tracer.trace_context_filtering(...)
    tracer.trace_action_validation(...)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# GuardrailTracer
# ──────────────────────────────────────────────────────────────────────────────

class GuardrailTracer:
    """
    Wraps a TraceLedger and emits policy decision trace events.

    All methods are safe to call even if the ledger is None — they degrade
    to debug-level logging only.
    """

    def __init__(self, ledger: Optional[Any] = None):
        self.ledger = ledger

    def _emit(self, *, event_type: str, identity: Dict[str, str], **kwargs) -> None:
        """Internal: emit a trace event via the ledger or fall back to logging."""
        if self.ledger is None:
            logger.debug("[guardrail trace] %s %s", event_type, kwargs)
            return

        try:
            self.ledger.emit(
                identity   = identity,
                event_type = event_type,
                **kwargs,
            )
        except Exception as exc:
            logger.warning("[guardrail trace] emit failed: %s", exc)

    # ── Public trace methods ──────────────────────────────────────────────────

    def trace_catalog_loaded(
        self,
        request_id:     str,
        policy_version: str,
        model_count:    int,
        identity:       Dict[str, str],
    ) -> None:
        self._emit(
            event_type      = "policy_catalog_loaded",
            identity        = identity,
            request_id      = request_id,
            ai_action_summary = (
                f"AI policy catalog loaded (version={policy_version}, "
                f"models={model_count})"
            ),
        )

    def trace_context_filtering(
        self,
        request_id:  str,
        use_case:    str,
        included:    List[str],
        identity:    Dict[str, str],
    ) -> None:
        self._emit(
            event_type       = "context_filtered",
            identity         = identity,
            request_id       = request_id,
            ai_action_summary = (
                f"Context filtered for use_case='{use_case}': "
                f"{len(included)} model(s) included [{', '.join(included[:5])}]"
            ),
        )

    def trace_action_validation(
        self,
        request_id: str,
        use_case:   str,
        result:     Any,  # GuardrailResult
        identity:   Dict[str, str],
    ) -> None:
        if result.allowed:
            self._emit(
                event_type       = "policy_check_passed",
                identity         = identity,
                request_id       = request_id,
                ai_action_summary = (
                    f"Policy check passed for use_case='{use_case}' "
                    f"(answer_mode={result.answer_mode.value})"
                ),
            )
        else:
            violations_summary = "; ".join(
                v.message for v in (result.violations or [])[:3]
            )
            self._emit(
                event_type       = "policy_check_failed",
                identity         = identity,
                request_id       = request_id,
                ai_action_summary = (
                    f"Policy check FAILED for use_case='{use_case}': {violations_summary}"
                ),
                boundary_violated = True,
                risk_flags        = [v.code for v in (result.violations or [])],
            )

    def trace_blocked_action(
        self,
        request_id: str,
        use_case:   str,
        result:     Any,  # GuardrailResult
        identity:   Dict[str, str],
    ) -> None:
        violations_summary = "; ".join(
            v.message for v in (result.violations or [])[:3]
        )
        self._emit(
            event_type         = "action_blocked",
            identity           = identity,
            request_id         = request_id,
            ai_action_summary  = (
                f"Action blocked: use_case='{use_case}' — {violations_summary}"
            ),
            boundary_violated  = True,
            risk_flags         = [v.code for v in (result.violations or [])],
        )

    def trace_response_safety_adjustment(
        self,
        request_id:  str,
        use_case:    str,
        adjustment:  str,  # e.g. "masked", "aggregate_only", "metadata_only"
        identity:    Dict[str, str],
    ) -> None:
        self._emit(
            event_type        = "action_modified_for_safety",
            identity          = identity,
            request_id        = request_id,
            ai_action_summary = (
                f"Response adjusted for safety: use_case='{use_case}', mode={adjustment}"
            ),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Standalone trace helper functions (for use outside GovernedAction)
# ──────────────────────────────────────────────────────────────────────────────

def trace_policy_decision(
    ledger:      Optional[Any],
    request_id:  str,
    use_case:    str,
    allowed:     bool,
    reasons:     List[str],
    identity:    Dict[str, str],
) -> None:
    """
    Emit a single policy decision trace event (pass or fail).

    Convenience function for callers that don't want to instantiate GuardrailTracer.
    """
    tracer = GuardrailTracer(ledger)
    if allowed:
        tracer.trace_action_validation(
            request_id = request_id,
            use_case   = use_case,
            result     = _SimpleResult(allowed=True, message="; ".join(reasons)),
            identity   = identity,
        )
    else:
        tracer.trace_blocked_action(
            request_id = request_id,
            use_case   = use_case,
            result     = _SimpleResult(allowed=False, message="; ".join(reasons)),
            identity   = identity,
        )


class _SimpleResult:
    """Minimal mock of GuardrailResult for standalone trace helpers."""
    def __init__(self, allowed: bool, message: str):
        self.allowed    = allowed
        self.violations = [type("V", (), {"message": message, "code": "POLICY"})()]
        from .metadata_schema import AnswerMode
        self.answer_mode = AnswerMode.FULL if allowed else AnswerMode.DENY

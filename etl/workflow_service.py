# etl/workflow_service.py
from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from etl.contracts import WorkflowRequest
from etl.workflow_runner import run_ingest_to_dbt
from etl.plans import WorkflowPlan

base_dir = os.environ.get("DATAPAI_RUN_DIR", "runs")

def _json_safe(obj: Any):
    """Best-effort conversion for JSON serialization."""
    try:
        if is_dataclass(obj):
            return asdict(obj)
    except Exception:
        pass

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    # fallback
    return str(obj)


def _save_run_artifacts(run_id: str, plan: WorkflowPlan, ctx: Any, base_dir: str = "runs") -> None:
    os.makedirs(base_dir, exist_ok=True)

    plan_path = os.path.join(base_dir, f"{run_id}_plan.json")
    ctx_path = os.path.join(base_dir, f"{run_id}_ctx.json")

    # plan
    plan_payload: Dict[str, Any] = asdict(plan) if is_dataclass(plan) else dict(plan)  # type: ignore
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(plan_payload), f, indent=2, ensure_ascii=False)

    # ctx: try to capture standard fields if present
    ctx_payload = {
        "run_id": getattr(ctx, "run_id", run_id),
        "artifacts": getattr(ctx, "artifacts", {}),
        "metrics": getattr(ctx, "metrics", {}),
        "logs": getattr(ctx, "logs", []),
    }
    with open(ctx_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(ctx_payload), f, indent=2, ensure_ascii=False)


def execute_plan(plan: WorkflowPlan):
    if plan.workflow != "ingest_to_dbt":
        raise ValueError(f"Unsupported workflow: {plan.workflow}")

    req = WorkflowRequest(
        source_type=plan.source_type,
        source=plan.source,
        target=plan.target,
        target_schema=plan.target_schema,
        target_table=plan.target_table,
        mode=plan.mode,
        pk=plan.pk,
        options={"tests": plan.tests, "notes": plan.notes},
    )

    ctx = run_ingest_to_dbt(req)

    # Save artifacts (you preferred it here)
    run_id = getattr(ctx, "run_id", None) or "run"
    try:
        _save_run_artifacts(run_id=run_id, plan=plan, ctx=ctx)
    except Exception as e:
        # Do not fail the workflow if saving fails
        try:
            if hasattr(ctx, "logs"):
                ctx.logs.append(f"[WARN] Failed to save run artifacts: {e}")
        except Exception:
            pass

    return ctx

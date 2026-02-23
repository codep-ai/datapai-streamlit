"""
run_context.py

Single source of truth for a DataPAI execution run.

Design goals:
- Deterministic and replayable
- Environment-aware (local / CI / Airflow / PROD)
- LLM usage explicitly controlled
- One run_id to tie logs, dbt artifacts, plans, and metadata together
"""

from __future__ import annotations

import os
import uuid
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _now_utc() -> str:
    return datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _expand(path: str) -> str:
    return os.path.expanduser(path)


# -------------------------------------------------------------------
# RunContext
# -------------------------------------------------------------------

@dataclass(frozen=True)
class RunContext:
    """
    Immutable context for a single execution run.
    """

    # Identity
    run_id: str
    started_at_utc: str

    # Environment
    env: str                 # local | ci | airflow | prod
    is_ci: bool
    is_airflow: bool

    # Paths
    base_run_dir: Path       # DATAPAI_RUN_DIR/<run_id>
    logs_dir: Path
    plans_dir: Path
    artifacts_dir: Path

    # AI control
    llm_enabled: bool

    # dbt
    dbt_project_dir: Path
    dbt_manifest_path: Path


# -------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------

def create_run_context(
    *,
    run_id: Optional[str] = None,
) -> RunContext:
    """
    Create a new deterministic RunContext.

    This should be called ONCE per execution (CLI, Airflow task, Streamlit action).
    """

    # -------- identity --------
    rid = run_id or str(uuid.uuid4())
    started = _now_utc()

    # -------- environment detection --------
    env = os.getenv("ENV", "local").lower()
    is_ci = os.getenv("CI", "false").lower() == "true"
    is_airflow = "AIRFLOW_HOME" in os.environ

    # -------- AI control --------
    llm_enabled = os.getenv("DATAPAI_LLM_ENABLED", "false").lower() == "true"

    # -------- run directory --------
    base_dir = _expand(os.getenv("DATAPAI_RUN_DIR", ".datapai/run"))
    base_run_dir = Path(base_dir) / rid

    logs_dir = base_run_dir / "logs"
    plans_dir = base_run_dir / "plans"
    artifacts_dir = base_run_dir / "artifacts"

    for d in (logs_dir, plans_dir, artifacts_dir):
        d.mkdir(parents=True, exist_ok=True)

    # -------- dbt --------
    dbt_project_dir = Path(
        _expand(os.getenv("DBT_PROJECT_DIR", "dbt-demo"))
    )

    dbt_manifest_path = Path(
        _expand(
            os.getenv(
                "DBT_MANIFEST_PATH",
                str(dbt_project_dir / "target" / "manifest.json"),
            )
        )
    )

    return RunContext(
        run_id=rid,
        started_at_utc=started,
        env=env,
        is_ci=is_ci,
        is_airflow=is_airflow,
        base_run_dir=base_run_dir,
        logs_dir=logs_dir,
        plans_dir=plans_dir,
        artifacts_dir=artifacts_dir,
        llm_enabled=llm_enabled,
        dbt_project_dir=dbt_project_dir,
        dbt_manifest_path=dbt_manifest_path,
    )


# -------------------------------------------------------------------
# Guardrails
# -------------------------------------------------------------------

def assert_llm_allowed(ctx: RunContext) -> None:
    """
    Hard guard to prevent accidental LLM usage in deterministic environments.
    """
    if not ctx.llm_enabled:
        raise RuntimeError(
            "LLM usage is disabled for this run "
            "(DATAPAI_LLM_ENABLED=false)."
        )

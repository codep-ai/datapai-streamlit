from .contracts import WorkflowRequest
from .run_context import create_run_context
from agents.tooling.loader import load_all_tools
from agents.tooling.registry import _TOOL_REGISTRY  # or better: get_tool()

def run_ingest_to_dbt(req: WorkflowRequest) -> RunContext:
    ctx = create_run_context()
    load_all_tools()

    # 1) Ingest
    # call your existing tool functions directly (no LLM) for determinism
    # e.g. ingest_file(...)
    # ctx.artifacts["ingest"] = ...

    # 2) Validate (row counts, schema drift rules)
    # ctx.metrics["rowcount"] = ...

    # 3) Generate dbt source + staging models
    # ctx.artifacts["dbt_files"] = ...

    # 4) Run dbt build/test
    # ctx.artifacts["dbt_run_results"] = ...

    return ctx

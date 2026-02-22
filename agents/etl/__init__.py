"""
agents.etl — AG2 Swarm ETL pipeline for DataPAI.

Compliance-first architecture for regulated industries (finance, healthcare, legal).

Pipeline:
  ingest_agent → compliance_agent → quality_agent → transform_agent

Public API:
    run_etl_pipeline(request, context_variables, llm_config, max_rounds) -> dict
    build_etl_swarm(llm_config) -> (orchestrator, agents)

Quick start:
    from agents.etl import run_etl_pipeline

    result = run_etl_pipeline(
        "Load /tmp/transactions.csv into DuckDB and generate a dbt staging model"
    )
    print(result["compliance_status"])   # "COMPLIANT" | "NEEDS_REVIEW" | "NON_COMPLIANT"
    print(result["pii_columns"])         # columns with detected PII
    print(result["masked_columns"])      # columns that were masked
    print(result["quality_issues"])      # data quality warnings
    print(result["dbt_artifacts"])       # generated file paths
    print(result["audit_run_id"])        # UUID for audit trail lookup
"""

from .pipeline import build_etl_swarm, run_etl_pipeline

__all__ = ["run_etl_pipeline", "build_etl_swarm"]

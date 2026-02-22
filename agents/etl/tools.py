"""
ETL tool functions for the AG2 Swarm pipeline.

Convention:
  - Every tool accepts `context_variables: dict` as its first parameter.
  - Every tool returns `SwarmResult(values=<str>, context_variables=<dict>)`.
  - Tools mutate context_variables in-place and return the updated dict,
    so downstream agents can read results without extra lookups.

Tool groups:
  1. File Ingestion  — ingest_file
  2. Data Quality    — profile_table, check_primary_key
  3. dbt Generation  — generate_staging_sql, generate_schema_yaml, save_dbt_artifacts
  4. Reporting       — get_pipeline_summary
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from autogen import SwarmResult

# ── Defaults (overridable via env vars) ──────────────────────────────────────
DB_PATH = os.getenv("DUCKDB_PATH", "datapai.duckdb")
DBT_PROJECT_DIR = os.getenv("DBT_PROJECT_DIR", "dbt_project")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FILE INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

def ingest_file(
    context_variables: dict,
    file_path: str,
    table_name: Optional[str] = None,
    mode: str = "replace",
) -> SwarmResult:
    """
    Load a CSV, Excel (.xlsx/.xls), or Parquet file into DuckDB.

    Args:
        file_path:   Absolute or relative path to the source file.
        table_name:  Target DuckDB table (defaults to the filename stem).
        mode:        'replace' (default) or 'append'.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return SwarmResult(
                values=f"File not found: {file_path}",
                context_variables=context_variables,
            )

        tbl = table_name or path.stem.lower().replace("-", "_").replace(" ", "_")
        db = context_variables.get("db_path", DB_PATH)
        con = duckdb.connect(db)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            sql = (
                f"CREATE OR REPLACE TABLE \"{tbl}\" AS "
                f"SELECT * FROM read_csv_auto('{path}', SAMPLE_SIZE=-1)"
                if mode == "replace"
                else
                f"INSERT INTO \"{tbl}\" SELECT * FROM read_csv_auto('{path}', SAMPLE_SIZE=-1)"
            )
            con.execute(sql)

        elif suffix == ".parquet":
            sql = (
                f"CREATE OR REPLACE TABLE \"{tbl}\" AS SELECT * FROM read_parquet('{path}')"
                if mode == "replace"
                else
                f"INSERT INTO \"{tbl}\" SELECT * FROM read_parquet('{path}')"
            )
            con.execute(sql)

        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
            con.register("_excel_tmp", df)
            if mode == "replace":
                con.execute(f"CREATE OR REPLACE TABLE \"{tbl}\" AS SELECT * FROM _excel_tmp")
            else:
                con.execute(f"INSERT INTO \"{tbl}\" SELECT * FROM _excel_tmp")

        else:
            con.close()
            return SwarmResult(
                values=f"Unsupported format '{suffix}'. Supported: .csv, .parquet, .xlsx, .xls",
                context_variables=context_variables,
            )

        row_count = con.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
        schema_df = con.execute(f'DESCRIBE "{tbl}"').fetchdf()
        schema = schema_df.set_index("column_name")["column_type"].to_dict()
        con.close()

        context_variables.update(
            {
                "file_path": str(path.resolve()),
                "table_name": tbl,
                "row_count": row_count,
                "schema": schema,
                "db_path": db,
                "ingestion_status": "success",
            }
        )

        summary = (
            f"Ingested '{path.name}' → DuckDB table '{tbl}'\n"
            f"Rows: {row_count:,}  |  Columns: {len(schema)}\n"
            f"Schema: {json.dumps(schema, indent=2)}"
        )
        return SwarmResult(values=summary, context_variables=context_variables)

    except Exception as exc:
        context_variables["ingestion_status"] = "failed"
        return SwarmResult(
            values=f"Ingestion failed: {exc}",
            context_variables=context_variables,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

def profile_table(context_variables: dict) -> SwarmResult:
    """
    Profile the loaded DuckDB table.

    Computes per-column null rate, distinct count, and low-cardinality flag.
    Also detects duplicate rows across the whole table.
    Results are stored in context_variables['quality_report'] and ['quality_issues'].
    """
    table_name = context_variables.get("table_name")
    db = context_variables.get("db_path", DB_PATH)
    schema = context_variables.get("schema", {})
    row_count = context_variables.get("row_count", 0)

    if not table_name:
        return SwarmResult(
            values="No table found in context. Run ingest_file first.",
            context_variables=context_variables,
        )

    try:
        con = duckdb.connect(db)
        issues: list[str] = []
        col_stats: dict = {}

        for col, dtype in schema.items():
            stats: dict = {"type": dtype}

            # Null rate
            null_count = con.execute(
                f'SELECT COUNT(*) FROM "{table_name}" WHERE "{col}" IS NULL'
            ).fetchone()[0]
            null_pct = round(null_count / max(row_count, 1) * 100, 1)
            stats["null_pct"] = null_pct
            stats["null_count"] = null_count
            if null_pct > 0:
                issues.append(f"'{col}': {null_pct}% nulls ({null_count:,} rows)")

            # Distinct count
            distinct = con.execute(
                f'SELECT COUNT(DISTINCT "{col}") FROM "{table_name}"'
            ).fetchone()[0]
            stats["distinct_count"] = distinct

            # Low-cardinality hint (< 1% unique values, more than 1 row)
            if row_count > 1 and distinct < max(row_count * 0.01, 2):
                stats["low_cardinality"] = True

            col_stats[col] = stats

        # Duplicate row detection
        unique_rows = con.execute(
            f'SELECT COUNT(*) FROM (SELECT DISTINCT * FROM "{table_name}")'
        ).fetchone()[0]
        dup_count = row_count - unique_rows
        if dup_count > 0:
            issues.append(f"{dup_count:,} fully duplicate rows detected")

        con.close()

        quality_report = {
            "table": table_name,
            "row_count": row_count,
            "duplicate_rows": dup_count,
            "columns": col_stats,
        }

        context_variables["quality_report"] = quality_report
        context_variables["quality_issues"] = issues
        context_variables["quality_status"] = "issues_found" if issues else "clean"

        issue_str = (
            "\n".join(f"  ⚠ {i}" for i in issues) if issues else "  ✓ No issues found"
        )
        return SwarmResult(
            values=(
                f"Quality Profile — '{table_name}'\n"
                f"{json.dumps(quality_report, indent=2)}\n\n"
                f"Issues:\n{issue_str}"
            ),
            context_variables=context_variables,
        )

    except Exception as exc:
        return SwarmResult(
            values=f"Profiling failed: {exc}",
            context_variables=context_variables,
        )


def check_primary_key(
    context_variables: dict,
    column: str,
) -> SwarmResult:
    """
    Check whether a column qualifies as a primary key (unique + not null).

    Args:
        column: Column name to test.
    """
    table_name = context_variables.get("table_name")
    db = context_variables.get("db_path", DB_PATH)

    if not table_name:
        return SwarmResult(
            values="No table in context.",
            context_variables=context_variables,
        )

    try:
        con = duckdb.connect(db)
        row_count = con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        unique_count = con.execute(
            f'SELECT COUNT(DISTINCT "{column}") FROM "{table_name}"'
        ).fetchone()[0]
        null_count = con.execute(
            f'SELECT COUNT(*) FROM "{table_name}" WHERE "{column}" IS NULL'
        ).fetchone()[0]
        con.close()

        is_pk = unique_count == row_count and null_count == 0
        verdict = "VALID primary key" if is_pk else "NOT a valid primary key"

        if is_pk:
            context_variables["primary_key"] = column

        return SwarmResult(
            values=(
                f"PK check — '{column}': {verdict}\n"
                f"  Total rows:    {row_count:,}\n"
                f"  Unique values: {unique_count:,}\n"
                f"  Null count:    {null_count:,}"
            ),
            context_variables=context_variables,
        )

    except Exception as exc:
        return SwarmResult(
            values=f"PK check failed: {exc}",
            context_variables=context_variables,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DBT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_staging_sql(context_variables: dict) -> SwarmResult:
    """
    Generate a dbt staging model SQL file for the loaded table.

    Uses the CTE pattern:  source → renamed → select *
    Output is stored in context_variables['staging_sql'] and ['model_name'].
    """
    table_name = context_variables.get("table_name")
    schema = context_variables.get("schema", {})

    if not table_name or not schema:
        return SwarmResult(
            values="Missing table_name or schema in context. Run ingest_file first.",
            context_variables=context_variables,
        )

    model_name = f"stg_{table_name}"

    # Build column select list with proper quoting
    cols = [f'    "{col}"' for col in schema]
    col_block = ",\n".join(cols)

    model_sql = f"""{{{{ config(materialized='view') }}}}

with source as (
    select * from {{{{ source('raw', '{table_name}') }}}}

),

renamed as (
    select
{col_block}
    from source

)

select * from renamed
"""

    context_variables["staging_sql"] = model_sql
    context_variables["model_name"] = model_name

    return SwarmResult(
        values=f"Generated staging SQL for '{model_name}':\n\n{model_sql}",
        context_variables=context_variables,
    )


def generate_schema_yaml(context_variables: dict) -> SwarmResult:
    """
    Generate a dbt schema.yml that doubles as an AI Governance Guide.

    Every column carries:
      - sensitivity classification (PII_HIGH / FINANCIAL / PII_MEDIUM / PII_LOW / PUBLIC)
      - detected PII type (ssn, credit_card, email, …)
      - masking status (masked_hash | masked_redact | unmasked)
      - applicable regulations (GDPR, PCI-DSS, SOX, HIPAA, CCPA)
      - review_required flag + business-user confirmation placeholders
      - dbt tags for filtering (pii_high, financial, gdpr, …)
      - data quality tests (unique, not_null, accepted_values)

    The model-level meta.governance block is the single source of truth for
    data stewards to review, complete, and approve before production promotion.

    Output stored in context_variables['schema_yaml'].
    """
    from datetime import datetime, timezone  # local import — tools.py has no datetime import

    table_name = context_variables.get("table_name")
    schema = context_variables.get("schema", {})
    model_name = context_variables.get("model_name", f"stg_{table_name}")
    primary_key = context_variables.get("primary_key")
    quality_report = context_variables.get("quality_report", {})
    col_quality = quality_report.get("columns", {})

    # Compliance context (populated by compliance_agent)
    pii_scan: dict = context_variables.get("pii_scan", {})
    masked_columns: list[dict] = context_variables.get("masked_columns", [])
    masked_map: dict[str, str] = {
        m["column"]: m["strategy"] for m in masked_columns
    }
    compliance_status: str = context_variables.get("compliance_status", "UNKNOWN")
    audit_run_id: str = context_variables.get("compliance_run_id", "N/A")
    high_risk_cols: list = context_variables.get("high_risk_columns", [])
    pii_cols: list = context_variables.get("pii_columns", [])

    if not table_name:
        return SwarmResult(
            values="No table in context.",
            context_variables=context_variables,
        )

    display_name = table_name.replace("_", " ").title()
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Derive model-level tags ────────────────────────────────────────────
    model_tags: list[str] = []
    regulations_detected: list[str] = []
    if pii_cols:
        model_tags.append("pii")
        regulations_detected.append("GDPR")
        regulations_detected.append("CCPA")
    if any(pii_scan.get(c, {}).get("pii_type") == "credit_card" for c in pii_scan):
        model_tags.append("pci_dss")
        if "PCI-DSS" not in regulations_detected:
            regulations_detected.append("PCI-DSS")
    if any(pii_scan.get(c, {}).get("sensitivity") == "FINANCIAL" for c in pii_scan):
        model_tags.append("sox")
        if "SOX" not in regulations_detected:
            regulations_detected.append("SOX")
    if any(pii_scan.get(c, {}).get("pii_type") == "health" for c in pii_scan):
        model_tags.append("hipaa")
        if "HIPAA" not in regulations_detected:
            regulations_detected.append("HIPAA")
    if model_tags:
        model_tags.append("compliance")
    model_tags_yaml = (
        "\n".join(f"      - {t}" for t in model_tags) if model_tags else "      []"
    )
    regs_str = json.dumps(regulations_detected)

    # ── Per-column YAML builder ────────────────────────────────────────────
    _REGULATION_MAP: dict[str, list[str]] = {
        "ssn":          ["GDPR", "CCPA"],
        "credit_card":  ["PCI-DSS", "GDPR"],
        "gov_id":       ["GDPR", "CCPA"],
        "credential":   ["GDPR"],
        "bank_account": ["SOX", "GDPR"],
        "financial_data": ["SOX"],
        "health":       ["HIPAA", "GDPR"],
        "email":        ["GDPR", "CCPA"],
        "phone":        ["GDPR", "CCPA"],
        "name":         ["GDPR", "CCPA"],
        "demographics": ["GDPR", "CCPA"],
        "address":      ["GDPR", "CCPA"],
        "ip_address":   ["GDPR"],
        "geo_data":     ["GDPR"],
    }

    _SENSITIVITY_TAGS: dict[str, list[str]] = {
        "PII_HIGH":  ["pii_high", "masked"],
        "FINANCIAL": ["financial", "sox"],
        "PII_MEDIUM": ["pii_medium"],
        "PII_LOW":   ["pii_low"],
        "PUBLIC":    [],
    }

    def _col_yaml(col: str, dtype: str) -> str:
        label = col.replace("_", " ").title()
        col_info = pii_scan.get(col, {})
        sensitivity = col_info.get("sensitivity", "PUBLIC")
        pii_type = col_info.get("pii_type", "none")
        is_masked = col in masked_map
        masking_strategy = masked_map.get(col, "unmasked")
        col_regulations = json.dumps(_REGULATION_MAP.get(pii_type, []))
        cq = col_quality.get(col, {})
        null_pct = cq.get("null_pct", 100)
        review_required = sensitivity != "PUBLIC"

        # Description — include masking note for high-risk columns
        if is_masked:
            desc = (
                f"{label} — {sensitivity}: {pii_type}. "
                f"Masked by ETL pipeline ({masking_strategy})."
            )
        elif sensitivity != "PUBLIC":
            desc = (
                f"{label} — {sensitivity}: {pii_type}. "
                f"UNMASKED — review required before production."
            )
        else:
            desc = label

        # dbt tags
        col_tags = _SENSITIVITY_TAGS.get(sensitivity, [])
        if pii_type != "none" and pii_type not in col_tags:
            col_tags = [pii_type.replace("_", "")] + col_tags

        # dbt tests
        tests: list[str] = []
        if col == primary_key:
            tests = ["unique", "not_null"]
        elif null_pct == 0:
            tests.append("not_null")

        # Build YAML block with governance meta
        lines = [
            f"      - name: {col}",
            f'        description: "{desc}"',
        ]

        # Meta block (governance data)
        lines.append("        meta:")
        lines.append(f"          sensitivity: {sensitivity}")
        lines.append(f"          pii_type: {pii_type}")
        lines.append(
            f"          masking_status: {'masked_' + masking_strategy if is_masked else 'unmasked'}"
        )
        lines.append(f"          regulations: {col_regulations}")
        lines.append(f"          review_required: {'true' if review_required else 'false'}")
        if review_required:
            action = (
                "APPROVED — masking applied"
                if is_masked
                else "PENDING — decide: mask | pseudonymize | document lawful basis"
            )
            lines.append(f"          review_action: \"{action}\"")
            if not is_masked and sensitivity in ("PII_HIGH", "FINANCIAL"):
                lines.append(
                    '          review_question: "This column contains high-risk data. '
                    "Confirm masking strategy or document the lawful basis for "
                    'retaining in clear text.\""'
                )

        # Tags block
        if col_tags:
            lines.append("        tags:")
            lines.extend(f"          - {t}" for t in col_tags)

        # Tests block
        if tests:
            lines.append("        tests:")
            lines.extend(f"          - {t}" for t in tests)

        return "\n".join(lines)

    columns_block = "\n\n".join(_col_yaml(c, t) for c, t in schema.items())

    yaml_content = f"""# ════════════════════════════════════════════════════════════════════════════
# AI GOVERNANCE GUIDE — DataPAI ETL Pipeline
# ════════════════════════════════════════════════════════════════════════════
#
# This file was auto-generated and contains:
#   1. dbt model definition (source, staging model, column catalog)
#   2. Compliance metadata per column (sensitivity, PII type, masking status)
#   3. Business-user review checklist (complete before production promotion)
#
# Generated:        {generated_at}
# Audit run ID:     {audit_run_id}
# Compliance:       {compliance_status}
# Regulations:      {regs_str}
# PII columns:      {json.dumps(pii_cols)}
# Masked columns:   {json.dumps(list(masked_map.keys()))}
#
# INSTRUCTIONS FOR BUSINESS USERS
# ─────────────────────────────────────────────────────────────────────────────
# 1. Review every column with  review_required: true
# 2. For UNMASKED sensitive columns, either:
#    a) Ask data engineering to enable masking in the pipeline, OR
#    b) Document the lawful basis (GDPR Article 6 / CCPA §1798.100) in
#       review_notes below and set approved_for_production: true
# 3. Complete reviewed_by, reviewed_at, review_notes at the model level
# 4. Set approved_for_production: true to signal production readiness
# ════════════════════════════════════════════════════════════════════════════

version: 2

sources:
  - name: raw
    database: datapai
    schema: main
    tables:
      - name: {table_name}
        description: "Raw {display_name} data — loaded by DataPAI ETL pipeline"
        meta:
          pii_detected: {str(bool(pii_cols)).lower()}
          compliance_status: {compliance_status}
          audit_run_id: "{audit_run_id}"
          loaded_at: "{generated_at}"

models:
  - name: {model_name}
    description: >
      Staging model for {display_name}.
      Auto-generated by DataPAI ETL pipeline — review compliance metadata
      and complete the governance checklist before promoting to production.

    # ── Governance metadata (AI-generated — business user confirmation required) ──
    meta:
      governance:
        pii_detected: {str(bool(pii_cols)).lower()}
        regulations: {regs_str}
        compliance_status: "{compliance_status}"
        high_risk_columns: {json.dumps(high_risk_cols)}
        masked_columns: {json.dumps(list(masked_map.keys()))}
        audit_run_id: "{audit_run_id}"
        generated_by: "datapai_etl_v1"
        generated_at: "{generated_at}"
        # ── COMPLETE THE FIELDS BELOW BEFORE PRODUCTION ──────────────────
        review_required: true
        reviewed_by: null           # Required: name or email of reviewer
        reviewed_at: null           # Required: ISO date (YYYY-MM-DD)
        review_notes: null          # Optional: observations, decisions, lawful basis
        approved_for_production: false   # Set to true when review is complete

    tags:
{model_tags_yaml}

    columns:
{columns_block}
"""

    context_variables["schema_yaml"] = yaml_content

    return SwarmResult(
        values=f"Generated governance-annotated schema.yml for '{model_name}':\n\n{yaml_content}",
        context_variables=context_variables,
    )


def save_dbt_artifacts(context_variables: dict) -> SwarmResult:
    """
    Write the generated dbt SQL model and schema YAML to the dbt project directory.

    Target paths:
      <dbt_project_dir>/models/staging/<model_name>.sql
      <dbt_project_dir>/models/staging/schema.yml   (appended if exists)
    """
    model_name = context_variables.get("model_name")
    staging_sql = context_variables.get("staging_sql")
    schema_yaml = context_variables.get("schema_yaml")
    dbt_dir = context_variables.get("dbt_project_dir", DBT_PROJECT_DIR)

    if not model_name or not staging_sql:
        return SwarmResult(
            values="Missing model_name or staging_sql. Run generate_staging_sql first.",
            context_variables=context_variables,
        )

    staging_dir = Path(dbt_dir) / "models" / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []

    sql_path = staging_dir / f"{model_name}.sql"
    sql_path.write_text(staging_sql, encoding="utf-8")
    saved.append(str(sql_path))

    if schema_yaml:
        yaml_path = staging_dir / "schema.yml"
        if yaml_path.exists():
            existing = yaml_path.read_text(encoding="utf-8")
            yaml_path.write_text(existing.rstrip() + "\n\n" + schema_yaml, encoding="utf-8")
        else:
            yaml_path.write_text(schema_yaml, encoding="utf-8")
        saved.append(str(yaml_path))

    context_variables["dbt_artifacts_saved"] = saved
    context_variables["pipeline_status"] = "complete"

    return SwarmResult(
        values="dbt artifacts saved:\n" + "\n".join(f"  {p}" for p in saved),
        context_variables=context_variables,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def get_pipeline_summary(context_variables: dict) -> SwarmResult:
    """
    Return a human-readable summary of the completed ETL pipeline run.
    """
    lines: list[str] = ["ETL Pipeline Summary", "=" * 50]

    if context_variables.get("ingestion_status") == "success":
        lines.append(
            f"[INGEST]  {context_variables['table_name']} "
            f"({context_variables['row_count']:,} rows) → DuckDB '{context_variables['db_path']}'"
        )
    else:
        lines.append("[INGEST]  Not completed or failed")

    if "quality_report" in context_variables:
        issues = context_variables.get("quality_issues", [])
        status = "CLEAN" if not issues else f"{len(issues)} issue(s)"
        lines.append(f"[QUALITY] {status}")
        for issue in issues:
            lines.append(f"          - {issue}")
        pk = context_variables.get("primary_key")
        if pk:
            lines.append(f"          Primary key: '{pk}'")
    else:
        lines.append("[QUALITY] Not run")

    if context_variables.get("dbt_artifacts_saved"):
        artifacts = context_variables["dbt_artifacts_saved"]
        lines.append(f"[DBT]     {len(artifacts)} artifact(s) generated:")
        for a in artifacts:
            lines.append(f"          {a}")
    else:
        lines.append("[DBT]     Not run")

    lines.append("=" * 50)

    return SwarmResult(
        values="\n".join(lines),
        context_variables=context_variables,
    )

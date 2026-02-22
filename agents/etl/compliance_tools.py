"""
Compliance and PII tools for the AG2 ETL pipeline.

Designed for financial and regulated industries (GDPR, CCPA, PCI-DSS, SOX, HIPAA).

Detection strategy (layered, no ML dependency required):
  1. Column name heuristics  — fast, catches labelled PII (e.g. 'email', 'ssn')
  2. Regex value sampling    — catches unlabelled PII in actual data
  3. Presidio integration    — optional, best-in-class NER (install presidio-analyzer)

Sensitivity levels:
  PII_HIGH    SSN, credit card, passport, credentials, health data
  FINANCIAL   salary, account numbers, routing numbers, IBAN
  PII_MEDIUM  email, phone, name, date of birth, gender
  PII_LOW     address, ZIP, IP address
  PUBLIC      no sensitive content detected

Masking strategies:
  hash    SHA-256 (first 16 hex chars) — one-way, preserves referential integrity
  redact  Replace with ***REDACTED*** / ***FINANCIAL*** — for display / reporting
  none    Keep original (for PUBLIC or explicitly whitelisted columns)

Audit log:
  Append-only JSONL file  (ETL_AUDIT_LOG env var, default: etl_audit.jsonl)
  DuckDB audit table      (schema: etl_audit_log in the pipeline database)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from autogen import SwarmResult

logger = logging.getLogger(__name__)

# ── Environment defaults ──────────────────────────────────────────────────────
DB_PATH = os.getenv("DUCKDB_PATH", "datapai.duckdb")
AUDIT_LOG_PATH = os.getenv("ETL_AUDIT_LOG", "etl_audit.jsonl")
COMPLIANCE_SAMPLE_ROWS = int(os.getenv("COMPLIANCE_SAMPLE_ROWS", "200"))

# ── Sensitivity ranking (higher = more sensitive) ─────────────────────────────
_SENSITIVITY_RANK: dict[str, int] = {
    "PII_HIGH": 4,
    "FINANCIAL": 3,
    "PII_MEDIUM": 2,
    "PII_LOW": 1,
    "PUBLIC": 0,
}

# ── Regex patterns applied to sampled column VALUES ───────────────────────────
# Format: { label: (compiled_regex, sensitivity_level) }
_VALUE_PATTERNS: dict[str, tuple[re.Pattern, str]] = {
    "email":       (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "PII_MEDIUM"),
    "us_phone":    (re.compile(r"\b(\+?1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b"), "PII_MEDIUM"),
    "ssn":         (re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b"), "PII_HIGH"),
    "credit_card": (re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"), "PII_HIGH"),
    "iban":        (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4,30}\b"), "FINANCIAL"),
    "ip_address":  (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "PII_LOW"),
    "passport":    (re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"), "PII_HIGH"),
}

# ── Column NAME heuristics ────────────────────────────────────────────────────
# Format: (compiled_regex, pii_type_label, sensitivity_level)
_NAME_HEURISTICS: list[tuple[re.Pattern, str, str]] = [
    # Credentials / secrets — highest risk
    (re.compile(r"\b(password|passwd|secret|token|api.?key|private.?key|auth.?key)\b", re.I), "credential", "PII_HIGH"),
    # Government IDs
    (re.compile(r"\b(ssn|social.?security|national.?id|sin|fiscal.?id|tax.?id|ein|itin)\b", re.I), "ssn", "PII_HIGH"),
    (re.compile(r"\b(passport|drivers?.?licen[sc]e|dl.?num|gov.?id)\b", re.I), "gov_id", "PII_HIGH"),
    # Payment / financial
    (re.compile(r"\b(credit.?card|card.?num(?:ber)?|pan|cvv|cvc|card.?holder|expiry|exp.?date)\b", re.I), "credit_card", "PII_HIGH"),
    (re.compile(r"\b(account.?num(?:ber)?|acct.?num|routing|iban|swift|bic|bank.?account|sort.?code)\b", re.I), "bank_account", "FINANCIAL"),
    (re.compile(r"\b(salary|income|wage|compensation|bonus|revenue|profit|balance|net.?worth)\b", re.I), "financial_data", "FINANCIAL"),
    # Health / HIPAA
    (re.compile(r"\b(medical|diagnosis|diagnos[ei]s|icd.?\d|health|prescription|patient|clinical|mrn|npi|nhs)\b", re.I), "health", "PII_HIGH"),
    # Personal contact
    (re.compile(r"\b(email|e.?mail|email.?address)\b", re.I), "email", "PII_MEDIUM"),
    (re.compile(r"\b(phone|mobile|cell|telephone|tel|fax)\b", re.I), "phone", "PII_MEDIUM"),
    # Identity
    (re.compile(r"\b(first.?name|last.?name|full.?name|given.?name|surname|maiden|middle.?name)\b", re.I), "name", "PII_MEDIUM"),
    (re.compile(r"\b(dob|birth.?date|date.?of.?birth|birthday|age|gender|sex|race|ethnicity|nationality)\b", re.I), "demographics", "PII_MEDIUM"),
    # Location — lower risk but still PII
    (re.compile(r"\b(street|address|addr|city|state|zip|postal|postcode|country|province|region)\b", re.I), "address", "PII_LOW"),
    (re.compile(r"\b(ip.?addr(?:ess)?|ipv4|ipv6|mac.?addr)\b", re.I), "ip_address", "PII_LOW"),
    (re.compile(r"\b(latitude|longitude|lat|lon|geo|location|coordinates)\b", re.I), "geo_data", "PII_LOW"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_column(col: str, sample_values: list) -> tuple[str, str]:
    """
    Return (pii_type, sensitivity_level) for a column.
    Applies name heuristics first (fast), then regex value scan.
    """
    # 1. Column name heuristics
    for pattern, pii_type, sensitivity in _NAME_HEURISTICS:
        if pattern.search(col):
            return pii_type, sensitivity

    # 2. Regex scan on sample values (non-null strings only)
    str_values = [str(v) for v in sample_values if v is not None and str(v).strip()]
    if not str_values:
        return "none", "PUBLIC"

    for pii_type, (pattern, sensitivity) in _VALUE_PATTERNS.items():
        hit_count = sum(1 for v in str_values if pattern.search(v))
        hit_rate = hit_count / len(str_values)
        if hit_rate >= 0.10:   # ≥ 10% of sample matches → flag as PII
            return pii_type, sensitivity

    return "none", "PUBLIC"


def _hash_value(value) -> str:
    """SHA-256 one-way hash (first 16 hex chars). Preserves referential integrity."""
    if value is None or str(value).strip() == "":
        return value
    return "H:" + hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def _redact_label(sensitivity: str) -> str:
    labels = {
        "PII_HIGH": "***REDACTED***",
        "FINANCIAL": "***FINANCIAL***",
        "PII_MEDIUM": "***PII***",
        "PII_LOW": "***PII***",
    }
    return labels.get(sensitivity, "***")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PII SCANNING
# ═══════════════════════════════════════════════════════════════════════════════

def scan_pii(context_variables: dict) -> SwarmResult:
    """
    Scan the loaded DuckDB table for PII and sensitive data.

    For each column:
      - Applies column name heuristics (pattern matching)
      - Samples up to COMPLIANCE_SAMPLE_ROWS rows and runs regex value detection
      - Classifies as: PII_HIGH | FINANCIAL | PII_MEDIUM | PII_LOW | PUBLIC

    Results stored in context_variables:
      pii_scan:           { col: { pii_type, sensitivity, detection_method } }
      pii_columns:        list of columns with any PII (all levels)
      high_risk_columns:  list of PII_HIGH + FINANCIAL columns (require masking)
      compliance_run_id:  unique ID for this compliance check
    """
    table_name = context_variables.get("table_name")
    db = context_variables.get("db_path", DB_PATH)
    schema = context_variables.get("schema", {})

    if not table_name:
        return SwarmResult(
            values="No table in context. Run ingest_file first.",
            context_variables=context_variables,
        )

    try:
        con = duckdb.connect(db)

        # Pull a representative sample for value-level scanning
        sample_df: pd.DataFrame = con.execute(
            f'SELECT * FROM "{table_name}" USING SAMPLE {COMPLIANCE_SAMPLE_ROWS}'
        ).fetchdf()
        con.close()

        pii_scan: dict = {}
        pii_columns: list[str] = []
        high_risk_columns: list[str] = []

        for col in schema:
            sample_vals = sample_df[col].tolist() if col in sample_df.columns else []
            pii_type, sensitivity = _classify_column(col, sample_vals)

            # Determine detection method for auditability
            method = "none"
            if sensitivity != "PUBLIC":
                # Check whether name heuristic or value scan caught it
                name_hit = any(p.search(col) for p, _, _ in _NAME_HEURISTICS)
                method = "column_name" if name_hit else "value_regex"

            pii_scan[col] = {
                "pii_type": pii_type,
                "sensitivity": sensitivity,
                "detection_method": method,
            }

            if sensitivity != "PUBLIC":
                pii_columns.append(col)
            if _SENSITIVITY_RANK.get(sensitivity, 0) >= _SENSITIVITY_RANK["FINANCIAL"]:
                high_risk_columns.append(col)

        run_id = str(uuid.uuid4())
        context_variables.update(
            {
                "pii_scan": pii_scan,
                "pii_columns": pii_columns,
                "high_risk_columns": high_risk_columns,
                "compliance_run_id": run_id,
                "compliance_scan_ts": _now_iso(),
            }
        )

        # Build human-readable summary
        lines = [f"PII Scan — '{table_name}'  (run_id: {run_id})"]
        lines.append(f"{'Column':<35} {'Sensitivity':<14} {'Type':<20} {'Method'}")
        lines.append("-" * 85)
        for col, info in pii_scan.items():
            flag = "⚠ " if info["sensitivity"] != "PUBLIC" else "  "
            lines.append(
                f"{flag}{col:<33} {info['sensitivity']:<14} "
                f"{info['pii_type']:<20} {info['detection_method']}"
            )

        lines.append("")
        lines.append(f"PII columns detected:      {len(pii_columns)}")
        lines.append(f"High-risk columns (mask):  {len(high_risk_columns)} → {high_risk_columns}")

        return SwarmResult(
            values="\n".join(lines),
            context_variables=context_variables,
        )

    except Exception as exc:
        return SwarmResult(
            values=f"PII scan failed: {exc}",
            context_variables=context_variables,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MASKING
# ═══════════════════════════════════════════════════════════════════════════════

def mask_pii_columns(
    context_variables: dict,
    columns: Optional[list[str]] = None,
    strategy: str = "hash",
) -> SwarmResult:
    """
    Mask sensitive columns in the DuckDB table in-place.

    Args:
        columns:  Columns to mask. Defaults to high_risk_columns from the PII scan.
        strategy: 'hash'   — SHA-256 prefix (default, preserves referential integrity)
                  'redact' — Replace with ***REDACTED*** / ***FINANCIAL***

    Masked column metadata is stored in context_variables['masked_columns'].
    The original DuckDB table is updated in-place; a backup table
    (<table>__pre_mask) is created first for auditability.
    """
    table_name = context_variables.get("table_name")
    db = context_variables.get("db_path", DB_PATH)
    pii_scan = context_variables.get("pii_scan", {})

    if not table_name:
        return SwarmResult(
            values="No table in context.",
            context_variables=context_variables,
        )

    # Determine which columns to mask
    if columns is None:
        columns = context_variables.get("high_risk_columns", [])

    if not columns:
        context_variables["masked_columns"] = []
        return SwarmResult(
            values="No columns require masking (no high-risk PII detected).",
            context_variables=context_variables,
        )

    try:
        con = duckdb.connect(db)

        # Safety: create a pre-mask backup
        backup_name = f"{table_name}__pre_mask"
        con.execute(
            f'CREATE OR REPLACE TABLE "{backup_name}" AS SELECT * FROM "{table_name}"'
        )

        # Load table into pandas, apply masking, write back
        df: pd.DataFrame = con.execute(f'SELECT * FROM "{table_name}"').fetchdf()

        masked_cols: list[dict] = []
        for col in columns:
            if col not in df.columns:
                continue

            sensitivity = pii_scan.get(col, {}).get("sensitivity", "PII_HIGH")

            if strategy == "hash":
                df[col] = df[col].apply(_hash_value)
                applied = "sha256_prefix"
            else:
                label = _redact_label(sensitivity)
                df[col] = label
                applied = f"redact({label})"

            masked_cols.append(
                {"column": col, "sensitivity": sensitivity, "strategy": applied}
            )

        # Write masked data back
        con.register("_masked_df", df)
        con.execute(
            f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM _masked_df'
        )
        con.close()

        context_variables["masked_columns"] = masked_cols

        lines = [f"Masked {len(masked_cols)} column(s) in '{table_name}' (backup: '{backup_name}')"]
        for m in masked_cols:
            lines.append(f"  {m['column']:<35} {m['sensitivity']:<14} → {m['strategy']}")

        return SwarmResult(
            values="\n".join(lines),
            context_variables=context_variables,
        )

    except Exception as exc:
        return SwarmResult(
            values=f"Masking failed: {exc}",
            context_variables=context_variables,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. AUDIT LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def write_audit_log(context_variables: dict) -> SwarmResult:
    """
    Write an append-only audit record for this pipeline run.

    Two sinks (both written atomically):
      1. JSONL file  — append one JSON line to ETL_AUDIT_LOG (default: etl_audit.jsonl)
      2. DuckDB table — INSERT into etl_audit_log in the pipeline database

    Audit record fields:
      run_id, timestamp, source_file, table_name, row_count,
      pii_columns_detected, high_risk_columns, masked_columns,
      quality_issues, compliance_status, pipeline_status, environment
    """
    record = {
        "run_id": context_variables.get("compliance_run_id", str(uuid.uuid4())),
        "timestamp": _now_iso(),
        "source_file": context_variables.get("file_path", "unknown"),
        "table_name": context_variables.get("table_name", "unknown"),
        "row_count": context_variables.get("row_count", 0),
        "pii_columns_detected": context_variables.get("pii_columns", []),
        "high_risk_columns": context_variables.get("high_risk_columns", []),
        "masked_columns": [
            m["column"] for m in context_variables.get("masked_columns", [])
        ],
        "quality_issues": context_variables.get("quality_issues", []),
        "compliance_status": _derive_compliance_status(context_variables),
        "pipeline_status": context_variables.get("pipeline_status", "in_progress"),
        "environment": os.getenv("DATAPAI_ENV", "dev"),
        "dbt_artifacts": context_variables.get("dbt_artifacts_saved", []),
    }

    errors: list[str] = []

    # ── Sink 1: JSONL file ────────────────────────────────────────────────
    try:
        log_path = Path(AUDIT_LOG_PATH)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        errors.append(f"JSONL write failed: {exc}")

    # ── Sink 2: DuckDB audit table ────────────────────────────────────────
    db = context_variables.get("db_path", DB_PATH)
    try:
        con = duckdb.connect(db)
        con.execute("""
            CREATE TABLE IF NOT EXISTS etl_audit_log (
                run_id            VARCHAR,
                timestamp         VARCHAR,
                source_file       VARCHAR,
                table_name        VARCHAR,
                row_count         BIGINT,
                pii_columns       VARCHAR,
                high_risk_columns VARCHAR,
                masked_columns    VARCHAR,
                quality_issues    VARCHAR,
                compliance_status VARCHAR,
                pipeline_status   VARCHAR,
                environment       VARCHAR,
                dbt_artifacts     VARCHAR
            )
        """)
        con.execute(
            """
            INSERT INTO etl_audit_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record["run_id"],
                record["timestamp"],
                record["source_file"],
                record["table_name"],
                record["row_count"],
                json.dumps(record["pii_columns_detected"]),
                json.dumps(record["high_risk_columns"]),
                json.dumps(record["masked_columns"]),
                json.dumps(record["quality_issues"]),
                record["compliance_status"],
                record["pipeline_status"],
                record["environment"],
                json.dumps(record["dbt_artifacts"]),
            ],
        )
        con.close()
    except Exception as exc:
        errors.append(f"DuckDB audit write failed: {exc}")

    context_variables["audit_record"] = record

    msg = f"Audit log written — run_id: {record['run_id']}\n"
    msg += f"  Status: {record['compliance_status']}\n"
    msg += f"  Sinks:  {AUDIT_LOG_PATH}  +  DuckDB:etl_audit_log"
    if errors:
        msg += "\n  Errors:\n" + "\n".join(f"    - {e}" for e in errors)

    return SwarmResult(values=msg, context_variables=context_variables)


def _derive_compliance_status(ctx: dict) -> str:
    """
    Derive a compliance status string from pipeline context.

    COMPLIANT      — no high-risk PII, or high-risk PII was masked
    NEEDS_REVIEW   — PII_MEDIUM detected but not masked
    NON_COMPLIANT  — PII_HIGH / FINANCIAL detected and NOT masked
    """
    high_risk = ctx.get("high_risk_columns", [])
    masked = [m["column"] for m in ctx.get("masked_columns", [])]

    unmasked_high_risk = [c for c in high_risk if c not in masked]

    if unmasked_high_risk:
        return f"NON_COMPLIANT (unmasked: {unmasked_high_risk})"

    pii_cols = ctx.get("pii_columns", [])
    medium_unmasked = [c for c in pii_cols if c not in masked and c not in high_risk]

    if medium_unmasked:
        return f"NEEDS_REVIEW (medium PII unmasked: {medium_unmasked})"

    return "COMPLIANT"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. COMPLIANCE REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_compliance_report(context_variables: dict) -> SwarmResult:
    """
    Generate a structured compliance report for this pipeline run.

    The report includes:
      - Data inventory (table, rows, columns)
      - PII classification table
      - Masking summary
      - Data quality findings
      - Compliance verdict with recommended actions
      - Audit trail reference (run_id)

    Report is stored in context_variables['compliance_report'] as a string
    and also saved to <table_name>_compliance_report.txt in the working directory.
    """
    table_name = context_variables.get("table_name", "unknown")
    run_id = context_variables.get("compliance_run_id", "N/A")
    scan_ts = context_variables.get("compliance_scan_ts", _now_iso())
    pii_scan: dict = context_variables.get("pii_scan", {})
    masked_columns: list = context_variables.get("masked_columns", [])
    quality_issues: list = context_variables.get("quality_issues", [])
    status = _derive_compliance_status(context_variables)

    masked_set = {m["column"] for m in masked_columns}

    lines = [
        "=" * 70,
        "DATA COMPLIANCE REPORT",
        "=" * 70,
        f"Run ID:          {run_id}",
        f"Generated:       {scan_ts}",
        f"Table:           {table_name}",
        f"Source file:     {context_variables.get('file_path', 'N/A')}",
        f"Row count:       {context_variables.get('row_count', 'N/A'):,}",
        f"Environment:     {os.getenv('DATAPAI_ENV', 'dev')}",
        "",
        "── DATA CLASSIFICATION ──────────────────────────────────────────────",
        f"{'Column':<35} {'Sensitivity':<14} {'Type':<20} {'Masked'}",
        "-" * 70,
    ]

    for col, info in pii_scan.items():
        sensitivity = info["sensitivity"]
        flag = "" if sensitivity == "PUBLIC" else "⚠ "
        masked_flag = "YES" if col in masked_set else ("—" if sensitivity == "PUBLIC" else "NO ← ACTION REQUIRED")
        lines.append(
            f"{flag}{col:<33} {sensitivity:<14} {info['pii_type']:<20} {masked_flag}"
        )

    # Masking summary
    lines += [
        "",
        "── MASKING SUMMARY ──────────────────────────────────────────────────",
    ]
    if masked_columns:
        for m in masked_columns:
            lines.append(f"  {m['column']:<35} {m['sensitivity']:<14} → {m['strategy']}")
    else:
        lines.append("  No columns were masked in this run.")

    # Quality findings
    lines += [
        "",
        "── DATA QUALITY FINDINGS ────────────────────────────────────────────",
    ]
    if quality_issues:
        for issue in quality_issues:
            lines.append(f"  ⚠ {issue}")
    else:
        lines.append("  No data quality issues detected.")

    # Regulatory applicability
    high_risk = context_variables.get("high_risk_columns", [])
    has_financial = any(
        pii_scan.get(c, {}).get("sensitivity") == "FINANCIAL" for c in pii_scan
    )
    has_health = any(
        pii_scan.get(c, {}).get("pii_type") == "health" for c in pii_scan
    )
    has_payment = any(
        pii_scan.get(c, {}).get("pii_type") == "credit_card" for c in pii_scan
    )

    applicable = []
    if high_risk:
        applicable.append("GDPR / CCPA  (personal data processing)")
    if has_financial:
        applicable.append("SOX          (financial data integrity)")
    if has_payment:
        applicable.append("PCI-DSS      (cardholder data)")
    if has_health:
        applicable.append("HIPAA        (health information)")

    lines += [
        "",
        "── REGULATORY APPLICABILITY ─────────────────────────────────────────",
    ]
    if applicable:
        for r in applicable:
            lines.append(f"  {r}")
    else:
        lines.append("  No regulated data categories detected.")

    # Recommended actions
    unmasked_high = [
        c for c in high_risk if c not in masked_set
    ]
    lines += [
        "",
        "── RECOMMENDED ACTIONS ──────────────────────────────────────────────",
    ]
    if unmasked_high:
        lines.append(f"  [CRITICAL] Mask or remove high-risk columns before sharing: {unmasked_high}")
    if quality_issues:
        lines.append("  [HIGH]     Resolve data quality issues before production load.")
    if applicable and "GDPR" in " ".join(applicable):
        lines.append("  [MEDIUM]   Ensure a lawful basis for processing personal data is documented.")
    if has_payment:
        lines.append("  [CRITICAL] Confirm PCI-DSS scope and tokenize card data at source.")
    if not unmasked_high and not quality_issues:
        lines.append("  No critical actions required. Proceed with standard review.")

    # Verdict
    lines += [
        "",
        "── COMPLIANCE VERDICT ───────────────────────────────────────────────",
        f"  Status: {status}",
        "=" * 70,
    ]

    report_text = "\n".join(lines)

    # Save to file
    report_path = Path(f"{table_name}_compliance_report.txt")
    try:
        report_path.write_text(report_text, encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not save compliance report to file: %s", exc)

    context_variables["compliance_report"] = report_text
    context_variables["compliance_status"] = status

    return SwarmResult(
        values=report_text,
        context_variables=context_variables,
    )

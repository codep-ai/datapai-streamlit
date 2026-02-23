from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import boto3


@dataclass
class AthenaTableInfo:
    database: str
    name: str
    table_type: str  # e.g., EXTERNAL_TABLE
    format: str      # iceberg|parquet|csv|orc|json|avro|unknown
    columns: List[Tuple[str, str]]
    location: Optional[str] = None


def _normalize_str(v: Optional[str]) -> str:
    return (v or "").strip()


def _safe_lower(v: Optional[str]) -> str:
    return _normalize_str(v).lower()


def _is_iceberg_from_glue(params: Dict[str, str], table_type: str) -> bool:
    """
    Best-effort detection. Different creation paths may set:
      - Parameters['table_type'] = 'ICEBERG'
      - Parameters['TABLE_TYPE'] = 'ICEBERG'
      - other keys containing 'iceberg'
      - TableType itself sometimes shows 'ICEBERG'
    """
    tt = _safe_lower(table_type)
    if "iceberg" in tt:
        return True

    if not params:
        return False

    # Direct keys commonly used
    for k in ("table_type", "TABLE_TYPE", "table.format", "table_format", "TABLE_FORMAT"):
        if _safe_lower(params.get(k)) == "iceberg":
            return True

    # Fallback: any key/value mentions iceberg
    blob = " ".join([f"{k}={v}" for k, v in params.items()]).lower()
    return "iceberg" in blob


def _infer_format_from_classification(params: Dict[str, str]) -> Optional[str]:
    """
    Glue classifiers often set Parameters['classification'] to parquet/csv/orc/json/avro, etc.
    """
    if not params:
        return None
    cls = _safe_lower(params.get("classification") or params.get("Classification"))
    if cls in {"parquet", "csv", "orc", "json", "avro"}:
        return cls
    # Some connectors use slightly different names
    if cls in {"text", "textfile"}:
        return "csv"
    return None


def _infer_format_from_storage_descriptor(sd: Dict) -> Optional[str]:
    """
    Fallback inference using InputFormat and SerDe library.
    """
    input_fmt = _safe_lower(sd.get("InputFormat"))
    if input_fmt:
        if "parquet" in input_fmt:
            return "parquet"
        if "orc" in input_fmt:
            return "orc"
        # CSV in Athena is usually TextInputFormat
        if "textinputformat" in input_fmt:
            return "csv"
        if "json" in input_fmt:
            return "json"
        if "avro" in input_fmt:
            return "avro"

    serde = sd.get("SerdeInfo") or {}
    serde_lib = _safe_lower(serde.get("SerializationLibrary"))
    if serde_lib:
        if "parquet" in serde_lib:
            return "parquet"
        if "orc" in serde_lib:
            return "orc"
        if "json" in serde_lib:
            return "json"
        if "avro" in serde_lib:
            return "avro"
        # Common CSV SerDes
        if "lazy" in serde_lib or "csv" in serde_lib:
            return "csv"

    return None


def detect_table_format(table: Dict) -> str:
    """
    Return a stable, lowercase table format label:
      iceberg|parquet|csv|orc|json|avro|unknown
    Never raises.
    """
    params = table.get("Parameters") or {}
    table_type = _normalize_str(table.get("TableType"))
    sd = table.get("StorageDescriptor") or {}

    if _is_iceberg_from_glue(params=params, table_type=table_type):
        return "iceberg"

    fmt = _infer_format_from_classification(params)
    if fmt:
        return fmt

    fmt = _infer_format_from_storage_descriptor(sd)
    if fmt:
        return fmt

    return "unknown"


def list_tables_from_glue(
    databases: List[str],
    region_name: Optional[str] = None,
    max_tables_per_db: int = 50,
) -> List[AthenaTableInfo]:
    glue = boto3.client("glue", region_name=region_name) if region_name else boto3.client("glue")
    out: List[AthenaTableInfo] = []

    for db in databases:
        next_token = None
        fetched = 0

        while True:
            remaining = max_tables_per_db - fetched
            if remaining <= 0:
                break

            kwargs = {"DatabaseName": db, "MaxResults": min(100, remaining)}
            if next_token:
                kwargs["NextToken"] = next_token

            resp = glue.get_tables(**kwargs)

            for t in resp.get("TableList", []) or []:
                if fetched >= max_tables_per_db:
                    break

                name = t.get("Name") or ""
                sd = t.get("StorageDescriptor") or {}
                cols = [(c.get("Name") or "", c.get("Type") or "") for c in (sd.get("Columns") or [])]
                cols = [(c, ty) for (c, ty) in cols if c]  # drop empty

                table_type = _normalize_str(t.get("TableType"))
                fmt = detect_table_format(t)

                location = sd.get("Location")
                out.append(
                    AthenaTableInfo(
                        database=db,
                        name=name,
                        table_type=table_type,
                        format=fmt,
                        columns=cols,
                        location=location,
                    )
                )
                fetched += 1

            next_token = resp.get("NextToken")
            if not next_token:
                break

    return out


def build_schema_context(
    databases: List[str],
    region_name: Optional[str] = None,
    max_tables_per_db: int = 25,
    max_columns_per_table: int = 30,
) -> str:
    """
    Return a compact, LLM-friendly schema description.
    Iceberg hints are included only if Iceberg tables are detected.
    Safe for CSV/Parquet/etc.
    """
    tables = list_tables_from_glue(
        databases=databases,
        region_name=region_name,
        max_tables_per_db=max_tables_per_db,
    )

    formats_present: Set[str] = {t.format for t in tables if t.format}
    iceberg_tables = [t for t in tables if t.format == "iceberg"]

    lines: List[str] = []
    lines.append("You are generating SQL for Amazon Athena.")
    lines.append("Use Athena SQL dialect (Presto/Trino-style).")
    lines.append("Prefer partition filters and LIMIT when exploring to reduce scanned data.")

    # Only mention Iceberg capabilities if Iceberg tables exist
    if iceberg_tables:
        lines.append(
            "Some tables are Apache Iceberg. You MAY use Iceberg features when relevant:\n"
            "- Time travel: FOR TIMESTAMP AS OF / FOR VERSION AS OF\n"
            "- Iceberg metadata tables: $snapshots, $files (if supported)\n"
            "- MERGE/UPDATE/DELETE may be supported depending on Athena capabilities\n"
            "Do NOT use Iceberg-specific syntax on non-Iceberg tables."
        )

    # Optional: short statement of other formats (useful for grounding)
    non_unknown = sorted([f for f in formats_present if f and f != "unknown" and f != "iceberg"])
    if non_unknown:
        lines.append(f"Other table formats present: {', '.join(non_unknown)}.")

    lines.append("\nSchema (from AWS Glue Data Catalog):")
    for t in tables:
        tag = f" [{t.format.upper()}]" if t.format and t.format != "unknown" else ""
        cols = t.columns[:max_columns_per_table]
        col_str = ", ".join([f"{c}:{ty}" for c, ty in cols])
        if len(t.columns) > max_columns_per_table:
            col_str += ", ..."
        lines.append(f"- {t.database}.{t.name}{tag}({col_str})")

    if iceberg_tables:
        lines.append("\nIceberg tables:")
        for t in iceberg_tables[:30]:
            lines.append(f"- {t.database}.{t.name}")

    return "\n".join(lines)


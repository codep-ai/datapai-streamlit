# agents/file_ingest_agent.py

from __future__ import annotations

import os
import io
import typing as t
import mimetypes
import json
import datetime as dt

import pandas as pd
import boto3
from botocore.exceptions import ClientError

# from agent_base import BaseAgent
from llm_client import BaseChatClient, RouterChatClient
from tools import tool  # your global @tool decorator + registry


# -------------------------------------------------------------------
# CONFIG / DEFAULTS
# -------------------------------------------------------------------

DEFAULT_BUCKET = os.environ.get("FILE_INGEST_DEFAULT_BUCKET", "codepais3")
RAW_PREFIX = os.environ.get("FILE_INGEST_RAW_PREFIX", "upload/")
CLEANED_PREFIX = os.environ.get("FILE_INGEST_CLEANED_PREFIX", "cleaned/")
DEFAULT_GLUE_DB = os.environ.get("FILE_INGEST_DEFAULT_GLUE_DB", "datapai_raw")

# Metadata store for file status tracking
FILE_INGEST_META_STORE = os.environ.get("FILE_INGEST_META_STORE", "dynamodb")  # "none" | "dynamodb"
FILE_INGEST_DDB_TABLE = os.environ.get("FILE_INGEST_DDB_TABLE", "file_ingest_metadata")

# Optional JSON config for table metadata (PK, load mode, etc.)
FILE_INGEST_CONFIG_S3 = os.environ.get("FILE_INGEST_CONFIG_S3", "")  # e.g. s3://.../table_ingest_config.json

_AWS_REGION = os.environ.get("AWS_REGION")
_DDB_TABLE_CACHE = None  # lazy cache for the DynamoDB table handle


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def _parse_s3_uri(s3_uri: str) -> t.Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {s3_uri}")
    _, _, path = s3_uri.partition("s3://")
    bucket, _, key = path.partition("/")
    return bucket, key


def _load_table_ingest_config() -> dict:
    """
    Load JSON-based ingest config from S3, if configured.
    Structure example:
      {
        "customers": {
          "primary_keys": ["customer_id"],
          "load_mode": "delta",          # or "full"
          "partition_spec": ["ingest_date"],
          "date_columns": ["created_at", "updated_at"],
          "numeric_columns": ["amount"]
        },
        "orders": { ... }
      }
    """
    if not FILE_INGEST_CONFIG_S3:
        return {}

    if _AWS_REGION:
        s3 = boto3.client("s3", region_name=_AWS_REGION)
    else:
        s3 = boto3.client("s3")

    bucket, key = _parse_s3_uri(FILE_INGEST_CONFIG_S3)
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8")
    return json.loads(body)


def _get_table_config(table_name: str) -> dict:
    cfg = _load_table_ingest_config()
    return cfg.get(table_name, {})


def _ddb_table():
    """
    Return the DynamoDB table used for file ingest metadata.
    - Respects FILE_INGEST_META_STORE env.
    - Lazily creates the table on first use if it does not exist.
    """
    global _DDB_TABLE_CACHE

    if FILE_INGEST_META_STORE != "dynamodb":
        return None

    if _DDB_TABLE_CACHE is not None:
        return _DDB_TABLE_CACHE

    # Build clients/resources with optional region
    if _AWS_REGION:
        ddb_resource = boto3.resource("dynamodb", region_name=_AWS_REGION)
        ddb_client = boto3.client("dynamodb", region_name=_AWS_REGION)
    else:
        ddb_resource = boto3.resource("dynamodb")
        ddb_client = boto3.client("dynamodb")

    # Check if table exists
    try:
        ddb_client.describe_table(TableName=FILE_INGEST_DDB_TABLE)
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            # Some other error (permissions, throttling, etc.)
            raise

        # Table does not exist: create it
        ddb_client.create_table(
            TableName=FILE_INGEST_DDB_TABLE,
            AttributeDefinitions=[
                {"AttributeName": "file_id", "AttributeType": "S"},
            ],
            KeySchema=[
                {"AttributeName": "file_id", "KeyType": "HASH"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        # Wait until table is ACTIVE
        waiter = ddb_client.get_waiter("table_exists")
        waiter.wait(TableName=FILE_INGEST_DDB_TABLE)

    _DDB_TABLE_CACHE = ddb_resource.Table(FILE_INGEST_DDB_TABLE)
    return _DDB_TABLE_CACHE


# -------------------------------------------------------------------
# LOW-LEVEL STEP TOOLS (fi_*)  – these are the "steps"
# -------------------------------------------------------------------

@tool()
def fi_detect_file_type(source_path: str) -> dict:
    """
    Detect file kind (structured/unstructured) and format based on extension.
    Returns:
      {
        "kind": "structured" | "unstructured",
        "format": "csv" | "json" | "txt" | "parquet" | "pdf" | "unknown"
      }
    """
    ext = os.path.splitext(source_path)[1].lower().strip(".")
    structured_exts = {"csv", "json", "txt", "parquet"}
    unstructured_exts = {"pdf", "doc", "docx", "ppt", "pptx", "md"}

    if ext in structured_exts:
        kind = "structured"
    elif ext in unstructured_exts:
        kind = "unstructured"
    else:
        mime, _ = mimetypes.guess_type(source_path)
        if mime and mime.startswith(("text/", "application/json")):
            kind = "structured"
        else:
            kind = "unstructured"

    fmt = ext if ext in structured_exts.union(unstructured_exts) else "unknown"

    return {
        "kind": kind,
        "format": fmt,
        "source_path": source_path,
    }


@tool()
def fi_check_file_already_processed(
    table_name: str,
    source_path: str,
) -> dict:
    """
    Check metadata store to see if the file has already been processed
    (idempotency / reload).

    Returns:
      {
        "should_process": True|False,
        "reason": "..."
      }
    """
    ddb = _ddb_table()
    if ddb is None:
        # If no metadata store configured, always process
        return {"should_process": True, "reason": "meta_store_disabled"}

    pk = f"{table_name}#{source_path}"
    resp = ddb.get_item(Key={"file_id": pk})
    item = resp.get("Item")
    if item and item.get("status") == "processed":
        return {
            "should_process": False,
            "reason": "already_processed",
        }
    return {"should_process": True, "reason": "not_processed_yet"}


@tool()
def fi_mark_file_status(
    table_name: str,
    source_path: str,
    status: str,
    extra: t.Optional[dict] = None,
) -> dict:
    """
    Record file ingestion status in metadata store (e.g. DynamoDB).
    Status examples: "processing", "processed", "failed".
    """
    ddb = _ddb_table()
    if ddb is None:
        return {
            "status": "meta_store_disabled",
            "table_name": table_name,
            "source_path": source_path,
        }

    pk = f"{table_name}#{source_path}"
    item = {
        "file_id": pk,
        "table_name": table_name,
        "source_path": source_path,
        "status": status,
        "updated_at": dt.datetime.utcnow().isoformat() + "Z",
    }
    if extra:
        item["extra"] = extra

    ddb.put_item(Item=item)
    return {"status": "ok", "stored_status": status}


@tool()
def fi_upload_to_s3_raw(
    source_path: str,
    bucket: t.Optional[str] = None,
    prefix: str = RAW_PREFIX,
) -> dict:
    """
    Upload a local file to S3 under the 'raw/upload' area.

    If source_path already starts with 's3://', it is returned unchanged.
    """
    bucket = bucket or DEFAULT_BUCKET

    if source_path.startswith("s3://"):
        return {"raw_s3_uri": source_path}

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Local file not found: {source_path}")

    filename = os.path.basename(source_path)
    key = prefix.rstrip("/") + "/" + filename

    if _AWS_REGION:
        s3 = boto3.client("s3", region_name=_AWS_REGION)
    else:
        s3 = boto3.client("s3")

    with open(source_path, "rb") as f:
        s3.upload_fileobj(f, bucket, key)

    raw_s3_uri = f"s3://{bucket}/{key}"
    return {"raw_s3_uri": raw_s3_uri}


@tool()
def fi_convert_structured_to_parquet_cleaned(
    raw_s3_uri: str,
    table_name: t.Optional[str] = None,
    bucket: t.Optional[str] = None,
    cleaned_prefix: str = CLEANED_PREFIX,
) -> dict:
    """
    Load structured data from raw S3 (CSV/JSON/TXT), apply:
      - PK-aware dedupe (if primary_keys defined in config)
      - date/numeric normalization (if configured)
    and write back as Parquet under the 'cleaned' prefix.

    Uses table metadata from FILE_INGEST_CONFIG_S3 if available.
    """
    bucket = bucket or DEFAULT_BUCKET

    if not raw_s3_uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {raw_s3_uri}")

    src_bucket, src_key = _parse_s3_uri(raw_s3_uri)

    if _AWS_REGION:
        s3 = boto3.client("s3", region_name=_AWS_REGION)
    else:
        s3 = boto3.client("s3")

    obj = s3.get_object(Bucket=src_bucket, Key=src_key)
    body = obj["Body"].read()

    ext = os.path.splitext(src_key)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(io.BytesIO(body))
    elif ext == ".json":
        df = pd.read_json(io.BytesIO(body), lines=True)
    elif ext in (".txt", ".log"):
        df = pd.read_csv(io.BytesIO(body), sep="\t", header=None)
    else:
        df = pd.read_csv(io.BytesIO(body))

    table_cfg = _get_table_config(table_name or "")

    # PK de-duplication
    pk_cols = table_cfg.get("primary_keys") or []
    if pk_cols and all(c in df.columns for c in pk_cols):
        df = df.drop_duplicates(subset=pk_cols)
    else:
        # fallback global dedupe
        df = df.drop_duplicates()

    # Date normalization
    date_cols = table_cfg.get("date_columns") or []
    for col in date_cols:
        if col in df.columns:
            df[col] = (
                pd.to_datetime(df[col], errors="coerce")
                .dt.strftime("%Y-%m-%dT%H:%M:%S")
            )

    # Numeric normalization
    num_cols = table_cfg.get("numeric_columns") or []
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build cleaned key: cleaned/<table_name>/<table_name>.parquet
    effective_table_name = table_name or "unknown_table"
    cleaned_prefix = cleaned_prefix.rstrip("/") + f"/{effective_table_name}/"
    cleaned_key = f"{cleaned_prefix}{effective_table_name}.parquet"

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)

    # BUGFIX: use 'buf' instead of undefined 'out_buf'
    s3.put_object(Bucket=bucket, Key=cleaned_key, Body=buf.getvalue())

    cleaned_s3_uri = f"s3://{bucket}/{cleaned_key}"
    return {
        "cleaned_s3_uri": cleaned_s3_uri,
        "row_count": len(df),
        "primary_keys_used": pk_cols,
    }


@tool()
def fi_run_data_validation(
    table_name: str,
    cleaned_s3_uri: str,
    min_rows: int = 1,
) -> dict:
    """
    Basic data validation step:
      - ensure row_count >= min_rows
      - check no-null PK (if pk defined in config)
      - can be extended to more complex rules / Great Expectations.
    """
    table_cfg = _get_table_config(table_name)
    pk_cols = table_cfg.get("primary_keys") or []

    bucket, key = _parse_s3_uri(cleaned_s3_uri)

    if _AWS_REGION:
        s3 = boto3.client("s3", region_name=_AWS_REGION)
    else:
        s3 = boto3.client("s3")

    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    df = pd.read_parquet(io.BytesIO(body))

    issues: t.List[str] = []
    row_count = len(df)

    if row_count < min_rows:
        issues.append(f"Row count {row_count} < min_rows {min_rows}")

    if pk_cols:
        for col in pk_cols:
            if col not in df.columns:
                issues.append(f"PK column '{col}' missing from data")
            else:
                nulls = df[col].isna().sum()
                if nulls > 0:
                    issues.append(f"PK column '{col}' has {nulls} null values")

    status = "passed" if not issues else "failed"

    return {
        "status": status,
        "row_count": row_count,
        "issues": issues,
    }


@tool()
def fi_register_glue_table(
    database_name: str,
    table_name: str,
    s3_location: str,
    classification: str = "parquet",
) -> dict:
    """
    Register or update a Glue EXTERNAL table pointing to the cleaned S3 location.
    Generic Parquet Glue table (non-Iceberg).
    """
    if _AWS_REGION:
        glue = boto3.client("glue", region_name=_AWS_REGION)
    else:
        glue = boto3.client("glue")

    storage_descriptor = {
        "Columns": [
            # Placeholder schema; real impl should infer from df
            {"Name": "placeholder_col", "Type": "string"},
        ],
        "Location": s3_location,
        "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
        "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
        "SerdeInfo": {
            "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
            "Parameters": {"classification": classification},
        },
    }

    try:
        glue.create_table(
            DatabaseName=database_name,
            TableInput={
                "Name": table_name,
                "StorageDescriptor": storage_descriptor,
                "TableType": "EXTERNAL_TABLE",
                "Parameters": {"classification": classification},
            },
        )
        action = "created"
    except glue.exceptions.AlreadyExistsException:
        glue.update_table(
            DatabaseName=database_name,
            TableInput={
                "Name": table_name,
                "StorageDescriptor": storage_descriptor,
                "TableType": "EXTERNAL_TABLE",
                "Parameters": {"classification": classification},
            },
        )
        action = "updated"

    return {
        "status": "ok",
        "action": action,
        "database_name": database_name,
        "table_name": table_name,
        "s3_location": s3_location,
        "table_type": "external_parquet",
    }


@tool()
def fi_create_iceberg_table(
    database_name: str,
    table_name: str,
    s3_location: str,
) -> dict:
    """
    Create or update an AWS Glue Iceberg table over the cleaned Parquet location.
    Uses simple table parameters; real impl should infer schema properly.
    """
    if _AWS_REGION:
        glue = boto3.client("glue", region_name=_AWS_REGION)
    else:
        glue = boto3.client("glue")

    iceberg_parameters = {
        "table_type": "ICEBERG",
        "EXTERNAL": "TRUE",
        "format": "iceberg",
        "write_compression": "SNAPPY",
    }

    storage_descriptor = {
        "Columns": [
            # Placeholder; you can later replace with real schema
            {"Name": "placeholder_col", "Type": "string"},
        ],
        "Location": s3_location,
        "InputFormat": "org.apache.iceberg.mr.hive.HiveIcebergInputFormat",
        "OutputFormat": "org.apache.iceberg.mr.hive.HiveIcebergOutputFormat",
        "SerdeInfo": {
            "SerializationLibrary": "org.apache.iceberg.mr.hive.HiveIcebergSerDe",
            "Parameters": {},
        },
    }

    table_input = {
        "Name": table_name,
        "StorageDescriptor": storage_descriptor,
        "TableType": "EXTERNAL_TABLE",
        "Parameters": iceberg_parameters,
    }

    try:
        glue.create_table(
            DatabaseName=database_name,
            TableInput=table_input,
        )
        action = "created"
    except glue.exceptions.AlreadyExistsException:
        glue.update_table(
            DatabaseName=database_name,
            TableInput=table_input,
        )
        action = "updated"

    return {
        "status": "ok",
        "action": action,
        "database_name": database_name,
        "table_name": table_name,
        "s3_location": s3_location,
        "table_type": "iceberg",
    }


@tool()
def fi_delegate_unstructured_to_knowledge_agent(
    s3_uri: str,
    collection_name: str = "default_knowledge",
) -> dict:
    """
    Placeholder tool to delegate unstructured content to the knowledge ingest pipeline.
    """
    # TODO: integrate with your real knowledge_ingest_agent tool
    return {
        "status": "pending_integration",
        "info": f"Unstructured file at {s3_uri} should be sent to knowledge ingest agent.",
        "collection_name": collection_name,
    }


@tool()
def fi_run_dlt_pipeline(
    pipeline_name: str,
    config: t.Optional[dict] = None,
) -> dict:
    """
    Optional: hook into a dlt pipeline for more advanced or parallel ingestion.

    This is a stub. You can implement:
      import dlt
      pipeline = dlt.pipeline(pipeline_name, ...)
      pipeline.run(...)

    For now, it just returns a placeholder.
    """
    return {
        "status": "not_implemented",
        "pipeline_name": pipeline_name,
        "config": config or {},
    }


# -------------------------------------------------------------------
# FILE INGEST AGENT SYSTEM PROMPT
# -------------------------------------------------------------------

FILE_INGEST_SYSTEM_PROMPT = """
You are the FILE_INGEST_AGENT for a modern data lakehouse.

Your responsibilities for EACH file:
  - Check metadata to see if it was already processed (idempotency).
  - If not processed:
      * detect file type (structured / unstructured)
      * for structured:
          - upload/normalize to S3 'raw'
          - clean data (PK-based dedupe if configured; otherwise global dedupe)
          - normalize dates & numeric formats using config
          - write cleaned Parquet to 'cleaned' area
          - run basic validation (row count, PK nulls)
          - register/update Glue table
          - create/update Iceberg table
      * for unstructured:
          - upload/normalize to S3
          - delegate to knowledge ingest pipeline
  - Mark file status as processing / processed / failed in metadata store.

IMPORTANT CONSTRAINTS:
  - NEVER call the same tool with the same arguments more than once.
  - In particular, call fi_check_file_already_processed AT MOST ONCE per
    (table_name, source_path) combination. After you have its result in history,
    you must move on to the next step using that result.
  - Always read the 'history' list and base your next decision on it.

You MUST respond in STRICT JSON ONLY via the BaseAgent protocol:

To call a tool:
{
  "type": "tool_call",
  "tool_name": "<tool_name>",
  "args": { ... }
}

To finish:
{
  "type": "final_answer",
  "result": "Human-readable summary of what you did, including S3 paths, Glue/Iceberg tables, validation result, and metadata status."
}

Typical sequence for STRUCTURED data:
  1) fi_check_file_already_processed
      - if should_process=false, finish with final_answer
  2) fi_detect_file_type
  3) fi_mark_file_status(status="processing")
  4) fi_upload_to_s3_raw
  5) fi_convert_structured_to_parquet_cleaned
  6) fi_run_data_validation
  7) fi_register_glue_table
  8) fi_create_iceberg_table
  9) fi_mark_file_status(status="processed")

Typical sequence for UNSTRUCTURED data:
  1) fi_check_file_already_processed
  2) fi_detect_file_type
  3) fi_mark_file_status(status="processing")
  4) fi_upload_to_s3_raw
  5) fi_delegate_unstructured_to_knowledge_agent
  6) fi_mark_file_status(status="processed")

Use `history` to see previous results and continue the pipeline.
Never invent tool names. Use only the ones in available_tools.
""".strip()


# -------------------------------------------------------------------
# FILE INGEST AGENT CLASS (domain orchestrator)
# -------------------------------------------------------------------

class FileIngestAgent:
    """
    Domain-specific agent that orchestrates fi_* tools to build
    a realistic file ingestion pipeline into your lakehouse.

    It wraps BaseAgent internally (composition) to avoid circular imports.
    """

    def __init__(
        self,
        llm: t.Optional[BaseChatClient] = None,
        max_steps: int = 12,
        temperature: float = 0.1,
    ):
        self.llm = llm or RouterChatClient()
        self.max_steps = max_steps
        self.temperature = temperature

    def run(self, goal: str, context: t.Optional[dict] = None) -> dict:
        # Lazy import here to break circular dependency:
        from agent_base import BaseAgent

        base = BaseAgent(
            name="file_ingest_agent",
            llm=self.llm,
            system_prompt=FILE_INGEST_SYSTEM_PROMPT,
            max_steps=self.max_steps,
            temperature=self.temperature,
        )
        return base.run(goal=goal, context=context or {})


# -------------------------------------------------------------------
# HIGH-LEVEL TOOL ENTRYPOINT FOR SUPERVISOR
# -------------------------------------------------------------------

@tool()
def file_ingest_agent(
    source_path: str,
    table_name: t.Optional[str] = None,
    database_name: t.Optional[str] = None,
    bucket: t.Optional[str] = None,
) -> dict:
    """
    High-level tool that runs the FileIngestAgent on a given file.

    This is what SupervisorAgent will typically call.
    """
    bucket = bucket or DEFAULT_BUCKET
    database_name = database_name or DEFAULT_GLUE_DB

    goal = (
        f"Ingest file {source_path} into S3 bucket '{bucket}' with a cleaned Parquet "
        f"layer and an Iceberg table in Glue catalog. "
        f"Table name: {table_name}, database: {database_name}. "
        f"Use PK/metadata from ingest config if available."
    )

    context = {
        "source_path": source_path,
        "table_name": table_name,
        "database_name": database_name,
        "bucket": bucket,
    }

    agent = FileIngestAgent(llm=RouterChatClient())
    return agent.run(goal=goal, context=context)


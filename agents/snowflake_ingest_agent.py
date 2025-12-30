# agents/snowflake_ingest_agent.py

from __future__ import annotations

import io
import typing as t
import os

import boto3
import pandas as pd

from tools import tool
from connect_db import connect_to_db


def _parse_s3_uri(s3_uri: str) -> t.Tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Expected s3:// URI, got: {s3_uri}")
    _, _, path = s3_uri.partition("s3://")
    bucket, _, key = path.partition("/")
    return bucket, key


def _get_glue_table(glue_database: str, glue_table: str) -> dict:
    glue = boto3.client("glue")
    resp = glue.get_table(DatabaseName=glue_database, Name=glue_table)
    return resp["Table"]


# -------------------------------------------------------------------
# 1) Snowflake ingestion as a NORMAL internal table
# -------------------------------------------------------------------

@tool()
def sf_ingest_from_glue_to_table(
    glue_database: str,
    glue_table: str,
    sf_database: t.Optional[str] = None,
    sf_schema: t.Optional[str] = None,
    sf_table: t.Optional[str] = None,
    load_mode: str = "append",
) -> dict:
    """
    Ingest data defined by a Glue table into a NORMAL Snowflake table.

    Steps:
      1) Read Glue catalog to get the S3 location.
      2) Load the data from S3 into a pandas DataFrame (assumes Parquet or CSV).
      3) Use Snowflake `write_pandas` to load into a regular Snowflake table.

    Notes / simplifications:
      - Assumes a single file at the Glue location (or one main Parquet file).
      - `load_mode` currently supports:
          * "append": append to existing table or auto-create.
          * "truncate": truncate table first, then load.
      - Uses env vars:
          * SNOWFLAKE_DATABASE (fallback if sf_database is None)
          * SNOWFLAKE_SCHEMA   (fallback if sf_schema is None)
    """
    # Resolve Snowflake DB/Schema
    sf_database = sf_database or os.environ.get("SNOWFLAKE_DATABASE", "DATAPAI")
    sf_schema = sf_schema or os.environ.get("SNOWFLAKE_SCHEMA", "DATAPAI")

    # Resolve target table name
    sf_table = sf_table or glue_table

    # 1) Get Glue table metadata
    table = _get_glue_table(glue_database, glue_table)
    location = table["StorageDescriptor"]["Location"]  # e.g. s3://bucket/path/file.parquet

    bucket, key = _parse_s3_uri(location)

    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()

    # Simple format inference from extension
    ext = os.path.splitext(key)[1].lower()

    if ext == ".parquet":
        df = pd.read_parquet(io.BytesIO(body))
    elif ext == ".csv":
        df = pd.read_csv(io.BytesIO(body))
    else:
        # fallback (you can extend this logic)
        df = pd.read_parquet(io.BytesIO(body))

    # 2) Connect to Snowflake
    from snowflake.connector.pandas_tools import write_pandas as sf_write_pandas

    conn = connect_to_db("Snowflake")

    # 3) Optional truncate if load_mode == "truncate"
    if load_mode.lower() == "truncate":
        with conn.cursor() as cur:
            cur.execute(f'USE DATABASE "{sf_database}"')
            cur.execute(f'USE SCHEMA "{sf_schema}"')
            cur.execute(f'TRUNCATE TABLE IF EXISTS "{sf_table.upper()}"')

    # 4) Write via write_pandas
    success, _, num_rows, _ = sf_write_pandas(
        conn,
        df,
        sf_table.upper(),
        database=sf_database,
        schema=sf_schema,
        auto_create_table=True,
    )

    return {
        "status": "ok" if success else "failed",
        "rows_written": int(num_rows),
        "sf_database": sf_database,
        "sf_schema": sf_schema,
        "sf_table": sf_table.upper(),
        "glue_database": glue_database,
        "glue_table": glue_table,
        "s3_location": location,
        "load_mode": load_mode,
    }


# -------------------------------------------------------------------
# 2) Snowflake EXTERNAL TABLE using Glue metadata
# -------------------------------------------------------------------

@tool()
def sf_register_external_table_from_glue(
    glue_database: str,
    glue_table: str,
    sf_database: t.Optional[str] = None,
    sf_schema: t.Optional[str] = None,
    sf_table: t.Optional[str] = None,
    sf_external_stage: t.Optional[str] = None,
    file_format: str = "PARQUET",
) -> dict:
    """
    Create or replace a Snowflake EXTERNAL TABLE that points to the S3
    location defined in the Glue catalog.

    Assumptions:
      - A Snowflake STORAGE INTEGRATION + EXTERNAL STAGE already exist.
      - The external stage points to the same S3 bucket as the Glue table.
      - We'll derive a sub-path from the Glue location and use:
          LOCATION = @<stage>/<subdir>

    Env vars used:
      - SNOWFLAKE_DATABASE (fallback if sf_database is None)
      - SNOWFLAKE_SCHEMA   (fallback if sf_schema is None)
      - SNOWFLAKE_EXTERNAL_STAGE (fallback if sf_external_stage is None)

    This does NOT infer a full schema; it creates a simple VARIANT-based table
    or a placeholder definition that you can extend later.
    """

    sf_database = sf_database or os.environ.get("SNOWFLAKE_DATABASE", "DATAPAI")
    sf_schema = sf_schema or os.environ.get("SNOWFLAKE_SCHEMA", "DATAPAI")
    sf_external_stage = sf_external_stage or os.environ.get("SNOWFLAKE_EXTERNAL_STAGE", "EXT_STAGE_DATAPAI")
    sf_table = sf_table or glue_table

    table = _get_glue_table(glue_database, glue_table)
    location = table["StorageDescriptor"]["Location"]  # e.g. s3://bucket/cleaned/kc_house_data.parquet

    bucket, key = _parse_s3_uri(location)

    # Derive a subpath for the external table location. For simplicity, use the directory.
    # e.g. s3://bucket/cleaned/kc_house_data.parquet -> cleaned/
    subdir = os.path.dirname(key)  # "cleaned"
    # Build Snowflake LOCATION string: @STAGE/subdir
    if subdir:
        stage_location = f'@"{sf_external_stage}"/{subdir}'
    else:
        stage_location = f'@"{sf_external_stage}"'

    file_format = file_format.upper()

    conn = connect_to_db("Snowflake")

    ddl = f'''
    USE DATABASE "{sf_database}";
    USE SCHEMA "{sf_schema}";

    CREATE OR REPLACE EXTERNAL TABLE "{sf_table.upper()}"
    WITH LOCATION = {stage_location}
    AUTO_REFRESH = FALSE
    FILE_FORMAT = (TYPE = {file_format})
    ;
    '''

    # Note: This is a simplified external table definition.
    # In a more advanced setup you would:
    #  - define explicit columns
    #  - use PATTERN, PARTITION BY, etc.

    with conn.cursor() as cur:
        for stmt in ddl.split(";"):
            s = stmt.strip()
            if s:
                cur.execute(s)

    return {
        "status": "ok",
        "sf_database": sf_database,
        "sf_schema": sf_schema,
        "sf_table": sf_table.upper(),
        "sf_external_stage": sf_external_stage,
        "glue_database": glue_database,
        "glue_table": glue_table,
        "s3_location": location,
        "stage_location": stage_location,
        "file_format": file_format,
    }


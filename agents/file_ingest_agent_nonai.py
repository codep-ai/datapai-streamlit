# agents/file_ingest_agent.py

from __future__ import annotations
import io
import os
import typing as t
import pandas as pd
from connect_db import connect_to_db

#try:
#    from snowflake.connector.pandas_tools import write_pandas as sf_write_pandas
#except ImportError:
#    sf_write_pandas = None

# --------------------
# Readers
# --------------------

def _read_csv(file: t.Union[str, io.BytesIO], **kwargs) -> pd.DataFrame:
    if isinstance(file, io.BytesIO):
        file.seek(0)
        return pd.read_csv(file, **kwargs)
    return pd.read_csv(str(file), **kwargs)

def _read_parquet(file: t.Union[str, io.BytesIO], **kwargs) -> pd.DataFrame:
    if isinstance(file, io.BytesIO):
        file.seek(0)
        return pd.read_parquet(file, **kwargs)
    return pd.read_parquet(str(file), **kwargs)

def _read_iceberg(table_location: str) -> pd.DataFrame:
    import kdb
    con = kdb.connect()
    return con.execute(f"SELECT * FROM read_iceberg('{table_location}')").df()

# --------------------
# Writers
# --------------------

def _write_to_snowflake(df: pd.DataFrame, table_name: str):
    from snowflake.connector.pandas_tools import write_pandas as sf_write_pandas
    conn = connect_to_db("Snowflake")
#    if sf_write_pandas is None:
#        raise ImportError("Snowflake-connector-python[pandas] is not installed")
    success, _, num_rows, _ = sf_write_pandas(conn, df, table_name.upper(),database='DATAPAI',schema='DATAPAI',auto_create_table=True)
    print(f"✅ Snowflake: {success}, rows written: {num_rows}")


def _write_to_Redshift(df: pd.DataFrame, table_name: str):
    import psycopg2
    from sqlalchemy import create_engine
    conn = connect_to_db("Redshift")
    df.to_sql(table_name, conn, index=False, if_exists="append")
    print(f"✅ Redshift: rows written: {len(df)}")

def _write_to_Bigquery(df: pd.DataFrame, table_name: str):
    from google.cloud import bigquery
    client = bigquery.Client()
    table_id = f"your_project.your_dataset.{table_name}"
    job = client.load_table_from_dataframe(df, table_id)
    job.result()
    print(f"✅ Bigquery: rows written: {len(df)}")

# --------------------
# Main Entry
# --------------------

def ingest_file(file: t.Union[str, io.BytesIO], destination: str, table_name: str):
    if hasattr(file, "name"):
        filename = file.name
    else:
        filename = str(file)

    extension = os.path.splitext(filename)[-1].lower()

    if extension == ".csv":
        df = _read_csv(file)
    elif extension == ".parquet":
        df = _read_parquet(file)
    elif extension == ".iceberg":
        df = _read_iceberg(str(file))
    else:
        raise ValueError(f"Unsupported file format: {extension}")

    if df.empty:
        raise ValueError("Ingested DataFrame is empty")

    if destination == "Snowflake":
        _write_to_snowflake(df, table_name)
    elif destination == "Redshift":
        _write_to_Redshift(df, table_name)
    elif destination == "Bigquery":
        _write_to_Bigquery(df, table_name)
    elif destination == "Duckdb":
        import kdb
        con = kdb.connect("duckdb.db")
        con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df")
        print(f"✅ DuckDB: rows written: {len(df)}")
    elif destination == "Sqlite3":
        import sqlite3
        conn = sqlite3.connect("local.db")
        df.to_sql(table_name, conn, index=False, if_exists="replace")
        print(f"✅ SQLite3: rows written: {len(df)}")
    else:
        raise ValueError(f"Unsupported destination: {destination}")


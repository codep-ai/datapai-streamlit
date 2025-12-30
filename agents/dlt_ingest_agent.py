# agents/dlt_ingest_agent.py

from __future__ import annotations

import dlt
import typing as t

def ingest_with_dlt(
    source_uri: str,
    destination: str,
    dataset_or_schema: str,
    table_name: str,
    pipeline_name: str = "dlt_csv_pipeline"
) -> tuple[bool, str]:
    try:
        if destination == "bigquery":
            pipeline = dlt.pipeline(
                pipeline_name=pipeline_name,
                destination="bigquery",
                dataset_name=dataset_or_schema,
                full_refresh=True,
            )
        elif destination == "snowflake":
            pipeline = dlt.pipeline(
                pipeline_name=pipeline_name,
                destination="snowflake",
                dataset_name=dataset_or_schema,
                full_refresh=True,
            )
        else:
            return False, f"Unsupported destination: {destination}"

        import pandas as pd
        if source_uri.startswith("s3://"):
            import s3fs
            fs = s3fs.S3FileSystem()
            with fs.open(source_uri, mode="rb") as f:
                df = pd.read_csv(f)
        else:
            df = pd.read_csv(source_uri)

        load_info = pipeline.run(df, table_name=table_name)
        print(load_info)
        return True, f"Ingested {len(df)} rows to {table_name} ({destination})"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


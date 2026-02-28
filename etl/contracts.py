from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class WorkflowRequest:
    source_type: str              # "s3_csv" | "s3_parquet" | "local" | "airbyte" ...
    source: str                   # path/uri
    target: str                   # "snowflake" | "redshift" | ...
    target_schema: str
    target_table: str
    mode: str = "upsert"          # or "overwrite"
    pk: Optional[List[str]] = None
    options: Dict[str, Any] = None

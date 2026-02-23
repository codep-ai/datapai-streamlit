from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class WorkflowPlan:
    workflow: str  # e.g. "ingest_to_dbt"
    source_type: str
    source: str
    target: str
    target_schema: str
    target_table: str
    mode: str = "upsert"  # or overwrite
    pk: Optional[List[str]] = None
    tests: List[Dict[str, Any]] = field(default_factory=list)
    notes: str = ""

# agents/dbt_agent.py
"""
dbt Agent (DataPAI) - Planner + Deterministic Executor

Design principle:
- LLM = planner (optional): propose pk/tests/columns/descriptions/metrics ideas
- Python = executor (deterministic): write files, run dbt, update metadata

Folder conventions (your repo):
- dbt project root:            dbt-demo/
- sources YAML output:         dbt-demo/source/<domain>/<source_name>.yml
- staging SQL output:          dbt-demo/models/<domain>/staging/stg_<table>.sql
- staging YAML output:         dbt-demo/models/<domain>/staging/_stg_<table>.yml
- canonical manifest for LD:   dbt-demo/target/manifest.json (one instance)

This module is tool-only; no agent_base imports (avoids circular imports).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .tooling.registry import tool

# ---------------------------------------------------------------------
# Environment / dbt project config
# ---------------------------------------------------------------------

DBT_PROJECT_DIR = os.environ.get("DBT_PROJECT_DIR", "dbt-demo")
DBT_PROFILES_DIR = os.environ.get("DBT_PROFILES_DIR", DBT_PROJECT_DIR)
DBT_TARGET = os.environ.get("DBT_TARGET", "")  # optional

# Import dbt_metadata pieces (must NOT auto-run on import)
try:
    from dbt_metadata import DBT_MANIFEST_PATH, update_dbt_metadata, get_dbt_metadata  # type: ignore
except Exception:
    DBT_MANIFEST_PATH = os.path.join(DBT_PROJECT_DIR, "target", "manifest.json")
    update_dbt_metadata = None
    get_dbt_metadata = None


# ---------------------------------------------------------------------
# Prompt: LLM planner (optional)
# ---------------------------------------------------------------------

DBT_AGENT_SYSTEM_PROMPT = """
You are a dbt expert for enterprise analytics engineering.

Goal:
Given an ingestion request + optional schema/columns + business context,
produce a PLAN for dbt assets (NOT file paths). The executor will handle paths.

Return ONLY valid JSON that matches this schema:

{
  "domain": "stock|chinook|full-jaffle-shop|<existing domain>",
  "source_name": "src_stock",
  "tables": {
    "trades": {
      "pk": ["trade_id"] | null,
      "columns": ["col1","col2", "..."] | null,
      "description": "short model description",
      "column_descriptions": {"col1":"...", "col2":"..."} | null,
      "tests": [
        {"type":"not_null","column":"col1"},
        {"type":"unique","column":"col1"},
        {"type":"accepted_values","column":"side","values":["BUY","SELL"]}
      ]
    }
  },
  "notes": "optional"
}

Rules:
- Do not invent tables that were not provided.
- If columns unknown, set columns=null (executor will use SELECT *).
- Prefer snake_case for names.
- Baseline tests: if pk present, include not_null + unique for pk columns.
- Keep it concise and realistic.
""".strip()


def _extract_first_json_object(text: str) -> str:
    """
    Best-effort extraction of the first JSON object from an LLM response.
    Handles cases where model wraps JSON in prose or markdown.
    """
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    # naive brace matching
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("Unbalanced JSON braces in model output.")


def plan_with_llm(
    llm_client: Any,
    request: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Optional planner: LLM returns a plan JSON. Executor applies it deterministically.
    Assumes llm_client has `.chat(messages=[...]) -> str` (your RouterChatClient / OpenAI / Ollama wrapper).
    """
    messages = [
        {"role": "system", "content": DBT_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(request, indent=2)},
    ]
    raw = llm_client.chat(messages=messages)
    obj_txt = _extract_first_json_object(raw)
    return json.loads(obj_txt)


# ---------------------------------------------------------------------
# Data classes for executor inputs
# ---------------------------------------------------------------------

@dataclass
class TablePlan:
    pk: Optional[List[str]] = None
    columns: Optional[List[str]] = None
    description: str = ""
    column_descriptions: Optional[Dict[str, str]] = None
    tests: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class DbtPlan:
    domain: Optional[str] = None
    source_name: Optional[str] = None
    tables: Dict[str, TablePlan] = field(default_factory=dict)
    notes: str = ""


def _dict_to_dbt_plan(d: Dict[str, Any]) -> DbtPlan:
    tables: Dict[str, TablePlan] = {}
    for t, v in (d.get("tables") or {}).items():
        tables[t] = TablePlan(
            pk=v.get("pk"),
            columns=v.get("columns"),
            description=v.get("description") or "",
            column_descriptions=v.get("column_descriptions"),
            tests=v.get("tests") or [],
        )
    return DbtPlan(
        domain=d.get("domain"),
        source_name=d.get("source_name"),
        tables=tables,
        notes=d.get("notes") or "",
    )


# ---------------------------------------------------------------------
# Path + naming helpers
# ---------------------------------------------------------------------

def _norm_path(p: str) -> str:
    return os.path.normpath(p)

def _dbt_project_dir() -> str:
    return _norm_path(DBT_PROJECT_DIR)

def _dbt_models_dir() -> str:
    return os.path.join(_dbt_project_dir(), "models")

def _dbt_source_dir() -> str:
    return os.path.join(_dbt_project_dir(), "source")

def _list_domains() -> List[str]:
    models_dir = _dbt_models_dir()
    if not os.path.isdir(models_dir):
        return []
    return sorted(
        d for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d)) and not d.startswith(".")
    )

def _resolve_domain(domain: Optional[str], schema: Optional[str], default_domain: str = "stock") -> str:
    domains = _list_domains()
    if domain and domain in domains:
        return domain
    if schema and schema in domains:
        return schema
    if default_domain in domains:
        return default_domain
    return domains[0] if domains else default_domain

def _snake_case(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower()

def _safe_identifier(name: str) -> str:
    return _snake_case(name)

def _ensure_dir_for_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _source_yaml_path(domain: str, source_name: str) -> str:
    out_dir = os.path.join(_dbt_source_dir(), domain)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{_safe_identifier(source_name)}.yml")

def _staging_paths(domain: str, table: str) -> Tuple[str, str]:
    t = _safe_identifier(table)
    sql_path = os.path.join(_dbt_models_dir(), domain, "staging", f"stg_{t}.sql")
    yml_path = os.path.join(_dbt_models_dir(), domain, "staging", f"_stg_{t}.yml")
    return sql_path, yml_path


# ---------------------------------------------------------------------
# YAML / SQL generators (executor side)
# ---------------------------------------------------------------------

def generate_source_yaml_text(
    source_name: str,
    schema: str,
    table_names: List[str],
    database: Optional[str] = None,
    include_columns: bool = False,
    tables_columns: Optional[Dict[str, List[str]]] = None,
) -> str:
    source_name = _safe_identifier(source_name)
    schema = schema.strip()

    tables = []
    for t in table_names:
        t_safe = _safe_identifier(t)
        table_obj: Dict[str, Any] = {"name": t_safe}
        if include_columns and tables_columns and t in tables_columns:
            cols = [{"name": _safe_identifier(c)} for c in (tables_columns.get(t) or [])]
            if cols:
                table_obj["columns"] = cols
        tables.append(table_obj)

    src: Dict[str, Any] = {
        "version": 2,
        "sources": [
            {
                "name": source_name,
                "schema": schema,
                "tables": tables,
            }
        ],
    }
    if database:
        src["sources"][0]["database"] = database

    return yaml.safe_dump(src, sort_keys=False, allow_unicode=True)

def generate_staging_sql_text(source_name: str, table: str, columns: Optional[List[str]] = None) -> str:
    source_name = _safe_identifier(source_name)
    table_safe = _safe_identifier(table)

    cols = [c for c in (columns or []) if c]
    if cols:
        select_list = ",\n    ".join([f'"{_safe_identifier(c)}"' for c in cols])
        return f"""select
    {select_list}
from {{{{ source('{source_name}', '{table_safe}') }}}}
"""
    return f"""select
    *
from {{{{ source('{source_name}', '{table_safe}') }}}}
"""

def _merge_tests(baseline: List[Dict[str, Any]], extra: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # de-dup (type,column,values)
    seen = set()
    out = []
    for t in baseline + extra:
        ttype = t.get("type")
        col = _safe_identifier(t.get("column", ""))
        vals = tuple(t.get("values") or [])
        key = (ttype, col, vals)
        if not ttype or not col:
            continue
        if key in seen:
            continue
        seen.add(key)
        norm = {"type": ttype, "column": col}
        if ttype == "accepted_values":
            norm["values"] = list(vals)
        out.append(norm)
    return out

def _baseline_tests_for_pk(pk: Optional[List[str]]) -> List[Dict[str, Any]]:
    if not pk:
        return []
    out = []
    for c in pk:
        col = _safe_identifier(c)
        out.append({"type": "not_null", "column": col})
        out.append({"type": "unique", "column": col})
    return out

def generate_staging_yml_text(
    model_name: str,
    columns: Optional[List[str]] = None,
    column_descriptions: Optional[Dict[str, str]] = None,
    tests: Optional[List[Dict[str, Any]]] = None,
    description: str = "",
) -> str:
    model_name = _safe_identifier(model_name)
    cols = [c for c in (columns or []) if c]

    # group tests by column
    tests_by_col: Dict[str, List[Any]] = {}
    accepted_values_by_col: Dict[str, List[str]] = {}
    for t in (tests or []):
        ttype = t.get("type")
        col = _safe_identifier(t.get("column", ""))
        if not ttype or not col:
            continue
        if ttype == "accepted_values":
            accepted_values_by_col[col] = list(t.get("values") or [])
        else:
            tests_by_col.setdefault(col, []).append(ttype)

    cols_list = []
    for c in cols:
        c_safe = _safe_identifier(c)
        col_obj: Dict[str, Any] = {"name": c_safe}
        if column_descriptions and c_safe in column_descriptions:
            col_obj["description"] = column_descriptions[c_safe]
        if c_safe in tests_by_col or c_safe in accepted_values_by_col:
            col_tests: List[Any] = []
            for x in tests_by_col.get(c_safe, []):
                col_tests.append(x)
            if c_safe in accepted_values_by_col:
                col_tests.append({"accepted_values": {"values": accepted_values_by_col[c_safe]}})
            col_obj["tests"] = col_tests
        cols_list.append(col_obj)

    yml_obj: Dict[str, Any] = {
        "version": 2,
        "models": [
            {
                "name": model_name,
                "description": description or "",
                "columns": cols_list,
            }
        ],
    }
    return yaml.safe_dump(yml_obj, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------
# dbt execution (deterministic)
# ---------------------------------------------------------------------

def _build_dbt_cmd(base_cmd: str) -> List[str]:
    cmd = base_cmd.strip()
    if not cmd.startswith("dbt "):
        cmd = "dbt " + cmd

    parts = cmd.split()

    if "--profiles-dir" not in parts:
        parts += ["--profiles-dir", DBT_PROFILES_DIR]

    if DBT_TARGET and "--target" not in parts:
        parts += ["--target", DBT_TARGET]

    return parts

def run_dbt_commands(commands: List[str]) -> Dict[str, Any]:
    project_dir = _dbt_project_dir()
    results = []

    for c in commands:
        parts = _build_dbt_cmd(c)
        proc = subprocess.Popen(
            parts,
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate()

        results.append(
            {
                "command": " ".join(parts),
                "cwd": project_dir,
                "returncode": proc.returncode,
                "stdout_tail": (stdout or "")[-4000:],
                "stderr_tail": (stderr or "")[-4000:],
            }
        )

    ok = all(r["returncode"] == 0 for r in results)
    return {"ok": ok, "results": results, "project_dir": project_dir, "profiles_dir": DBT_PROFILES_DIR, "target": DBT_TARGET}


# ---------------------------------------------------------------------
# Executor: generate files + run dbt + update metadata
# ---------------------------------------------------------------------

def run_dbt_agent(
    schema: str,
    table_names: List[str],
    domain: Optional[str] = None,
    source_name: Optional[str] = None,
    database: Optional[str] = None,
    # planner-derived fields:
    pk_by_table: Optional[Dict[str, List[str]]] = None,
    columns_by_table: Optional[Dict[str, List[str]]] = None,
    description_by_table: Optional[Dict[str, str]] = None,
    col_desc_by_table: Optional[Dict[str, Dict[str, str]]] = None,
    tests_by_table: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    # behavior:
    write_sources: bool = True,
    write_staging: bool = True,
    run_dbt: bool = True,
    dbt_cmd: str = "build",
) -> Dict[str, Any]:
    if not schema:
        raise ValueError("schema is required")
    if not table_names:
        raise ValueError("table_names is required")

    domain_resolved = _resolve_domain(domain, schema=schema, default_domain="stock")
    source_name = source_name or schema

    tables = [_safe_identifier(t) for t in table_names]

    out: Dict[str, Any] = {
        "ok": True,
        "domain": domain_resolved,
        "schema": schema,
        "source_name": _safe_identifier(source_name),
        "written": {"sources": None, "staging": []},
        "dbt": None,
        "artifacts": {
            "dbt_manifest_path": DBT_MANIFEST_PATH,
            "dbt_run_results_path": os.path.join(os.path.dirname(DBT_MANIFEST_PATH), "run_results.json"),
            "dbt_catalog_path": os.path.join(os.path.dirname(DBT_MANIFEST_PATH), "catalog.json"),
        },
        "notes": [],
    }

    # 1) Write sources YAML to dbt-demo/source/<domain>/<source_name>.yml
    if write_sources:
        src_yaml = generate_source_yaml_text(
            source_name=source_name,
            schema=schema,
            table_names=tables,
            database=database,
            include_columns=bool(columns_by_table),
            tables_columns=columns_by_table,
        )
        src_path = _source_yaml_path(domain_resolved, source_name)
        _ensure_dir_for_file(src_path)
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(src_yaml)
        out["written"]["sources"] = src_path

    # 2) Write staging SQL/YML to models/<domain>/staging/
    if write_staging:
        for t in tables:
            sql_path, yml_path = _staging_paths(domain_resolved, t)
            _ensure_dir_for_file(sql_path)
            _ensure_dir_for_file(yml_path)

            cols = (columns_by_table or {}).get(t)
            sql_text = generate_staging_sql_text(source_name, t, columns=cols)
            with open(sql_path, "w", encoding="utf-8") as f:
                f.write(sql_text)

            pk = (pk_by_table or {}).get(t)
            baseline_tests = _baseline_tests_for_pk(pk)
            extra_tests = (tests_by_table or {}).get(t, [])
            merged_tests = _merge_tests(baseline_tests, extra_tests)

            desc = (description_by_table or {}).get(t, "")
            col_desc = (col_desc_by_table or {}).get(t)

            yml_text = generate_staging_yml_text(
                model_name=f"stg_{t}",
                columns=cols or [],
                column_descriptions=col_desc,
                tests=merged_tests,
                description=desc,
            )
            with open(yml_path, "w", encoding="utf-8") as f:
                f.write(yml_text)

            out["written"]["staging"].append({"table": t, "sql": sql_path, "yml": yml_path})

    # 3) Run dbt (canonical)
    if run_dbt:
        dbt_res = run_dbt_commands([dbt_cmd])
        out["dbt"] = dbt_res
        if not dbt_res.get("ok"):
            out["ok"] = False
            out["notes"].append("dbt command failed; metadata not refreshed.")
            return out

        # 4) Refresh metadata index (FAISS/vector store) for RAG/Lightdash sync
        if update_dbt_metadata is not None:
            try:
                update_dbt_metadata()
                out["notes"].append("dbt metadata index refreshed.")
            except Exception as e:
                out["notes"].append(f"WARNING: failed to refresh dbt metadata index: {e}")

    return out


# ---------------------------------------------------------------------
# Combined: Plan (LLM) + Execute (deterministic)
# ---------------------------------------------------------------------

def run_dbt_agent_with_plan(
    schema: str,
    table_names: List[str],
    llm_client: Any,
    domain: Optional[str] = None,
    source_name: Optional[str] = None,
    database: Optional[str] = None,
    business_context: Optional[str] = None,
    run_dbt: bool = True,
    dbt_cmd: str = "build",
) -> Dict[str, Any]:
    """
    Convenience: uses LLM to propose plan fields, then executes deterministically.
    """
    request = {
        "schema": schema,
        "tables": table_names,
        "preferred_domain": domain,
        "preferred_source_name": source_name,
        "business_context": business_context or "",
        "existing_domains": _list_domains(),
        "conventions": {
            "source_yaml_folder": "dbt-demo/source/<domain>/",
            "staging_folder": "dbt-demo/models/<domain>/staging/",
            "staging_sql": "stg_<table>.sql",
            "staging_yml": "_stg_<table>.yml",
        },
    }

    plan_dict = plan_with_llm(llm_client, request)
    plan = _dict_to_dbt_plan(plan_dict)

    # Resolve final domain/source_name deterministically with fallback
    final_domain = _resolve_domain(plan.domain or domain, schema=schema, default_domain="stock")
    final_source = plan.source_name or source_name or schema

    # Convert plan tables to per-table maps
    pk_by_table: Dict[str, List[str]] = {}
    columns_by_table: Dict[str, List[str]] = {}
    desc_by_table: Dict[str, str] = {}
    col_desc_by_table: Dict[str, Dict[str, str]] = {}
    tests_by_table: Dict[str, List[Dict[str, Any]]] = {}

    # planner may only include subset; keep deterministic table list from input
    for t in table_names:
        t_key = _safe_identifier(t)
        tp = plan.tables.get(t) or plan.tables.get(t_key)
        if not tp:
            continue
        if tp.pk:
            pk_by_table[t_key] = tp.pk
        if tp.columns:
            columns_by_table[t_key] = tp.columns
        if tp.description:
            desc_by_table[t_key] = tp.description
        if tp.column_descriptions:
            # normalize keys
            col_desc_by_table[t_key] = { _safe_identifier(k): v for k, v in tp.column_descriptions.items() }
        if tp.tests:
            tests_by_table[t_key] = tp.tests

    result = run_dbt_agent(
        schema=schema,
        table_names=table_names,
        domain=final_domain,
        source_name=final_source,
        database=database,
        pk_by_table=pk_by_table or None,
        columns_by_table=columns_by_table or None,
        description_by_table=desc_by_table or None,
        col_desc_by_table=col_desc_by_table or None,
        tests_by_table=tests_by_table or None,
        write_sources=True,
        write_staging=True,
        run_dbt=run_dbt,
        dbt_cmd=dbt_cmd,
    )

    # attach plan for traceability
    result["plan"] = plan_dict
    return result


# ---------------------------------------------------------------------
# Tools (for your tool registry / Streamlit / supervisor)
# ---------------------------------------------------------------------

@tool()
def dbt_generate_sources_and_staging(
    schema: str,
    tables: str,
    domain: Optional[str] = None,
    source_name: Optional[str] = None,
    database: Optional[str] = None,
    run_build: bool = True,
    dbt_cmd: str = "build",
) -> str:
    """
    Deterministic executor tool: generate sources + staging for given schema/tables
    and optionally run dbt (default build).
    """
    table_list = [t.strip() for t in tables.split(",") if t.strip()]
    res = run_dbt_agent(
        schema=schema,
        table_names=table_list,
        domain=domain,
        source_name=source_name,
        database=database,
        run_dbt=run_build,
        dbt_cmd=dbt_cmd,
    )
    return json.dumps(res, indent=2)

@tool()
def dbt_run(command: str = "build") -> str:
    """
    Run dbt command deterministically (cwd=dbt-demo, profiles-dir configured).
    Example: command="build" or "run -s stg_trades"
    """
    res = run_dbt_commands([command])
    return json.dumps(res, indent=2)

@tool()
def dbt_read_manifest_summary() -> str:
    """
    Read canonical manifest and return a small summary (counts).
    """
    try:
        with open(DBT_MANIFEST_PATH, "r", encoding="utf-8") as f:
            m = json.load(f)
        return json.dumps(
            {
                "ok": True,
                "manifest_path": DBT_MANIFEST_PATH,
                "counts": {
                    "nodes": len(m.get("nodes", {}) or {}),
                    "sources": len(m.get("sources", {}) or {}),
                    "macros": len(m.get("macros", {}) or {}),
                },
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"ok": False, "manifest_path": DBT_MANIFEST_PATH, "error": str(e)}, indent=2)

@tool()
def dbt_rag_metadata(query: str, top_k: int = 3) -> str:
    """
    Retrieve relevant dbt metadata via your FAISS/vector index built from manifest.
    """
    if get_dbt_metadata is None:
        return json.dumps({"ok": False, "message": "dbt_metadata.get_dbt_metadata not available"}, indent=2)
    return get_dbt_metadata(query=query, top_k=top_k)  # type: ignore

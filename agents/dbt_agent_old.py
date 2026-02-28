# dbt_agent.py

import os
import re
import subprocess
import typing as t
from typing import List, Dict, Any, Optional

import sqlalchemy as sa
from sqlalchemy.engine import Engine
import yaml

#from tools import tool
from .tooling.registry import tool

#/from agent_base import BaseAgent, DEFAULT_SYSTEM_PROMPT
from .llm_client import BaseChatClient
from .dbt_mcp_client import DbtMcpClient


# ---------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------

def _get_engine(engine_name: str) -> Engine:
    engine_name = engine_name.lower()

    if engine_name == "snowflake":
        from snowflake.sqlalchemy import URL  # type: ignore

        user = os.environ["SF_USER"]
        password = os.environ["SF_PASSWORD"]
        account = os.environ["SF_ACCOUNT"]
        database = os.environ["SF_DATABASE"]
        schema = os.environ["SF_SCHEMA"]
        warehouse = os.environ["SF_WAREHOUSE"]
        role = os.environ.get("SF_ROLE")

        url = URL(
            user=user,
            password=password,
            account=account,
            database=database,
            schema=schema,
            warehouse=warehouse,
            role=role,
        )
        return sa.create_engine(url)

    if engine_name == "redshift":
        host = os.environ["RS_HOST"]
        port = os.environ.get("RS_PORT", "5439")
        db = os.environ["RS_DB"]
        user = os.environ["RS_USER"]
        password = os.environ["RS_PASSWORD"]

        url = sa.engine.URL.create(
            drivername="redshift+psycopg2",
            username=user,
            password=password,
            host=host,
            port=port,
            database=db,
        )
        return sa.create_engine(url)

    if engine_name == "duckdb":
        db_path = os.environ.get("DUCKDB_PATH", "file_ingest.duckdb")
        url = sa.engine.URL.create(
            drivername="duckdb",
            database=db_path,
        )
        return sa.create_engine(url)

    raise ValueError(f"Unsupported engine_name: {engine_name}")


def _split_schema_table(table_name: str) -> (Optional[str], str):
    if "." in table_name:
        s, t = table_name.split(".", 1)
        return s, t
    return None, table_name


def _snake_case(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
    return s2.lower().strip("_")


def _get_project_root() -> str:
    return os.environ.get("DBT_PROJECT_ROOT", os.getcwd())


# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------

@tool()
def get_table_schema(table_name: str, engine_name: str) -> Dict[str, Any]:
    engine = _get_engine(engine_name)
    inspector = sa.inspect(engine)

    schema, table = _split_schema_table(table_name)

    columns_raw = inspector.get_columns(table_name=table, schema=schema)
    pk_info = inspector.get_pk_constraint(table_name=table, schema=schema)
    pk_cols = set(pk_info.get("constrained_columns") or [])

    columns = []
    for col in columns_raw:
        name = col["name"]
        coltype = str(col["type"])
        nullable = col.get("nullable", True)
        is_pk = name in pk_cols
        columns.append(
            {
                "name": name,
                "type": coltype,
                "nullable": nullable,
                "is_pk": is_pk,
            }
        )

    return {
        "schema": schema,
        "table": table,
        "columns": columns,
    }


@tool()
def generate_dbt_source_yaml(
    source_name: str,
    database: Optional[str],
    schema: str,
    table: str,
    columns: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
) -> str:
    project_root = _get_project_root()
    if output_dir is None:
        output_dir = os.path.join(project_root, "models", "sources", source_name)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sources.yml")

    yaml_obj = {
        "version": 2,
        "sources": [
            {
                "name": source_name,
                "database": database,
                "schema": schema,
                "tables": [
                    {
                        "name": table,
                        "description": f"Source table for {table}.",
                        "columns": [
                            {
                                "name": c["name"],
                                "description": "",
                                "tests": ["not_null"]
                                if (c.get("is_pk") or not c.get("nullable", True))
                                else [],
                            }
                            for c in columns
                        ],
                    }
                ],
            }
        ],
    }

    if database is None:
        del yaml_obj["sources"][0]["database"]

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_obj, f, sort_keys=False)

    return output_path


@tool()
def generate_dbt_staging_model_sql(
    source_name: str,
    table: str,
    columns: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
) -> str:
    project_root = _get_project_root()
    if output_dir is None:
        output_dir = os.path.join(project_root, "models", "staging", source_name)

    os.makedirs(output_dir, exist_ok=True)

    model_name = f"stg_{table}"
    output_path = os.path.join(output_dir, f"{model_name}.sql")

    select_lines = []
    for c in columns:
        src = c["name"]
        tgt = _snake_case(src)
        if src == tgt:
            select_lines.append(f'    "{src}"')
        else:
            select_lines.append(f'    "{src}" as {tgt}')

    select_clause = ",\n".join(select_lines)

    sql = f"""{{{{ config(materialized='view') }}}}

select
{select_clause}
from {{{{ source('{source_name}', '{table}') }}}}
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(sql)

    return output_path


@tool()
def generate_dbt_staging_model_yaml(
    source_name: str,
    table: str,
    columns: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
) -> str:
    project_root = _get_project_root()
    if output_dir is None:
        output_dir = os.path.join(project_root, "models", "staging", source_name)

    os.makedirs(output_dir, exist_ok=True)

    model_name = f"stg_{table}"
    output_path = os.path.join(output_dir, f"{model_name}.yml")

    yaml_obj = {
        "version": 2,
        "models": [
            {
                "name": model_name,
                "description": f"Staging model for {table}.",
                "columns": [],
            }
        ],
    }

    cols_yaml = []
    for c in columns:
        src = c["name"]
        tgt = _snake_case(src)
        tests = []
        if c.get("is_pk"):
            tests.extend(["unique", "not_null"])
        elif not c.get("nullable", True):
            tests.append("not_null")

        cols_yaml.append(
            {
                "name": tgt,
                "description": "",
                "tests": tests,
            }
        )

    yaml_obj["models"][0]["columns"] = cols_yaml

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_obj, f, sort_keys=False)

    return output_path


@tool()
def run_dbt_commands(commands: List[str]) -> Dict[str, Any]:
    project_root = _get_project_root()
    results = []

    for cmd in commands:
        parts = cmd.split()
        proc = subprocess.Popen(
            parts,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate()
        results.append(
            {
                "command": cmd,
                "returncode": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
            }
        )

    return {"commands": commands, "results": results}


# ---------------------------------------------------------------------
# MCP-based tools (optional)
# ---------------------------------------------------------------------

@tool()
def mcp_dbt_run(commands: List[str]) -> Dict[str, Any]:
    client = DbtMcpClient()
    results = []
    for cmd in commands:
        try:
            res = client.run_dbt_command(cmd)
        except Exception as e:
            res = {"error": str(e)}
        results.append({"command": cmd, "result": res})

    return {"via": "mcp", "commands": commands, "results": results}


@tool()
def mcp_dbt_get_manifest_summary() -> Dict[str, Any]:
    client = DbtMcpClient()
    try:
        result = client.get_manifest_summary()
    except Exception as e:
        result = {"error": str(e)}
    return result


@tool()
def mcp_dbt_get_model_info(node_name: str) -> Dict[str, Any]:
    client = DbtMcpClient()
    try:
        result = client.get_model_info(node_name=node_name)
    except Exception as e:
        result = {"error": str(e)}
    return result


# ---------------------------------------------------------------------
# Agent specialisation
# ---------------------------------------------------------------------

DBT_AGENT_SYSTEM_PROMPT = (
     """
You are the dbtAgent.

Responsibilities:
  - Given a warehouse engine and table name, inspect the table schema.
  - Generate dbt source definition (sources.yml).
  - Generate staging model SQL (stg_<table>.sql).
  - Generate staging model YAML (stg_<table>.yml).
  - Optionally run dbt commands and/or query dbt metadata.

Tools:
  Direct dbt + warehouse tools:
    - get_table_schema
    - generate_dbt_source_yaml
    - generate_dbt_staging_model_sql
    - generate_dbt_staging_model_yaml
    - run_dbt_commands

  MCP dbt tools (use when needed, don't spam):
    - mcp_dbt_run
    - mcp_dbt_get_manifest_summary
    - mcp_dbt_get_model_info

Use context:
  - engine_name
  - database (optional)
  - schema
  - table
  - source_name

In final_answer, summarise:
  - which table you processed
  - what dbt files you created (paths)
  - whether you used MCP tools, and what they did
  - any dbt commands run and their outcomes.
"""
)

class DbtAgent:
    """
    Domain-specific agent that orchestrates dbt-related tools
    (generate sources, staging models, tests, run dbt, etc.).

    It wraps BaseAgent internally (composition) to avoid circular imports.
    """

    def __init__(
        self,
        llm: t.Optional[BaseChatClient] = None,
        max_steps: int = 10,
        temperature: float = 0.1,
    ):
        self.llm = llm or RouterChatClient()
        self.max_steps = max_steps
        self.temperature = temperature

    def run(self, goal: str, context: t.Optional[dict] = None) -> dict:
        # Lazy import here to break circular dependency:
        from agent_base import BaseAgent

        base = BaseAgent(
            name="dbt_agent",
            llm=self.llm,
            system_prompt=DBT_AGENT_SYSTEM_PROMPT,
            max_steps=self.max_steps,
            temperature=self.temperature,
        )
        return base.run(goal=goal, context=context or {})

class DbtAgent_old:
    def __init__(
        self,
        llm: Optional[BaseChatClient] = None,
        max_steps: int = 10,
        temperature: float = 0.1,
    ):
        # Default to shared router if none provided
        if llm is None:
            llm = RouterChatClient()

        super().__init__(
            name="dbt_agent",
            llm=llm,
            system_prompt=DBT_AGENT_SYSTEM_PROMPT,
            max_steps=max_steps,
            temperature=temperature,
        )


if __name__ == "__main__":
    agent = DbtAgent()
    goal = (
        "Inspect the table 'raw.kc_house_data' in the 'duckdb' engine, "
        "generate a dbt source definition and a staging model (SQL + YAML). "
        "Do NOT run dbt yet, just generate the files."
    )

    context = {
        "engine_name": "duckdb",
        "database": None,
        "schema": "raw",
        "table": "kc_house_data",
        "source_name": "raw_kc_house",
    }

    result = agent.run(goal=goal, context=context)
    print(result)


#!/usr/bin/env python3
from __future__ import annotations
import os, subprocess, sys

try:
    import tomllib as toml
except Exception:
    import tomli as toml  # pip install tomli

TOOLS_FILE = "mcp/tools.yaml"
TOOLBOX_BIN = "./toolbox"  # adjust if the binary is elsewhere

def main():
    # Load secrets
    with open(".streamlit/secrets.toml", "rb") as f:
        secrets = toml.load(f)
    env = os.environ.copy()

    # Snowflake
    for k in ["SNOWFLAKE_USER","SNOWFLAKE_PASSWORD","SNOWFLAKE_ACCOUNT",
              "SNOWFLAKE_WAREHOUSE","SNOWFLAKE_DATABASE","SNOWFLAKE_SCHEMA","SNOWFLAKE_ROLE"]:
        v = secrets.get(k)
        if v is not None: env[k] = str(v)

    # Redshift
    for k in ["REDSHIFT_DBNAME","REDSHIFT_SCHEMA","REDSHIFT_USER",
              "REDSHIFT_PASSWORD","REDSHIFT_HOST","REDSHIFT_PORT"]:
        v = secrets.get(k)
        if v is not None: env[k] = str(v)

    # Local DBs
    for k in ["SQLITE3_DB_PATH","DUCKDB_DB_PATH"]:
        v = secrets.get(k)
        if v is not None: env[k] = str(v)

    # BigQuery (nested)
    gcp = secrets.get("gcp_service_account") or {}
    if "project_id" in gcp:
        env["BIGQUERY_PROJECT"] = str(gcp["project_id"])
    if "BIGQUERY_SCHEMA" in gcp:
        env["BIGQUERY_SCHEMA"] = str(gcp["BIGQUERY_SCHEMA"])
    if "BIGQUERY_LOCATION" in gcp:
        env["BIGQUERY_LOCATION"] = str(gcp["BIGQUERY_LOCATION"])

    import json
    if gcp:
        env["BIGQUERY_CREDENTIALS_JSON"] = json.dumps(gcp, separators=(",", ":"))

    # Run toolbox
    cmd = [TOOLBOX_BIN, "--tools-file", TOOLS_FILE, "--host", "127.0.0.1", "--port", "5000"]
    print("Launching:", " ".join(cmd))
    p = subprocess.Popen(cmd, env=env)
    p.wait()
    return p.returncode

if __name__ == "__main__":
    sys.exit(main())


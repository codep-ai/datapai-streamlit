#!/usr/bin/env bash
set -euo pipefail

# --- Paths (adjust if needed)
VENV="/home/ec2-user/venvs/dbt-mcp-env"
DBT_PROJECT_DIR="/home/ec2-user/git/dbt-demo"
DBT_PROFILES_DIR="/home/ec2-user/.dbt"
HOST="127.0.0.1"
PORT="6000"
# add to your start script before Exec:
export DBT_HOST="https://cloud.getdbt.com"
export DBT_PROD_ENV_ID="1"
export DBT_TOKEN="dummy"


# --- Activate Python 3.12 venv
source "${VENV}/bin/activate"

# --- CLI-only mode (no dbt Cloud): disable cloud features so it doesn't ask for DBT_HOST/DBT_TOKEN/DBT_PROD_ENV_ID
export DBT_DISABLE_SEMANTIC_LAYER=1
export DBT_DISABLE_DISCOVERY=1
export DBT_DISABLE_SQL_TOOLS=1

# --- Required for dbt CLI tools
export DBT_PROJECT_DIR="${DBT_PROJECT_DIR}"
export DBT_PROFILES_DIR="${DBT_PROFILES_DIR}"

# If your profiles.yml uses Snowflake env vars, set them here (examples):
# export SNOWFLAKE_ACCOUNT="..."
# export SNOWFLAKE_USER="..."
# export SNOWFLAKE_PASSWORD="..."
# export SNOWFLAKE_ROLE="..."
# export SNOWFLAKE_WAREHOUSE="..."
# export SNOWFLAKE_DATABASE="..."
# export SNOWFLAKE_SCHEMA="..."

# --- (Optional) log what config we’re using
echo "Starting dbt-mcp on ${HOST}:${PORT}"
echo "DBT_PROJECT_DIR=${DBT_PROJECT_DIR}"
echo "DBT_PROFILES_DIR=${DBT_PROFILES_DIR}"
echo "Cloud features disabled: SL=${DBT_DISABLE_SEMANTIC_LAYER:-}, DISC=${DBT_DISABLE_DISCOVERY:-}, SQL_TOOLS=${DBT_DISABLE_SQL_TOOLS:-}"

# --- Start server
exec "${VENV}/bin/dbt-mcp" serve \
  --host "${HOST}" \
  --port "${PORT}" \
  --project-dir "${DBT_PROJECT_DIR}" \
  --profiles-dir "${DBT_PROFILES_DIR}"


"""
Airbyte Docker Integration — AG2 SwarmResult Tools for E2E Data Flow.

Architecture:
  User (natural language) → ingest_agent → Airbyte Docker API
    → Source connector (PostgreSQL, Salesforce, S3, MSSQL, MySQL, …)
    → Destination connector (Snowflake, BigQuery, Redshift, DuckDB, …)
    → Sync job → data lands in warehouse
    → compliance_agent → quality_agent → transform_agent (dbt)

Airbyte Docker API used: Config API v1 (http://localhost:8000/api/v1/...)
  — same API used in the existing airbyte_ingest_agent.py.
  — Compatible with Airbyte OSS Docker Compose deployments.

Environment variables:
  AIRBYTE_API_URL      Airbyte base URL          default: http://localhost:8000/api
  AIRBYTE_USERNAME     Basic auth user           default: airbyte
  AIRBYTE_PASSWORD     Basic auth password       default: password
  AIRBYTE_WORKSPACE_ID Workspace ID (optional)   auto-detected if not set

Platform API (platform.datap.ai) bridge:
  DATAPAI_API_URL      Laravel backend URL       default: https://platform.datap.ai/api
  DATAPAI_API_TOKEN    Bearer token for API      optional

SwarmResult tools (registered on ingest_agent in pipeline.py):
  list_connections_tool        → list all Airbyte source→destination connections
  trigger_sync_tool            → trigger a sync by connection_id and poll to completion
  create_full_pipeline_tool    → create source + destination + connection then sync
  get_source_schema_tool       → discover schema (tables/streams) from a source
  list_sources_tool            → list all configured sources in Airbyte
  list_destinations_tool       → list all configured destinations in Airbyte
  submit_platform_job_tool     → submit a job via the platform Laravel API

Usage:
    from agents.etl.airbyte_tools import register_airbyte_tools
    register_airbyte_tools(ingest_agent)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import requests
from autogen import ConversableAgent, SwarmResult

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────
_AIRBYTE_BASE     = os.getenv("AIRBYTE_API_URL",      "http://localhost:8000/api")
_AIRBYTE_USER     = os.getenv("AIRBYTE_USERNAME",     "airbyte")
_AIRBYTE_PASS     = os.getenv("AIRBYTE_PASSWORD",     "password")
_DATAPAI_API      = os.getenv("DATAPAI_API_URL",      "https://platform.datap.ai/api")
_DATAPAI_TOKEN    = os.getenv("DATAPAI_API_TOKEN",    "")
_POLL_INTERVAL_S  = int(os.getenv("AIRBYTE_POLL_INTERVAL", "10"))
_POLL_TIMEOUT_S   = int(os.getenv("AIRBYTE_POLL_TIMEOUT",  "1800"))   # 30 min max

# Terminal job statuses
_TERMINAL = {"succeeded", "failed", "cancelled", "incomplete", "error"}


# ═══════════════════════════════════════════════════════════════════════════════
# Low-level Airbyte API client
# ═══════════════════════════════════════════════════════════════════════════════

class AirbyteClient:
    """
    Thin wrapper around the Airbyte Config API v1.

    All methods raise AirbyteAPIError on HTTP errors.
    Workspace ID is auto-discovered on first call if not provided.
    """

    def __init__(
        self,
        base_url: str = _AIRBYTE_BASE,
        username: str = _AIRBYTE_USER,
        password: str = _AIRBYTE_PASS,
        workspace_id: Optional[str] = None,
    ) -> None:
        self.base_url     = base_url.rstrip("/")
        self.auth         = (username, password)
        self._workspace_id = workspace_id or os.getenv("AIRBYTE_WORKSPACE_ID")

    # ── Internals ──────────────────────────────────────────────────────────

    def _post(self, path: str, body: dict) -> dict:
        url = f"{self.base_url}/v1/{path}"
        try:
            r = requests.post(url, json=body, auth=self.auth, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError as exc:
            raise AirbyteAPIError(
                f"Cannot reach Airbyte at {url}. "
                f"Is the Airbyte Docker stack running? ({exc})"
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise AirbyteAPIError(
                f"Airbyte API error [{r.status_code}] {path}: {r.text[:300]}"
            ) from exc

    def _get_workspace_id(self) -> str:
        if self._workspace_id:
            return self._workspace_id
        data = self._post("workspaces/list", {})
        workspaces = data.get("workspaces", [])
        if not workspaces:
            raise AirbyteAPIError("No Airbyte workspaces found.")
        self._workspace_id = workspaces[0]["workspaceId"]
        logger.info("[Airbyte] Using workspace: %s", self._workspace_id)
        return self._workspace_id

    # ── Sources ────────────────────────────────────────────────────────────

    def list_sources(self) -> list[dict]:
        """Return all sources in the workspace."""
        ws = self._get_workspace_id()
        return self._post("sources/list", {"workspaceId": ws}).get("sources", [])

    def create_source(
        self,
        name: str,
        source_definition_id: str,
        connection_configuration: dict,
    ) -> dict:
        """Create a new Airbyte source."""
        ws = self._get_workspace_id()
        return self._post("sources/create", {
            "workspaceId": ws,
            "name": name,
            "sourceDefinitionId": source_definition_id,
            "connectionConfiguration": connection_configuration,
        })

    def get_source_schema(self, source_id: str) -> dict:
        """Discover the schema (tables/streams) of a source."""
        return self._post("sources/discover_schema", {"sourceId": source_id})

    def list_source_definitions(self) -> list[dict]:
        """List all available source connector types."""
        ws = self._get_workspace_id()
        data = self._post("source_definitions/list_for_workspace", {"workspaceId": ws})
        return data.get("sourceDefinitions", [])

    # ── Destinations ───────────────────────────────────────────────────────

    def list_destinations(self) -> list[dict]:
        """Return all destinations in the workspace."""
        ws = self._get_workspace_id()
        return self._post("destinations/list", {"workspaceId": ws}).get("destinations", [])

    def create_destination(
        self,
        name: str,
        destination_definition_id: str,
        connection_configuration: dict,
    ) -> dict:
        """Create a new Airbyte destination."""
        ws = self._get_workspace_id()
        return self._post("destinations/create", {
            "workspaceId": ws,
            "name": name,
            "destinationDefinitionId": destination_definition_id,
            "connectionConfiguration": connection_configuration,
        })

    def list_destination_definitions(self) -> list[dict]:
        """List all available destination connector types."""
        ws = self._get_workspace_id()
        data = self._post("destination_definitions/list_for_workspace", {"workspaceId": ws})
        return data.get("destinationDefinitions", [])

    # ── Connections ────────────────────────────────────────────────────────

    def list_connections(self) -> list[dict]:
        """Return all source→destination connections in the workspace."""
        ws = self._get_workspace_id()
        return self._post("connections/list", {"workspaceId": ws}).get("connections", [])

    def create_connection(
        self,
        source_id: str,
        destination_id: str,
        name: str,
        streams: Optional[list[dict]] = None,
        sync_mode: str = "full_refresh",
        schedule_type: str = "manual",
    ) -> dict:
        """
        Create a source→destination connection.

        Args:
            source_id:       Airbyte source UUID
            destination_id:  Airbyte destination UUID
            name:            Human-readable connection name
            streams:         List of stream configs.  If None, all streams are synced.
            sync_mode:       "full_refresh" | "incremental"
            schedule_type:   "manual" | "cron" | "basic"

        Returns the created connection dict including connectionId.
        """
        body: dict[str, Any] = {
            "sourceId": source_id,
            "destinationId": destination_id,
            "name": name,
            "scheduleType": schedule_type,
            "status": "active",
        }

        if streams:
            body["syncCatalog"] = {"streams": streams}

        if schedule_type == "manual":
            body["scheduleType"] = "manual"

        return self._post("connections/create", body)

    def get_connection(self, connection_id: str) -> dict:
        """Get connection details including syncCatalog."""
        return self._post("connections/get", {"connectionId": connection_id})

    # ── Jobs ───────────────────────────────────────────────────────────────

    def trigger_sync(self, connection_id: str) -> dict:
        """Trigger a sync and return the job object."""
        data = self._post("connections/sync", {"connectionId": connection_id})
        return data.get("job", data)

    def get_job(self, job_id: int) -> dict:
        """Get job status."""
        data = self._post("jobs/get", {"id": job_id})
        return data.get("job", data)

    def trigger_and_poll(
        self,
        connection_id: str,
        poll_interval: int = _POLL_INTERVAL_S,
        timeout: int = _POLL_TIMEOUT_S,
    ) -> dict:
        """
        Trigger a sync and poll until terminal status.

        Returns final job dict with fields: id, status, createdAt, updatedAt.
        Raises AirbyteAPIError on timeout or failure.
        """
        job = self.trigger_sync(connection_id)
        job_id = job["id"]
        logger.info("[Airbyte] Triggered sync  connection=%s  job=%s", connection_id, job_id)

        elapsed = 0
        while elapsed < timeout:
            time.sleep(poll_interval)
            elapsed += poll_interval
            job = self.get_job(job_id)
            status = job.get("status", "")
            logger.info("[Airbyte] Job %s status=%s  elapsed=%ds", job_id, status, elapsed)
            if status in _TERMINAL:
                break

        if job.get("status") not in _TERMINAL:
            raise AirbyteAPIError(
                f"Airbyte job {job_id} timed out after {timeout}s. "
                f"Last status: {job.get('status')}. "
                f"Check Airbyte UI for details."
            )

        return job


class AirbyteAPIError(Exception):
    """Raised when the Airbyte API returns an error or is unreachable."""


# ═══════════════════════════════════════════════════════════════════════════════
# Connector definition ID helpers
# ═══════════════════════════════════════════════════════════════════════════════

# Well-known Airbyte source definition IDs (stable across OSS versions)
SOURCE_DEFINITION_IDS: dict[str, str] = {
    "postgres":           "decd338e-5647-4c0b-adf4-da0e75f5a750",
    "postgresql":         "decd338e-5647-4c0b-adf4-da0e75f5a750",
    "mysql":              "435bb9a5-7887-4809-aa58-28c27df0d7ad",
    "mssql":              "b5ea17b1-f170-46dc-bc31-cc744ca984c1",
    "sqlserver":          "b5ea17b1-f170-46dc-bc31-cc744ca984c1",
    "salesforce":         "b117307c-14b6-41aa-9422-947e34922962",
    "s3":                 "69589781-7828-43c5-9f63-8925b1c1ccc2",
    "gcs":                "56d0f5d4-4a0e-421d-9a99-f168a01d19f4",
    "bigquery":           "6217e33e-9924-4b7c-a4e0-04b9d2e80f2e",  # BigQuery as source
    "hubspot":            "36c891d9-4bd9-43ac-bad2-10e12756272c",
    "shopify":            "9da77001-af33-4bcd-be46-6252bf9342b9",
    "stripe":             "e094cb9a-26de-4645-8761-65c0c425d1de",
    "github":             "ef69ef6e-aa7f-4af1-a01d-ef775033524e",
    "google_sheets":      "71607ba1-c0ac-4799-8049-7f4b90dd50f7",
    "mongodb":            "b2e713cd-cc36-4c0a-b5bd-b47cb8a0561e",
    "snowflake":          "b2e713cd-cc36-4c0a-b5bd-b47cb8a0561e",  # Snowflake as source
    "csv":                "8be1cf83-fde1-477f-a4ad-318d23c9f3c6",
    "file":               "8be1cf83-fde1-477f-a4ad-318d23c9f3c6",
    "rest_api":           "13a656f7-5e4e-4f0b-8249-6e58a5a2bb63",
    "oracle":             "d2147be3-767b-4a5e-9ed3-d1d8de3be2a9",
}

# Well-known Airbyte destination definition IDs
DESTINATION_DEFINITION_IDS: dict[str, str] = {
    "snowflake":          "424892c4-daac-4491-b35d-c6688ba547ba",
    "bigquery":           "22f6c74f-5699-40ff-833c-4a879ea40133",
    "redshift":           "f7a7d195-377f-cf77-88d7-a7a17b069932",
    "postgres":           "25c5221d-dce2-4163-ade9-739ef790f503",
    "postgresql":         "25c5221d-dce2-4163-ade9-739ef790f503",
    "duckdb":             "97f7b02e-ece1-4c93-9fd8-7a42e3fe5eca",
    "s3":                 "4816b78f-1489-44c1-9060-4b19d5fa9362",
    "mysql":              "ca8f1e21-78ad-4b3c-a975-01dd92a4d18b",
    "local_csv":          "b5ea17b1-f170-46dc-bc31-cc744ca984c1",
    "databricks":         "e5f38c12-f25e-4bc5-9777-c07bf4c0e1a0",
}


def _resolve_source_def_id(source_type: str, client: AirbyteClient) -> str:
    """Resolve source definition ID: try lookup table first, then search Airbyte catalog."""
    key = source_type.lower().replace(" ", "_").replace("-", "_")
    if key in SOURCE_DEFINITION_IDS:
        return SOURCE_DEFINITION_IDS[key]
    # Fallback: search the live catalog
    definitions = client.list_source_definitions()
    for d in definitions:
        if source_type.lower() in d.get("name", "").lower():
            return d["sourceDefinitionId"]
    raise AirbyteAPIError(
        f"Unknown source type '{source_type}'. "
        f"Available: {', '.join(SOURCE_DEFINITION_IDS.keys())}"
    )


def _resolve_dest_def_id(dest_type: str, client: AirbyteClient) -> str:
    """Resolve destination definition ID."""
    key = dest_type.lower().replace(" ", "_").replace("-", "_")
    if key in DESTINATION_DEFINITION_IDS:
        return DESTINATION_DEFINITION_IDS[key]
    definitions = client.list_destination_definitions()
    for d in definitions:
        if dest_type.lower() in d.get("name", "").lower():
            return d["destinationDefinitionId"]
    raise AirbyteAPIError(
        f"Unknown destination type '{dest_type}'. "
        f"Available: {', '.join(DESTINATION_DEFINITION_IDS.keys())}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SwarmResult tools — registered on ingest_agent
# ═══════════════════════════════════════════════════════════════════════════════

def _client(ctx: dict) -> AirbyteClient:
    """Build an AirbyteClient from context_variables or env vars."""
    return AirbyteClient(
        base_url=ctx.get("airbyte_api_url", _AIRBYTE_BASE),
        username=ctx.get("airbyte_username", _AIRBYTE_USER),
        password=ctx.get("airbyte_password", _AIRBYTE_PASS),
        workspace_id=ctx.get("airbyte_workspace_id"),
    )


def list_connections_tool(context_variables: dict) -> SwarmResult:
    """
    List all Airbyte source→destination connections in the workspace.

    Returns a summary table with: connection_id, name, source, destination, status.
    Sets context_variables['airbyte_connections'] with the raw list.
    """
    try:
        client = _client(context_variables)
        connections = client.list_connections()

        if not connections:
            return SwarmResult(
                values="No Airbyte connections found in this workspace. "
                       "Use create_full_pipeline_tool to set up a new source→destination connection.",
                context_variables={**context_variables, "airbyte_connections": []},
            )

        lines = [f"Found {len(connections)} Airbyte connection(s):\n"]
        for c in connections:
            lines.append(
                f"  ID:   {c.get('connectionId', 'N/A')}\n"
                f"  Name: {c.get('name', 'N/A')}\n"
                f"  Status: {c.get('status', 'N/A')}\n"
                f"  Schedule: {c.get('scheduleType', 'manual')}\n"
                f"  ---"
            )

        ctx = {**context_variables, "airbyte_connections": connections}
        return SwarmResult(values="\n".join(lines), context_variables=ctx)

    except AirbyteAPIError as exc:
        return SwarmResult(
            values=f"[Airbyte] Connection list failed: {exc}",
            context_variables=context_variables,
        )


def list_sources_tool(context_variables: dict) -> SwarmResult:
    """
    List all configured Airbyte sources.

    Returns source names, IDs, and connector types.
    """
    try:
        client = _client(context_variables)
        sources = client.list_sources()

        if not sources:
            return SwarmResult(
                values="No Airbyte sources configured yet.",
                context_variables={**context_variables, "airbyte_sources": []},
            )

        lines = [f"Found {len(sources)} Airbyte source(s):\n"]
        for s in sources:
            lines.append(
                f"  ID:   {s.get('sourceId', 'N/A')}\n"
                f"  Name: {s.get('name', 'N/A')}\n"
                f"  Type: {s.get('sourceName', 'N/A')}\n"
                f"  ---"
            )

        ctx = {**context_variables, "airbyte_sources": sources}
        return SwarmResult(values="\n".join(lines), context_variables=ctx)

    except AirbyteAPIError as exc:
        return SwarmResult(
            values=f"[Airbyte] Source list failed: {exc}",
            context_variables=context_variables,
        )


def list_destinations_tool(context_variables: dict) -> SwarmResult:
    """
    List all configured Airbyte destinations.

    Returns destination names, IDs, and connector types.
    """
    try:
        client = _client(context_variables)
        destinations = client.list_destinations()

        if not destinations:
            return SwarmResult(
                values="No Airbyte destinations configured yet.",
                context_variables={**context_variables, "airbyte_destinations": []},
            )

        lines = [f"Found {len(destinations)} Airbyte destination(s):\n"]
        for d in destinations:
            lines.append(
                f"  ID:   {d.get('destinationId', 'N/A')}\n"
                f"  Name: {d.get('name', 'N/A')}\n"
                f"  Type: {d.get('destinationName', 'N/A')}\n"
                f"  ---"
            )

        ctx = {**context_variables, "airbyte_destinations": destinations}
        return SwarmResult(values="\n".join(lines), context_variables=ctx)

    except AirbyteAPIError as exc:
        return SwarmResult(
            values=f"[Airbyte] Destination list failed: {exc}",
            context_variables=context_variables,
        )


def get_source_schema_tool(
    context_variables: dict,
    source_id: str,
) -> SwarmResult:
    """
    Discover the schema (tables/streams) of an Airbyte source.

    Args:
        source_id: The Airbyte source UUID.

    Returns stream names, field names, and sync modes available.
    Sets context_variables['source_catalog'] with the full catalog.
    """
    try:
        client = _client(context_variables)
        catalog = client.get_source_schema(source_id)

        streams = catalog.get("catalog", {}).get("streams", [])
        if not streams:
            return SwarmResult(
                values=f"No streams found for source {source_id}.",
                context_variables={**context_variables, "source_catalog": catalog},
            )

        lines = [f"Discovered {len(streams)} stream(s) from source {source_id}:\n"]
        for s in streams[:20]:  # cap at 20 for readability
            stream = s.get("stream", {})
            fields = list(stream.get("jsonSchema", {}).get("properties", {}).keys())
            lines.append(
                f"  Stream: {stream.get('name', 'N/A')}\n"
                f"  Namespace: {stream.get('namespace', 'public')}\n"
                f"  Fields ({len(fields)}): {', '.join(fields[:10])}"
                + (" …" if len(fields) > 10 else "") + "\n"
                f"  Sync modes: {stream.get('supportedSyncModes', ['full_refresh'])}\n"
                f"  ---"
            )

        ctx = {**context_variables, "source_catalog": catalog, "source_id": source_id}
        return SwarmResult(values="\n".join(lines), context_variables=ctx)

    except AirbyteAPIError as exc:
        return SwarmResult(
            values=f"[Airbyte] Schema discovery failed for source {source_id}: {exc}",
            context_variables=context_variables,
        )


def trigger_sync_tool(
    context_variables: dict,
    connection_id: str,
    wait_for_completion: bool = True,
) -> SwarmResult:
    """
    Trigger an Airbyte sync for an existing connection and optionally wait for completion.

    Args:
        connection_id:        The Airbyte connection UUID to sync.
        wait_for_completion:  If True (default), poll until the job completes.
                              If False, return immediately with the job ID.

    Returns sync status, rows synced, and any error details.
    Sets context_variables['airbyte_job'] and 'sync_status'.
    """
    try:
        client = _client(context_variables)

        if wait_for_completion:
            job = client.trigger_and_poll(connection_id)
        else:
            job = client.trigger_sync(connection_id)

        status    = job.get("status", "unknown")
        job_id    = job.get("id", "")
        succeeded = status == "succeeded"

        summary = (
            f"Airbyte sync {'completed' if wait_for_completion else 'triggered'}.\n"
            f"  Job ID:     {job_id}\n"
            f"  Connection: {connection_id}\n"
            f"  Status:     {status}\n"
        )

        if not succeeded and wait_for_completion:
            summary += (
                f"\n⚠ Sync did not succeed (status={status}). "
                f"Check Airbyte UI at {_AIRBYTE_BASE.replace('/api', '')} "
                f"for job {job_id} details."
            )

        ctx = {
            **context_variables,
            "airbyte_job": job,
            "airbyte_job_id": str(job_id),
            "sync_status": status,
            "sync_succeeded": succeeded,
            "airbyte_connection_id": connection_id,
        }
        return SwarmResult(values=summary, context_variables=ctx)

    except AirbyteAPIError as exc:
        return SwarmResult(
            values=f"[Airbyte] Sync failed for connection {connection_id}: {exc}",
            context_variables={**context_variables, "sync_status": "error", "sync_succeeded": False},
        )


def create_full_pipeline_tool(
    context_variables: dict,
    source_type: str,
    source_name: str,
    source_config: str,
    destination_type: str,
    destination_name: str,
    destination_config: str,
    connection_name: str = "",
    streams: str = "",
    sync_mode: str = "full_refresh",
    trigger_sync: bool = True,
) -> SwarmResult:
    """
    Create a complete Airbyte pipeline (source + destination + connection) and trigger sync.

    This is the primary E2E ingestion tool. Pass source and destination configs as
    JSON strings. The tool will:
      1. Create the source connector
      2. Discover the schema (optional stream filtering)
      3. Create the destination connector
      4. Create the source→destination connection
      5. Trigger the first sync and poll to completion

    Args:
        source_type:         Source connector type. Examples:
                               "postgres", "mysql", "mssql", "salesforce",
                               "s3", "hubspot", "shopify", "stripe", "github",
                               "google_sheets", "mongodb", "rest_api", "file"
        source_name:         Human-readable name for this source (e.g. "Production Postgres")
        source_config:       JSON string with connector-specific config. Examples:

                             Postgres:
                               {"host":"db.example.com","port":5432,"database":"mydb",
                                "username":"user","password":"pass","ssl_mode":{"mode":"prefer"}}

                             MSSQL / SQL Server:
                               {"host":"sql.example.com","port":1433,"database":"mydb",
                                "username":"sa","password":"pass"}

                             MySQL:
                               {"host":"mysql.example.com","port":3306,"database":"mydb",
                                "username":"root","password":"pass"}

                             S3 (CSV):
                               {"dataset":"my_data","path_pattern":"**/*.csv",
                                "format":{"filetype":"csv"},"url":"s3://bucket/prefix/",
                                "provider":{"storage":"S3","aws_access_key_id":"...","aws_secret_access_key":"..."}}

                             Salesforce:
                               {"client_id":"...","client_secret":"...","refresh_token":"...",
                                "is_sandbox":false,"start_date":"2024-01-01T00:00:00Z"}

        destination_type:    Destination connector type. Examples:
                               "snowflake", "bigquery", "redshift", "postgres", "duckdb", "s3"
        destination_name:    Human-readable name for this destination (e.g. "Snowflake DW")
        destination_config:  JSON string with connector-specific config. Examples:

                             Snowflake:
                               {"host":"acct.snowflakecomputing.com","role":"AIRBYTE_ROLE",
                                "warehouse":"AIRBYTE_WH","database":"RAW","schema":"PUBLIC",
                                "username":"airbyte","credentials":{"password":"pass"}}

                             BigQuery:
                               {"project_id":"my-project","dataset_id":"raw_data",
                                "credentials_json":"<service-account-json>"}

                             Redshift:
                               {"host":"cluster.region.redshift.amazonaws.com","port":5439,
                                "database":"mydb","username":"user","password":"pass",
                                "schema":"public","uploading_method":{"method":"Standard"}}

                             Postgres (as destination):
                               {"host":"db.example.com","port":5432,"database":"mydb",
                                "username":"user","password":"pass","schema":"public"}

                             DuckDB:
                               {"destination_path":"/local/datapai.duckdb"}

        connection_name:     Name for the connection (auto-generated if empty).
        streams:             Comma-separated list of table/stream names to sync.
                             If empty, all streams are synced.
        sync_mode:           "full_refresh" (default) | "incremental"
        trigger_sync:        If True (default), immediately trigger the first sync.

    Returns creation status, connection_id, and sync results.
    """
    ctx = dict(context_variables)

    try:
        client = _client(ctx)

        # Parse configs
        try:
            src_cfg  = json.loads(source_config)  if isinstance(source_config,  str) else source_config
            dst_cfg  = json.loads(destination_config) if isinstance(destination_config, str) else destination_config
        except json.JSONDecodeError as exc:
            return SwarmResult(
                values=f"Invalid JSON in source_config or destination_config: {exc}\n"
                       f"Please provide valid JSON strings.",
                context_variables=ctx,
            )

        # Step 1: Resolve definition IDs
        try:
            src_def_id  = _resolve_source_def_id(source_type, client)
            dst_def_id  = _resolve_dest_def_id(destination_type, client)
        except AirbyteAPIError as exc:
            return SwarmResult(values=str(exc), context_variables=ctx)

        # Step 2: Create source
        logger.info("[Airbyte] Creating source: %s (%s)", source_name, source_type)
        source = client.create_source(source_name, src_def_id, src_cfg)
        source_id = source["sourceId"]
        ctx["airbyte_source_id"]   = source_id
        ctx["airbyte_source_name"] = source_name
        logger.info("[Airbyte] Source created: %s", source_id)

        # Step 3: Create destination
        logger.info("[Airbyte] Creating destination: %s (%s)", destination_name, destination_type)
        destination = client.create_destination(destination_name, dst_def_id, dst_cfg)
        destination_id = destination["destinationId"]
        ctx["airbyte_destination_id"]   = destination_id
        ctx["airbyte_destination_name"] = destination_name
        logger.info("[Airbyte] Destination created: %s", destination_id)

        # Step 4: Discover schema + build stream list
        catalog_streams = None
        if streams:
            stream_names = {s.strip() for s in streams.split(",") if s.strip()}
            try:
                catalog = client.get_source_schema(source_id)
                all_streams = catalog.get("catalog", {}).get("streams", [])
                catalog_streams = [
                    {
                        "stream": s["stream"],
                        "config": {
                            "syncMode": sync_mode,
                            "destinationSyncMode": "overwrite" if sync_mode == "full_refresh" else "append_dedup",
                            "selected": True,
                        },
                    }
                    for s in all_streams
                    if s.get("stream", {}).get("name", "") in stream_names
                ]
            except AirbyteAPIError as exc:
                logger.warning("[Airbyte] Schema discovery failed (will sync all streams): %s", exc)

        # Step 5: Create connection
        conn_name = connection_name or f"{source_name} → {destination_name}"
        logger.info("[Airbyte] Creating connection: %s", conn_name)
        connection = client.create_connection(
            source_id=source_id,
            destination_id=destination_id,
            name=conn_name,
            streams=catalog_streams,
            sync_mode=sync_mode,
        )
        connection_id = connection["connectionId"]
        ctx["airbyte_connection_id"] = connection_id
        ctx["airbyte_connection_name"] = conn_name
        logger.info("[Airbyte] Connection created: %s", connection_id)

        # Step 6: Trigger sync
        sync_summary = ""
        if trigger_sync:
            logger.info("[Airbyte] Triggering first sync for connection %s", connection_id)
            job = client.trigger_and_poll(connection_id)
            status = job.get("status", "unknown")
            ctx["airbyte_job"]     = job
            ctx["sync_status"]     = status
            ctx["sync_succeeded"]  = status == "succeeded"
            sync_summary = (
                f"\n  First sync status: {status} (job {job.get('id')})"
            )
            if status != "succeeded":
                sync_summary += (
                    f"\n  ⚠ Sync did not succeed. Check Airbyte UI for job details."
                )
        else:
            ctx["sync_status"] = "pending"

        # Update pipeline context for downstream agents
        ctx["pipeline_status"]   = "ingested" if ctx.get("sync_succeeded") else "sync_pending"
        ctx["ingestion_method"]  = "airbyte"
        ctx["source_type"]       = source_type
        ctx["destination_type"]  = destination_type

        summary = (
            f"✅ Airbyte pipeline created successfully.\n"
            f"  Source:      {source_name} ({source_type}) — ID: {source_id}\n"
            f"  Destination: {destination_name} ({destination_type}) — ID: {destination_id}\n"
            f"  Connection:  {conn_name} — ID: {connection_id}\n"
            f"  Streams:     {'all' if not streams else streams}\n"
            f"  Sync mode:   {sync_mode}"
            f"{sync_summary}"
        )
        return SwarmResult(values=summary, context_variables=ctx)

    except AirbyteAPIError as exc:
        return SwarmResult(
            values=f"[Airbyte] Pipeline creation failed: {exc}",
            context_variables=ctx,
        )
    except Exception as exc:
        logger.exception("Unexpected error in create_full_pipeline_tool")
        return SwarmResult(
            values=f"[Airbyte] Unexpected error: {type(exc).__name__}: {exc}",
            context_variables=ctx,
        )


def submit_platform_job_tool(
    context_variables: dict,
    dataflow_id: str = "",
    connection_id: str = "",
) -> SwarmResult:
    """
    Submit a sync job via the DataPAI platform API (platform.datap.ai/api/datasource/submit_job).

    This is an alternative to calling Airbyte directly — it goes through the Laravel
    backend which may apply additional business logic, logging, or notifications.

    Args:
        dataflow_id:    The platform dataflow ID (from /api/dataflow/list).
        connection_id:  The platform connection ID (from /api/connection/list).
                        One of dataflow_id or connection_id must be provided.
    """
    if not dataflow_id and not connection_id:
        return SwarmResult(
            values="Either dataflow_id or connection_id is required to submit a platform job.",
            context_variables=context_variables,
        )

    api_base = context_variables.get("datapai_api_url", _DATAPAI_API)
    token    = context_variables.get("datapai_api_token", _DATAPAI_TOKEN)

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    body: dict[str, str] = {}
    if dataflow_id:
        body["dataflow_id"] = dataflow_id
    if connection_id:
        body["connection_id"] = connection_id

    try:
        r = requests.post(
            f"{api_base}/datasource/submit_job",
            json=body,
            headers=headers,
            timeout=30,
        )
        if r.status_code in (200, 201):
            result = r.json() if r.text else {}
            return SwarmResult(
                values=(
                    f"Platform job submitted.\n"
                    f"  Status: {result.get('status', 'submitted')}\n"
                    f"  Job ID: {result.get('job_id', 'N/A')}\n"
                    f"  Message: {result.get('message', '')}"
                ),
                context_variables={
                    **context_variables,
                    "platform_job_id": result.get("job_id", ""),
                    "platform_job_status": result.get("status", "submitted"),
                },
            )
        else:
            return SwarmResult(
                values=f"Platform job submission failed [{r.status_code}]: {r.text[:300]}",
                context_variables=context_variables,
            )
    except Exception as exc:
        return SwarmResult(
            values=f"[Platform API] submit_job error: {exc}",
            context_variables=context_variables,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Registration helper — call this from pipeline.py
# ═══════════════════════════════════════════════════════════════════════════════

def register_airbyte_tools(
    ingest_agent: ConversableAgent,
    audit_decorator=None,
) -> None:
    """
    Register all Airbyte SwarmResult tools on the ingest_agent.

    Args:
        ingest_agent:    The AG2 ConversableAgent to register tools on.
        audit_decorator: Optional audit_tool wrapper (e.g. audit_tool("ingest_agent")).
                         If provided, every tool is wrapped before registration.
    """
    def _wrap(fn):
        return audit_decorator(fn) if audit_decorator else fn

    tools = [
        (
            list_connections_tool,
            "List all Airbyte source→destination connections configured in the workspace. "
            "Use this to discover existing pipelines before creating new ones.",
        ),
        (
            list_sources_tool,
            "List all Airbyte source connectors configured in the workspace.",
        ),
        (
            list_destinations_tool,
            "List all Airbyte destination connectors configured in the workspace.",
        ),
        (
            get_source_schema_tool,
            "Discover the schema (tables/streams) of an Airbyte source connector. "
            "Use source_id from list_sources_tool. Returns stream names, field names, sync modes.",
        ),
        (
            trigger_sync_tool,
            "Trigger an Airbyte sync for an existing connection and wait for completion. "
            "Use connection_id from list_connections_tool. "
            "Set wait_for_completion=False to fire-and-forget.",
        ),
        (
            create_full_pipeline_tool,
            "Create a complete Airbyte E2E pipeline: source + destination + connection, "
            "then trigger the first sync. "
            "Supports Postgres, MySQL, MSSQL, Salesforce, S3, HubSpot, Stripe, BigQuery, "
            "Google Sheets, REST API, MongoDB → Snowflake, BigQuery, Redshift, Postgres, DuckDB, S3. "
            "Provide source_config and destination_config as JSON strings.",
        ),
        (
            submit_platform_job_tool,
            "Submit a data sync job via the DataPAI platform API (platform.datap.ai). "
            "Requires dataflow_id or connection_id from the platform. "
            "Use as an alternative to direct Airbyte API calls.",
        ),
    ]

    for fn, description in tools:
        wrapped = _wrap(fn)
        ingest_agent.register_for_llm(description=description)(wrapped)
        ingest_agent.register_for_execution()(wrapped)

    logger.info(
        "[Airbyte] Registered %d tools on %s", len(tools), ingest_agent.name
    )

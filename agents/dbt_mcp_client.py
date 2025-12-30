# dbt_mcp_client.py

import os
import json
import uuid
from typing import Dict, Any, Optional

import requests


class DbtMcpClient:
    """
    Minimal JSON-RPC style client to talk to a dbt MCP server.

    Assumptions:
      - HTTP MCP bridge/server
      - URL: MCP_DBT_SERVER_URL env var
      - Uses JSON-RPC 2.0:
          { "jsonrpc": "2.0", "id": "...", "method": "<tool_name>", "params": {...} }

    Adjust method names to match your actual dbt MCP server.
    """

    def __init__(self, base_url: Optional[str] = None, timeout: int = 600):
        self.base_url = base_url or os.environ.get("MCP_DBT_SERVER_URL")
        if not self.base_url:
            raise ValueError("MCP_DBT_SERVER_URL env var not set and no base_url provided.")
        self.base_url = self.base_url.rstrip("/")
        self.timeout = timeout

    def _rpc(self, method: str, params: Dict[str, Any]) -> Any:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params,
        }
        resp = requests.post(self.base_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        if "error" in data and data["error"] is not None:
            raise RuntimeError(f"MCP error from {method}: {data['error']}")

        return data.get("result")

    def run_dbt_command(self, command: str) -> Dict[str, Any]:
        # Change method="dbt_cli.run" to your server's method name
        return self._rpc(method="dbt_cli.run", params={"command": command})

    def get_manifest_summary(self) -> Dict[str, Any]:
        return self._rpc(method="dbt.get_manifest", params={})

    def get_model_info(self, node_name: str) -> Dict[str, Any]:
        return self._rpc(method="dbt.get_node", params={"node_name": node_name})


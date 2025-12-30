# mcp/__init__.py
# Initialize MCP package (Model Context Protocol)

# This can remain empty or be used to expose shared utilities


# mcp/model_registry.py
import json
import os

REGISTRY_PATH = "mcp/registry.json"


def load_tool_registry(path: str = REGISTRY_PATH) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_tool_registry(registry: dict, path: str = REGISTRY_PATH):
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def register_tool(name: str, metadata: dict, registry_path: str = REGISTRY_PATH):
    registry = load_tool_registry(registry_path)
    registry[name] = metadata
    save_tool_registry(registry, registry_path)


def list_registered_tools(registry_path: str = REGISTRY_PATH) -> list:
    registry = load_tool_registry(registry_path)
    return list(registry.keys())



# tools.py

from typing import Callable, Dict, Any
import inspect
import json

_TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def tool(name: str = None):
    """
    Decorator to register a function as a tool.

    Usage:
        @tool()
        def list_s3_files(prefix: str) -> list[str]: ...
    """
    def decorator(func: Callable[..., Any]):
        tool_name = name or func.__name__
        if tool_name in _TOOL_REGISTRY:
            raise ValueError(f"Tool '{tool_name}' already registered")
        _TOOL_REGISTRY[tool_name] = func
        return func
    return decorator


def get_tool(name: str) -> Callable[..., Any]:
    return _TOOL_REGISTRY[name]


def list_tools() -> Dict[str, Dict[str, Any]]:
    """
    Returns a schema-like summary of available tools for the LLM.
    """
    result = {}
    for name, fn in _TOOL_REGISTRY.items():
        sig = inspect.signature(fn)
        params = []
        for pname, p in sig.parameters.items():
            params.append({
                "name": pname,
                "kind": str(p.kind),
                "default": p.default if p.default is not inspect._empty else None,
                "annotation": str(p.annotation) if p.annotation is not inspect._empty else "Any",
            })
        result[name] = {
            "doc": inspect.getdoc(fn) or "",
            "parameters": params,
            "returns": str(sig.return_annotation) if sig.return_annotation is not inspect._empty else "Any",
        }
    return result


def call_tool_from_json_call(tool_call: str) -> Any:
    """
    Parse LLM-produced JSON describing a tool call and execute it.

    Expected JSON format:
    {
      "tool_name": "list_s3_files",
      "args": {
        "prefix": "s3://bucket/path"
      }
    }
    """
    obj = json.loads(tool_call)
    tool_name = obj["tool_name"]
    args = obj.get("args", {}) or {}
    fn = get_tool(tool_name)
    return fn(**args)

import file_ingest_agent  # noqa: F401
import dbt_agent          # noqa: F401
import snowflake_ingest_agent # noqa: F401   

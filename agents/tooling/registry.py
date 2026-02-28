from typing import Callable, Dict, Any, List
import json

_TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {}

def tool(name: str):
    def deco(fn: Callable[..., Any]):
        _TOOL_REGISTRY[name or fn.__name__] = fn
        return fn
    return deco

def list_tools() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for n, fn in _TOOL_REGISTRY.items():
        out.append({"name": n, "doc": (fn.__doc__ or "").strip()})
    return out

def call_tool_from_json_call(tool_call: str) -> Any:
    obj = json.loads(tool_call)
    fn = _TOOL_REGISTRY[obj["tool_name"]]
    return fn(**(obj.get("args") or {}))

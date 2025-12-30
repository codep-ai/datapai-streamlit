def _show_mcp_health(registry: dict) -> dict:
    """
    Dummy implementation that assumes each tool is reachable
    In real scenarios, you might ping endpoints or check DB connectivity
    """
    return {tool: "✅ OK" for tool in registry.keys()}


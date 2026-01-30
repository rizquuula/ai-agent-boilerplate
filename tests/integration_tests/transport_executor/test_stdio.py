"""Integration tests for StdioTransport using localtime MCP server."""


from agent.mcp.transport_executor.stdio import StdioTransport


def test_list_tools_success(localtime_mcp_path):
    """Test successfully listing tools from the localtime MCP server."""
    transport = StdioTransport()

    try:
        transport.start("uvx", [localtime_mcp_path])
        tools = transport.list_tools()

        assert isinstance(tools, list)
        assert "get_current_time" in tools
    finally:
        transport.stop()

def test_get_current_time_success(localtime_mcp_path):
    """Test successfully getting current time from the localtime MCP server."""
    transport = StdioTransport()

    try:
        transport.start("uvx", [localtime_mcp_path])
        result = transport.execute_tool("get_current_time")

        assert "result" in result
        time_data = result["result"]
        assert "iso_datetime" in time_data
        assert "unix_timestamp" in time_data
        assert "timezone" in time_data
        assert isinstance(time_data["unix_timestamp"], int)
    finally:
        transport.stop()

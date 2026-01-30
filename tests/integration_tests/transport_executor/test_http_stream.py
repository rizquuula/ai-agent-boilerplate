"""Integration tests for HTTPStreamTransport using localtime MCP server."""

from agent.mcp.transport_executor.http_stream import HTTPStreamTransport


def test_list_tools_success(http_stream_server_url):
    """Test successfully listing tools from the localtime MCP server via HTTP Stream."""
    transport = HTTPStreamTransport()

    try:
        transport.start("http", [http_stream_server_url])
        tools = transport.list_tools()

        assert isinstance(tools, list)
        assert "get_current_time" in tools
    finally:
        transport.stop()


def test_get_current_time_success(http_stream_server_url):
    """Test successfully getting current time from the localtime MCP server via HTTP Stream."""
    transport = HTTPStreamTransport()

    try:
        transport.start("http", [http_stream_server_url])
        response = transport.execute_tool("get_current_time")

        assert response["success"] is True
        time_data = response["result"]
        assert "iso_datetime" in time_data
        assert "unix_timestamp" in time_data
        assert "timezone" in time_data
        assert isinstance(time_data["unix_timestamp"], int)
    finally:
        transport.stop()

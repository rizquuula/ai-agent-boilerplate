"""Shared fixtures for integration tests."""

import pytest


@pytest.fixture
def localtime_mcp_path():
    """Return the path to the localtime MCP server."""
    return "external_resources/mcp_servers/stdio/localtime_mcp"


@pytest.fixture(scope="module")
def sse_server_url():
    """Return the URL for the SSE MCP server."""
    return "http://localhost:3000"


@pytest.fixture(scope="module")
def http_stream_server_url():
    """Return the URL for the HTTP Stream MCP server."""
    return "http://localhost:3001"

"""Shared fixtures for unit tests."""


import pytest


@pytest.fixture
def localtime_mcp_path():
    """Return the path to the localtime MCP server."""
    return "external_resources/mcp_servers/stdio/localtime_mcp"

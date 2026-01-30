"""Test MCP configuration system."""

from unittest.mock import patch

import pytest

from agent.mcp.config import MCPConfig


@patch("agent.mcp.config.MCPConfig.load_config")
def test_load_config(mock_load):
    """Test loading MCP configuration from file."""
    mock_config = {
        "mcpServers": {
            "filesystem": {
                "command": "uvx",
                "args": ["external_resources/mcp_servers/stdio/filesystem_mcp"],
                "transport": "stdio",
            }
        }
    }
    mock_load.return_value = mock_config

    config = MCPConfig()
    loaded_config = config.load_config()

    assert loaded_config == mock_config
    assert "mcpServers" in loaded_config
    assert "filesystem" in loaded_config["mcpServers"]


@patch("agent.mcp.config.MCPConfig.load_config")
def test_get_server_metadata(mock_load):
    """Test getting server metadata."""
    mock_config = {
        "mcpServers": {
            "filesystem": {
                "command": "uvx",
                "args": ["external_resources/mcp_servers/stdio/filesystem_mcp"],
                "transport": "stdio",
            }
        }
    }
    mock_load.return_value = mock_config

    config = MCPConfig()
    metadata = config.get_server_metadata("filesystem")

    assert metadata == {
        "command": "uvx",
        "args": ["external_resources/mcp_servers/stdio/filesystem_mcp"],
        "transport": "stdio",
    }
    assert config.get_server_metadata("non_existent") is None


@patch("agent.mcp.config.MCPConfig.load_config")
def test_get_available_servers(mock_load):
    """Test getting list of available servers."""
    mock_config = {"mcpServers": {"filesystem": {}, "code_parser": {}}}
    mock_load.return_value = mock_config

    config = MCPConfig()
    servers = config.get_available_servers()

    assert "filesystem" in servers
    assert "code_parser" in servers
    assert len(servers) == 2


@patch("agent.mcp.config.MCPConfig.load_config")
def test_get_enabled_servers(mock_load):
    """Test getting list of enabled servers."""
    mock_config = {"mcpServers": {"filesystem": {"enabled": True}, "code_parser": {"enabled": False}}}
    mock_load.return_value = mock_config

    config = MCPConfig()
    enabled_servers = config.get_enabled_servers()

    assert "filesystem" in enabled_servers
    assert "code_parser" not in enabled_servers


if __name__ == "__main__":
    pytest.main([__file__])

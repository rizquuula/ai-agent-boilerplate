"""MCP Configuration Management.

This module provides functionality to load and manage MCP server configurations
from a JSON file, enabling flexible and dynamic MCP server management.
"""

import json
from pathlib import Path
from typing import Any

MCP_SERVERS_KEY = "mcpServers"


class MCPConfig:
    """MCP server configuration manager."""

    def __init__(self, config_path: str | None = None):
        """
        Initialize MCP configuration.

        Args:
            config_path: Path to the MCP configuration file. If None, uses default location.
        """
        if config_path is None:
            # Use the project root config directory
            config_path = Path(__file__).parent.parent.parent / "config" / "mcp_servers.json"

        self.config_path = Path(config_path)
        self._config: dict[str, Any] | None = None

    def load_config(self) -> dict[str, Any]:
        """
        Load MCP server configuration from file.

        Returns:
            Dictionary containing the loaded configuration.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file has invalid YAML syntax.
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"MCP configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, encoding="utf-8") as f:
                config = json.load(f)

            if not config or MCP_SERVERS_KEY not in config:
                raise ValueError("Invalid MCP configuration: missing 'mcpServers' section")

            self._config = config
            return config

        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing MCP configuration file: {e}") from e

    def get_config(self) -> dict[str, Any]:
        """
        Get the loaded MCP configuration.

        Returns:
            Dictionary containing the MCP configuration.

        Raises:
            RuntimeError: If configuration hasn't been loaded yet.
        """
        if self._config is None:
            return self.load_config()
        return self._config

    def get_server_config(self, server_name: str) -> dict[str, Any] | None:
        """
        Get configuration for a specific MCP server.

        Args:
            server_name: Name of the MCP server.

        Returns:
            Dictionary containing server configuration, or None if not found.
        """
        config = self.get_config()
        servers = config.get(MCP_SERVERS_KEY, {})

        if server_name not in servers:
            return None

        server_config = servers[server_name]

        # Validate required fields
        if "tools" not in server_config:
            server_config["tools"] = []

        if "enabled" not in server_config:
            server_config["enabled"] = True

        if "connection" not in server_config:
            server_config["connection"] = {"type": "local"}

        return server_config

    def get_available_servers(self) -> list[str]:
        """
        Get list of all available MCP servers.

        Returns:
            List of server names.
        """
        config = self.get_config()
        servers = config.get(MCP_SERVERS_KEY, {})
        return list(servers.keys())

    def get_enabled_servers(self) -> list[str]:
        """
        Get list of enabled MCP servers.

        Returns:
            List of enabled server names.
        """
        config = self.get_config()
        servers = config.get(MCP_SERVERS_KEY, {})

        enabled_servers = []
        for server_name, server_config in servers.items():
            if server_config.get("enabled", True):
                enabled_servers.append(server_name)

        return enabled_servers

    def is_server_enabled(self, server_name: str) -> bool:
        """
        Check if a specific MCP server is enabled.

        Args:
            server_name: Name of the MCP server.

        Returns:
            True if server is enabled, False otherwise.
        """
        server_config = self.get_server_config(server_name)
        if server_config is None:
            return False
        return server_config.get("enabled", True)

    def get_server_metadata(self, server_name: str) -> dict[str, Any] | None:
        """
        Get server metadata including command and transport type.

        Args:
            server_name: Name of the MCP server.

        Returns:
            Dictionary containing server metadata, or None if server not found.
        """
        server_config = self.get_server_config(server_name)
        if server_config is None:
            return None

        return {
            "command": server_config.get("command"),
            "args": server_config.get("args", []),
            "transport": server_config.get("transport", "stdio"),
        }


# Global MCP configuration instance
_mcp_config: MCPConfig | None = None


def get_mcp_config() -> MCPConfig:
    """
    Get the global MCP configuration instance.

    Returns:
        MCPConfig instance.
    """
    global _mcp_config
    if _mcp_config is None:
        _mcp_config = MCPConfig()
    return _mcp_config


def load_mcp_config(config_path: str | None = None) -> MCPConfig:
    """
    Load MCP configuration from file.

    Args:
        config_path: Path to the MCP configuration file. If None, uses default location.

    Returns:
        MCPConfig instance with loaded configuration.
    """
    global _mcp_config
    _mcp_config = MCPConfig(config_path)
    _mcp_config.load_config()
    return _mcp_config

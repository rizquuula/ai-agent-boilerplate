"""MCP Tool Execution Interface.

This module provides a dynamic interface for executing MCP tools based on
configuration, replacing the hardcoded implementations in the executor node.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from asterism.mcp.transport_executor.base import BaseTransport

from .config import MCPConfig, get_mcp_config
from .transport_executor import create_transport


class MCPExecutor:
    """Dynamic MCP tool executor that uses configuration-based tool routing."""

    def __init__(self, config_path: str | None = None):
        """
        Initialize the MCP executor.

        Args:
            config_path: Path to the MCP configuration file. If None, uses default location.
        """
        if config_path:
            self.config = MCPConfig(config_path)
            self.config.load_config()
        else:
            self.config = get_mcp_config()
        self.transports: dict[str, BaseTransport | None] = {}
        self.tool_cache: dict[str, list] = {}
        self.tool_schema_cache: dict[str, list[dict[str, Any]]] = {}

        self._log = logging.getLogger(self.__class__.__name__)

    def _get_transport(self, server_name: str) -> BaseTransport:
        """Get or create transport for a server."""
        if server_name not in self.transports:
            metadata = self.config.get_server_metadata(server_name)
            if not metadata:
                raise ValueError(f"No metadata found for server: {server_name}")

            transport = create_transport(metadata["transport"])
            transport.start(metadata["command"], metadata["args"], metadata.get("cwd"))
            self.transports[server_name] = transport

            # Cache tools for this server
            self.tool_cache[server_name] = transport.list_tools()
        return self.transports[server_name]

    def execute_tool(self, server_name: str, tool_name: str, **kwargs) -> dict[str, Any]:
        """
        Execute an MCP tool dynamically based on configuration.

        Args:
            server_name: Name of the MCP server.
            tool_name: Name of the tool to execute.
            **kwargs: Additional arguments for the tool.

        Returns:
            Dictionary containing execution result with keys:
                - success: Boolean indicating if execution succeeded
                - result: The tool result or None if failed
                - error: Error message if failed, None if succeeded
                - tool: The tool identifier used
                - tool_call: The original tool call string
        """
        try:
            # Validate server is enabled
            if not self.config.is_server_enabled(server_name):
                return {
                    "success": False,
                    "error": f"MCP server '{server_name}' is not enabled",
                    "result": None,
                    "tool": f"{server_name}:{tool_name}",
                    "tool_call": f"{server_name}:{tool_name}",
                }

            # Get transport and validate tool
            transport = self._get_transport(server_name)
            if tool_name not in self.tool_cache.get(server_name, []):
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' not found on server '{server_name}'",
                    "result": None,
                    "tool": f"{server_name}:{tool_name}",
                    "tool_call": f"{server_name}:{tool_name}",
                }

            # Execute the tool via transport
            result = transport.execute_tool(tool_name, **kwargs)
            return {
                "success": True,
                "result": result,
                "error": None,
                "tool": f"{server_name}:{tool_name}",
                "tool_call": f"{server_name}:{tool_name}",
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Error executing tool: {str(e)}",
                "result": None,
                "tool": f"{server_name}:{tool_name}",
                "tool_call": f"{server_name}:{tool_name}",
            }

    def get_available_tools(self) -> dict[str, list]:
        """
        Get all available tools organized by server.

        Returns:
            Dictionary mapping server names to lists of available tool names.
        """
        available_tools = {}
        enabled_servers = self.config.get_enabled_servers()

        for server_name in enabled_servers:
            # Initialize transport to populate tool cache
            try:
                self._get_transport(server_name)
                available_tools[server_name] = self.tool_cache.get(server_name, [])
            except Exception:
                available_tools[server_name] = []

        return available_tools

    def get_tool_schemas(self) -> dict[str, list[dict[str, Any]]]:
        """
        Get detailed tool information for all enabled servers.

        Returns:
            Dictionary mapping server names to lists of tool schema objects.
            Each tool object contains: name, description, inputSchema.
        """
        tool_schemas = {}
        enabled_servers = self.config.get_enabled_servers()

        for server_name in enabled_servers:
            # Check if we already have schemas cached
            if server_name in self.tool_schema_cache:
                tool_schemas[server_name] = self.tool_schema_cache[server_name]
                continue

            # Initialize transport to get tool schemas
            try:
                transport = self._get_transport(server_name)
                schemas = transport.get_tool_schemas()
                self.tool_schema_cache[server_name] = schemas
                tool_schemas[server_name] = schemas
            except Exception as e:
                self._log.error(f"Fail to get tool schemes: {e}")
                tool_schemas[server_name] = []

        return tool_schemas

    def validate_tool_call(self, server_name: str, tool_name: str) -> bool:
        """
        Validate if a tool call is valid.

        Args:
            server_name: Name of the MCP server.
            tool_name: Name of the tool.

        Returns:
            True if the tool call is valid, False otherwise.
        """
        try:
            return tool_name in self.tool_cache.get(server_name, [])
        except Exception:
            return False

    def shutdown(self):
        """Clean up all transport connections."""
        for name, transport in self.transports.items():
            if transport:
                transport.stop()
        self.transports = {}
        self.tool_cache = {}
        self.tool_schema_cache = {}


# Global MCP executor instance
_mcp_executor: MCPExecutor | None = None


def get_mcp_executor() -> MCPExecutor:
    """
    Get the global MCP executor instance.

    Returns:
        MCPExecutor instance.
    """
    global _mcp_executor
    if _mcp_executor is None:
        _mcp_executor = MCPExecutor()
    return _mcp_executor


def execute_mcp_tool(server_name: str, tool_name: str, **kwargs) -> dict[str, Any]:
    """
    Execute an MCP tool using the global executor instance.

    This function provides a simple interface for executing MCP tools that
    can be used as a drop-in replacement for the hardcoded implementations.

    Args:
        server_name: Name of the MCP server.
        tool_name: Name of the tool to execute.
        **kwargs: Additional arguments for the tool.

    Returns:
        Dictionary containing execution result.
    """
    executor = get_mcp_executor()
    return executor.execute_tool(server_name, tool_name, **kwargs)


@contextmanager
def mcp_executor_session() -> Generator[MCPExecutor]:
    """
    Context manager for MCP executor lifecycle management.

    This context manager ensures proper cleanup of transport connections
    when the executor is no longer needed, providing graceful shutdown
    of all MCP server connections.

    Yields:
        MCPExecutor: The global MCP executor instance.

    Example:
        with mcp_executor_session() as executor:
            result = executor.execute_tool("server", "tool")
        # Connections are automatically cleaned up on exit
    """
    executor = get_mcp_executor()
    try:
        yield executor
    finally:
        executor.shutdown()

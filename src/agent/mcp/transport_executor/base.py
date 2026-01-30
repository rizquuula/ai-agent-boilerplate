import abc
from typing import Any


class BaseTransport(abc.ABC):
    """Abstract base class for MCP server transports."""

    @abc.abstractmethod
    def start(self, command: str, args: list[str]) -> None:
        """Start the server process/connection.

        Args:
            command: The command to execute
            args: List of arguments for the command
        """
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop the server process/connection."""
        pass

    @abc.abstractmethod
    def execute_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a tool on the MCP server.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool-specific arguments

        Returns:
            Dictionary containing the execution result
        """
        pass

    @abc.abstractmethod
    def list_tools(self) -> list[str]:
        """List available tools from the MCP server.

        Returns:
            List of available tool names
        """
        pass

    @abc.abstractmethod
    def is_alive(self) -> bool:
        """Check if the server connection/process is alive.

        Returns:
            True if alive, False otherwise
        """
        pass

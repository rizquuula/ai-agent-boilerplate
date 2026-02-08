"""Utility functions for the executor module."""


def parse_tool_call(tool_call: str) -> tuple[str, str]:
    """Parse tool call string into server and tool names.

    Args:
        tool_call: Tool call string in format "server:tool".

    Returns:
        Tuple of (server_name, tool_name).

    Raises:
        ValueError: If format is invalid.
    """
    if ":" not in tool_call:
        raise ValueError(f"Invalid tool_call format: {tool_call}. Expected 'server:tool'")

    server_name, tool_name = tool_call.split(":", 1)
    return server_name, tool_name

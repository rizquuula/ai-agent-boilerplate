"""Simple MCP server that returns current local time using FastMCP."""

import os
from datetime import UTC, datetime

from mcp.server.fastmcp import FastMCP

transport = os.environ.get("MCP_TRANSPORT", "stdio")
port = int(os.environ.get("MCP_PORT", "3000"))

# Create the FastMCP instance
mcp = FastMCP(
    "localtime-mcp",
    host="0.0.0.0",
    port=port,
)

@mcp.tool()
def get_current_time() -> dict:
    """Get the current local time with timezone information.

    Returns:
        Dictionary containing ISO datetime string, UNIX timestamp, and timezone.
    """
    now = datetime.now(UTC).astimezone()
    return {
        "iso_datetime": now.isoformat(),
        "unix_timestamp": int(now.timestamp()),
        "timezone": str(now.tzinfo),
    }


def main():
    """Run the MCP server using FastMCP with configurable transport.

    Transport can be configured via environment variables:
    - MCP_TRANSPORT: Transport type (stdio, sse, http). Defaults to "stdio".
    - MCP_PORT: Port for HTTP-based transports. Defaults to 3000.
    """

    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        mcp.run(transport="sse")
    elif transport == "http":
        mcp.run(transport="streamable-http")
    else:
        raise ValueError(f"Unknown transport: {transport}. Use 'stdio', 'sse', or 'http'.")


if __name__ == "__main__":
    main()

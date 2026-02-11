"""Simple MCP server that returns current local time using FastMCP."""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

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
def get_current_time(timezone: str | None = None) -> dict:
    """
    Get the current time for a specific timezone.

    Args:
        timezone: An IANA timezone string (e.g., 'America/Los_Angeles').
                 Defaults to UTC if not provided or invalid.
    """
    try:
        # Default to UTC if no timezone is provided
        tz_name = timezone if timezone else "UTC"
        tz_info = ZoneInfo(tz_name)
    except Exception:
        # Fallback if the user provides a junk string
        tz_info = ZoneInfo("UTC")
        tz_name = "UTC (fallback)"

    now = datetime.now(tz_info)

    return {
        "requested_timezone": tz_name,
        "iso_datetime": now.isoformat(),
        "unix_timestamp": int(now.timestamp()),
        "year": now.year,
        "month": now.month,
        "day": now.day,
        "hour": now.hour,
        "minute": now.minute,
        "weekday": now.strftime("%A"),
        "is_dst": bool(now.dst()),  # Crucial for scheduling!
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

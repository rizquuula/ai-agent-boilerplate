"""Simple MCP server that returns current local time."""

import json
import sys
from datetime import datetime, timezone


def get_current_time():
    """Get the current local time with timezone information."""
    now = datetime.now(timezone.utc).astimezone()
    return {
        "iso_datetime": now.isoformat(),
        "unix_timestamp": int(now.timestamp()),
        "timezone": str(now.tzinfo),
    }


def _list_tools():
    """List available tools."""
    return ["get_current_time"]


def main():
    """Run the MCP server using simple JSON-RPC over stdio."""
    # Map of available methods
    methods = {
        "get_current_time": get_current_time,
        "_list_tools": _list_tools,
    }

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            method_name = request.get("method")
            request_id = request.get("id")
            params = request.get("params", {})

            if method_name in methods:
                result = methods[method_name](**params)
                response = {
                    "jsonrpc": "2.0",
                    "result": result,
                    "id": request_id,
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "error": {"code": -32601, "message": f"Method not found: {method_name}"},
                    "id": request_id,
                }
        except Exception as e:
            response = {
                "jsonrpc": "2.0",
                "error": {"code": -32603, "message": str(e)},
                "id": request.get("id") if "request" in locals() else None,
            }

        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()

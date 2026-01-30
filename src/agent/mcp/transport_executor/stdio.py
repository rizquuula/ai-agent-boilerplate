import json
import subprocess
from typing import Any

from .base import BaseTransport


class StdioTransport(BaseTransport):
    """Transport for MCP servers using stdio communication."""

    def __init__(self):
        self._process = None
        self._request_id = 0

    def start(self, command: str, args: list[str]) -> None:
        """Start the MCP server process."""
        try:
            self._process = subprocess.Popen(
                [command] + args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start MCP server: {str(e)}") from e

    def stop(self) -> None:
        """Stop the MCP server process."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

    def execute_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a tool via JSON-RPC over stdio."""
        if not self.is_alive():
            raise RuntimeError("Server process is not running")

        self._request_id += 1
        request = {"jsonrpc": "2.0", "method": tool_name, "params": kwargs, "id": self._request_id}

        try:
            self._process.stdin.write(json.dumps(request) + "\n")
            self._process.stdin.flush()
            response = self._process.stdout.readline()
            return json.loads(response)
        except Exception as e:
            raise RuntimeError(f"Tool execution failed: {str(e)}") from e

    def list_tools(self) -> list[str]:
        """List available tools using JSON-RPC."""
        result = self.execute_tool("_list_tools")
        return result.get("result", [])

    def is_alive(self) -> bool:
        """Check if server process is running."""
        return self._process is not None and self._process.poll() is None

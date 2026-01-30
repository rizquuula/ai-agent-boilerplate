import ast
import json
import subprocess
from typing import Any

from .base import BaseTransport


class StdioTransport(BaseTransport):
    """Transport for MCP servers using stdio communication."""

    def __init__(self):
        self._process = None
        self._request_id = 0
        self._initialized = False

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
            # Perform MCP initialization handshake
            self._initialize()
        except Exception as e:
            raise RuntimeError(f"Failed to start MCP server: {str(e)}") from e

    def _initialize(self) -> None:
        """Perform MCP initialization handshake."""
        self._request_id += 1
        request = self._build_init_request()

        self._send_json_request(request)
        response = self._read_json_response()
        result = json.loads(response)

        if "error" in result:
            raise RuntimeError(f"MCP initialization failed: {result['error']}")

        self._initialized = True

        # Send initialized notification
        self._send_notification()

    def _build_init_request(self) -> dict[str, Any]:
        """Build the MCP initialize request payload."""
        return {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ai-agent", "version": "0.1.0"},
            },
            "id": self._request_id,
        }

    def _send_json_request(self, request: dict[str, Any]) -> None:
        """Send a JSON-RPC request to the process stdin."""
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()

    def _read_json_response(self) -> str:
        """Read a JSON-RPC response from the process stdout."""
        return self._process.stdout.readline()

    def _send_notification(self) -> None:
        """Send the initialized notification to the server."""
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        self._send_json_request(notification)

    def stop(self) -> None:
        """Stop the MCP server process."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._initialized = False

    def _send_request(self, method: str, params: dict | None = None) -> dict[str, Any]:
        """Send a JSON-RPC request and return the response."""
        if not self.is_alive():
            raise RuntimeError("Server process is not running")

        self._request_id += 1
        request: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "id": self._request_id}
        if params is not None:
            request["params"] = params

        try:
            self._send_json_request(request)
            response = self._read_json_response()
            return json.loads(response)
        except Exception as e:
            raise RuntimeError(f"Request failed: {str(e)}") from e

    def execute_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a tool via MCP protocol over stdio with robust parsing."""
        if not self._initialized:
            raise RuntimeError("MCP server not initialized")

        response = self._send_request(
            "tools/call",
            {"name": tool_name, "arguments": kwargs or {}},
        )

        if "error" in response:
            raise RuntimeError(f"Tool execution failed: {response['error']}")

        result = response.get("result", {})
        contents = result.get("content", [])

        # Handle empty or unexpected content formats
        if not contents:
            return {}

        # Aggregate text from all content blocks
        text = self._extract_text_content(contents)

        return self._parse_tool_output(text)

    def _extract_text_content(self, contents: list[dict[str, Any]]) -> str:
        """Extract and aggregate text from content blocks."""
        text = ""
        for item in contents:
            if item.get("type") == "text":
                text += item.get("text", "")
        return text

    def _parse_tool_output(self, text: str) -> dict[str, Any]:
        """Parse tool output text into a dictionary."""
        try:
            # Try standard JSON first
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                # Fallback to Python literal parsing if the tool sent single quotes
                data = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                raise RuntimeError(f"Could not parse tool output: {text}")

        return data

    def list_tools(self) -> list[str]:
        """List available tools using MCP protocol."""
        if not self._initialized:
            raise RuntimeError("MCP server not initialized")

        response = self._send_request("tools/list")

        if "error" in response:
            raise RuntimeError(f"Failed to list tools: {response['error']}")

        return self._parse_tools_response(response)

    def _parse_tools_response(self, response: dict[str, Any]) -> list[str]:
        """Parse tools list from response."""
        tools = response.get("result", {}).get("tools", [])
        return [tool["name"] for tool in tools]

    def is_alive(self) -> bool:
        """Check if server process is running."""
        return self._process is not None and self._process.poll() is None

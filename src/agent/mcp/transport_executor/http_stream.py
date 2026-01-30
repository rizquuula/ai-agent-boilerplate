import json
from typing import Any

import requests

from .base import BaseTransport


class HTTPStreamTransport(BaseTransport):
    """Transport for MCP servers using HTTP streaming."""

    def __init__(self):
        self._session: requests.Session | None = None
        self._base_url: str | None = None
        self._timeout: int = 30
        self._request_id: int = 0
        self._initialized: bool = False
        self._session_id: str | None = None

    def start(self, command: str, args: list[str]) -> None:
        """Initialize HTTP session with base URL."""
        if not args:
            raise ValueError("HTTP transport requires server URL in args")

        self._base_url = args[0].rstrip("/")
        self._session = requests.Session()

        # Perform MCP initialization handshake
        self._initialize()

    def _initialize(self) -> None:
        """Perform MCP initialization handshake over HTTP Stream."""
        url = f"{self._base_url}/mcp"
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }

        # Send initialize request
        self._request_id += 1
        init_request = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ai-agent", "version": "0.1.0"},
            },
            "id": self._request_id,
        }

        try:
            response = self._session.post(
                url,
                json=init_request,
                headers=headers,
                stream=True,
                timeout=self._timeout,
            )
            if not response.ok:
                raise RuntimeError(f"HTTP initialization failed: {response.status_code}")

            # Extract session ID from response headers
            self._session_id = response.headers.get("mcp-session-id")
            if not self._session_id:
                raise RuntimeError("No session ID received from server")

            # Parse the SSE-formatted response
            result = self._parse_stream_response(response)

        except requests.RequestException as e:
            raise RuntimeError(f"HTTP connection failed: {str(e)}") from e

        if "error" in result:
            raise RuntimeError(f"MCP initialization failed: {result['error']}")

        self._initialized = True

        # Send initialized notification
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        self._send_message(notification)

    def _parse_stream_response(self, response: requests.Response) -> dict[str, Any]:
        """Parse SSE-formatted streaming response."""
        result = {}
        for chunk in response.iter_lines():
            if chunk:
                try:
                    line = chunk.decode("utf-8")
                    # Handle SSE format
                    if line.startswith("event: "):
                        # Event type line, skip
                        continue
                    elif line.startswith("data: "):
                        line = line[6:]  # Remove "data: " prefix
                    # Try to parse JSON
                    data = json.loads(line)
                    if isinstance(data, dict):
                        result = data
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
        return result

    def _send_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC message via HTTP POST with streaming."""
        if not self._session or not self._base_url:
            raise RuntimeError("HTTP transport not connected")

        url = f"{self._base_url}/mcp"
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "mcp-session-id": self._session_id or "",
        }

        try:
            with self._session.post(
                url,
                json=message,
                headers=headers,
                stream=True,
                timeout=self._timeout,
            ) as response:
                if not response.ok:
                    return {"error": f"HTTP error {response.status_code}: {response.text}"}

                return self._parse_stream_response(response)

        except requests.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}

    def stop(self) -> None:
        """Close HTTP session."""
        if self._session:
            self._session.close()
            self._session = None
        self._base_url = None
        self._session_id = None
        self._initialized = False

    def execute_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a tool via MCP protocol over HTTP streaming."""
        if not self.is_alive():
            raise RuntimeError("HTTP transport is not connected")

        if not self._initialized:
            raise RuntimeError("MCP server not initialized")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": kwargs or {}},
            "id": self._request_id,
        }

        response = self._send_message(request)

        if "error" in response:
            return {"success": False, "error": response["error"]}

        result = response.get("result", {})
        contents = result.get("content", [])

        if not contents:
            return {"success": True, "result": {}}

        # Aggregate text from all content blocks
        text = ""
        for item in contents:
            if item.get("type") == "text":
                text += item.get("text", "")

        try:
            parsed_result = json.loads(text) if text else {}
        except json.JSONDecodeError:
            parsed_result = {"text": text}

        return {"success": True, "result": parsed_result}

    def list_tools(self) -> list[str]:
        """List available tools via MCP protocol."""
        if not self.is_alive() or not self._initialized:
            return []

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": self._request_id,
        }

        response = self._send_message(request)

        if "error" in response:
            return []

        result = response.get("result", {})
        tools = result.get("tools", [])
        return [tool["name"] for tool in tools]

    def is_alive(self) -> bool:
        """Check if HTTP session is active."""
        return self._session is not None and self._base_url is not None

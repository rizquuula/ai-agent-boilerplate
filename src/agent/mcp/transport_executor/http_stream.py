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
        headers = self._build_request_headers()

        # Send initialize request
        self._request_id += 1
        init_request = self._build_init_request()

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
            self._session_id = self._extract_session_id(response)
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
        self._send_initialized_notification()

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

    def _build_request_headers(self) -> dict[str, str]:
        """Build HTTP request headers for MCP communication."""
        return {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }

    def _extract_session_id(self, response: requests.Response) -> str | None:
        """Extract session ID from response headers."""
        return response.headers.get("mcp-session-id")

    def _send_initialized_notification(self) -> None:
        """Send the initialized notification to the server."""
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
        headers = self._build_message_headers()

        return self._post_request(url, headers, message)

    def _build_message_headers(self) -> dict[str, str]:
        """Build HTTP headers for sending messages with session ID."""
        return {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "mcp-session-id": self._session_id or "",
        }

    def _post_request(
        self, url: str, headers: dict[str, str], message: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute HTTP POST request and parse response."""
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
        request = self._build_tool_request(tool_name, kwargs)

        response = self._send_message(request)

        if "error" in response:
            return {"success": False, "error": response["error"]}

        return self._parse_tool_result(response)

    def _build_tool_request(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Build the JSON-RPC request for tool execution."""
        return {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments or {}},
            "id": self._request_id,
        }

    def _parse_tool_result(self, response: dict[str, Any]) -> dict[str, Any]:
        """Parse tool execution result from response."""
        result = response.get("result", {})
        contents = result.get("content", [])

        if not contents:
            return {"success": True, "result": {}}

        # Aggregate text from all content blocks
        text = self._extract_text_content(contents)

        try:
            parsed_result = json.loads(text) if text else {}
        except json.JSONDecodeError:
            parsed_result = {"text": text}

        return {"success": True, "result": parsed_result}

    def _extract_text_content(self, contents: list[dict[str, Any]]) -> str:
        """Extract and aggregate text from content blocks."""
        text = ""
        for item in contents:
            if item.get("type") == "text":
                text += item.get("text", "")
        return text

    def list_tools(self) -> list[str]:
        """List available tools via MCP protocol."""
        if not self.is_alive() or not self._initialized:
            return []

        self._request_id += 1
        request = self._build_list_tools_request()

        response = self._send_message(request)

        if "error" in response:
            return []

        return self._parse_tools_response(response)

    def _build_list_tools_request(self) -> dict[str, Any]:
        """Build the JSON-RPC request for listing tools."""
        return {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "id": self._request_id,
        }

    def _parse_tools_response(self, response: dict[str, Any]) -> list[str]:
        """Parse tools list from response."""
        result = response.get("result", {})
        tools = result.get("tools", [])
        return [tool["name"] for tool in tools]

    def is_alive(self) -> bool:
        """Check if HTTP session is active."""
        return self._session is not None and self._base_url is not None

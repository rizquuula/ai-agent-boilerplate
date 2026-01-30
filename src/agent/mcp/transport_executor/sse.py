import json
import queue
import threading
from typing import Any

import requests

from .base import BaseTransport


class SSETransport(BaseTransport):
    """Transport for MCP servers using Server-Sent Events (SSE)."""

    def __init__(self):
        self._session: requests.Session | None = None
        self._base_url: str | None = None
        self._timeout: int = 30
        self._request_id: int = 0
        self._initialized: bool = False
        self._message_endpoint: str | None = None
        self._response_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._sse_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self, command: str, args: list[str]) -> None:
        """Initialize SSE connection to server."""
        if not args:
            raise ValueError("SSE transport requires server URL in args")

        self._base_url = args[0].rstrip("/")
        self._session = requests.Session()

        # Perform MCP initialization handshake
        self._initialize()

    def _initialize(self) -> None:
        """Perform MCP initialization handshake over SSE."""
        # First, connect to SSE endpoint to get the message endpoint
        sse_url = f"{self._base_url}/sse"
        try:
            # Start SSE listener thread first
            self._stop_event.clear()
            self._sse_thread = threading.Thread(target=self._listen_sse, args=(sse_url,))
            self._sse_thread.daemon = True
            self._sse_thread.start()

            # Wait a bit for the connection to establish and get the endpoint
            import time

            time.sleep(0.5)

            # The endpoint should have been extracted by the listener
            if not self._message_endpoint:
                self._extract_message_endpoint_fallback(sse_url)

            if not self._message_endpoint:
                raise RuntimeError("Failed to get message endpoint from SSE")

        except requests.RequestException as e:
            raise RuntimeError(f"SSE connection failed: {str(e)}") from e

        # Send initialize request
        self._request_id += 1
        init_request = self._build_init_request()

        result = self._send_request(init_request)
        self._handle_init_response(result)

    def _extract_message_endpoint_fallback(self, sse_url: str) -> None:
        """Fallback method to extract message endpoint directly."""
        response = self._session.get(sse_url, stream=True, timeout=self._timeout)
        if response.ok:
            for line in response.iter_lines():
                if line:
                    line_str = line.decode("utf-8")
                    if line_str.startswith("data: "):
                        endpoint = line_str[6:]
                        self._message_endpoint = self._build_full_endpoint(endpoint)
                        break
            response.close()

    def _build_full_endpoint(self, endpoint: str) -> str:
        """Build full endpoint URL from relative or absolute path."""
        if endpoint.startswith("http"):
            return endpoint
        return f"{self._base_url}{endpoint}"

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

    def _handle_init_response(self, result: dict[str, Any]) -> None:
        """Handle the initialization response from server."""
        if "error" in result:
            raise RuntimeError(f"MCP initialization failed: {result['error']}")

        self._initialized = True

        # Send initialized notification
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        self._send_message(notification)

    def _listen_sse(self, sse_url: str) -> None:
        """Listen for SSE events and put responses in queue."""
        try:
            with self._session.get(sse_url, stream=True, timeout=self._timeout) as response:
                for line in response.iter_lines():
                    if self._stop_event.is_set():
                        break
                    if line:
                        try:
                            line_str = line.decode("utf-8")
                            if line_str.startswith("data: "):
                                data_str = line_str[6:]
                                self._process_sse_data(data_str)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
        except Exception:
            # Thread will exit on errors
            pass

    def _process_sse_data(self, data_str: str) -> None:
        """Process a single SSE data payload."""
        # Check if this is the endpoint URL
        if data_str.startswith("/"):
            # This is the message endpoint
            self._message_endpoint = f"{self._base_url}{data_str}"
        else:
            # This is a JSON response
            data = json.loads(data_str)
            self._response_queue.put(data)

    def _send_message(self, message: dict[str, Any]) -> None:
        """Send a JSON-RPC message via HTTP POST (notification - no response expected)."""
        if not self._session or not self._message_endpoint:
            raise RuntimeError("SSE transport not connected")

        try:
            self._session.post(
                self._message_endpoint,
                json=message,
                timeout=self._timeout,
            )
        except requests.RequestException:
            pass  # Notifications don't wait for response

    def _send_request(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for response."""
        if not self._session or not self._message_endpoint:
            raise RuntimeError("SSE transport not connected")

        request_id = message.get("id")

        try:
            # Send the request
            response = self._session.post(
                self._message_endpoint,
                json=message,
                timeout=self._timeout,
            )
            if not response.ok:
                return {"error": f"HTTP error {response.status_code}: {response.text}"}

            # Wait for response with matching ID
            return self._wait_for_response(request_id)

        except requests.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}

    def _wait_for_response(self, request_id: Any) -> dict[str, Any]:
        """Wait for a response with the matching request ID."""
        timeout_count = 0
        max_timeout = self._timeout * 10  # 0.1s * 300 = 30 seconds
        while timeout_count < max_timeout:
            try:
                result = self._response_queue.get(timeout=0.1)
                if result.get("id") == request_id:
                    return result
                # Not our response, put it back? No, just continue waiting
            except queue.Empty:
                timeout_count += 1
                continue

        return {"error": "Timeout waiting for response"}

    def stop(self) -> None:
        """Close SSE connection."""
        self._stop_event.set()
        if self._sse_thread and self._sse_thread.is_alive():
            self._sse_thread.join(timeout=2)
        if self._session:
            self._session.close()
            self._session = None
        self._base_url = None
        self._message_endpoint = None
        self._initialized = False

    def execute_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a tool via MCP protocol over SSE."""
        if not self.is_alive():
            raise RuntimeError("SSE transport is not connected")

        if not self._initialized:
            raise RuntimeError("MCP server not initialized")

        self._request_id += 1
        request = self._build_tool_request(tool_name, kwargs)

        response = self._send_request(request)

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

        response = self._send_request(request)

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
        """Check if SSE connection is active."""
        return self._session is not None and self._base_url is not None

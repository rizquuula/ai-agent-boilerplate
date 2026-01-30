import json
from collections.abc import Generator
from typing import Any

import requests

from .base import BaseTransport


class SSETransport(BaseTransport):
    """Transport for MCP servers using Server-Sent Events (SSE)."""

    def __init__(self):
        self._session: requests.Session | None = None
        self._base_url: str | None = None
        self._timeout: int = 30
        self._event_stream: Generator | None = None

    def start(self, command: str, args: list[str]) -> None:
        """Initialize SSE connection to server."""
        if not args:
            raise ValueError("SSE transport requires server URL in args")

        self._base_url = args[0]
        self._session = requests.Session()

        # Verify connection
        try:
            response = self._session.get(f"{self._base_url}/health", timeout=self._timeout)
            if not response.ok:
                raise RuntimeError(f"Server health check failed: {response.text}")
        except requests.RequestException as e:
            raise RuntimeError(f"SSE connection failed: {str(e)}") from e

        # Open event stream
        self._event_stream = self._create_event_stream()

    def stop(self) -> None:
        """Close SSE connection."""
        if self._event_stream:
            try:
                self._event_stream.close()
            except Exception:
                pass
            self._event_stream = None

        if self._session:
            self._session.close()
            self._session = None
        self._base_url = None

    def _create_event_stream(self) -> Generator[dict[str, Any]]:
        """Create a generator for SSE events."""
        url = f"{self._base_url}/events"
        response = self._session.get(url, stream=True, timeout=self._timeout)

        for line in response.iter_lines():
            if line:
                yield json.loads(line.decode("utf-8"))

    def execute_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a tool via SSE connection."""
        if not self.is_alive():
            raise RuntimeError("SSE transport is not connected")

        url = f"{self._base_url}/tools/{tool_name}"

        try:
            response = self._session.post(url, json=kwargs, timeout=self._timeout)

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP error {response.status_code}: {response.text}",
                }

            # Wait for result through event stream
            for event in self._event_stream:
                if event.get("tool") == tool_name:
                    return {"success": True, "result": event.get("result")}

            return {"success": False, "error": "No response received from server"}

        except requests.RequestException as e:
            return {"success": False, "error": f"SSE request failed: {str(e)}"}

    def list_tools(self) -> list[str]:
        """List available tools via SSE endpoint."""
        try:
            response = self.execute_tool("_list_tools")
            if response["success"]:
                return response["result"].get("tools", [])
            return []
        except Exception:
            return []

    def is_alive(self) -> bool:
        """Check if SSE connection is active."""
        return self._session is not None and self._base_url is not None

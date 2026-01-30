"""Tool parsing and formatting utilities."""

from typing import Any

from agent.mcp.config import get_mcp_config
from agent.state import ExecutionRecord


def parse_tool_id(tool_id: str) -> tuple[str, str]:
    """Parse a tool identifier into server and tool names.

    Args:
        tool_id: Tool identifier in format "server:tool"

    Returns:
        Tuple of (server_name, tool_name)

    Raises:
        ValueError: If tool_id format is invalid
    """
    if ":" not in tool_id:
        raise ValueError(f"Invalid tool ID format: {tool_id}. Expected 'server:tool'")

    parts = tool_id.split(":", 1)
    return parts[0], parts[1]


def format_available_tools() -> str:
    """Get formatted description of available MCP tools.

    Returns:
        Multi-line string describing available tools by server
    """
    try:
        config = get_mcp_config()
        servers = config.get_enabled_servers()

        if not servers:
            return "No MCP servers enabled."

        lines = []
        for server_name in servers:
            lines.append(f"\n{server_name}:")
            server_config = config.get_server_config(server_name)
            if server_config and "tools" in server_config:
                for tool in server_config["tools"]:
                    lines.append(f"  - {tool}")
            else:
                lines.append("  - Tools loaded dynamically at runtime")

        return "\n".join(lines)

    except Exception as e:
        return f"Error loading tool configuration: {e}"


def format_execution_context(context: dict[str, Any]) -> str:
    """Format execution context for LLM prompts.

    Args:
        context: Dictionary of accumulated context

    Returns:
        Formatted string representation
    """
    if not context:
        return "No context accumulated yet."

    lines = []
    for key, value in context.items():
        if isinstance(value, list):
            lines.append(f"- {key}: {len(value)} items")
            for item in value[:5]:  # Show first 5 items
                lines.append(f"  - {item}")
            if len(value) > 5:
                lines.append(f"  ... and {len(value) - 5} more")
        elif isinstance(value, dict):
            lines.append(f"- {key}:")
            for k, v in value.items():
                lines.append(f"  - {k}: {v}")
        else:
            lines.append(f"- {key}: {value}")

    return "\n".join(lines)


def format_history_for_milestone(history: list[ExecutionRecord], milestone_id: str) -> list[ExecutionRecord]:
    """Filter history to get records for a specific milestone.

    Args:
        history: Full execution history
        milestone_id: Milestone ID to filter by

    Returns:
        List of execution records for the milestone
    """
    return [h for h in history if h.milestone_id == milestone_id]


def format_milestone_history(records: list[ExecutionRecord]) -> str:
    """Format execution records for display in prompts.

    Args:
        records: List of execution records

    Returns:
        Formatted string describing tool executions
    """
    if not records:
        return "No tools executed yet for this milestone."

    lines = []
    for i, record in enumerate(records, 1):
        status = "SUCCESS" if record.success else "FAILED"
        lines.append(f"{i}. [{status}] {record.tool_call.tool_id}")
        if record.tool_call.parameters:
            lines.append(f"   Parameters: {record.tool_call.parameters}")

    return "\n".join(lines)


def format_execution_results(records: list[ExecutionRecord]) -> str:
    """Format execution results for validation prompts.

    Args:
        records: List of execution records

    Returns:
        Formatted string with results
    """
    if not records:
        return "No results available."

    lines = []
    for record in records:
        if record.success:
            result_summary = str(record.result.get("result", "No result"))[:200]
            lines.append(f"- {record.tool_call.tool_id}: {result_summary}")
        else:
            error = record.result.get("error", "Unknown error")
            lines.append(f"- {record.tool_call.tool_id}: ERROR - {error}")

    return "\n".join(lines)


def format_errors(errors: list[dict[str, Any]]) -> str:
    """Format error records for retry prompts.

    Args:
        errors: List of error records

    Returns:
        Formatted string describing errors
    """
    if not errors:
        return "No previous errors."

    lines = ["Previous attempts failed:"]
    for i, error in enumerate(errors[-3:], 1):  # Show last 3 errors
        lines.append(f"{i}. {error.get('error_message', 'Unknown error')}")
        if error.get("suggested_fix"):
            lines.append(f"   Suggested fix: {error['suggested_fix']}")

    return "\n".join(lines)


def has_execution_failures(records: list[ExecutionRecord]) -> bool:
    """Check if any execution records show failures.

    Args:
        records: List of execution records

    Returns:
        True if any record failed
    """
    return any(not record.success for record in records)


def get_milestone_history(history: list[ExecutionRecord], milestone_id: str) -> list[ExecutionRecord]:
    """Get all execution records for a specific milestone.

    Args:
        history: Full execution history
        milestone_id: Milestone ID to filter by

    Returns:
        List of execution records for the milestone
    """
    return [h for h in history if h.milestone_id == milestone_id]

"""Utility functions for the planner node."""

from typing import Any


def generate_task_id(index: int, description: str) -> str:
    """Generate a unique task ID."""
    return f"task_{index}_{description.lower().replace(' ', '_')[:30]}"


def format_tools_context(tool_schemas: dict[str, list[dict[str, Any]]]) -> str:
    """Format tool schemas for inclusion in LLM prompt."""
    if not tool_schemas:
        return "No MCP tools available."

    lines = []
    for server_name, tools in tool_schemas.items():
        if not tools:
            continue

        lines.append(f"\n## Server: {server_name}")
        for tool in tools:
            name = tool.get("name", "unknown")
            description = tool.get("description", "No description")
            input_schema = tool.get("inputSchema", {})

            lines.append(f"\n### {server_name}:{name}")
            lines.append(f"Description: {description}")

            if input_schema:
                properties = input_schema.get("properties", {})
                required = input_schema.get("required", [])

                if properties:
                    lines.append("Parameters:")
                    for param_name, param_info in properties.items():
                        param_type = param_info.get("type", "any")
                        param_desc = param_info.get("description", "")
                        is_required = " (required)" if param_name in required else ""
                        lines.append(f"  - {param_name} ({param_type}){is_required}: {param_desc}")

    return "\n".join(lines)

"""Planner node for creating and updating task plans."""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from asterism.agent.models import LLMUsage, Plan
from asterism.agent.state import AgentState
from asterism.llm.base import BaseLLMProvider
from asterism.mcp.executor import MCPExecutor


def _generate_task_id(index: int, description: str) -> str:
    """Generate a unique task ID."""
    return f"task_{index}_{description.lower().replace(' ', '_')[:30]}"


def _format_tools_context(tool_schemas: dict[str, list[dict[str, Any]]]) -> str:
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


# Node-specific system prompt - this is combined with SOUL.md + AGENT.md
PLANNER_SYSTEM_PROMPT = """You are a task planning agent. Create a detailed plan to accomplish the user's request.

You have access to MCP tools. When planning tasks, specify tool calls in format:
- tool_call: "server_name:tool_name"
- tool_input: dictionary of parameters for the tool

You can also include LLM reasoning tasks (no tool_call) for analysis or synthesis.

Return a JSON with:
{
  "reasoning": "explanation of your approach",
  "tasks": [
    {
      "id": "unique_task_id",
      "description": "what this task does",
      "tool_call": "server:tool" or null,
      "tool_input": {} or null,
      "depends_on": ["task_id_1", ...]  // tasks that must complete first
    }
  ]
}

Guidelines:
- Order tasks logically, respecting dependencies
- Break complex tasks into smaller steps
- Use available MCP tools when appropriate
- Include verification tasks if needed
"""


def planner_node(llm: BaseLLMProvider, mcp_executor: MCPExecutor, state: AgentState) -> AgentState:
    """
    Create or update a plan based on the user request and execution history.

    The LLM will receive:
    1. SOUL.md + AGENT.md as a SystemMessage (loaded fresh from disk if configured)
    2. Node-specific planning instructions as a SystemMessage (including available tools)
    3. User request and execution context as a HumanMessage

    Args:
        llm: The LLM provider to use for planning.
        mcp_executor: The MCP executor for discovering available tools.
        state: Current agent state.

    Returns:
        Updated state with a new or updated plan.
    """
    # Get available MCP tools
    try:
        tool_schemas = mcp_executor.get_tool_schemas()
        tools_context = _format_tools_context(tool_schemas)
    except Exception as e:
        tools_context = f"Error getting tool information: {str(e)}"

    # Build context from messages
    user_message = ""
    if state["messages"]:
        # Get the latest user message
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

    execution_context = ""
    if state["execution_results"]:
        execution_context = "\n\nExecution History:\n"
        for result in state["execution_results"]:
            status = "✓" if result.success else "✗"
            execution_context += f"- {status} {result.task_id}: {result.result if result.success else result.error}\n"

    # Enhanced system prompt with tool information
    enhanced_system_prompt = f"""{PLANNER_SYSTEM_PROMPT}

Available MCP Tools:
{tools_context}

When creating tasks with tool calls:
- Use ONLY tools listed above
- Use exact format: "server_name:tool_name"
- Provide all required parameters in tool_input
- If a tool is not available, use LLM reasoning instead
"""

    user_prompt = f"""User Request: {user_message}

{execution_context if execution_context else ""}

Create a plan to accomplish this request using the available tools."""

    try:
        # Use structured output with message list
        # The provider will auto-prepend SOUL.md + AGENT.md as a SystemMessage
        messages = [
            SystemMessage(content=enhanced_system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke_structured(messages, Plan)
        plan = response.parsed

        # Ensure all tasks have IDs
        for i, task in enumerate(plan.tasks):
            if not task.id:
                task.id = _generate_task_id(i, task.description)

        # Update state
        new_state = state.copy()
        new_state["plan"] = plan
        new_state["current_task_index"] = 0
        new_state["error"] = None

        # Track LLM usage from structured output response
        usage = LLMUsage(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            model=llm.model,
            node_name="planner_node",
        )
        new_state["llm_usage"] = state.get("llm_usage", []) + [usage]

        return new_state

    except Exception as e:
        new_state = state.copy()
        new_state["error"] = f"Planning failed: {str(e)}"
        return new_state

"""Executor node: MCP tool execution loop with proper parameter handling."""

from typing import TYPE_CHECKING

from agent.state import AgentState, ExecutionRecord
from agent.utils.tool_parser import parse_tool_id

if TYPE_CHECKING:
    from agent.mcp.executor import MCPExecutor


def node(state: AgentState, mcp_executor: "MCPExecutor") -> AgentState:
    """Execute MCP tools from the tactical plan.

    Args:
        state: Current agent state with tactical_plan
        mcp_executor: MCP executor for tool execution

    Returns:
        Updated state with execution history
    """
    tactical_plan = state.get("tactical_plan", [])
    history = state.get("history", [])
    milestones = state.get("milestones", [])
    current_idx = state.get("current_idx", 0)

    # Validate we have an active milestone
    if not milestones or current_idx >= len(milestones):
        return {**state, "history": history}

    current_milestone = milestones[current_idx]

    # Execute each tool in the tactical plan
    for tool_call in tactical_plan:
        try:
            # Parse tool identifier
            server_name, tool_name = parse_tool_id(tool_call.tool_id)

            # Execute the tool with proper parameter handling
            result = mcp_executor.execute_tool(server_name=server_name, tool_name=tool_name, **tool_call.parameters)

            # Create execution record
            execution_record = ExecutionRecord(
                milestone_id=current_milestone.id,
                tool_call=tool_call,
                result=result,
                success=result.get("success", False),
            )

            history.append(execution_record)

        except Exception as e:
            # Record the failure
            error_result = {
                "success": False,
                "error": str(e),
                "tool": tool_call.tool_id,
            }

            execution_record = ExecutionRecord(
                milestone_id=current_milestone.id,
                tool_call=tool_call,
                result=error_result,
                success=False,
            )

            history.append(execution_record)

    return {**state, "history": history}


def update_execution_context(state: AgentState, tool_call, result: dict) -> dict:
    """Update execution context based on successful tool execution.

    Args:
        state: Current agent state
        tool_call: The tool call that was executed
        result: Execution result

    Returns:
        Updated execution context
    """
    context = state.get("execution_context", {}).copy()

    # Extract useful information from result
    if result.get("success"):
        tool_id = tool_call.tool_id
        result_data = result.get("result")

        # Store results by tool type
        if "list_files" in tool_id and isinstance(result_data, list):
            if "discovered_files" not in context:
                context["discovered_files"] = []
            context["discovered_files"].extend(result_data)

        elif "read_file" in tool_id and isinstance(result_data, str):
            if "file_contents" not in context:
                context["file_contents"] = {}
            # Extract filename from parameters
            filename = tool_call.parameters.get("path", "unknown")
            context["file_contents"][filename] = result_data[:500]  # Store preview

        elif "analyze" in tool_id and isinstance(result_data, dict):
            if "analysis_results" not in context:
                context["analysis_results"] = []
            context["analysis_results"].append(result_data)

    return context

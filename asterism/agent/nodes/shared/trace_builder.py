"""Execution trace building utilities."""

from typing import Any

from asterism.agent.state import AgentState


def build_execution_trace(state: AgentState) -> list[dict[str, Any]]:
    """Build execution trace from state.

    Creates a serializable trace of all task executions for the final response.

    Args:
        state: Current agent state.

    Returns:
        List of trace entry dictionaries.
    """
    execution_results = state.get("execution_results", [])
    trace = []

    for result in execution_results:
        trace_entry = {
            "task_id": result.task_id,
            "success": result.success,
            "result": result.result,
            "error": result.error,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None,
        }
        trace.append(trace_entry)

    return trace


def format_trace_for_display(trace: list[dict[str, Any]]) -> str:
    """Format trace for human-readable display.

    Args:
        trace: Execution trace from build_execution_trace.

    Returns:
        Formatted string representation.
    """
    if not trace:
        return "No execution trace available."

    lines = ["=== Execution Trace ==="]
    for entry in trace:
        status = "✓" if entry["success"] else "✗"
        lines.append(f"\n{status} Task: {entry['task_id']}")
        if entry["success"]:
            result_str = str(entry["result"])[:200] if entry["result"] else "No result"
            lines.append(f"  Result: {result_str}")
        else:
            lines.append(f"  Error: {entry.get('error', 'Unknown error')}")

    return "\n".join(lines)


def get_trace_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Get summary statistics from execution trace.

    Args:
        trace: Execution trace from build_execution_trace.

    Returns:
        Dictionary with summary statistics.
    """
    if not trace:
        return {"total": 0, "successful": 0, "failed": 0}

    successful = sum(1 for entry in trace if entry["success"])
    failed = len(trace) - successful

    return {
        "total": len(trace),
        "successful": successful,
        "failed": failed,
    }

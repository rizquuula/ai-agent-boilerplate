"""Streaming graph builder - stops before finalization."""

from typing import TYPE_CHECKING

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from asterism.agent.graph_builders.base import (
    add_common_edges,
    add_common_nodes,
    make_routing_function_with_end,
)
from asterism.agent.state import AgentState

if TYPE_CHECKING:
    from asterism.agent.agent import Agent


def build_streaming_graph(
    agent: "Agent",
    checkpointer: BaseCheckpointSaver | None = None,
) -> StateGraph:
    """Build the streaming agent graph (stops before finalizer).

    This graph includes: planner → executor → evaluator → END
    When evaluator decides to finalize, it routes to END instead of finalizer.
    Use this for astream() where you want to handle finalization manually.

    Args:
        agent: The Agent instance with dependencies.
        checkpointer: Optional checkpointer for state persistence.

    Returns:
        Compiled StateGraph ready for execution.
    """
    workflow = StateGraph(AgentState)

    # Add common nodes (planner, executor, evaluator) - NO finalizer
    add_common_nodes(workflow, agent)

    # Add common edges (START → planner → executor → evaluator)
    add_common_edges(workflow)

    # Add conditional edges from evaluator
    # Routes: planner_node | executor_node | END (when would go to finalizer)
    workflow.add_conditional_edges(
        "evaluator_node",
        make_routing_function_with_end(agent),
        {
            "planner_node": "planner_node",
            "executor_node": "executor_node",
            "finalizer_node": END,  # Stop here instead of going to finalizer
        },
    )

    # Compile with or without checkpointing
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()

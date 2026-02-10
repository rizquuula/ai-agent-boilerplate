"""Full graph builder - includes all nodes including finalizer."""

from typing import TYPE_CHECKING

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from asterism.agent.graph_builders.base import (
    _make_finalizer_node,
    add_common_edges,
    add_common_nodes,
    make_routing_function,
)
from asterism.agent.state import AgentState

if TYPE_CHECKING:
    from asterism.agent.agent import Agent


def build_full_graph(
    agent: "Agent",
    checkpointer: BaseCheckpointSaver | None = None,
) -> StateGraph:
    """Build the complete agent graph with all nodes.

    This graph includes: planner → executor → evaluator → finalizer → END
    Use this for standard invoke() operations.

    Args:
        agent: The Agent instance with dependencies.
        checkpointer: Optional checkpointer for state persistence.

    Returns:
        Compiled StateGraph ready for execution.
    """
    workflow = StateGraph(AgentState)

    # Add common nodes (planner, executor, evaluator)
    add_common_nodes(workflow, agent)

    # Add finalizer node
    workflow.add_node("finalizer_node", _make_finalizer_node(agent))

    # Add common edges (START → planner → executor → evaluator)
    add_common_edges(workflow)

    # Add conditional edges from evaluator
    # Routes: planner_node | executor_node | finalizer_node
    workflow.add_conditional_edges(
        "evaluator_node",
        make_routing_function(agent),
        {
            "planner_node": "planner_node",
            "executor_node": "executor_node",
            "finalizer_node": "finalizer_node",
        },
    )

    # Finalizer → END
    workflow.add_edge("finalizer_node", END)

    # Compile with or without checkpointing
    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()

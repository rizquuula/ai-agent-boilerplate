"""Utility functions for the finalizer node."""

from langchain_core.messages import HumanMessage

from asterism.agent.state import AgentState


def get_user_request(state: AgentState) -> str:
    """Extract the original user request from state messages."""
    messages = state.get("messages", [])
    if messages:
        # Find the first human message
        for msg in messages:
            if isinstance(msg, HumanMessage):
                return msg.content
    return "No user request found"

"""Integration test for the complete agent workflow."""

import os

import pytest

from asterism.agent.agent import Agent
from asterism.llm.openai_provider import OpenAIProvider
from asterism.mcp.executor import MCPExecutor


def test_agent_invoke():
    """Test agent invocation with real LLM."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    llm = OpenAIProvider(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        prompt_loader=None,
    )

    mcp_executor = MCPExecutor(
        config_path=os.getenv("MCP_CONFIG_PATH"),
    )

    agent = Agent(
        llm=llm,
        mcp_executor=mcp_executor,
        db_path=":memory:",
    )

    response = agent.invoke(
        session_id="test-session",
        user_message="Sekarang jam berapa ya?",
    )

    agent.close()

    assert response["session_id"] == "test-session"
    assert "message" in response
    assert response.get("error") is None

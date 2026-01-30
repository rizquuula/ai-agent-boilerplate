"""Unit tests for the router node."""

from unittest.mock import patch

from agent.nodes import router


class TestRouterNode:
    """Tests for the router node."""

    def test_router_loads_skill_from_skill_md(self, sample_agent_state, tmp_path):
        """Should load skill config from SKILL.md file."""
        # Arrange
        with patch("agent.nodes.router.load_skill_config") as mock_load:
            mock_load.return_value = type(
                "SkillConfig",
                (),
                {
                    "name": "filesystem",
                    "description": "File operations",
                    "content": "Skill guidance content",
                    "metadata": {},
                },
            )()

            # Act
            result = router.node(sample_agent_state)

            # Assert
            assert "active_skill_context" in result
            assert result["active_skill_context"] is not None
            mock_load.assert_called_once_with("filesystem")

    def test_router_normalizes_skill_names(self, sample_agent_state):
        """Should handle code-reader vs code_reader naming."""
        # Arrange
        sample_agent_state["milestones"][0].assigned_skill = "code_reader"

        with patch("agent.nodes.router.load_skill_config") as mock_load:
            mock_load.return_value = type(
                "SkillConfig",
                (),
                {
                    "name": "code-reader",
                    "description": "Code reading skill",
                    "content": "Guidance",
                    "metadata": {},
                },
            )()

            # Act
            result = router.node(sample_agent_state)

            # Assert
            assert "code-reader" in result["active_skill_context"]

    def test_router_includes_execution_context(self, sample_agent_state):
        """Should include previous execution results in context."""
        # Arrange
        sample_agent_state["execution_context"] = {
            "discovered_files": ["main.py", "utils.py"],
        }

        with patch("agent.nodes.router.load_skill_config") as mock_load:
            mock_load.return_value = type(
                "SkillConfig",
                (),
                {
                    "name": "filesystem",
                    "description": "File operations",
                    "content": "Guidance",
                    "metadata": {},
                },
            )()

            # Act
            result = router.node(sample_agent_state)

            # Assert
            assert "discovered_files" in result["active_skill_context"]
            assert "main.py" in result["active_skill_context"]

    def test_router_handles_missing_skill(self, sample_agent_state):
        """Should handle missing skill files gracefully."""
        # Arrange
        sample_agent_state["milestones"][0].assigned_skill = "nonexistent"

        # Act
        result = router.node(sample_agent_state)

        # Assert
        assert "active_skill_context" in result
        assert "nonexistent" in result["active_skill_context"]
        assert "not found" in result["active_skill_context"].lower()

    def test_router_resets_retry_count(self, sample_agent_state):
        """Should reset retry count for new milestone."""
        # Arrange
        sample_agent_state["retry_count"] = 2

        with patch("agent.nodes.router.load_skill_config") as mock_load:
            mock_load.return_value = type(
                "SkillConfig",
                (),
                {
                    "name": "filesystem",
                    "description": "File operations",
                    "content": "Guidance",
                    "metadata": {},
                },
            )()

            # Act
            result = router.node(sample_agent_state)

            # Assert
            assert result["retry_count"] == 0

    def test_router_no_milestones(self):
        """Should handle state with no milestones."""
        # Arrange
        state = {
            "objective": "Test",
            "milestones": [],
            "current_idx": 0,
        }

        # Act
        result = router.node(state)

        # Assert
        assert "No active milestone" in result["active_skill_context"]

    def test_router_includes_completed_milestones(self, sample_agent_state, completed_milestone):
        """Should include completed milestones in context."""
        # Arrange
        sample_agent_state["milestones"] = [completed_milestone, sample_agent_state["milestones"][0]]
        sample_agent_state["current_idx"] = 1

        with patch("agent.nodes.router.load_skill_config") as mock_load:
            mock_load.return_value = type(
                "SkillConfig",
                (),
                {
                    "name": "filesystem",
                    "description": "File operations",
                    "content": "Guidance",
                    "metadata": {},
                },
            )()

            # Act
            result = router.node(sample_agent_state)

            # Assert
            assert "COMPLETED MILESTONES" in result["active_skill_context"]
            assert "Find configuration files" in result["active_skill_context"]

    def test_router_includes_milestone_details(self, sample_agent_state):
        """Should include current milestone description and criteria."""
        # Arrange
        with patch("agent.nodes.router.load_skill_config") as mock_load:
            mock_load.return_value = type(
                "SkillConfig",
                (),
                {
                    "name": "filesystem",
                    "description": "File operations",
                    "content": "Guidance",
                    "metadata": {},
                },
            )()

            # Act
            result = router.node(sample_agent_state)

            # Assert
            assert "List all Python files" in result["active_skill_context"]
            assert "Success Criteria:" in result["active_skill_context"]

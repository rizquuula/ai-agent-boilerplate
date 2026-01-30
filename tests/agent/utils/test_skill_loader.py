"""Unit tests for the skill_loader utility module."""

from unittest.mock import mock_open, patch

import pytest

from agent.utils.skill_loader import (
    format_completed_milestones,
    get_available_skills,
    load_skill_config,
    normalize_skill_name,
)


class TestNormalizeSkillName:
    """Tests for normalize_skill_name function."""

    def test_converts_snake_case(self):
        """Should convert snake_case to kebab-case."""
        assert normalize_skill_name("code_reader") == "code-reader"
        assert normalize_skill_name("report_writer") == "report-writer"

    def test_converts_camel_case(self):
        """Should convert camelCase to kebab-case."""
        assert normalize_skill_name("CodeReader") == "code-reader"
        assert normalize_skill_name("ReportWriter") == "report-writer"

    def test_preserves_kebab_case(self):
        """Should preserve existing kebab-case."""
        assert normalize_skill_name("code-reader") == "code-reader"

    def test_handles_spaces(self):
        """Should convert spaces to hyphens."""
        assert normalize_skill_name("code reader") == "code-reader"


class TestLoadSkillConfig:
    """Tests for load_skill_config function."""

    def test_loads_skill_with_frontmatter(self, tmp_path):
        """Should load skill with YAML frontmatter."""
        # Arrange
        skill_content = """---
name: filesystem
description: File operations skill
---

# Filesystem

This skill handles file operations.
"""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=skill_content)):
                # Act
                config = load_skill_config("filesystem")

        # Assert
        assert config.name == "filesystem"
        assert config.description == "File operations skill"
        assert "file operations" in config.content.lower()

    def test_loads_skill_without_frontmatter(self, tmp_path):
        """Should load skill without YAML frontmatter."""
        # Arrange
        skill_content = "# Simple Skill\n\nContent here."

        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=skill_content)):
                # Act
                config = load_skill_config("test")

        # Assert
        assert config.name == "test"
        assert config.content == skill_content.strip()

    def test_raises_on_missing_skill(self):
        """Should raise FileNotFoundError for missing skill."""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                load_skill_config("nonexistent")


class TestGetAvailableSkills:
    """Tests for get_available_skills function."""

    def test_returns_skill_directories(self, tmp_path):
        """Should return list of skill directory names."""
        # This test is difficult without actual filesystem
        # Just verify the function doesn't crash
        skills = get_available_skills()
        assert isinstance(skills, list)


class TestFormatCompletedMilestones:
    """Tests for format_completed_milestones function."""

    def test_formats_completed_milestones(self):
        """Should format completed milestones with status."""
        from agent.state import SubGoal

        milestones = [
            SubGoal(id="1", description="Task 1", assigned_skill="test", status="completed"),
            SubGoal(id="2", description="Task 2", assigned_skill="test", status="failed"),
            SubGoal(id="3", description="Task 3", assigned_skill="test", status="pending"),
        ]

        result = format_completed_milestones(milestones, 2)

        assert "Task 1" in result
        assert "Task 2" in result
        assert "Task 3" not in result  # Not completed yet
        assert "PASS" in result
        assert "FAIL" in result

    def test_handles_no_completed_milestones(self):
        """Should handle empty completed list."""
        from agent.state import SubGoal

        milestones = [
            SubGoal(id="1", description="Task 1", assigned_skill="test", status="pending"),
        ]

        result = format_completed_milestones(milestones, 0)

        assert "No milestones completed" in result

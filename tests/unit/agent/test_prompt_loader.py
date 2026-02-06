"""Unit tests for SystemPromptLoader."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from asterism.agent.prompt_loader import SystemPromptLoader


def test_load_default_paths():
    """Test loading with default paths."""
    loader = SystemPromptLoader()
    assert loader.soul_path == "workspace/SOUL.md"
    assert loader.agent_path == "workspace/AGENT.md"


def test_load_custom_paths():
    """Test loading with custom paths."""
    loader = SystemPromptLoader(soul_path="/custom/soul.md", agent_path="/custom/agent.md")
    assert loader.soul_path == "/custom/soul.md"
    assert loader.agent_path == "/custom/agent.md"


def test_with_paths_returns_new_instance():
    """Test that with_paths returns a new instance."""
    loader = SystemPromptLoader()
    new_loader = loader.with_paths("/new/soul.md", "/new/agent.md")

    assert new_loader is not loader
    assert new_loader.soul_path == "/new/soul.md"
    assert new_loader.agent_path == "/new/agent.md"


def test_load_combines_soul_and_agent():
    """Test that load() combines SOUL and AGENT content."""
    with TemporaryDirectory() as tmpdir:
        soul_path = Path(tmpdir) / "SOUL.md"
        agent_path = Path(tmpdir) / "AGENT.md"

        soul_path.write_text("# Soul\nCore values here.")
        agent_path.write_text("# Agent\nCapabilities here.")

        loader = SystemPromptLoader(soul_path=str(soul_path), agent_path=str(agent_path))

        result = loader.load()

        assert "# SOUL (Core Values & Philosophy)" in result
        assert "# AGENT (Identity & Capabilities)" in result
        assert "Core values here." in result
        assert "Capabilities here." in result


def test_load_separate():
    """Test load_separate returns tuple of contents."""
    with TemporaryDirectory() as tmpdir:
        soul_path = Path(tmpdir) / "SOUL.md"
        agent_path = Path(tmpdir) / "AGENT.md"

        soul_content = "# Soul\nCore values here."
        agent_content = "# Agent\nCapabilities here."

        soul_path.write_text(soul_content)
        agent_path.write_text(agent_content)

        loader = SystemPromptLoader(soul_path=str(soul_path), agent_path=str(agent_path))

        soul, agent = loader.load_separate()

        assert soul == soul_content
        assert agent == agent_content


def test_load_missing_soul_raises_error():
    """Test that missing SOUL.md raises FileNotFoundError."""
    with TemporaryDirectory() as tmpdir:
        agent_path = Path(tmpdir) / "AGENT.md"
        agent_path.write_text("# Agent\nContent")

        loader = SystemPromptLoader(soul_path=str(Path(tmpdir) / "MISSING_SOUL.md"), agent_path=str(agent_path))

        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load()

        assert "MISSING_SOUL.md" in str(exc_info.value)


def test_load_missing_agent_raises_error():
    """Test that missing AGENT.md raises FileNotFoundError."""
    with TemporaryDirectory() as tmpdir:
        soul_path = Path(tmpdir) / "SOUL.md"
        soul_path.write_text("# Soul\nContent")

        loader = SystemPromptLoader(soul_path=str(soul_path), agent_path=str(Path(tmpdir) / "MISSING_AGENT.md"))

        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load()

        assert "MISSING_AGENT.md" in str(exc_info.value)


def test_validate_files_exist_returns_true():
    """Test validate_files_exist returns True when files exist."""
    with TemporaryDirectory() as tmpdir:
        soul_path = Path(tmpdir) / "SOUL.md"
        agent_path = Path(tmpdir) / "AGENT.md"

        soul_path.write_text("# Soul\nContent")
        agent_path.write_text("# Agent\nContent")

        loader = SystemPromptLoader(soul_path=str(soul_path), agent_path=str(agent_path))

        assert loader.validate_files_exist() is True


def test_validate_files_exist_returns_false():
    """Test validate_files_exist returns False when files missing."""
    with TemporaryDirectory() as tmpdir:
        loader = SystemPromptLoader(soul_path=str(Path(tmpdir) / "SOUL.md"), agent_path=str(Path(tmpdir) / "AGENT.md"))

        assert loader.validate_files_exist() is False


def test_load_with_empty_files():
    """Test loading with empty files still works."""
    with TemporaryDirectory() as tmpdir:
        soul_path = Path(tmpdir) / "SOUL.md"
        agent_path = Path(tmpdir) / "AGENT.md"

        soul_path.write_text("")
        agent_path.write_text("")

        loader = SystemPromptLoader(soul_path=str(soul_path), agent_path=str(agent_path))

        result = loader.load()

        assert "# SOUL (Core Values & Philosophy)" in result
        assert "# AGENT (Identity & Capabilities)" in result
        # Empty files result in newlines, not "Capabilities here."
        assert result.endswith("\n\n")


def test_load_reads_fresh_each_time():
    """Test that load() reads files fresh each time (not cached)."""
    with TemporaryDirectory() as tmpdir:
        soul_path = Path(tmpdir) / "SOUL.md"
        agent_path = Path(tmpdir) / "AGENT.md"

        soul_path.write_text("# Soul\nOriginal content")
        agent_path.write_text("# Agent\nOriginal content")

        loader = SystemPromptLoader(soul_path=str(soul_path), agent_path=str(agent_path))

        # First load
        result1 = loader.load()
        assert "Original content" in result1

        # Modify files
        soul_path.write_text("# Soul\nUpdated content")
        agent_path.write_text("# Agent\nUpdated content")

        # Second load should reflect updates
        result2 = loader.load()
        assert "Updated content" in result2
        assert "Original content" not in result2

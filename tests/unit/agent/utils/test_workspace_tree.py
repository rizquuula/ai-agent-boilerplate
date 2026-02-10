"""Test workspace tree generator."""

import tempfile
from pathlib import Path

from asterism.agent.utils.workspace_tree import (
    DEFAULT_IGNORE_PATTERNS,
    _should_ignore,
    generate_workspace_tree,
    get_workspace_tree_context,
)


def test_default_ignore_patterns():
    """Test default ignore patterns exist."""
    assert "__pycache__" in DEFAULT_IGNORE_PATTERNS
    assert ".git" in DEFAULT_IGNORE_PATTERNS
    assert "node_modules" in DEFAULT_IGNORE_PATTERNS
    assert ".venv" in DEFAULT_IGNORE_PATTERNS
    assert ".env" in DEFAULT_IGNORE_PATTERNS


def test_should_ignore_exact_match():
    """Test exact match ignore patterns."""
    assert _should_ignore("__pycache__", DEFAULT_IGNORE_PATTERNS) is True
    assert _should_ignore(".git", DEFAULT_IGNORE_PATTERNS) is True
    assert _should_ignore(".venv", DEFAULT_IGNORE_PATTERNS) is True


def test_should_ignore_wildcard_match():
    """Test wildcard pattern matching."""
    assert _should_ignore("test.egg-info", DEFAULT_IGNORE_PATTERNS) is True
    assert _should_ignore("package.egg-info", DEFAULT_IGNORE_PATTERNS) is True


def test_should_ignore_no_match():
    """Test that non-matching names are not ignored."""
    assert _should_ignore("asterism", DEFAULT_IGNORE_PATTERNS) is False
    assert _should_ignore("main.py", DEFAULT_IGNORE_PATTERNS) is False
    assert _should_ignore("README.md", DEFAULT_IGNORE_PATTERNS) is False


def test_should_ignore_custom_patterns():
    """Test custom ignore patterns."""
    custom_patterns = frozenset(["custom_dir", "*.log"])
    assert _should_ignore("custom_dir", custom_patterns) is True
    assert _should_ignore("debug.log", custom_patterns) is True
    assert _should_ignore("app.py", custom_patterns) is False


def test_generate_workspace_tree_nonexistent():
    """Test tree generation for non-existent directory."""
    result = generate_workspace_tree("/nonexistent/path")
    assert "Directory does not exist" in result


def test_generate_workspace_tree_not_a_directory():
    """Test tree generation for a file path."""
    with tempfile.NamedTemporaryFile() as f:
        result = generate_workspace_tree(f.name)
        assert "Not a directory" in result


def test_generate_workspace_tree_empty_directory():
    """Test tree generation for empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = generate_workspace_tree(tmpdir)
        assert "Workspace Directory:" in result
        assert Path(tmpdir).name in result


def test_generate_workspace_tree_with_files():
    """Test tree generation with files and directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create structure
        (Path(tmpdir) / "file1.txt").touch()
        (Path(tmpdir) / "file2.py").touch()

        result = generate_workspace_tree(tmpdir)

        assert "file1.txt" in result
        assert "file2.py" in result


def test_generate_workspace_tree_nested():
    """Test tree generation with nested directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested structure
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        (subdir / "nested_file.txt").touch()

        result = generate_workspace_tree(tmpdir, max_depth=3)

        assert "subdir/" in result
        assert "nested_file.txt" in result


def test_generate_workspace_tree_ignores_patterns():
    """Test that ignored patterns are not included."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files, including ignored ones
        (Path(tmpdir) / "main.py").touch()
        (Path(tmpdir) / ".env").touch()
        (Path(tmpdir) / "__pycache__").mkdir()

        result = generate_workspace_tree(tmpdir)

        assert "main.py" in result
        assert ".env" not in result
        assert "__pycache__" not in result


def test_generate_workspace_tree_max_depth():
    """Test max depth limitation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create deep structure
        current = Path(tmpdir)
        for i in range(5):
            current = current / f"level{i}"
            current.mkdir()
            (current / "file.txt").touch()

        # With max_depth=2, should not show level3 and beyond
        result = generate_workspace_tree(tmpdir, max_depth=2)

        # Should contain level0 and level1
        assert "level0/" in result
        assert "level1/" in result


def test_generate_workspace_tree_max_files():
    """Test max files limitation with truncation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create many files
        for i in range(25):
            (Path(tmpdir) / f"file{i}.txt").touch()

        result = generate_workspace_tree(tmpdir, max_files=10)

        # Should show truncation message
        assert "items total" in result


def test_generate_workspace_tree_permission_error():
    """Test handling of permission errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        subdir = Path(tmpdir) / "restricted"
        subdir.mkdir()
        (subdir / "file.txt").touch()

        # Remove read permissions
        original_mode = subdir.stat().st_mode
        try:
            subdir.chmod(0o000)
            result = generate_workspace_tree(tmpdir)
            # Should handle gracefully
            assert "Workspace Directory:" in result
        finally:
            # Restore permissions for cleanup
            subdir.chmod(original_mode)


def test_get_workspace_tree_context():
    """Test workspace tree context formatting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "main.py").touch()

        result = get_workspace_tree_context(tmpdir)

        assert "WORKSPACE STRUCTURE:" in result
        assert "main.py" in result
        assert "Note: Use this structure" in result


def test_get_workspace_tree_context_custom_params():
    """Test workspace tree context with custom parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested structure
        subdir = Path(tmpdir) / "deep"
        subdir.mkdir()
        (subdir / "file.txt").touch()

        result = get_workspace_tree_context(tmpdir, max_depth=1, max_files=5)

        assert "WORKSPACE STRUCTURE:" in result

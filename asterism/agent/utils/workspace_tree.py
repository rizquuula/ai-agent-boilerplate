"""Workspace directory tree generator for agent prompts."""

from pathlib import Path

# Default ignore patterns for common directories/files
DEFAULT_IGNORE_PATTERNS: frozenset[str] = frozenset(
    {
        # Python
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".tox",
        ".eggs",
        "*.egg-info",
        "dist",
        "build",
        # Node.js
        "node_modules",
        # Git
        ".git",
        ".gitignore",
        # IDE
        ".idea",
        ".vscode",
        ".vs",
        # OS
        ".DS_Store",
        "Thumbs.db",
        # Misc
        ".env",
        ".env.*",
        ".coverage",
        "htmlcov",
        ".ruff_cache",
    }
)

# Default configuration
DEFAULT_MAX_DEPTH = 3
DEFAULT_MAX_FILES = 20


def _should_ignore(name: str, ignore_patterns: frozenset[str]) -> bool:
    """Check if a file/directory should be ignored."""
    # Check exact match
    if name in ignore_patterns:
        return True
    # Check wildcard patterns
    for pattern in ignore_patterns:
        if pattern.startswith("*") and name.endswith(pattern[1:]):
            return True
    return False


def generate_workspace_tree(
    workspace_root: str | Path,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_files: int = DEFAULT_MAX_FILES,
    ignore_patterns: frozenset[str] | None = None,
) -> str:
    """Generate a directory tree string for the workspace.

    Args:
        workspace_root: Path to the workspace directory
        max_depth: Maximum depth to traverse (default: 3)
        max_files: Maximum files to list per directory before truncating (default: 20)
        ignore_patterns: Set of patterns to ignore (uses DEFAULT_IGNORE_PATTERNS if None)

    Returns:
        A formatted string representing the directory tree, or empty string if directory doesn't exist
    """
    root_path = Path(workspace_root)

    # Handle non-existent directory
    if not root_path.exists():
        return f"# Workspace Directory: {root_path}\n(Directory does not exist)"

    if not root_path.is_dir():
        return f"# Workspace Directory: {root_path}\n(Not a directory)"

    ignores = ignore_patterns if ignore_patterns is not None else DEFAULT_IGNORE_PATTERNS

    lines: list[str] = [f"# Workspace Directory: {root_path}", ""]

    def _build_tree(current_path: Path, prefix: str = "", current_depth: int = 0) -> None:
        """Recursively build the tree structure."""
        if current_depth > max_depth:
            return

        try:
            entries = list(current_path.iterdir())
        except PermissionError:
            lines.append(f"{prefix}[Permission Denied]")
            return
        except OSError:
            lines.append(f"{prefix}[Error reading directory]")
            return

        # Separate directories and files, filtering ignored items
        dirs: list[Path] = []
        files: list[Path] = []

        for entry in entries:
            if _should_ignore(entry.name, ignores):
                continue
            if entry.is_dir():
                dirs.append(entry)
            else:
                files.append(entry)

        # Sort alphabetically
        dirs.sort(key=lambda p: p.name.lower())
        files.sort(key=lambda p: p.name.lower())

        # Combine and check total count
        all_entries = dirs + files
        total_count = len(all_entries)
        show_truncated = total_count > max_files

        if show_truncated:
            # Show only directories when truncated
            display_entries = dirs
            lines.append(f"{prefix}... ({total_count} items total, showing {len(dirs)} directories)")
        else:
            display_entries = all_entries

        # Process entries
        for i, entry in enumerate(display_entries):
            is_last = (i == len(display_entries) - 1) and not show_truncated
            connector = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")

            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                if current_depth < max_depth:
                    _build_tree(entry, next_prefix, current_depth + 1)
            else:
                lines.append(f"{prefix}{connector}{entry.name}")

    # Start building from root
    lines.append(f"{root_path.name}/")
    _build_tree(root_path, current_depth=1)

    return "\n".join(lines)


def get_workspace_tree_context(
    workspace_root: str | Path,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_files: int = DEFAULT_MAX_FILES,
) -> str:
    """Get formatted workspace tree context for prompt injection.

    Args:
        workspace_root: Path to the workspace directory
        max_depth: Maximum depth to traverse
        max_files: Maximum files to list per directory

    Returns:
        Formatted tree string ready to inject into prompts
    """
    tree = generate_workspace_tree(workspace_root, max_depth, max_files)

    return f"""WORKSPACE STRUCTURE:
{tree}

Note: Use this structure to understand the workspace layout. For detailed file listings, use filesystem:list_files tool.
"""

"""System prompt loader for SOUL.md and AGENT.md files.

This module provides functionality to load and combine system prompts
from markdown files. SOUL.md contains the agent's core values and philosophy,
while AGENT.md contains the agent's identity and capabilities.

These files are read fresh on each call to ensure runtime updates are reflected.
"""

from pathlib import Path
from typing import Self


class SystemPromptLoader:
    """Loads SOUL.md and AGENT.md from disk and combines them.

    This loader reads both files fresh on each call, ensuring that any
    runtime updates to these files are immediately reflected in the
    agent's behavior.

    Attributes:
        soul_path: Path to the SOUL.md file (default: workspace/SOUL.md)
        agent_path: Path to the AGENT.md file (default: workspace/AGENT.md)
    """

    DEFAULT_SOUL_PATH = "workspace/SOUL.md"
    DEFAULT_AGENT_PATH = "workspace/AGENT.md"

    def __init__(
        self,
        soul_path: str | None = None,
        agent_path: str | None = None,
    ):
        """
        Initialize the system prompt loader.

        Args:
            soul_path: Path to the SOUL.md file. Defaults to "workspace/SOUL.md".
            agent_path: Path to the AGENT.md file. Defaults to "workspace/AGENT.md".
        """
        self.soul_path = soul_path or self.DEFAULT_SOUL_PATH
        self.agent_path = agent_path or self.DEFAULT_AGENT_PATH

    def with_paths(self, soul_path: str, agent_path: str) -> Self:
        """
        Create a new loader with specified file paths.

        Args:
            soul_path: Path to the SOUL.md file.
            agent_path: Path to the AGENT.md file.

        Returns:
            A new SystemPromptLoader instance with the specified paths.
        """
        return self.__class__(soul_path=soul_path, agent_path=agent_path)

    def _read_file(self, path: str) -> str:
        """
        Read a file from disk.

        Args:
            path: Path to the file to read.

        Returns:
            The file contents as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(path)
        if not file_path.is_absolute():
            # Resolve relative paths from the project root
            # Try to find the project root by looking for pyproject.toml
            current = Path.cwd()
            for parent in [current] + list(current.parents):
                if (parent / "pyproject.toml").exists():
                    file_path = parent / path
                    break
            else:
                # Fallback to current directory
                file_path = current / path

        with open(file_path, encoding="utf-8") as f:
            return f.read()

    def load(self) -> str:
        """
        Load and combine SOUL.md and AGENT.md content.

        Reads both files fresh from disk and combines them with
        a separator. This ensures runtime updates are reflected.

        Returns:
            Combined system prompt string from both files.

        Raises:
            FileNotFoundError: If either SOUL.md or AGENT.md is missing.
                               These files are mandatory for the agent.
        """
        soul_content = self._read_file(self.soul_path)
        agent_content = self._read_file(self.agent_path)

        # Combine with clear section headers
        combined = f"""# SOUL (Core Values & Philosophy)

{soul_content}

# AGENT (Identity & Capabilities)

{agent_content}
"""
        return combined

    def load_separate(self) -> tuple[str, str]:
        """
        Load SOUL.md and AGENT.md separately.

        Returns:
            A tuple of (soul_content, agent_content).

        Raises:
            FileNotFoundError: If either file is missing.
        """
        soul_content = self._read_file(self.soul_path)
        agent_content = self._read_file(self.agent_path)
        return soul_content, agent_content

    def validate_files_exist(self) -> bool:
        """
        Check if both required files exist.

        Returns:
            True if both files exist, False otherwise.
        """
        try:
            self._read_file(self.soul_path)
            self._read_file(self.agent_path)
            return True
        except FileNotFoundError:
            return False

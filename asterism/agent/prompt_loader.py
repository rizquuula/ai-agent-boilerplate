"""System prompt loader for SOUL.md and AGENT.md files.

This module re-exports SystemPromptLoader from asterism.core for backward compatibility.
The actual implementation has been moved to asterism.core.prompt_loader to avoid
circular import issues.
"""

from asterism.core.prompt_loader import SystemPromptLoader

__all__ = ["SystemPromptLoader"]

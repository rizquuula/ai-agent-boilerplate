"""Skill loading utilities for the agent framework."""

from pathlib import Path
from typing import Any

import yaml

from agent.state import SubGoal


class SkillConfig:
    """Configuration for a skill."""

    def __init__(self, name: str, description: str, content: str, metadata: dict[str, Any]):
        self.name = name
        self.description = description
        self.content = content
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"SkillConfig(name={self.name}, description={self.description[:50]}...)"


def normalize_skill_name(skill_name: str) -> str:
    """Normalize skill name to match directory naming.

    Handles conversions like:
    - code_reader -> code-reader
    - CodeReader -> code-reader
    - report-writer -> report-writer (unchanged)

    Args:
        skill_name: Raw skill name from milestone

    Returns:
        Normalized skill name matching directory structure
    """
    # Handle camelCase by inserting hyphens before capitals (except first char)
    result = skill_name[0].lower() if skill_name else ""
    for char in skill_name[1:]:
        if char.isupper():
            result += "-" + char.lower()
        else:
            result += char

    # Convert snake_case and spaces to kebab-case
    normalized = result.replace("_", "-").replace(" ", "-")
    return normalized


def load_skill_config(skill_name: str) -> SkillConfig:
    """Load skill configuration from SKILL.md file.

    Args:
        skill_name: Name of the skill (will be normalized)

    Returns:
        SkillConfig object with parsed metadata and content

    Raises:
        FileNotFoundError: If skill file doesn't exist
    """
    normalized_name = normalize_skill_name(skill_name)
    skill_path = Path(__file__).parent.parent.parent / "skills" / normalized_name / "SKILL.md"

    if not skill_path.exists():
        raise FileNotFoundError(f"Skill not found: {skill_name} (looked for {skill_path})")

    with open(skill_path, encoding="utf-8") as f:
        content = f.read()

    # Parse YAML frontmatter
    metadata: dict[str, Any] = {}
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            _, frontmatter, body = parts
            try:
                metadata = yaml.safe_load(frontmatter) or {}
            except yaml.YAMLError:
                metadata = {}

    return SkillConfig(
        name=metadata.get("name", normalized_name),
        description=metadata.get("description", ""),
        content=body.strip(),
        metadata=metadata,
    )


def get_available_skills() -> list[str]:
    """Get list of available skill names.

    Returns:
        List of skill directory names
    """
    skills_dir = Path(__file__).parent.parent.parent / "skills"

    if not skills_dir.exists():
        return []

    return [d.name for d in skills_dir.iterdir() if d.is_dir()]


def get_available_skills_description() -> str:
    """Get formatted description of all available skills.

    Returns:
        Multi-line string describing each skill
    """
    skills = get_available_skills()
    descriptions = []

    for skill_name in skills:
        try:
            config = load_skill_config(skill_name)
            descriptions.append(f"- {config.name}: {config.description}")
        except FileNotFoundError:
            continue

    return "\n".join(descriptions) if descriptions else "No skills available"


def format_completed_milestones(milestones: list[SubGoal], current_idx: int) -> str:
    """Format completed milestones for context.

    Args:
        milestones: List of all milestones
        current_idx: Current milestone index

    Returns:
        Formatted string of completed milestones
    """
    completed = milestones[:current_idx]

    if not completed:
        return "No milestones completed yet."

    lines = []
    for i, m in enumerate(completed, 1):
        status_icon = "PASS" if m.status == "completed" else "FAIL"
        lines.append(f"{i}. [{status_icon}] {m.description}")

    return "\n".join(lines)

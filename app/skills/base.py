from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Skill:
    """A modular, prompt-driven specialization that agents load on-demand."""

    name: str
    description: str
    domain: str  # "mesh" | "graphql" | "classification"
    prompt_template: str
    examples: list[dict] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    fallback_skill: Optional[str] = None

    def matches_query(self, query: str) -> float:
        """Score how well this skill matches a user query (0.0 to 1.0)."""
        query_lower = query.lower()
        matched = sum(1 for kw in self.keywords if kw in query_lower)
        if not self.keywords:
            return 0.0
        return matched / len(self.keywords)

    def render_prompt(self, **kwargs: str) -> str:
        """Render the prompt template with provided variables."""
        return self.prompt_template.format(**kwargs)

    def format_examples(self) -> str:
        """Format few-shot examples as a string for inclusion in prompts."""
        if not self.examples:
            return ""
        lines = ["Examples:"]
        for ex in self.examples:
            lines.append(f"  Q: {ex.get('query', '')}")
            lines.append(f"  A: {ex.get('output', ex.get('needed', ''))}")
            lines.append("")
        return "\n".join(lines)

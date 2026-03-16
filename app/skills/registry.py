from __future__ import annotations

from typing import Optional

from app.config import AppConfig, SkillConfig
from app.skills.base import Skill


class SkillRegistry:
    """Discovers, loads, and manages skills from configuration."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def load_from_config(self, config: AppConfig) -> None:
        """Load all skills defined in application.yml."""
        for name, skill_cfg in config.skills.items():
            self._skills[name] = self._build_skill(name, skill_cfg)

    def _build_skill(self, name: str, cfg: SkillConfig) -> Skill:
        return Skill(
            name=name,
            description=cfg.description,
            domain=cfg.domain,
            prompt_template=cfg.prompt_template,
            examples=cfg.examples,
            keywords=cfg.keywords,
            fallback_skill=cfg.fallback_skill,
        )

    def get_skill(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def find_skills(self, query: str, domain: Optional[str] = None) -> list[Skill]:
        """Find skills matching a query, optionally filtered by domain.
        Returns skills sorted by match score (best first)."""
        candidates = self._skills.values()
        if domain:
            candidates = [s for s in candidates if s.domain == domain]

        scored = [(s, s.matches_query(query)) for s in candidates]
        scored = [(s, score) for s, score in scored if score > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored]

    def list_skills_for_domain(self, domain: str) -> list[Skill]:
        return [s for s in self._skills.values() if s.domain == domain]

    def get_catalog(self) -> str:
        """Generate a skill catalog string for the supervisor prompt."""
        lines = []
        for skill in self._skills.values():
            fallback = f" (fallback: {skill.fallback_skill})" if skill.fallback_skill else ""
            lines.append(f"- {skill.name} [{skill.domain}]: {skill.description}{fallback}")
        return "\n".join(lines)

    @property
    def all_skills(self) -> dict[str, Skill]:
        return dict(self._skills)

"""Skills = config-driven prompt specialisations for worker agents.

A skill is chosen by the router (RouteDecision.skill) or, failing that, by
keyword match. The worker passes the rendered skill text in its agent state
(`skill_prompt`) and `SkillPromptMiddleware` appends it to the system prompt
right before each model call. Base prompts stay static (cache-friendly) and
new skills are pure YAML.
"""

from __future__ import annotations

from typing import Optional

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest
from langchain_core.messages import SystemMessage
from typing_extensions import NotRequired

from supervisor_template.config import SkillConfig


class SkillRegistry:
    def __init__(self, skills: dict[str, SkillConfig]):
        self._skills = dict(skills)

    def get(self, name: Optional[str]) -> Optional[SkillConfig]:
        return self._skills.get(name) if name else None

    def names(self) -> list[str]:
        return list(self._skills)

    def match(self, question: str, agent: Optional[str] = None) -> Optional[str]:
        """Best skill by keyword hits (deterministic fallback when the router gives none)."""
        q = question.lower()
        scored = [
            (sum(kw in q for kw in s.keywords), name)
            for name, s in self._skills.items()
            if agent is None or s.agent == agent
        ]
        scored = [x for x in scored if x[0] > 0]
        return max(scored)[1] if scored else None

    def allows_fallback(self, name: Optional[str]) -> bool:
        skill = self.get(name)
        return bool(skill and skill.fallback)

    def catalog(self) -> str:
        return "\n".join(
            f"- {n} [{s.agent}]{' (fallback allowed)' if s.fallback else ''}: {s.description}"
            for n, s in self._skills.items()
        )

    def render(self, name: Optional[str]) -> str:
        skill = self.get(name)
        if not skill:
            return ""
        parts = [f"## Skill: {name}", skill.description]
        if skill.instructions.strip():
            parts.append(skill.instructions.strip())
        if skill.examples:
            parts.append("Examples:")
            parts += [f"Q: {ex['q']}\nA: {ex['a']}" for ex in skill.examples]
        return "\n".join(parts)


class WorkerAgentState(AgentState):
    """Agent state extended with the skill text (middleware can extend state)."""

    skill_prompt: NotRequired[str]


class SkillPromptMiddleware(AgentMiddleware):
    """Appends `state["skill_prompt"]` to the system prompt on every model call."""

    state_schema = WorkerAgentState

    @staticmethod
    def _apply(request: ModelRequest) -> ModelRequest:
        extra = (request.state or {}).get("skill_prompt")
        if not extra:
            return request
        base = request.system_message.text if request.system_message else ""
        return request.override(system_message=SystemMessage(f"{base}\n\n{extra}"))

    def wrap_model_call(self, request, handler):
        return handler(self._apply(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._apply(request))

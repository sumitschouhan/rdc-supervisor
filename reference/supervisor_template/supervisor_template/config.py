"""Typed configuration loaded from application.yml."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    name: str
    effort: Optional[str] = None  # low | medium | high | xhigh | max
    max_tokens: int = 8192


class ModelsConfig(BaseModel):
    router: ModelSpec
    worker: ModelSpec
    synthesizer: ModelSpec
    fallback: Optional[str] = None


class Limits(BaseModel):
    max_supervisor_iterations: int = 6
    max_clarifications: int = 2
    worker_timeout_seconds: float = 90
    worker_max_attempts: int = 2
    worker_tool_calls_per_run: int = 4
    worker_model_calls_per_run: int = 6


class AgentConfig(BaseModel):
    description: str
    source_system: str
    tools: list[str]
    system_prompt: str
    fallback: Optional[str] = None


class SkillConfig(BaseModel):
    agent: str
    description: str
    keywords: list[str] = Field(default_factory=list)
    fallback: bool = False
    instructions: str = ""
    examples: list[dict[str, str]] = Field(default_factory=list)


class Prompts(BaseModel):
    metadata_agent: str
    router: str
    synthesizer: str
    no_data_question: str


class DeepAnalystConfig(BaseModel):
    enabled: bool = True
    require_sql_approval: bool = False
    system_prompt: str = ""
    guide: str = ""


class AppConfig(BaseModel):
    models: ModelsConfig
    limits: Limits = Field(default_factory=Limits)
    agents: dict[str, AgentConfig]
    prompts: Prompts
    skills: dict[str, SkillConfig] = Field(default_factory=dict)
    deep_analyst: DeepAnalystConfig = Field(default_factory=DeepAnalystConfig)

    def model_post_init(self, _ctx) -> None:  # fail fast on config typos
        for name, skill in self.skills.items():
            if skill.agent not in self.agents:
                raise ValueError(f"skill {name!r} references unknown agent {skill.agent!r}")
        for name, agent in self.agents.items():
            if agent.fallback and agent.fallback not in self.agents:
                raise ValueError(f"agent {name!r} has unknown fallback {agent.fallback!r}")


def load_config(path: str | Path | None = None) -> AppConfig:
    path = Path(path or Path(__file__).with_name("application.yml"))
    with open(path) as f:
        return AppConfig(**yaml.safe_load(f))

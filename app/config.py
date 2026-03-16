from __future__ import annotations

import yaml
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    provider: str = "anthropic"
    name: str = "claude-sonnet-4-20250514"
    temperature: float = 0.0
    max_tokens: int = 4096


class AgentsConfig(BaseModel):
    max_iterations: int = 5
    metadata_timeout_seconds: int = 30


class SkillConfig(BaseModel):
    description: str
    domain: str
    keywords: list[str]
    fallback_skill: Optional[str] = None
    prompt_template: str
    examples: list[dict] = Field(default_factory=list)


class PromptsConfig(BaseModel):
    supervisor: dict[str, str]
    metadata_agent: dict[str, str]
    mesh_agent: dict[str, str]
    graphql_agent: dict[str, str]
    human_feedback: dict[str, str]
    synthesizer: dict[str, str]


class AppConfig(BaseModel):
    prompts: PromptsConfig
    model: ModelConfig
    agents: AgentsConfig
    skills: dict[str, SkillConfig] = Field(default_factory=dict)


def load_config(path: Optional[str] = None) -> AppConfig:
    config_path = Path(path or Path(__file__).parent / "application.yml")
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)

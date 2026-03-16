from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import load_config
from app.prompts.loader import PromptLoader
from app.skills.registry import SkillRegistry


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def prompt_loader(config):
    return PromptLoader(config)


@pytest.fixture
def skill_registry(config):
    registry = SkillRegistry()
    registry.load_from_config(config)
    return registry

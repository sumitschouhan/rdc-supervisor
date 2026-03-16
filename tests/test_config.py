"""Tests for configuration loading."""

from app.config import load_config


def test_load_config():
    config = load_config()
    assert config.model.provider == "anthropic"
    assert config.agents.max_iterations == 5


def test_prompts_loaded():
    config = load_config()
    assert "system" in config.prompts.supervisor
    assert "system" in config.prompts.metadata_agent
    assert "system" in config.prompts.mesh_agent
    assert "system" in config.prompts.graphql_agent
    assert "classification" in config.prompts.human_feedback
    assert "no_data" in config.prompts.human_feedback
    assert "system" in config.prompts.synthesizer


def test_skills_loaded():
    config = load_config()
    assert "trade_query" in config.skills
    assert "instrument_mesh" in config.skills
    assert "morningstar_query" in config.skills
    assert config.skills["trade_query"].domain == "mesh"
    assert config.skills["morningstar_query"].domain == "graphql"


def test_skill_fallback_config():
    config = load_config()
    assert config.skills["instrument_mesh"].fallback_skill == "instrument_graphql"
    assert config.skills["trade_query"].fallback_skill is None


def test_prompt_loader(prompt_loader):
    assert "routing supervisor" in prompt_loader.supervisor_system()
    assert "metadata specialist" in prompt_loader.metadata_agent_system()
    assert "Snowflake SQL" in prompt_loader.mesh_agent_system()
    assert "GraphQL" in prompt_loader.graphql_agent_system()
    assert "Summarize" in prompt_loader.synthesizer_system()

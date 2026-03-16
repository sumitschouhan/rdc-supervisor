"""Tests for the skills framework."""

from app.skills.base import Skill


def test_skill_matches_query():
    skill = Skill(
        name="trade_query",
        description="Trade data",
        domain="mesh",
        prompt_template="Generate SQL for {query}",
        keywords=["trade", "trades", "execution"],
    )
    assert skill.matches_query("show me all trades") > 0
    assert skill.matches_query("weather forecast") == 0


def test_skill_render_prompt():
    skill = Skill(
        name="test",
        description="Test",
        domain="mesh",
        prompt_template="Tables: {tables}\nQuery: {query}",
    )
    result = skill.render_prompt(tables="[trades]", query="show trades")
    assert "Tables: [trades]" in result
    assert "Query: show trades" in result


def test_skill_format_examples():
    skill = Skill(
        name="test",
        description="Test",
        domain="mesh",
        prompt_template="",
        examples=[
            {"query": "show trades", "output": "SELECT * FROM trades"},
        ],
    )
    formatted = skill.format_examples()
    assert "show trades" in formatted
    assert "SELECT * FROM trades" in formatted


def test_skill_format_examples_empty():
    skill = Skill(name="test", description="Test", domain="mesh", prompt_template="")
    assert skill.format_examples() == ""


def test_registry_load_from_config(config, skill_registry):
    assert len(skill_registry.all_skills) > 0
    assert skill_registry.get_skill("trade_query") is not None


def test_registry_find_skills(skill_registry):
    skills = skill_registry.find_skills("show me all trades", domain="mesh")
    assert len(skills) > 0
    assert skills[0].name == "trade_query"


def test_registry_find_skills_graphql(skill_registry):
    skills = skill_registry.find_skills("morningstar ratings", domain="graphql")
    assert len(skills) > 0
    assert "morningstar" in skills[0].name


def test_registry_list_skills_for_domain(skill_registry):
    mesh_skills = skill_registry.list_skills_for_domain("mesh")
    graphql_skills = skill_registry.list_skills_for_domain("graphql")
    assert len(mesh_skills) >= 3  # trade, position, instrument_mesh, issuer
    assert len(graphql_skills) >= 2  # instrument_graphql, morningstar, pricing


def test_registry_get_catalog(skill_registry):
    catalog = skill_registry.get_catalog()
    assert "trade_query" in catalog
    assert "morningstar_query" in catalog
    assert "[mesh]" in catalog
    assert "[graphql]" in catalog


def test_instrument_mesh_has_fallback(skill_registry):
    skill = skill_registry.get_skill("instrument_mesh")
    assert skill is not None
    assert skill.fallback_skill == "instrument_graphql"


def test_trade_query_no_fallback(skill_registry):
    skill = skill_registry.get_skill("trade_query")
    assert skill is not None
    assert skill.fallback_skill is None

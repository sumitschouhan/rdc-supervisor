from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_anthropic import ChatAnthropic

from app.config import load_config
from app.prompts.loader import PromptLoader
from app.skills.registry import SkillRegistry
from app.api.router import router
from app.graph.supervisor import init_graph, build_graph, set_graph


def create_app() -> FastAPI:
    load_dotenv()

    app = FastAPI(
        title="RDC Supervisor - Multi-Agent Orchestrator",
        description="Chat-based data coach using LangGraph supervisor pattern with skills architecture",
        version="0.1.0",
    )

    config = load_config()

    model = ChatAnthropic(
        model=config.model.name,
        temperature=config.model.temperature,
        max_tokens=config.model.max_tokens,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
    )

    prompt_loader = PromptLoader(config)

    skill_registry = SkillRegistry()
    skill_registry.load_from_config(config)

    init_graph(config, model, prompt_loader, skill_registry)
    graph = build_graph()
    set_graph(graph)

    app.include_router(router)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "skills_loaded": len(skill_registry.all_skills),
            "model": config.model.name,
        }

    return app


app = create_app()

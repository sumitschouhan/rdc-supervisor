"""Model factory.

Claude 5.x gotchas handled here (all verified against langchain-anthropic 1.7):
- No `temperature`/`top_p`/`top_k`: non-default values are rejected. Tune
  quality/cost with `reasoning_effort` (-> output_config.effort) instead.
- Thinking is adaptive and on by default; don't send `{"type": "disabled"}`.
- Forced tool_choice ("any"/specific tool) is rejected on Opus 5.5 / Sonnet 5.5,
  so structured output uses Claude's native JSON-schema mode
  (`method="json_schema"`) rather than forced tool calling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

from supervisor_template.config import AppConfig, ModelSpec


@dataclass
class ModelSet:
    router: BaseChatModel
    worker: BaseChatModel
    synthesizer: BaseChatModel
    fallback: Optional[BaseChatModel] = None


def _claude(spec: ModelSpec) -> ChatAnthropic:
    return ChatAnthropic(
        model=spec.name,
        max_tokens=spec.max_tokens,
        reasoning_effort=spec.effort,
        max_retries=3,  # SDK-level retries for 429/5xx
    )


def build_models(cfg: AppConfig, *, demo: Optional[bool] = None) -> ModelSet:
    """Real Claude models, or the offline DemoChatModel when no API key is set
    (or SUPERVISOR_DEMO=1). Tests always use the demo model."""
    if demo is None:
        demo = os.getenv("SUPERVISOR_DEMO") == "1" or not os.getenv("ANTHROPIC_API_KEY")
    if demo:
        from supervisor_template.demo_model import DemoChatModel

        m = DemoChatModel()
        return ModelSet(router=m, worker=m, synthesizer=m, fallback=None)

    fallback = None
    if cfg.models.fallback:
        fallback = _claude(ModelSpec(name=cfg.models.fallback, max_tokens=cfg.models.worker.max_tokens))
    return ModelSet(
        router=_claude(cfg.models.router),
        worker=_claude(cfg.models.worker),
        synthesizer=_claude(cfg.models.synthesizer),
        fallback=fallback,
    )


def structured(model: BaseChatModel, schema: type) -> Runnable:
    """Structured output that works on every current Claude model."""
    if isinstance(model, ChatAnthropic):
        return model.with_structured_output(schema, method="json_schema")
    return model.with_structured_output(schema)

import sys
import uuid
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supervisor_template.config import load_config  # noqa: E402
from supervisor_template.graph import build_graph  # noqa: E402
from supervisor_template.llm import build_models  # noqa: E402


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def models(cfg):
    return build_models(cfg, demo=True)  # offline, deterministic


@pytest.fixture
def graph(cfg, models):
    return build_graph(cfg, models, checkpointer=InMemorySaver())


@pytest.fixture
def thread():
    return {"configurable": {"thread_id": str(uuid.uuid4())}}

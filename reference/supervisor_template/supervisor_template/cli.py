"""Interactive terminal chat.  python -m supervisor_template.cli [--demo]"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from supervisor_template.config import load_config
from supervisor_template.context import AppContext
from supervisor_template.graph import build_graph
from supervisor_template.llm import build_models


async def main(demo: bool) -> None:
    load_dotenv()
    cfg = load_config()
    graph = build_graph(cfg, build_models(cfg, demo=demo or None), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    ctx = AppContext(user_id="cli")
    print("Ask a data question (Ctrl-D to quit).")
    while True:
        try:
            text = input("\nyou> ").strip()
        except EOFError:
            return
        if not text:
            continue
        payload = {"messages": [HumanMessage(text)]}
        while True:
            # stream node-level progress, then inspect the final state
            async for part in graph.astream(payload, config, context=ctx, stream_mode="updates", version="v2"):
                nodes = [k for k in (part["data"] or {}) if k != "__interrupt__"]
                if nodes:
                    print("  ·", ", ".join(nodes))
            snap = await graph.aget_state(config)
            if not snap.interrupts:
                break
            req = snap.interrupts[0].value
            opts = f" {req['options']}" if req.get("options") else ""
            payload = Command(resume={"answer": input(f"assistant> {req['question']}{opts}\nyou> ")})
        print("assistant>", snap.values["messages"][-1].text)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="use the offline demo model")
    asyncio.run(main(p.parse_args().demo))

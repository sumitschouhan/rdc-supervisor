# What's new in LangChain / LangGraph (as of Nov 2026), and what it means for rdc-supervisor

Versions checked: **langgraph 1.2.24**, **langchain 1.5.2**, **langchain-core 1.8.4**,
**langchain-anthropic 1.7.12** (latest on PyPI, 2026-11-25). Every API below was
checked against the installed package source, and the template's tests run on it.

rdc-supervisor today pins `langgraph>=0.2`, `langchain-core>=0.3`, `langchain-anthropic>=0.3`
and uses the 0.2-era patterns. The table shows what to change.

## High-impact changes for this project

| Area | rdc-supervisor today | Current recommendation | Where in template |
|---|---|---|---|
| Building agents | hand-written `bind_tools` loop (`for _ in range(3)`) in each agent | `langchain.agents.create_agent(model, tools, system_prompt=, middleware=[...], response_format=, context_schema=)`. (`langgraph.prebuilt.create_react_agent` is deprecated.) | `agents.py` |
| Agent cross-cutting concerns | ad-hoc code | **Middleware**: `ToolErrorMiddleware`, `ToolCallLimitMiddleware`, `ModelCallLimitMiddleware`, `ModelFallbackMiddleware`, `ModelRetryMiddleware`, `ToolRetryMiddleware`, `HumanInTheLoopMiddleware`, `SummarizationMiddleware`, `PIIMiddleware`, `LLMToolSelectorMiddleware`, `ContextEditingMiddleware`, `ModelRouterMiddleware` (beta), plus custom `AgentMiddleware` / `@dynamic_prompt` / `@wrap_model_call` | `agents.py`, `skills.py` |
| Supervisor routing output | LLM returns JSON text, parsed by hand | Structured output (`with_structured_output(Schema, method="json_schema")`, `create_agent(response_format=Schema)`) | `schemas.py`, `graph.py` |
| Routing edges | `add_conditional_edges` + separate `routing.py` | Nodes return `Command(goto=..., update=...)`, with `destinations=` for rendering | `graph.py` |
| Cross-agent parallel | custom `CrossAgentDispatcher` with `ThreadPoolExecutor` | `Command(goto=[Send("mesh_agent", task), Send("graphql_agent", task)])`. LangGraph runs both in one super-step, merges with reducers, and checkpoints each | `graph.py` step 5 |
| Cross-agent sequential | `pending_agents` + manual state copy | `plan` queue + `Send(next, task_with_prior_results)` | `graph.py` step 2 |
| Worker inputs | every agent reads the whole state | Per-node `input_schema=WorkerTask` delivered via `Send` | `state.py` |
| Per-user data (entitlements, user id) | not modelled / globals | `StateGraph(..., context_schema=AppContext)`; nodes get `runtime: Runtime[AppContext]`, tools get `runtime: ToolRuntime[AppContext]`; call with `ainvoke(..., context=AppContext(...))` | `context.py`, `tools/*` |
| Resetting per-turn lists | `operator.add` lists grow forever across turns | `Overwrite([])` bypasses the reducer | `graph.py start_turn` |
| Failures | an exception kills the run | `add_node(..., retry_policy=RetryPolicy(...), timeout=..., error_handler=fn)`; the handler gets `error: NodeError` and returns a `Command`. Also `set_node_defaults(...)` | `graph.py` wiring |
| Human in the loop | `interrupt(dict)` + `get_state().tasks` probing | `interrupt(payload, response_schema=PydanticModel)`: the client gets a JSON schema and the resume value is validated. `ainvoke(..., version="v2")` returns `GraphOutput(.value, .interrupts)` | `graph.py human_review`, `api.py` |
| Large tool results | rows go through the LLM | `@tool(response_format="content_and_artifact")`: a short summary goes to the LLM, full rows to code | `tools/*`, worker node |
| Model params | `temperature: 0.0`, `claude-sonnet-4-20250514` | Claude 5.x **rejects** non-default `temperature/top_p/top_k` and **forced tool_choice**. Use `reasoning_effort` and native JSON-schema structured output. Current models: `claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5` | `llm.py`, `application.yml` |
| Sync API in async FastAPI | `graph.invoke` inside `async def` (blocks the event loop) | `await graph.ainvoke(...)`, built once in FastAPI `lifespan` | `api.py` |

## Other notable 2026 additions (worth knowing)

- **LangChain 1.0 → 1.5** (Oct 2025 → 2026): `create_agent` plus middleware became *the*
  agent API. Standard content blocks were added (`message.content_blocks`, `message.text` is a property).
  Structured-output strategies are `ToolStrategy`, `ProviderStrategy` and `AutoStrategy`.
- **Multi-agent guidance**: the docs now centre on *subagents as tools* (a supervisor
  `create_agent` whose tools call other agents) plus handoffs/router patterns. The
  `langgraph-supervisor` package (`create_supervisor`, still at 0.0.31) is no longer actively
  maintained. See `patterns/subagents_as_tools.py`.
- **LangGraph 1.1** (Mar 2026): opt-in `version="v2"` output for `invoke`/`stream`:
  typed `GraphOutput` and `StreamPart` dicts (`{"type","ns","data"}`).
- **LangGraph 1.2** (May 2026): node `timeout` / `TimeoutPolicy` (async nodes), node
  `error_handler` + `NodeError`, `set_node_defaults`, **DeltaChannel** (beta), and **v3
  streaming** (beta). Named `create_agent(name=...)` agents appear as typed
  `run.subagents` handles in the new stream protocol.
- **Sep 2026**: `interrupt(..., response_schema=...)`.
- **Oct 2026**: deterministic parallel DeltaChannel writes; Postgres checkpointer migration.
- **Durability modes**: `ainvoke(..., durability="sync" | "async" | "exit")`.
- **Node caching**: `add_node(..., cache_policy=CachePolicy(ttl=...))` + `compile(cache=...)`.
- **Deep Agents** (`deepagents` 0.7.x): a batteries-included harness (planning, virtual
  filesystem, subagents, skills) on top of `create_agent`. Consider it for open-ended research
  agents. It is overkill for rule-driven routing like RDC.

## Sources

- PyPI version listings for langgraph, langchain, langchain-anthropic, langgraph-supervisor
  and deepagents (queried 2026-11-25).
- Installed package source and docstrings (`StateGraph.add_node`, `set_node_defaults`,
  `interrupt`, `GraphOutput`, `NodeError`, middleware classes, `ChatAnthropic`).
- [LangChain release notes (Releasebot)](https://releasebot.io/updates/langchain-ai)
- [LangChain and LangGraph in Late 2026 (Developer's Digest)](https://www.developersdigest.tech/blog/langchain-langgraph-whats-new-2026)
- [LangChain docs: subagents personal assistant](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant)
- [LangChain docs: agents](https://docs.langchain.com/oss/python/langchain/agents)

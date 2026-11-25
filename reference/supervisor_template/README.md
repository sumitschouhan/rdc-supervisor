# Supervisor template: reference multi-agent project (LangGraph 1.2 / LangChain 1.5)

A small, fully tested reference implementation of a supervisor-based multi-agent
system on the current LangChain/LangGraph APIs. It uses the RDC domain (metadata → Snowflake
"mesh" agent / GraphQL agent → synthesizer, with human-in-the-loop). It is meant to be
**copied and adapted** for similar projects.

- What changed in the libraries, and the mapping to the current `app/` code: **[WHATS_NEW.md](WHATS_NEW.md)**
- Deep Agents research (features, use cases, best practices, roadmap): **[DEEPAGENTS_RESEARCH.md](DEEPAGENTS_RESEARCH.md)**
- Runs fully **offline** with a deterministic demo model (no API key needed), so the graph,
  agents, middleware, fan-out, interrupts and checkpointing are exercised exactly as in
  production.

## Quick start

```bash
cd reference/supervisor_template
pip install -r requirements.txt

python -m pytest -q                          # 28 tests, offline
python -m supervisor_template.cli --demo     # chat in the terminal, offline
export ANTHROPIC_API_KEY=...                 # then without --demo for real Claude
uvicorn supervisor_template.api:app --reload # HTTP API (demo mode if no key)
```

Try: `Show me trades for AAPL`, `Show my positions and their Morningstar ratings` (parallel),
`For my top holdings then get current prices` (sequential), `Show instrument details for NVDA`
(Snowflake empty → GraphQL fallback), `Show instruments in the sector` (clarification interrupt),
`Compare my positions with their Morningstar ratings` (→ Deep Agents analyst mode).

## Architecture

```
START → start_turn → supervisor ─┬─→ metadata_agent ───────────────┐
                        ▲        ├─→ Send(mesh_agent, WorkerTask) ──┤   parallel: several Sends at once
                        │        ├─→ Send(graphql_agent, WorkerTask)┤   sequential: plan queue, one at a time
                        │        ├─→ human_review  (interrupt)      ┤
                        └────────┴──────────────────────────────────┘
                                 ├─→ synthesizer → END
                                 └─→ deep_analyst (Deep Agent) → END     mode="deep"
```

**Analyst mode (Deep Agents).** For open-ended, multi-step analysis the router picks
`mode="deep"`, and `deep_analyst.py` hands the question to a `create_deep_agent` that:
- delegates to `mesh_analyst` / `graphql_analyst` subagents (in parallel, each in its own
  context, returning `WorkerResult` JSON)
- reads SKILL.md skills rendered from `application.yml`
- always loads org conventions (`/guides/AGENTS.md`, read-only) and per-user memory
  (`/memories/AGENTS.md`, a Store namespaced by `user_id`)
- can require human approval for risky SQL (`require_sql_approval`)

Plain lookups never go there: it costs more model calls.

**"Deterministic shell, LLM core."** Rules that must always hold live in code in the
`supervisor` node, so they are testable and auditable:

1. Loop guard (`max_supervisor_iterations`).
2. Metadata first.
3. Continue a sequential plan, passing prior results.
4. Data found → synthesize.
5. No data → configured fallback agent (only if the skill allows it), else ask the human.
6. Otherwise ask the **router LLM** for a validated `RouteDecision` (agents, mode, skill,
   clarification). Code then constrains that decision to the agents that are available
   *and* the user is entitled to.

Each worker is a `create_agent` agent with middleware (tool-error recovery, call limits,
model fallback, skill injection). Rows come back as **tool artifacts**, so `has_data` is
ground truth, not an LLM claim.

| Concern | Mechanism |
|---|---|
| Parallel cross-agent | `Command(goto=[Send(a, task), Send(b, task)])` |
| Sequential cross-agent | `plan` + `Send(next, task with prior_results)` |
| Worker isolation | `add_node(..., input_schema=WorkerTask)` |
| Per-user context / entitlements | `context_schema=AppContext`, `Runtime` / `ToolRuntime` |
| Transient failures | `RetryPolicy` → `timeout` → `error_handler(NodeError)` |
| Typed human input | `interrupt(payload, response_schema=ClarificationAnswer)` |
| Turn reset | `Overwrite([])` on reducer channels |
| Durable threads | checkpointer (`InMemorySaver` → `AsyncPostgresSaver`) |
| API | `ainvoke(version="v2")` → `.value` / `.interrupts`; SSE via `astream(version="v2")` |

## Layout

```
supervisor_template/
  application.yml        ← prompts, agents, skills, models, limits  (most edits go here)
  config.py              typed config + cross-reference validation
  context.py             AppContext (runtime context: user, entitlements)
  schemas.py             RouteDecision, MetadataResult, WorkerResult, Clarification*
  state.py               OrchestratorState (+ reducers), WorkerTask
  llm.py                 Claude model factory (effort, no temperature, json_schema output)
  skills.py              SkillRegistry + SkillPromptMiddleware
  agents.py              create_agent workers + middleware stack
  graph.py               supervisor graph  ← the core
  api.py / cli.py        FastAPI (send/resume/stream/history) and terminal chat
  tools/                 search_catalog, run_snowflake_sql, run_graphql_query (mocks)
  deep_analyst.py        Deep Agents analyst mode (subagents, skills, memory, permissions, HITL)
  patterns/subagents_as_tools.py   alternative tool-calling supervisor
  demo_model.py          offline scripted model (tests/demo only)
tests/                   graph paths, error handler, HITL, entitlements, API, SSE
```

## Using it for a new project (checklist)

1. **Copy** `reference/supervisor_template/` and rename the package.
2. **Tools** (`tools/`): replace the mocks with real clients. Keep:
   - `response_format="content_and_artifact"`: a short summary for the LLM, full rows as the artifact.
   - `runtime: ToolRuntime[AppContext]` for entitlement checks.
   - Raise `ValueError`/`PermissionError` for errors the model should see and fix
     (`ToolErrorMiddleware`). Let infrastructure errors propagate, so the graph retries
     them and then calls `error_handler`.
   - Register each tool in `tools/__init__.py`.
3. **Agents** (`application.yml → agents`): one entry per data agent: `source_system`,
   `tools`, `system_prompt`, optional `fallback`. Each entry automatically becomes a graph
   node with retry, timeout and error handler.
4. **Skills** (`application.yml → skills`): keywords, instructions and few-shot examples
   per query type. Set `fallback: true` only where a fallback agent is valid.
5. **Router rules** (`prompts.router`): business routing rules in plain language. Put rules
   that must *never* be violated in `supervisor()` in `graph.py`.
6. **Context** (`context.py`): add whatever tools and nodes need per request (tenant,
   roles, locale). Build it from the authenticated user in `api.py`.
7. **Models** (`application.yml → models`): choose per role and tune with `effort`.
8. **Checkpointer**: switch to `AsyncPostgresSaver` (see `api.py` lifespan).
9. **Tests**: keep the demo model for graph-path tests. Add rules to `demo_model.py` for
   new tools and schemas. Add evals against the real model for prompt quality.

## Choosing a pattern

| | Explicit graph (`graph.py`) | Subagents as tools (`patterns/`) |
|---|---|---|
| Routing | code rules + one structured LLM decision | model chooses tools each turn |
| Guarantees (metadata-first, fallbacks, entitlements) | enforced | prompt-dependent |
| Parallelism | `Send` fan-out | parallel tool calls |
| Retries / timeouts / error handler per agent | per node | per tool (middleware) |
| Best for | regulated, rule-heavy data assistants (RDC) | open-ended assistants, quick prototypes |

Both share the same workers, tools, skills and config.

## Claude model notes (langchain-anthropic 1.7)

- Defaults: router `claude-opus-5-5` (effort `medium`); workers and synthesizer
  `claude-sonnet-5-5`. Change these in YAML.
- Do **not** set `temperature`/`top_p`/`top_k`: Claude 5.x rejects non-default values.
  Use `effort`.
- Forced `tool_choice` is rejected on Opus 5.5 / Sonnet 5.5. `llm.structured()` uses
  `method="json_schema"`, and `create_agent(response_format=Model)` picks native
  structured output automatically.
- Thinking is adaptive and always on for these models. Don't send `thinking={"type": "disabled"}`.

## Possible extensions

- `HumanInTheLoopMiddleware(interrupt_on={"run_snowflake_sql": True})` to approve
  expensive SQL before it runs.
- `SummarizationMiddleware` on long threads; `PIIMiddleware` on outputs.
- `cache_policy=CachePolicy(ttl=300)` on `metadata_agent` (metadata rarely changes).
- LangSmith tracing: set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`.

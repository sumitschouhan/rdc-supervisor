# Deep Agents and the 2026 agent stack: research for rdc-supervisor

*Research date: 2026-11-25. Sources:*
- *LangChain's docs source on GitHub (`langchain-ai/docs`)*
- *the deepagents CHANGELOG and README*
- *the installed packages (deepagents 0.7.41, langgraph 1.2.24, langchain 1.5.2)*
- *LangChain blog posts, via search summaries (langchain.com was not reachable from the research environment)*
- *third-party reviews and 2026 papers*

*Vendor claims are marked as such. Everything marked "verified" was run in this repo's tests.*

---

## 1. TL;DR and recommendation

**Use Deep Agents, but not as the front door.** Keep the deterministic LangGraph supervisor
for routing. Add a Deep Agent as an **"analyst mode"** for open-ended, multi-step analytical
questions. That is what `supervisor_template/deep_analyst.py` now does, with tests.

Why the hybrid:

| Question type (RDC) | Best engine | Reason |
|---|---|---|
| "Trades for AAPL last week" | Supervisor → one worker | 1 query. Deep Agents adds planning/subagent overhead (≈4 model calls vs 3, per LangChain's own benchmark) and cost. |
| "Positions + their Morningstar ratings" | Supervisor, `Send` fan-out | Fixed, rule-driven, parallel. Code-enforced metadata-first, entitlements and fallback. |
| "Compare my top holdings' ratings vs last quarter and flag risks" / "Why did exposure to tech rise?" | **Deep analyst** | Unknown number of queries, iteration, intermediate notes, synthesis. This is the use case the harness was built for. |

Deep Agents' own README says: use Deep Agents for the full harness, `create_agent` for a
lighter one, and LangGraph when the loop needs a custom shape. RDC needs all three, and
they compose: a Deep Agent is a LangGraph graph, so it can be a node, and any compiled
graph can be a Deep Agents subagent.

---

## 2. What Deep Agents is (in one paragraph)

`deepagents` is LangChain's open-source **agent harness**. It is `create_agent` plus a
curated middleware stack:
- **filesystem tools** (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `delete`) over pluggable **backends**
- **subagents** through a `task` tool (isolated context windows)
- **skills** (SKILL.md, loaded on demand)
- **memory** (AGENTS.md, always loaded)
- **summarization** and **large-tool-result offloading** to files
- automatic **Anthropic/Bedrock prompt caching**
- **HITL** through `interrupt_on`
- tool-call repair and model-refusal handling
- optionally **sandboxes** (`execute`), a **code interpreter** and **async (remote) subagents**

The design idea is that context engineering happens through the filesystem: offload, write
notes, read back only what's needed.

---

## 3. What shipped in 2026 (timeline)

| Version / date | Highlights |
|---|---|
| **0.5** (Apr 2026) | **Async subagents** (background tasks on any Agent Protocol server, returning a task id immediately). Multimodal `read_file` (PDF, audio, video). Filesystem **permissions**. Structured output for subagents (0.5.3). **Profiles API** (`HarnessProfile`, `ProviderProfile`, 0.5.4). Backend factories deprecated. |
| **0.6** (May 2026, at Interrupt) | Experimental **code interpreter** (`CodeInterpreterMiddleware`, QuickJS). **Harness profiles** for open-weight models. **v3 streaming**. **Delta channels** for checkpoints (vendor claim: up to 100× less checkpoint storage for long runs). **ContextHub** (versioned AGENTS.md and skills in LangSmith). `RubricMiddleware` (self-grading loop). `state_schema` param. Interrupt mode for filesystem permissions. |
| **0.7** (late Jul 2026), breaking | **Leaner harness**: the base system prompt was removed and tool descriptions were cut 43%, for **65% fewer base input tokens** at comparable performance (vendor; it is a per-turn base-token number, not total cost). **TodoListMiddleware / `write_todos` is now opt-in**: LangChain's evals were slightly better and cheaper without it. `virtual_mode=True` is the default for local backends. Recursive `delete` tool. |
| 0.7.12 – 0.7.24 | Subagent **fork mode** (shares the parent's context and prompt cache). Skill-scoped tools (`include_tools`, which load only after SKILL.md is read). `pinned_skills`. Skills hot reload. Blob offload. Fork mode for async subagents. |
| 0.7.26 – 0.7.33 | Async subagents report "waiting for human". `notify_parent` and `webhook` on `AsyncSubAgent`. Queued follow-ups. **Model refusals raise by default and are reported to the parent** (0.7.33). |
| 0.7.36 – **0.7.41** (Nov 20 2026) | Backend passed to middleware. Subagent checkpoint namespace recorded. Bug fixes. |
| **Managed Deep Agents** (hosted) | Unveiled at Interrupt (May), public beta (Aug). Covers user-owned credentials, user-level memory, HTTP channels, built-in web search (v0.8), plus schedules, per-run model/skills/tools config and Slack reactions (v0.9). |
| Related platform (Interrupt 2026) | LangSmith **Context Hub**, **LLM Gateway** (spend limits, PII redaction; beta), LangSmith Sandboxes, LangSmith Engine, **Trajectories** view and trajectory online evaluators (Sep 2026). |

---

## 4. Feature catalogue: value for RDC

Legend: **H** = high value now, **M** = useful later, **L** = low or not applicable. ✅ = in the template (tested).

| Feature | What it gives you | RDC value | Template |
|---|---|---|---|
| **Subagents** (`task`) | Isolated context per specialist. Parallel when several `task` calls are made in one turn. | **H**: mesh/GraphQL analysts keep raw rows out of the planner's context. | ✅ `mesh_analyst`, `graphql_analyst` |
| Subagent `response_format` | The parent receives compact JSON, not prose. | **H**: same `WorkerResult` contract as the graph workers. | ✅ |
| **Skills** (SKILL.md, progressive disclosure) | Only name and description sit in the prompt; the full body is read on demand. | **H**: this is the 2026 standard for your YAML "skills". Custom subagents don't inherit skills. | ✅ rendered from `application.yml`, scoped per subagent |
| Skill-scoped tools (`include_tools`) | A tool appears only after its skill is read. | M: e.g. an `explain_query_plan` tool only inside a "performance" skill. | — |
| **Memory** (AGENTS.md) | Always-loaded instructions. Read-write by the agent. | **H**: org data conventions (read-only) plus per-user preferences. | ✅ `/guides/` (read-only), `/memories/` (per user) |
| **Backends**: State / Store / Composite | Scratch files per thread; durable cross-thread files; routing by path. | **H** | ✅ `CompositeBackend(StateBackend, {"/memories/": StoreBackend(namespace=user)})` |
| **Permissions** (`FilesystemPermission`) | Path rules: allow / deny / interrupt. | **H**: stop prompt-injected writes to policy and skills. | ✅ deny writes to `/skills/**`, `/guides/**` |
| **HITL** `interrupt_on` + `when` predicate | Approve, edit or reject tool calls, only when the predicate fires. | **H**: approve risky SQL (no WHERE/LIMIT) and auto-approve the rest. | ✅ `require_sql_approval` |
| Summarization + tool-result offload | Long analyses don't overflow context. | **H** for analyst mode. | ✅ (built in) |
| Prompt caching (Anthropic) | Automatic with Claude. | **H**: cost. | ✅ (built in) |
| `context_schema` propagation | User and entitlements reach subagents and tools. | **H** | ✅ `AppContext` |
| Model-refusal handling | The parent gets "subagent refused" instead of a crash. | M | ✅ (built in) |
| **Async subagents** | Long jobs in the background on an Agent Server, with notify and webhook. | M: "run the month-end exposure report" while the user keeps chatting. Needs LangGraph/LangSmith Agent Server. | — |
| Fork mode | A subagent shares the parent context and cache. | L–M | — |
| Code interpreter / sandboxes (`execute`) | Run Python/JS over results (pandas-style analysis). | M: calculations over query results. **Never put DB credentials in a sandbox** (docs). Keep Snowflake access in host-side tools. | — |
| `RubricMiddleware` | Self-grading iteration against a rubric. | M: e.g. "answer cites source for every number". | — |
| Harness/Provider profiles | Per-model prompt and tool tweaks. | L (single provider). | — |
| ContextHub backend | Versioned AGENTS.md and skills in LangSmith. | M: lets non-engineers edit skills with review and versioning. | — |
| Managed Deep Agents | Hosted runtime, schedules, Slack. | M: only if you move to LangSmith hosting. | — |
| `TodoListMiddleware` | Planning list. | L: off by default since 0.7 (evals were better without it). | — |

---

## 5. Where Deep Agents fits and where it doesn't (market view)

**Good fits reported in 2026:**
- research and report generation
- coding agents (Deep Agents CLI/Code)
- multi-step data analysis
- long-running ops tasks with checkpoints
- personal and assistant agents with memory and schedules (Managed Deep Agents + Slack)

Text-to-SQL guides build on `create_deep_agent` plus a SQL toolkit (SQLite in the public
examples; no public Snowflake + deepagents example was found). Snowflake's own approach is
Cortex Agents (Cortex Analyst over semantic views).

**Poor fits:**
- simple single-shot lookups (extra model calls and latency)
- strict, auditable routing that must never deviate (put it in code)
- cases where you can't afford "trust the LLM" security

Deep Agents' security model explicitly relies on tool- and sandbox-level boundaries, not on
the model behaving.

**Independent critiques** (treat as opinion; some are from competitors):
- It only pays off when the task really needs a long-running harness.
- Governance (central credentials, budgets, audit, approvals) sits outside the library. You
  must add identity, tenant isolation, tool permissions, budgets and observability yourself.
- Token cost grows with multi-agent fan-out and retries. One estimate puts supervisor
  systems at ~3× a single agent; LangChain's own benchmark: subagents ~9K tokens vs skills
  ~15K on multi-domain tasks, but 4 vs 3 model calls on simple ones.
- The 0.7 leaner harness "was not cheaper on every model". Measure on your own model and
  workload.

---

## 6. Best practices (consolidated)

### Architecture
1. **Start simple; split only when roles are clearly separable.** Subagents for context
   isolation first, skills for progressive disclosure second (LangChain's recommended order).
2. **Keep hard rules in code**: iteration caps, metadata-first, entitlements, fallback. Not
   in prompts. ✅ supervisor graph.
3. **Treat an empty worker result as "no data", not success.** This is cited as the #1
   production bug in naive supervisors. ✅ `has_data` comes from tool artifacts.
4. Avoid over-decomposition. Don't create subagents for trivial single steps.
5. The supervisor should not hold dangerous tools. Worker output can carry prompt injection.
   ✅ the deep main agent has no data tools.

### Subagents
- Specific, **distinct, action-oriented descriptions**. The description is the routing signal.
- **Minimal tool sets** per subagent (3–5 tools max is a common rule of thumb).
- Ask for **concise results, not raw data**. Put large data in files. ✅ `WorkerResult` + "short summary".
- Custom subagents **don't inherit** the parent's tools, middleware or skills. Configure each. ✅
- Runtime context propagates to subagents. Use namespaced keys for per-subagent settings.

### Skills
- `name` must match its directory (lowercase, hyphens). The skills path must be a
  **directory of skill directories**. Last source wins on name collisions.
- Description = **what plus when, with keywords**. Keep sibling descriptions distinct. Prefer
  fewer, well-scoped skills.
- Body under ~5K tokens (spec: under 500 lines). Keep references one level deep. Include
  procedures, decision criteria, examples and edge cases.
- With a checkpointer, skills load once per thread. Reset `skills_metadata=None` to reload.

### Memory
- **Default to user-scoped memory.** Shared memory is a prompt-injection vector. ✅
- Make org policy files read-only through permissions. ✅
- Writes to the same file are last-write-wins. Use per-topic files or background
  consolidation (a cron "consolidation agent" whose lookback matches the cron interval).

### Backends and sandboxes
- Production: `StateBackend` (scratch) + `StoreBackend` (durable) behind `CompositeBackend`.
  **Never `FilesystemBackend`/`LocalShellBackend` in a deployed server.**
- `StoreBackend`: **always set `namespace`** (otherwise all users share files).
- Sandboxes: **no secrets inside**. Prefer a "sandbox as tool" over an "agent in sandbox".
  Use TTLs and cleanup. Treat sandbox output as untrusted.

### HITL
- Always use a checkpointer and the same `thread_id` on resume. Decisions are matched to
  actions in order.
- Tailor by risk: approve/edit/reject for side-effecting or expensive tools, nothing for
  reads. Use a `when` predicate to avoid approval fatigue. ✅
- Rejection messages should say "not executed, don't retry, do X instead".

### Operations
- Durable checkpointer (Postgres). Never `InMemorySaver` in production.
- **Prefer async** tools, middleware hooks and graph calls.
- Pass `thread_id` (conversation) and `context` (per-run identity) on **every** call.
- Trace with LangSmith. Subagents carry `lc_agent_name` metadata. Use
  `stream(subgraphs=True)` to stream subagent progress.
- Use `PIIMiddleware` and an LLM Gateway for redaction and spend limits where required.

### Text-to-SQL on financial data (research, 2026)
- A **semantic layer / governed context** is the biggest accuracy lever. One 2026 paired
  benchmark: Claude Sonnet 4.6 went from 90% to 98% accuracy, and failures became refusals
  instead of confident wrong numbers. That matters most in finance.
- Watch **semantically wrong but executable SQL** (YoY vs QoQ, fiscal calendars, units,
  time alignment). Add explicit checks.
- Use a **validation chain**: deterministic checks (read-only, allowed tables, LIMIT) →
  execute → LLM judge on the question, SQL and preview → regenerate on failure. ✅ the
  read-only/LIMIT guards and the HITL predicate exist; adding the judge is the next step
  (`RubricMiddleware` or a post-tool check).

### Evaluation
- Evaluate at three levels: **single step** (right tool or agent?), **full turn** (final
  answer, trajectory, state) and **multi-turn thread**.
- Build a **routing-accuracy dataset** before adding agents. Use `agentevals` trajectory
  match (strict / unordered / subset) or LLM-as-judge.
- Keep a regression suite that should pass at ~100%. Use LangSmith trajectory online
  evaluators in production.

---

## 7. Recommended target architecture for RDC

```
                    ┌──────────── deterministic LangGraph supervisor ────────────┐
user ─▶ start_turn ─▶ supervisor ─▶ metadata ─▶ router (RouteDecision)            │
                    │      ├─ single/parallel/sequential ─▶ Send(mesh|graphql) ─▶ synthesizer
                    │      ├─ clarification / no data ─▶ human_review (interrupt)  │
                    │      └─ mode="deep" ─▶ deep_analyst (Deep Agent) ───────────▶ END
                    └──────────────────────────────────────────────────────────────┘
deep_analyst = create_deep_agent(
   subagents = [mesh_analyst(skills=/skills/mesh_agent/), graphql_analyst(skills=/skills/graphql_agent/)],
   memory    = [/guides/AGENTS.md (read-only org conventions), /memories/AGENTS.md (per user, Store)],
   backend   = CompositeBackend(StateBackend, {"/memories/": StoreBackend(namespace=user_id)}),
   permissions = deny writes to /skills/** and /guides/**,
   interrupt_on = approve risky SQL (optional), context_schema = AppContext)
```

### Roadmap
1. **Now**: adopt the template patterns (done in `reference/`). Port `app/` to
   `create_agent`, `Command`/`Send`, typed HITL and `context_schema`.
2. **Next**: build a routing-accuracy and answer-correctness eval set (50–100 real RDC
   questions) in LangSmith. Tune the router prompt and `effort`.
3. **Next**: add a SQL validation chain (deterministic checks plus an LLM judge) and a
   semantic-layer document per domain, shipped as skills `references/` files.
4. **Later**:
   - async subagents for long reports, on Agent Server
   - a code interpreter for calculations (no credentials inside)
   - `RubricMiddleware` for "every number cites its source"
   - ContextHub so data owners can edit skills with versioning
5. **Production**: Postgres checkpointer and store, LangSmith tracing, PII middleware or
   LLM Gateway, cost budgets.

---

## 8. Sources

**Primary (LangChain):**
- deepagents CHANGELOG: https://github.com/langchain-ai/deepagents/blob/main/libs/deepagents/CHANGELOG.md
- deepagents README: https://github.com/langchain-ai/deepagents/blob/main/README.md
- Deep Agents docs source: https://github.com/langchain-ai/docs/tree/main/src/oss/deepagents
  (overview, quickstart, customization, subagents, async-subagents, skills, backends,
  memory, human-in-the-loop, profiles, sandboxes, streaming, going-to-production)
- Deep Agents v0.5: https://www.langchain.com/blog/deep-agents-v0-5
- New in Deep Agents v0.6: https://www.langchain.com/blog/deep-agents-0-6
- Deep Agents v0.7: https://www.langchain.com/blog/deep-agents-v0-7
- Building Multi-Agent Applications with Deep Agents: https://www.langchain.com/blog/building-multi-agent-applications-with-deep-agents
- Choosing the Right Multi-Agent Architecture: https://www.langchain.com/blog/choosing-the-right-multi-agent-architecture
- Deep Agents vs LangChain vs LangGraph: https://www.langchain.com/blog/deep-agents-vs-langchain-vs-langgraph
- Context engineering with Deep Agents: https://www.langchain.com/blog/context-engineering-deep-agents-remember
- Introducing Context Hub: https://www.langchain.com/blog/introducing-context-hub
- Managed Deep Agents public beta: https://www.langchain.com/blog/managed-deep-agents-is-now-in-public-beta
- Managed Deep Agents (v0.8): https://www.langchain.com/blog/langsmith-managed-deep-agents-whats-new
- Managed Deep Agents schedules (v0.9): https://www.langchain.com/blog/managed-deep-agents-schedules-per-run-configuration-slack
- August 2026 newsletter: https://www.langchain.com/blog/august-2026-langchain-newsletter
- October 2026 newsletter (Interrupt NYC/London): https://www.langchain.com/blog/october-2026-langchain-newsletter
- LangSmith trajectories: https://www.langchain.com/blog/langsmith-trajectories-tracing
- Agent evals: https://www.langchain.com/resources/agent-evals
- agentevals: https://github.com/langchain-ai/agentevals

**Third-party and research:**
- Deep Agents review (wavect): https://wavect.io/blog/langchain-deep-agents-review/
- Deep Agents vs production reality (TrueFoundry; a competitor): https://www.truefoundry.com/blog/langchain-deep-agents-vs-production-reality
- Gloss, on the 0.7 deletions: https://gloss.run/post/harness-scaffolding-nobody-deletes
- Deep Agents v0.7 analysis: https://apimart.ai/blog/deep-agents-v0-7-cuts-per-turn-input-tokens-65-configurable-harness
- Async subagents: https://refactix.com/ai/deep-agents-async-subagents-orchestrating-long-running-work
- Text-to-SQL with Deep Agents: https://oss.vstorm.co/guides/text-to-sql-with-deepagents
- LangGraph multi-agent patterns and pitfalls: https://dev.to/ayinedjimi-consultants/multi-agent-orchestration-with-langgraph-patterns-and-pitfalls-3k2p
- Supervisor vs swarm: https://dev.to/focused_dot_io/multi-agent-orchestration-in-langgraph-supervisor-vs-swarm-tradeoffs-and-architecture-1b7e
- Semantic layers benchmark: https://arxiv.org/pdf/2604.25149
- Cost-aware NL-to-SQL: https://arxiv.org/pdf/2609.04641
- Atlan on enterprise text-to-SQL: https://atlan.com/know/ai-agent/data-for-ai/text-to-sql-for-enterprise/
- Snowflake + LangGraph multi-agent guide: https://www.snowflake.com/en/developers/guides/build-and-evaluate-agents-with-langgraph-and-snowflake/

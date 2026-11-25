"""Per-run runtime context (LangGraph `context_schema`).

Context is static, per-invocation data that is NOT part of the checkpointed
state: who the user is, what they may access, request ids, feature flags.
Nodes receive it as `runtime: Runtime[AppContext]`; tools receive it as
`runtime: ToolRuntime[AppContext]` -> `runtime.context`.

Pass it on every call:  graph.ainvoke(input, config, context=AppContext(...))
This replaces the old pattern of stuffing user data into `configurable`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AppContext:
    user_id: str = "anonymous"
    # Entitlements: the supervisor never routes to a source the user can't access,
    # and tools double-check (defence in depth).
    allowed_sources: frozenset[str] = field(
        default_factory=lambda: frozenset({"snowflake", "graphql"})
    )
    request_id: str | None = None

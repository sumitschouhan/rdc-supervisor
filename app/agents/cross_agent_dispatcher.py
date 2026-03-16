from __future__ import annotations

import concurrent.futures
from typing import Any

from langchain_anthropic import ChatAnthropic

from app.agents.base import BaseAgent
from app.agents.mesh_agent import MeshAgent
from app.agents.graphql_agent import GraphQLAgent
from app.graph.state import AgentResult, SupervisorState
from app.prompts.loader import PromptLoader
from app.skills.registry import SkillRegistry


class CrossAgentDispatcher(BaseAgent):
    """Runs multiple data agents (mesh + graphql) for questions that need both sources.

    Supports two execution modes:
    - parallel:    Both agents run concurrently (I/O-bound LLM calls via ThreadPoolExecutor).
    - sequential:  Agents run one after another; useful when the second agent needs
                   context (e.g. security IDs) produced by the first.
    """

    def __init__(
        self,
        model: ChatAnthropic,
        prompt_loader: PromptLoader,
        skill_registry: SkillRegistry,
    ):
        self.model = model
        self.prompt_loader = prompt_loader
        self.skill_registry = skill_registry

    def invoke(self, state: SupervisorState) -> dict:
        mode = state.get("execution_mode") or "parallel"
        targets = state.get("cross_agent_targets") or ["mesh_agent", "graphql_agent"]

        agent_factories: dict[str, Any] = {
            "mesh_agent": lambda s: MeshAgent(
                self.model, self.prompt_loader, self.skill_registry
            ).invoke(s),
            "graphql_agent": lambda s: GraphQLAgent(
                self.model, self.prompt_loader, self.skill_registry
            ).invoke(s),
        }

        valid_targets = [t for t in targets if t in agent_factories]

        if mode == "parallel":
            results = self._run_parallel(agent_factories, valid_targets, state)
        else:
            results = self._run_sequential(agent_factories, valid_targets, state)

        all_agent_results: list[AgentResult] = []
        for r in results:
            all_agent_results.extend(r.get("agent_results", []))

        has_any_data = any(r.get("has_data") for r in all_agent_results)

        return {
            "agent_results": all_agent_results,
            "is_complete": has_any_data,
            # Clear cross-agent fields after execution
            "execution_mode": None,
            "cross_agent_targets": None,
            "pending_agents": None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_parallel(
        self,
        factories: dict,
        targets: list[str],
        state: SupervisorState,
    ) -> list[dict]:
        """Execute agents concurrently via threads (LLM calls are I/O-bound)."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as executor:
            future_map = {
                executor.submit(factories[t], state): t for t in targets
            }
            results = []
            for future in concurrent.futures.as_completed(future_map):
                try:
                    results.append(future.result())
                except Exception as exc:
                    agent_name = future_map[future]
                    results.append({
                        "agent_results": [{
                            "agent_name": agent_name,
                            "skill_used": None,
                            "query_generated": "",
                            "data": None,
                            "error": str(exc),
                            "has_data": False,
                        }]
                    })
        return results

    def _run_sequential(
        self,
        factories: dict,
        targets: list[str],
        state: SupervisorState,
    ) -> list[dict]:
        """Execute agents one after another; later agents receive accumulated state."""
        results = []
        # Build a mutable copy of state so later agents see earlier results
        accumulated_results: list[AgentResult] = list(state.get("agent_results") or [])

        for target in targets:
            # Pass updated agent_results so the next agent can reference prior data
            enriched_state = dict(state)
            enriched_state["agent_results"] = accumulated_results
            result = factories[target](enriched_state)
            results.append(result)
            accumulated_results.extend(result.get("agent_results", []))

        return results

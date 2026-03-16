from __future__ import annotations

from app.config import AppConfig


class PromptLoader:
    """Typed access to prompts from application.yml."""

    def __init__(self, config: AppConfig):
        self._prompts = config.prompts

    def supervisor_system(self, skill_catalog: str = "") -> str:
        return self._prompts.supervisor["system"].replace(
            "{skill_catalog}", skill_catalog
        )

    def supervisor_output_format(self) -> str:
        return self._prompts.supervisor["output_format"]

    def metadata_agent_system(self) -> str:
        return self._prompts.metadata_agent["system"]

    def mesh_agent_system(self) -> str:
        return self._prompts.mesh_agent["system"]

    def graphql_agent_system(self) -> str:
        return self._prompts.graphql_agent["system"]

    def human_feedback_classification(self, query: str, needed_fields: str) -> str:
        return self._prompts.human_feedback["classification"].format(
            query=query, needed_fields=needed_fields
        )

    def human_feedback_no_data(self, attempts_summary: str) -> str:
        return self._prompts.human_feedback["no_data"].format(
            attempts_summary=attempts_summary
        )

    def synthesizer_system(self) -> str:
        return self._prompts.synthesizer["system"]

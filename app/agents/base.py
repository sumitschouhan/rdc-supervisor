from __future__ import annotations

from abc import ABC, abstractmethod

from app.graph.state import SupervisorState


class BaseAgent(ABC):
    """Protocol that all agents follow."""

    @abstractmethod
    def invoke(self, state: SupervisorState) -> dict:
        """Process state and return state updates."""
        ...

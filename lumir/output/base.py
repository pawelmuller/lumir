"""Abstract base for output backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from lumir.models import LightCommand


class Output(ABC):
    """Receives LightCommands and sends them somewhere."""

    @abstractmethod
    def send(self, command: LightCommand) -> None:
        """Deliver a single lighting command."""

    def close(self) -> None:
        """Release resources and perform clean-up."""
        pass

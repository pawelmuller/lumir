"""Abstract base for audio sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from lumir.models import AudioChunk


class AudioSource(ABC):
    """Yields AudioChunks in chronological order."""

    @abstractmethod
    def iter_chunks(self) -> Iterator[AudioChunk]:
        """Generate successive audio chunks."""

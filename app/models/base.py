from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.transcript import Transcript


class ASRModel(ABC):
    """Replaceable speech-to-text backend. Pipeline code depends only on this interface."""

    name: str = "base"

    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> Transcript:
        """Return transcript text and word-level timestamps when available."""

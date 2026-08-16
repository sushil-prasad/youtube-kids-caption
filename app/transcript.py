from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Word:
    word: str
    start: float
    end: float
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "confidence": self.confidence,
        }


@dataclass
class Transcript:
    text: str
    words: list[Word] = field(default_factory=list)
    language: str | None = None
    model: str | None = None
    device: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["words"] = [w.to_dict() if isinstance(w, Word) else w for w in self.words]
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transcript:
        words = [
            Word(
                word=item["word"],
                start=float(item["start"]),
                end=float(item["end"]),
                confidence=item.get("confidence"),
            )
            for item in data.get("words") or []
        ]
        return cls(
            text=data.get("text") or "",
            words=words,
            language=data.get("language"),
            model=data.get("model"),
            device=data.get("device"),
        )

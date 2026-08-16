from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Word:
    word: str
    start: float
    end: float
    confidence: float | None = None
    speaker: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "word": self.word,
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "confidence": self.confidence,
        }
        if self.speaker is not None:
            payload["speaker"] = self.speaker
        return payload


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
                speaker=item.get("speaker"),
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


def _word_from_item(item: Any) -> Word:
    if isinstance(item, Word):
        return item
    return Word(
        word=item["word"],
        start=float(item["start"]),
        end=float(item["end"]),
        confidence=item.get("confidence"),
        speaker=item.get("speaker"),
    )


@dataclass
class Caption:
    index: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    speaker: str | None = None
    cps: float = 0.0
    wps: float = 0.0
    reading_status: str = "OK"
    flags: list[str] = field(default_factory=list)
    mean_confidence: float | None = None
    confidence_band: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "text": self.text,
            "lines": [line for line in self.text.split("\n") if line] if self.text else [],
            "words": [w.to_dict() if isinstance(w, Word) else w for w in self.words],
            "speaker": self.speaker,
            "reading_speed": {
                "cps": round(self.cps, 2),
                "wps": round(self.wps, 2),
                "status": self.reading_status,
            },
            "flags": list(self.flags),
            "mean_confidence": self.mean_confidence,
            "confidence_band": self.confidence_band,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Caption:
        reading = data.get("reading_speed") or {}
        return cls(
            index=int(data.get("index") or 0),
            start=float(data["start"]),
            end=float(data["end"]),
            text=data.get("text") or "",
            words=[_word_from_item(item) for item in data.get("words") or []],
            speaker=data.get("speaker"),
            cps=float(reading.get("cps") or data.get("cps") or 0.0),
            wps=float(reading.get("wps") or data.get("wps") or 0.0),
            reading_status=str(reading.get("status") or data.get("reading_status") or "OK"),
            flags=list(data.get("flags") or []),
            mean_confidence=data.get("mean_confidence"),
            confidence_band=data.get("confidence_band"),
        )

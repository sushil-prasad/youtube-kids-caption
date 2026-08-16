from __future__ import annotations

from types import SimpleNamespace

from app.pipeline.timestamps import words_from_aligner, words_from_text
from app.transcript import Transcript


def test_words_from_text_even_spacing() -> None:
    words = words_from_text("Look at the dinosaur", 4.0)
    assert [w.word for w in words] == ["Look", "at", "the", "dinosaur"]
    assert words[0].start == 0.0
    assert words[-1].end == 4.0
    for previous, current in zip(words, words[1:]):
        assert current.start >= previous.end - 1e-6


def test_words_from_aligner_items_object() -> None:
    stamps = SimpleNamespace(
        items=[
            SimpleNamespace(text="Hello", start_time=0.0, end_time=0.4, confidence=0.9),
            SimpleNamespace(text="world", start_time=0.4, end_time=0.8, confidence=0.8),
        ]
    )
    words = words_from_aligner(stamps)
    assert words[0].word == "Hello"
    assert words[0].start == 0.0
    assert words[1].end == 0.8
    assert words[1].confidence == 0.8


def test_transcript_roundtrip() -> None:
    original = Transcript.from_dict(
        {
            "text": "Look at the purple dinosaur!",
            "words": [{"word": "Look", "start": 10.10, "end": 10.42, "confidence": 0.97}],
        }
    )
    cloned = Transcript.from_dict(original.to_dict())
    assert cloned.text == original.text
    assert cloned.words[0].word == "Look"
    assert cloned.words[0].start == 10.1

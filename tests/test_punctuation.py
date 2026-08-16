from __future__ import annotations

from pathlib import Path

from app.pipeline.punctuation import punctuate_job, punctuate_words
from app.transcript import Word


def _words(text: str, start: float = 0.0, step: float = 0.2) -> list[Word]:
    tokens = text.split()
    words: list[Word] = []
    cursor = start
    for token in tokens:
        words.append(Word(token, cursor, cursor + step, 0.9))
        cursor += step
    return words


def test_greeting_and_sentence_punctuation() -> None:
    words = _words("hey guys today we're going to build a castle")
    result = punctuate_words(words)
    text = " ".join(word.word for word in result)
    assert text == "Hey guys! Today we're going to build a castle."
    assert [word.word for word in result][3] == "we're"


def test_does_not_rewrite_word_stems() -> None:
    words = _words("look at the fluffer noodle")
    result = punctuate_words(words)
    stems = [word.word.strip(".!?").lower() for word in result]
    assert stems == ["look", "at", "the", "fluffer", "noodle"]


def test_question_from_pause_boundary() -> None:
    words = [
        Word("look", 0.0, 0.3, 0.9),
        Word("at", 0.3, 0.45, 0.9),
        Word("that", 0.45, 0.7, 0.9),
        Word("isn't", 1.4, 1.7, 0.9),
        Word("he", 1.7, 1.85, 0.9),
        Word("huge", 1.85, 2.2, 0.9),
    ]
    result = punctuate_words(words, pause_gap=0.45)
    text = " ".join(word.word for word in result)
    assert text == "Look at that. Isn't he huge?"


def test_keeps_existing_terminals() -> None:
    words = _words("Look at the purple dinosaur!")
    result = punctuate_words(words)
    assert result[-1].word.endswith("!")
    assert result[-1].word.count("!") == 1


def test_preserves_raw_transcript(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    raw = (
        '{"text": "hey guys today we are here", "words": ['
        '{"word": "hey", "start": 0.0, "end": 0.2, "confidence": 0.9},'
        '{"word": "guys", "start": 0.2, "end": 0.4, "confidence": 0.9},'
        '{"word": "today", "start": 0.4, "end": 0.6, "confidence": 0.9},'
        '{"word": "we", "start": 0.6, "end": 0.7, "confidence": 0.9},'
        '{"word": "are", "start": 0.7, "end": 0.8, "confidence": 0.9},'
        '{"word": "here", "start": 0.8, "end": 1.0, "confidence": 0.9}'
        "]}"
    )
    (job / "raw_transcript.json").write_text(raw, encoding="utf-8")
    (job / "word_timestamps.json").write_text(raw, encoding="utf-8")
    dest = punctuate_job(job)
    assert dest.name == "punctuated_transcript.json"
    assert (job / "raw_transcript.json").read_text(encoding="utf-8") == raw
    punctuated = dest.read_text(encoding="utf-8")
    assert "Hey guys!" in punctuated
    assert "hey guys today" in (job / "raw_transcript.json").read_text(encoding="utf-8")

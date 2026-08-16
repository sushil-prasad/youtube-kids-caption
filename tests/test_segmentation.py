from __future__ import annotations

from app.config import Settings
from app.pipeline.segmentation import (
    apply_reading_speed,
    build_captions,
    caption_from_words,
    wrap_caption,
)
from app.transcript import Caption, Word


def _words(text: str, start: float = 0.0, step: float = 0.25, speaker: str | None = None) -> list[Word]:
    tokens = text.split()
    words: list[Word] = []
    cursor = start
    for token in tokens:
        words.append(Word(token, cursor, cursor + step, 0.95, speaker=speaker))
        cursor += step
    return words


def test_wrap_prefers_giant_purple_dinosaur_split() -> None:
    wrapped = wrap_caption("Look at this giant purple dinosaur!", max_chars=32)
    lines = wrapped.split("\n")
    assert lines[0] == "Look at this giant"
    assert "dinosaur" in lines[1]
    assert not lines[0].endswith("purple")


def test_max_line_length() -> None:
    settings = Settings(max_chars_per_line=20, max_caption_lines=2, pause_gap=10.0, max_cue_duration=30.0)
    words = _words("Look at this giant purple dinosaur!", step=0.4)
    captions = build_captions(words, settings=settings)
    for caption in captions:
        for line in caption.text.split("\n"):
            assert len(line) <= 20


def test_sentence_boundaries_split_cues() -> None:
    words = (
        _words("Look at the purple dinosaur!", start=10.1, step=0.4)
        + _words("Isn't he huge?", start=13.1, step=0.4)
    )
    captions = build_captions(words)
    texts = [caption.text.replace("\n", " ") for caption in captions]
    assert any("dinosaur" in text for text in texts)
    assert any("huge" in text for text in texts)
    assert len(captions) >= 2
    assert captions[0].end <= captions[1].start + 1e-6


def test_speaker_changes_split_cues() -> None:
    mom = _words("Look at that", start=0.0, step=0.5, speaker="1")
    kid = _words("Wow a dinosaur", start=1.6, step=0.5, speaker="2")
    captions = build_captions(mom + kid)
    speakers = [caption.speaker for caption in captions]
    assert speakers[0] == "1"
    assert "2" in speakers
    change_at = next(i for i, speaker in enumerate(speakers) if speaker == "2")
    assert all(speaker == "1" for speaker in speakers[:change_at])


def test_reading_speed_flags_dense_caption() -> None:
    words = _words(
        "Look at this enormous sparkling purple dinosaur dancing around",
        start=0.0,
        step=0.05,
    )
    caption = caption_from_words(words, 1, Settings(max_cps=17.0, max_wps=4.0, max_cue_duration=7.0))
    flagged = apply_reading_speed([caption], settings=Settings(max_cps=17.0, max_wps=4.0, max_cue_duration=7.0))
    assert flagged
    assert any(item.reading_status == "too_fast" or "reading_speed" in item.flags for item in flagged) or len(flagged) > 1


def test_reading_speed_splits_or_extends() -> None:
    first = _words("Look at this giant purple dinosaur and also the sparkling castle nearby", start=0.0, step=0.08)
    second = _words("Wow", start=8.0, step=0.4)
    captions = [
        caption_from_words(first, 1, Settings(max_cps=17.0, max_wps=3.0)),
        caption_from_words(second, 2, Settings(max_cps=17.0, max_wps=3.0)),
    ]
    result = apply_reading_speed(captions, settings=Settings(max_cps=17.0, max_wps=3.0, max_cue_duration=7.0))
    assert result
    # Either split into more cues, extended the first cue, or flagged it.
    denser = captions[0]
    changed = len(result) > len(captions) or result[0].end > denser.end + 0.01
    flagged = any("reading_speed" in item.flags or item.reading_status == "too_fast" for item in result)
    assert changed or flagged

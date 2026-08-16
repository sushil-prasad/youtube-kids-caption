from __future__ import annotations

from pathlib import Path

from app.pipeline.diarization import (
    SpeakerSegment,
    apply_speakers,
    diarization_from_job,
    rename_speakers,
    segments_from_pauses,
)
from app.pipeline.segmentation import build_captions
from app.transcript import Word


def _words(text: str, start: float, step: float = 0.25, speaker: str | None = None) -> list[Word]:
    words: list[Word] = []
    cursor = start
    for token in text.split():
        words.append(Word(token, cursor, cursor + step, 0.95, speaker=speaker))
        cursor += step
    return words


def test_pause_creates_speaker_change() -> None:
    words = _words("Hello there", 0.0, 0.4) + _words("Hi mom", 2.5, 0.4)
    segments = segments_from_pauses(words, gap=0.8)
    assert len(segments) == 2
    assert segments[0].speaker == "1"
    assert segments[1].speaker == "2"
    assert segments[0].label == "Speaker 1"


def test_rename_speakers_uses_creator_map() -> None:
    segments = [
        SpeakerSegment("1", 0.0, 1.0, "Speaker 1"),
        SpeakerSegment("2", 1.5, 2.5, "Speaker 2"),
    ]
    renamed = rename_speakers(segments, {"1": "Mom", "2": "Liam"})
    assert renamed[0].speaker == "Mom"
    assert renamed[1].speaker == "Liam"


def test_speaker_labels_affect_segmentation() -> None:
    mom = _words("Look at that", 0.0, 0.5)
    kid = _words("Wow a dinosaur", 1.6, 0.5)
    labeled = apply_speakers(
        mom + kid,
        [SpeakerSegment("1", 0.0, 1.5), SpeakerSegment("2", 1.6, 3.2)],
    )
    captions = build_captions(labeled)
    speakers = [caption.speaker for caption in captions]
    assert speakers[0] == "1"
    assert "2" in speakers


def test_diarization_job_writes_segments(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    words = _words("Hello there", 0.0, 0.4) + _words("Hi friends", 2.4, 0.4)
    (job / "punctuated_transcript.json").write_text(
        '{"text": "Hello there Hi friends", "words": ['
        + ",".join(
            f'{{"word": "{w.word}", "start": {w.start}, "end": {w.end}, "confidence": 0.9}}' for w in words
        )
        + "]}",
        encoding="utf-8",
    )
    path = diarization_from_job(job)
    assert path.name == "speaker_segments.json"
    assert (job / "annotated_transcript.json").is_file()
    payload = path.read_text(encoding="utf-8")
    assert "Speaker 1" in payload
    assert "segments" in payload

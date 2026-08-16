from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from app.pipeline.sound_events import (
    SoundEvent,
    detect_nonspeech_events,
    filter_events,
    format_sound_event,
    insert_sound_events,
)
from app.transcript import Word


def test_events_use_square_brackets() -> None:
    assert format_sound_event("laughter") == "[laughter]"
    assert format_sound_event("[Music]") == "[music]"
    assert format_sound_event("dog barking") == "[dog barking]"


def test_below_threshold_events_are_dropped() -> None:
    events = [
        SoundEvent("laughter", 1.0, 1.4, 0.79),
        SoundEvent("music", 2.0, 3.0, 0.81),
    ]
    kept = filter_events(events, threshold=0.80)
    assert [event.label for event in kept] == ["[music]"]


def test_events_insert_in_gaps_not_over_speech() -> None:
    words = [
        Word("Hello", 0.0, 0.4, 1.0),
        Word("world", 0.4, 0.8, 1.0),
        Word("again", 2.0, 2.4, 1.0),
    ]
    overlapping = insert_sound_events(words, [SoundEvent("[laughter]", 0.2, 0.5, 0.99)])
    assert [word.word for word in overlapping] == ["Hello", "world", "again"]
    gapped = insert_sound_events(words, [SoundEvent("[laughter]", 1.0, 1.4, 0.99)])
    labels = [word.word for word in gapped]
    assert "[laughter]" in labels
    assert labels.index("[laughter]") == 2


def _write_burst_wav(path: Path) -> Path:
    rate = 16000
    total = int(rate * 1.5)
    burst_start = int(rate * 0.6)
    burst_end = int(rate * 1.0)
    frames = bytearray()
    for index in range(total):
        if burst_start <= index < burst_end:
            sample = int(12000 * math.sin(2 * math.pi * 220 * index / rate))
        else:
            sample = 0
        frames.extend(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))
    return path


def test_detector_does_not_hallucinate_on_silence(tmp_path: Path) -> None:
    path = tmp_path / "silent.wav"
    rate = 16000
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<h", 0) * rate)
    assert detect_nonspeech_events(path, words=[], threshold=0.80) == []


def test_detector_emits_background_noise_for_loud_gap(tmp_path: Path) -> None:
    path = _write_burst_wav(tmp_path / "burst.wav")
    events = detect_nonspeech_events(path, words=[], threshold=0.80)
    assert events
    assert all(event.label == "[background noise]" for event in events)
    assert all(event.confidence > 0.80 for event in events)
    assert not any("laughter" in event.label for event in events)

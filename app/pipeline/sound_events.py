from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from app.config import Settings, load_settings
from app.pipeline.audio import read_wav_mono, write_json
from app.pipeline.timestamps import job_transcript_path, load_transcript, write_word_timestamps
from app.transcript import Transcript, Word

CANONICAL_EVENTS = (
    "laughter",
    "giggles",
    "crying",
    "screaming",
    "applause",
    "music",
    "dog barking",
    "doorbell",
    "background noise",
)


@dataclass
class SoundEvent:
    label: str
    start: float
    end: float
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["label"] = format_sound_event(self.label)
        payload["confidence"] = round(float(self.confidence), 4)
        payload["start"] = round(float(self.start), 3)
        payload["end"] = round(float(self.end), 3)
        return payload


def format_sound_event(label: str) -> str:
    inner = " ".join((label or "").strip().strip("[]").split()).lower()
    return f"[{inner}]" if inner else ""


def filter_events(events: Iterable[SoundEvent], threshold: float) -> list[SoundEvent]:
    """Drop guesses. Never emit an event at or below the confidence threshold."""
    kept: list[SoundEvent] = []
    for event in events:
        if event.confidence > threshold and event.end > event.start and format_sound_event(event.label):
            kept.append(
                SoundEvent(
                    label=format_sound_event(event.label),
                    start=event.start,
                    end=event.end,
                    confidence=min(1.0, float(event.confidence)),
                )
            )
    return kept


def insert_sound_events(words: list[Word], events: list[SoundEvent]) -> list[Word]:
    """Insert [event] tokens into gaps. Do not overlay events on top of speech."""
    if not events:
        return list(words)
    merged = list(words)
    for event in sorted(events, key=lambda item: item.start):
        label = format_sound_event(event.label)
        if not label:
            continue
        overlaps_speech = False
        insert_at = len(merged)
        for index, word in enumerate(merged):
            if word.end <= event.start + 1e-3:
                insert_at = index + 1
                continue
            if word.start >= event.end - 1e-3:
                insert_at = index
                break
            overlaps_speech = True
            break
        if overlaps_speech:
            continue
        token = Word(label, event.start, max(event.end, event.start + 0.2), event.confidence)
        merged.insert(insert_at, token)
    return merged


def detect_nonspeech_events(
    audio_path: str | Path,
    words: list[Word] | None = None,
    threshold: float = 0.80,
) -> list[SoundEvent]:
    """Conservative detector: only [background noise] in energetic non-speech gaps.

    Specific labels such as [laughter] are never guessed from energy alone.
    """
    samples, rate = read_wav_mono(audio_path)
    import numpy as np

    if samples.size == 0:
        return []
    frame = max(1, int(rate * 0.02))
    rms = []
    for index in range(0, samples.size - frame + 1, frame):
        window = samples[index:index + frame]
        rms.append(float(np.sqrt(np.mean(window * window) + 1e-12)))
    energy = np.array(rms, dtype=np.float32)
    if energy.size == 0:
        return []
    speech_floor = max(float(np.median(energy) * 2.5), 0.015)
    speech_peak = float(np.percentile(energy, 90))
    candidates: list[SoundEvent] = []
    start = None
    for i, value in enumerate(energy):
        time = i * frame / float(rate)
        in_speech = False
        if words:
            in_speech = any(word.start - 0.05 <= time <= word.end + 0.05 for word in words)
        noisy = (not in_speech) and value >= speech_floor * 1.8
        if noisy and start is None:
            start = time
        elif not noisy and start is not None:
            end = time
            if end - start >= 0.25:
                peak = float(energy[max(0, int(start * rate / frame)): i + 1].max()) if i > 0 else value
                confidence = min(1.0, 0.45 + 0.55 * (peak / max(speech_peak, 1e-4)))
                candidates.append(SoundEvent("[background noise]", start, end, confidence))
            start = None
    if start is not None:
        end = len(energy) * frame / float(rate)
        if end - start >= 0.25:
            i1 = len(energy)
            i0 = max(0, int(start * rate / frame))
            peak = float(energy[i0:i1].max()) if i1 > i0 else 0.0
            confidence = min(1.0, 0.45 + 0.55 * (peak / max(speech_peak, 1e-4)))
            candidates.append(SoundEvent("[background noise]", start, end, confidence))
    return filter_events(candidates, threshold)


def sound_events_from_job(job_dir: str | Path, settings: Settings | None = None) -> Path:
    settings = settings or load_settings()
    job_dir = Path(job_dir)
    source = job_transcript_path(job_dir, stage="annotate")
    transcript = load_transcript(source)
    audio = job_dir / "audio.wav"
    events: list[SoundEvent] = []
    if audio.is_file():
        try:
            events = detect_nonspeech_events(audio, words=transcript.words, threshold=settings.sound_event_threshold)
        except Exception:
            events = []
    annotated = insert_sound_events(transcript.words, events)
    write_word_timestamps(
        Transcript(
            text=" ".join(word.word for word in annotated),
            words=annotated,
            language=transcript.language,
            model=transcript.model,
            device=transcript.device,
        ),
        job_dir / "annotated_transcript.json",
    )
    dest = job_dir / "sound_events.json"
    write_json(
        dest,
        {
            "enabled": True,
            "threshold": settings.sound_event_threshold,
            "events": [event.to_dict() for event in events],
        },
    )
    return dest


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Optional sound-event detection. Emits [events] only above the confidence threshold."
    )
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)
    print(sound_events_from_job(args.job_dir))


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.config import Settings, load_settings
from app.pipeline.audio import read_json, read_wav_mono, write_json
from app.pipeline.timestamps import job_transcript_path, load_transcript, write_word_timestamps
from app.transcript import Transcript, Word


@dataclass
class SpeakerSegment:
    speaker: str
    start: float
    end: float
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not payload.get("label"):
            payload["label"] = f"Speaker {self.speaker}"
        return payload


def load_speaker_map(job_dir: str | Path | None = None) -> dict[str, str]:
    if not job_dir:
        return {}
    path = Path(job_dir) / "speaker_map.json"
    if not path.is_file():
        return {}
    data = read_json(path)
    if not isinstance(data, dict):
        return {}
    return {str(key): str(value) for key, value in data.items() if str(value).strip()}


def rename_speakers(segments: list[SpeakerSegment], mapping: dict[str, str]) -> list[SpeakerSegment]:
    if not mapping:
        return segments
    renamed: list[SpeakerSegment] = []
    for segment in segments:
        display = mapping.get(segment.speaker) or mapping.get(segment.label)
        label = display or segment.label or f"Speaker {segment.speaker}"
        speaker = display or segment.speaker
        renamed.append(SpeakerSegment(speaker=speaker, start=segment.start, end=segment.end, label=label))
    return renamed


def segments_from_pauses(words: list[Word], gap: float = 0.80) -> list[SpeakerSegment]:
    """Minimum speaker-change detector: a long pause starts a new speaker turn."""
    if not words:
        return []
    segments: list[SpeakerSegment] = []
    speaker_index = 1
    start = words[0].start
    prev_end = words[0].end
    for word in words[1:]:
        if word.start - prev_end >= gap:
            sid = str(speaker_index)
            segments.append(SpeakerSegment(sid, start, prev_end, f"Speaker {sid}"))
            speaker_index = 2 if speaker_index == 1 else 1
            start = word.start
        prev_end = max(prev_end, word.end)
    sid = str(speaker_index)
    segments.append(SpeakerSegment(sid, start, prev_end, f"Speaker {sid}"))
    return segments


def _speech_regions(samples, rate: int, frame_ms: float = 20.0) -> list[tuple[float, float]]:
    import numpy as np

    frame = max(1, int(rate * frame_ms / 1000.0))
    if samples.size < frame:
        duration = samples.size / float(rate or 1)
        return [(0.0, duration)] if duration > 0 else []
    rms = []
    for index in range(0, samples.size - frame + 1, frame):
        window = samples[index:index + frame]
        rms.append(float(np.sqrt(np.mean(window * window) + 1e-12)))
    energy = np.array(rms, dtype=np.float32)
    threshold = max(float(np.median(energy) * 3.0), 0.02)
    speech = energy >= threshold
    regions: list[tuple[float, float]] = []
    start = None
    for i, flag in enumerate(speech):
        time = i * frame / float(rate)
        if flag and start is None:
            start = time
        elif not flag and start is not None:
            end = time
            if end - start >= 0.12:
                regions.append((start, end))
            start = None
    if start is not None:
        end = len(speech) * frame / float(rate)
        if end - start >= 0.12:
            regions.append((start, end))
    return regions


def _cluster_two(features) -> list[int]:
    import numpy as np

    data = np.asarray(features, dtype=np.float32)
    if len(data) < 2:
        return [0] * len(data)
    centers = np.array([data[0], data[-1]], dtype=np.float32)
    labels = np.zeros(len(data), dtype=int)
    for _ in range(8):
        distances = ((data[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        for k in range(2):
            members = data[labels == k]
            if len(members):
                centers[k] = members.mean(axis=0)
    return [int(x) + 1 for x in labels]


def diarize_audio(audio_path: str | Path, gap: float = 0.80) -> list[SpeakerSegment]:
    samples, rate = read_wav_mono(audio_path)
    regions = _speech_regions(samples, rate)
    if not regions:
        return []
    import numpy as np

    features = []
    merged: list[tuple[float, float]] = [regions[0]]
    for start, end in regions[1:]:
        if start - merged[-1][1] < min(0.35, gap):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    for start, end in merged:
        i0 = int(start * rate)
        i1 = max(i0 + 1, int(end * rate))
        chunk = samples[i0:i1]
        rms = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
        zcr = float(np.mean(np.abs(np.diff(np.sign(chunk)))) / 2.0) if chunk.size > 1 else 0.0
        features.append([np.log(rms + 1e-6), zcr])
    labels = _cluster_two(features) if len(merged) >= 2 else [1] * len(merged)
    # Long pauses still force a speaker change even if the cluster id repeats.
    segments: list[SpeakerSegment] = []
    last_end = None
    last_label = None
    for (start, end), label in zip(merged, labels):
        sid = str(label)
        if last_end is not None and start - last_end >= gap and sid == last_label:
            sid = "2" if sid == "1" else "1"
        segments.append(SpeakerSegment(sid, start, end, f"Speaker {sid}"))
        last_end = end
        last_label = sid
    return segments


def apply_speakers(words: list[Word], segments: list[SpeakerSegment]) -> list[Word]:
    if not words:
        return []
    if not segments:
        return [Word(w.word, w.start, w.end, w.confidence, w.speaker) for w in words]
    labeled: list[Word] = []
    for word in words:
        mid = (word.start + word.end) / 2.0
        speaker = word.speaker
        for segment in segments:
            if segment.start - 1e-3 <= mid <= segment.end + 1e-3:
                speaker = segment.speaker
                break
        labeled.append(Word(word.word, word.start, word.end, word.confidence, speaker))
    return labeled


def diarize_words(words: list[Word], settings: Settings | None = None, audio_path: str | Path | None = None) -> list[SpeakerSegment]:
    settings = settings or load_settings()
    gap = settings.speaker_change_gap
    if audio_path and Path(audio_path).is_file():
        try:
            segments = diarize_audio(audio_path, gap=gap)
            if segments:
                return segments
        except Exception:
            pass
    return segments_from_pauses(words, gap=gap)


def diarization_from_job(job_dir: str | Path, settings: Settings | None = None) -> Path:
    settings = settings or load_settings()
    job_dir = Path(job_dir)
    source = job_transcript_path(job_dir, stage="annotate")
    transcript = load_transcript(source)
    audio = job_dir / "audio.wav"
    segments = diarize_words(transcript.words, settings=settings, audio_path=audio if audio.is_file() else None)
    mapping = load_speaker_map(job_dir) if settings.enable_speaker_labels else {}
    segments = rename_speakers(segments, mapping)
    labeled = apply_speakers(transcript.words, segments)
    write_word_timestamps(
        Transcript(text=transcript.text, words=labeled, language=transcript.language, model=transcript.model, device=transcript.device),
        job_dir / "annotated_transcript.json",
    )
    speakers = []
    seen: set[str] = set()
    for segment in segments:
        if segment.speaker in seen:
            continue
        seen.add(segment.speaker)
        speakers.append({"id": segment.speaker, "label": segment.label or f"Speaker {segment.speaker}"})
    dest = job_dir / "speaker_segments.json"
    write_json(
        dest,
        {
            "enabled": True,
            "speakers": speakers,
            "segments": [segment.to_dict() for segment in segments],
        },
    )
    return dest


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Optional speaker-change detection. Not required for the basic pipeline.")
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)
    print(diarization_from_job(args.job_dir))


if __name__ == "__main__":
    main()

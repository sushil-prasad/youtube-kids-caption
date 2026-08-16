from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from app.transcript import Transcript, Word

_TOKEN_RE = re.compile(r"\S+")


def words_from_aligner(stamps: Any) -> list[Word]:
    """Normalize Qwen forced-aligner output into Word objects."""
    items = _iter_stamp_items(stamps)
    words: list[Word] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("word") or "").strip()
            start = item.get("start_time", item.get("start"))
            end = item.get("end_time", item.get("end"))
        else:
            text = str(getattr(item, "text", None) or getattr(item, "word", "") or "").strip()
            start = getattr(item, "start_time", None)
            if start is None:
                start = getattr(item, "start", None)
            end = getattr(item, "end_time", None)
            if end is None:
                end = getattr(item, "end", None)
        if not text or start is None or end is None:
            continue
        confidence = None
        if isinstance(item, dict):
            confidence = item.get("confidence")
        else:
            confidence = getattr(item, "confidence", None)
        words.append(
            Word(
                word=text,
                start=float(start),
                end=float(end),
                confidence=float(confidence) if confidence is not None else 1.0,
            )
        )
    return words


def _iter_stamp_items(stamps: Any) -> Iterable[Any]:
    if stamps is None:
        return []
    if hasattr(stamps, "items") and not isinstance(stamps, dict):
        try:
            items = stamps.items
            if callable(items):
                # dict.items — treat stamps as a sequence instead
                pass
            else:
                return list(items)
        except TypeError:
            pass
    if isinstance(stamps, dict) and "items" in stamps:
        return list(stamps["items"])
    if isinstance(stamps, (list, tuple)):
        return list(stamps)
    return [stamps]


def words_from_text(text: str, duration: float) -> list[Word]:
    """Evenly space tokens across the audio duration when alignment is unavailable."""
    tokens = _TOKEN_RE.findall(text or "")
    if not tokens:
        return []
    duration = max(float(duration), 0.01)
    slot = duration / len(tokens)
    words: list[Word] = []
    for index, token in enumerate(tokens):
        start = round(index * slot, 3)
        end = round(min(duration, (index + 1) * slot), 3)
        if end <= start:
            end = round(start + 0.01, 3)
        words.append(Word(word=token, start=start, end=end, confidence=None))
    return words


def write_word_timestamps(transcript: Transcript, path: str | Path) -> Path:
    path = Path(path)
    payload = {
        "text": transcript.text,
        "words": [word.to_dict() for word in transcript.words],
        "language": transcript.language,
        "model": transcript.model,
        "device": transcript.device,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_transcript(path: str | Path) -> Transcript:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Transcript.from_dict(data)


def build_word_timestamps(job_dir: str | Path) -> Path:
    job_dir = Path(job_dir)
    raw_path = job_dir / "raw_transcript.json"
    existing = job_dir / "word_timestamps.json"
    if not raw_path.is_file():
        if existing.is_file():
            return existing
        raise FileNotFoundError(f"Missing {raw_path}. Run ASR first.")
    transcript = load_transcript(raw_path)
    audio_path = job_dir / "audio.wav"
    if not transcript.words and audio_path.is_file():
        import wave

        with wave.open(str(audio_path), "rb") as handle:
            duration = handle.getnframes() / float(handle.getframerate() or 1)
        transcript.words = words_from_text(transcript.text, duration)
    out = job_dir / "word_timestamps.json"
    write_word_timestamps(transcript, out)
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write word_timestamps.json from raw_transcript.json")
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)
    print(build_word_timestamps(args.job_dir))


if __name__ == "__main__":
    main()

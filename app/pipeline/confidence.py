from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import Settings, load_settings
from app.pipeline.audio import write_json
from app.pipeline.timestamps import load_transcript
from app.transcript import Word


def classify_confidence(score: float, high: float = 0.90, medium: float = 0.70) -> str:
    """Map a 0–1 score onto high / medium / low using configurable thresholds."""
    value = _clamp01(score)
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    return "low"


def _clamp01(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def asr_confidence(word: Word, unknown: float = 0.50) -> float:
    if word.confidence is None:
        return _clamp01(unknown)
    return _clamp01(word.confidence)


def word_confidence_record(
    word: Word,
    settings: Settings | None = None,
    *,
    correction_confidence: float | None = None,
    profanity_risk: float | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    asr = asr_confidence(word, settings.confidence_unknown)
    overall = asr
    if profanity_risk is not None:
        overall = min(overall, 1.0 - _clamp01(profanity_risk) * 0.25)
    if correction_confidence is not None:
        overall = (overall + _clamp01(correction_confidence)) / 2.0
    overall = _clamp01(overall)
    return {
        "word": word.word,
        "start": round(float(word.start), 3),
        "end": round(float(word.end), 3),
        "asr_confidence": round(asr, 4),
        "correction_confidence": correction_confidence,
        "profanity_risk": profanity_risk,
        "overall_confidence": round(overall, 4),
        "band": classify_confidence(overall, settings.confidence_high, settings.confidence_medium),
        "speaker": word.speaker,
    }


def estimate_confidence(words: list[Word], settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    records = [word_confidence_record(word, settings) for word in words]
    bands = {"high": 0, "medium": 0, "low": 0}
    total = 0.0
    for record in records:
        bands[record["band"]] += 1
        total += record["overall_confidence"]
    count = len(records)
    mean = (total / count) if count else 0.0
    return {
        "thresholds": {
            "high": settings.confidence_high,
            "medium": settings.confidence_medium,
            "unknown": settings.confidence_unknown,
        },
        "words": records,
        "summary": {
            "word_count": count,
            "mean_confidence": round(mean, 4),
            "overall_caption_confidence": round(mean, 4),
            "high": bands["high"],
            "medium": bands["medium"],
            "low": bands["low"],
        },
    }


def confidence_from_job(job_dir: str | Path) -> Path:
    job_dir = Path(job_dir)
    timestamps_path = job_dir / "word_timestamps.json"
    if not timestamps_path.is_file():
        raise FileNotFoundError(f"Missing {timestamps_path}. Run timestamps first.")
    transcript = load_transcript(timestamps_path)
    payload = estimate_confidence(transcript.words)
    dest = job_dir / "confidence.json"
    write_json(dest, payload)
    return dest


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Estimate per-word confidence bands.")
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)
    path = confidence_from_job(args.job_dir)
    print(path)


if __name__ == "__main__":
    main()

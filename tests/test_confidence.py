from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.pipeline.confidence import classify_confidence, confidence_from_job, estimate_confidence
from app.transcript import Word


def test_bands_use_inclusive_thresholds() -> None:
    assert classify_confidence(1.0) == "high"
    assert classify_confidence(0.90) == "high"
    assert classify_confidence(0.89) == "medium"
    assert classify_confidence(0.70) == "medium"
    assert classify_confidence(0.69) == "low"
    assert classify_confidence(0.0) == "low"


def test_custom_thresholds() -> None:
    assert classify_confidence(0.80, high=0.85, medium=0.60) == "medium"
    assert classify_confidence(0.85, high=0.85, medium=0.60) == "high"


def test_scores_are_clamped_to_unit_interval() -> None:
    payload = estimate_confidence(
        [Word("wow", 0.0, 0.3, 1.4), Word("nope", 0.3, 0.6, -0.2)],
        Settings(confidence_high=0.90, confidence_medium=0.70, confidence_unknown=0.50),
    )
    scores = [item["overall_confidence"] for item in payload["words"]]
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert payload["words"][0]["band"] == "high"
    assert payload["words"][1]["band"] == "low"


def test_missing_asr_confidence_uses_unknown_default() -> None:
    settings = Settings(confidence_unknown=0.50, confidence_high=0.90, confidence_medium=0.70)
    payload = estimate_confidence([Word("hello", 0.0, 0.4, None)], settings)
    record = payload["words"][0]
    assert record["asr_confidence"] == 0.5
    assert record["correction_confidence"] is None
    assert record["profanity_risk"] is None
    assert record["band"] == "low"


def test_confidence_json_written_for_job(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "word_timestamps.json").write_text(
        '{"text": "Look at that", "words": ['
        '{"word": "Look", "start": 0.0, "end": 0.3, "confidence": 0.97},'
        '{"word": "at", "start": 0.3, "end": 0.4, "confidence": 0.80},'
        '{"word": "that", "start": 0.4, "end": 0.7, "confidence": 0.40}'
        "]}",
        encoding="utf-8",
    )
    path = confidence_from_job(job)
    assert path.name == "confidence.json"
    payload = estimate_confidence(
        [
            Word("Look", 0.0, 0.3, 0.97),
            Word("at", 0.3, 0.4, 0.80),
            Word("that", 0.4, 0.7, 0.40),
        ]
    )
    assert payload["summary"]["high"] == 1
    assert payload["summary"]["medium"] == 1
    assert payload["summary"]["low"] == 1
    assert 0.0 <= payload["summary"]["overall_caption_confidence"] <= 1.0

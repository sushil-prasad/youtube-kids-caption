from __future__ import annotations

from pathlib import Path

import pytest

from app.config import ROOT
from app.transcript import Word
from evaluate import (
    ABLATION_CONFIGS,
    apply_pipeline,
    character_error_rate,
    evaluate_config,
    format_report,
    load_dataset,
    low_confidence_rate,
    run_evaluation,
    safety_rates,
    timestamp_accuracy,
    word_error_rate,
)


DATASET = ROOT / "evaluation_data"


def test_wer_and_cer_on_known_strings() -> None:
    assert word_error_rate("look at bingo", "look at bingo")["wer"] == 0
    wer = word_error_rate("Bingo found a Fluffernoodle", "bingo found a fluffernoddle")
    assert wer["wer"] > 0
    assert character_error_rate("abc", "abc")["cer"] == 0
    assert character_error_rate("cat", "car")["cer"] > 0


def test_timestamp_accuracy_within_tolerance() -> None:
    reference = [Word("hello", 0.0, 0.4), Word("world", 0.4, 0.8)]
    hypothesis = [Word("hello", 0.05, 0.41), Word("world", 0.42, 0.81)]
    result = timestamp_accuracy(reference, hypothesis)
    assert result["pairs_compared"] == 2
    assert result["within_200ms"] == 1.0


def test_allowlist_words_are_not_false_positives() -> None:
    rates = safety_rates(hits_matched=[], gold_profane=[], gold_clean=["bass", "class"])
    assert rates["profanity_false_positive_rate"] == 0
    classroom = next(clip for clip in load_dataset(DATASET) if clip.clip_id == "classroom")
    produced = apply_pipeline(classroom.hypothesis_words, ABLATION_CONFIGS["full_pipeline"])
    scored_hits = produced["hit_terms"]
    fp = safety_rates(scored_hits, classroom.gold_profane, classroom.gold_clean)
    assert fp["false_positives"] == 0


def test_low_confidence_and_reading_speed_on_fixtures() -> None:
    clips = load_dataset(DATASET)
    playground = next(clip for clip in clips if clip.clip_id == "playground")
    low = low_confidence_rate(playground.hypothesis_words)
    assert low["low"] >= 1
    speed = next(clip for clip in clips if clip.clip_id == "speed")
    produced = apply_pipeline(speed.hypothesis_words, ABLATION_CONFIGS["full_pipeline"])
    assert produced["captions"]
    assert any(caption.reading_status != "OK" or "reading_speed" in caption.flags for caption in produced["captions"])


def test_ablation_runs_every_config() -> None:
    clips = load_dataset(DATASET)
    names = list(ABLATION_CONFIGS)
    assert "asr_only" in names
    assert "full_pipeline" in names
    summaries = {name: evaluate_config(clips, name)["summary"] for name in names}
    assert summaries["asr_only"]["wer"] >= summaries["full_pipeline"]["wer"]
    assert summaries["full_pipeline"]["mean_correction_accuracy"] >= summaries["asr_only"]["mean_correction_accuracy"]


def test_evaluate_cli_does_not_call_asr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("evaluation fixtures must not load ASR")

    monkeypatch.setattr("app.models.registry.get_asr_model", forbidden)
    payload = run_evaluation(DATASET, model="pasketti_first", output=tmp_path / "metrics.json")
    report = format_report(payload)
    assert "WER" in report
    assert "not guaranteed safe" in report.lower() or "Creator review" in report
    assert (tmp_path / "metrics.json").is_file()


def test_ablation_report_lists_all_configs(tmp_path: Path) -> None:
    payload = run_evaluation(
        DATASET,
        model="pasketti_first",
        ablation=True,
        output=tmp_path / "ablation.json",
    )
    configs = [item["config"] for item in payload["reports"]]
    assert configs == list(ABLATION_CONFIGS)
    text = format_report(payload)
    assert "asr_only" in text
    assert "full_pipeline" in text

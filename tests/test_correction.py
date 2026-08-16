from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.pipeline.correction import (
    CENSOR_MARK,
    correct_words,
    correction_from_job,
    decide_safety,
    phonetic_similarity,
)
from app.pipeline.profanity import SafetyHit, SafetyLexicon
from app.transcript import Word


def _hit(
    word: str,
    *,
    confidence: float | None = 0.4,
    severity: str = "strong",
    allowlisted: bool = False,
    before: list[str] | None = None,
    after: list[str] | None = None,
    matched: str | None = None,
) -> SafetyHit:
    return SafetyHit(
        index=4,
        end_index=5,
        word=word,
        start=1.0,
        end=1.3,
        match_type="word",
        matched=matched or word.lower().strip(".,!?"),
        severity=severity,
        allowlisted=allowlisted,
        asr_confidence=confidence,
        context_before=before or ["That", "was", "a", "really"],
        context_after=after or ["idea"],
    )


def _settings(**kwargs: object) -> Settings:
    values = {
        "safety_mode": "standard",
        "unknown_profanity": "flag",
        "correction_threshold": 0.65,
        "confidence_high": 0.90,
        "confidence_medium": 0.70,
        "context_window": 5,
    }
    values.update(kwargs)
    return Settings(**values)  # type: ignore[arg-type]


def test_case_a_allowlisted_word_is_kept() -> None:
    decision = decide_safety(_hit("badword", allowlisted=True), None, 0.0, _settings())
    assert decision.case == "A"
    assert decision.action == "keep"
    assert decision.replacement == "badword"


def test_case_b_replaces_likely_asr_error() -> None:
    lexicon = SafetyLexicon(
        words={"badword": "strong"},
        confusions={"badword": ["bad"]},
        collocations={"really|idea": ["bad", "good", "great"]},
    )
    words = [
        Word("That", 0.0, 0.2, 0.95),
        Word("was", 0.2, 0.35, 0.95),
        Word("a", 0.35, 0.4, 0.95),
        Word("really", 0.4, 0.7, 0.95),
        Word("badword", 0.7, 1.0, 0.40),
        Word("idea", 1.0, 1.4, 0.95),
    ]
    corrected, decisions, _hits = correct_words(words, settings=_settings(safety_mode="strict"), lexicon=lexicon)
    assert any(item.case == "B" and item.action == "replace" for item in decisions)
    assert [word.word for word in corrected][4].lower().startswith("bad")


def test_case_c_censors_unresolved_strong_word() -> None:
    lexicon = SafetyLexicon(words={"badword": "strong"})
    words = [Word("badword", 0.0, 0.4, 0.99)]
    _corrected, decisions, _hits = correct_words(
        words,
        settings=_settings(safety_mode="strict", unknown_profanity="censor"),
        lexicon=lexicon,
    )
    assert decisions[0].case == "C"
    assert decisions[0].action == "censor"
    assert decisions[0].replacement == CENSOR_MARK


def test_case_d_flags_uncertain_mild_word() -> None:
    lexicon = SafetyLexicon(words={"dang": "mild"})
    words = [Word("dang", 0.0, 0.3, 0.8)]
    _corrected, decisions, _hits = correct_words(
        words,
        settings=_settings(safety_mode="standard", unknown_profanity="flag"),
        lexicon=lexicon,
    )
    assert decisions[0].case == "D"
    assert decisions[0].action == "flag"
    assert decisions[0].replacement == "dang"
    assert decisions[0].needs_review is True


def test_review_mode_never_censors() -> None:
    lexicon = SafetyLexicon(words={"badword": "strong"})
    words = [Word("badword", 0.0, 0.4, 0.99)]
    corrected, decisions, _hits = correct_words(
        words,
        settings=_settings(safety_mode="review-only"),
        lexicon=lexicon,
    )
    assert all(item.action != "censor" for item in decisions)
    assert corrected[0].word == "badword"
    assert decisions[0].needs_review is True


def test_high_asr_confidence_does_not_silently_replace() -> None:
    decision = decide_safety(
        _hit("badword", confidence=0.97),
        "bad",
        0.70,
        _settings(safety_mode="strict", correction_threshold=0.65),
    )
    assert decision.action != "replace"
    assert decision.case == "D"


def test_phonetic_similarity_ranks_near_misses() -> None:
    assert phonetic_similarity("shit", "ship") > phonetic_similarity("shit", "banana")


def test_correction_job_keeps_raw_transcript(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    raw = (
        '{"text": "That was a really badword idea", "words": ['
        '{"word": "That", "start": 0.0, "end": 0.2, "confidence": 0.9},'
        '{"word": "was", "start": 0.2, "end": 0.4, "confidence": 0.9},'
        '{"word": "a", "start": 0.4, "end": 0.5, "confidence": 0.9},'
        '{"word": "really", "start": 0.5, "end": 0.8, "confidence": 0.9},'
        '{"word": "badword", "start": 0.8, "end": 1.1, "confidence": 0.4},'
        '{"word": "idea", "start": 1.1, "end": 1.5, "confidence": 0.9}'
        "]}"
    )
    (job / "raw_transcript.json").write_text(raw, encoding="utf-8")
    (job / "punctuated_transcript.json").write_text(raw, encoding="utf-8")
    (job / "word_timestamps.json").write_text(raw, encoding="utf-8")
    path = correction_from_job(job, safety_mode="review-only")
    assert path.name == "corrected_transcript.json"
    assert (job / "raw_transcript.json").read_text(encoding="utf-8") == raw
    assert (job / "safety_analysis.json").is_file()
    assert (job / "correction_log.json").is_file()
    log = (job / "correction_log.json").read_text(encoding="utf-8")
    assert "not guaranteed safe" in log.lower() or "Creator review" in log

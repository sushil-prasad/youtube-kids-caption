from __future__ import annotations

from pathlib import Path

from app.pipeline.srt import (
    format_srt_timestamp,
    parse_srt,
    render_srt,
    srt_from_job,
    validate_captions,
    wrap_caption,
    write_srt,
)
from app.transcript import Caption, Word


def test_format_srt_timestamp() -> None:
    assert format_srt_timestamp(0) == "00:00:00,000"
    assert format_srt_timestamp(10.1) == "00:00:10,100"
    assert format_srt_timestamp(12.9) == "00:00:12,900"
    assert format_srt_timestamp(3723.456) == "01:02:03,456"
    assert format_srt_timestamp(-1) == "00:00:00,000"


def test_render_srt_numbering_and_utf8(tmp_path: Path) -> None:
    words = [
        Word("Look", 10.10, 10.42, 0.97),
        Word("at", 10.42, 10.55, 0.99),
        Word("the", 10.55, 10.70, 0.99),
        Word("purple", 10.70, 11.10, 0.95),
        Word("dinosaur!", 11.10, 12.90, 0.94),
        Word("Isn't", 13.10, 13.40, 0.93),
        Word("he", 13.40, 13.55, 0.99),
        Word("huge?", 13.55, 15.20, 0.96),
    ]
    text = render_srt(words)
    assert text.startswith("1\n")
    assert "00:00:10,100 --> " in text
    assert "dinosaur" in text
    path = write_srt(words, tmp_path / "final.srt")
    raw = path.read_bytes()
    raw.decode("utf-8")
    assert raw.endswith(b"\n")
    cues = [block for block in text.strip().split("\n\n") if block.strip()]
    assert cues[0].splitlines()[0] == "1"
    assert "-->" in cues[0]


def test_wrap_prefers_phrase_boundary() -> None:
    wrapped = wrap_caption("Look at this giant purple dinosaur!", max_chars=20)
    assert wrapped.split("\n")[0] == "Look at this giant"
    assert "dinosaur" in wrapped.split("\n")[1]


def test_empty_words_make_empty_srt() -> None:
    assert render_srt([]) == ""


def test_skip_asr_artifacts_write_srt(tmp_path: Path) -> None:
    from app.pipeline.orchestrator import run_pipeline

    job = tmp_path / "job"
    job.mkdir()
    (job / "word_timestamps.json").write_text(
        '{"text": "Hello world", "words": ['
        '{"word": "Hello", "start": 0.0, "end": 0.4, "confidence": 1.0},'
        '{"word": "world", "start": 0.4, "end": 0.8, "confidence": 1.0}'
        "]}",
        encoding="utf-8",
    )
    srt_path = run_pipeline(skip_asr=True, job_dir=job, output=tmp_path / "hello.srt")
    text = Path(srt_path).read_text(encoding="utf-8")
    assert "Hello world" in text
    assert "00:00:00,000 --> " in text
    assert (job / "final.srt").is_file()
    assert (job / "confidence.json").is_file()
    assert (job / "punctuated_transcript.json").is_file()
    assert (job / "final_captions.json").is_file()
    assert (job / "timestamp_validation.json").is_file()
    assert (job / "safety_analysis.json").is_file()
    assert (job / "corrected_transcript.json").is_file()
    raw = job / "raw_transcript.json"
    if raw.is_file():
        raise AssertionError("skip-asr should not invent a raw transcript overwrite")


def test_validate_repairs_overlap_and_writes_report(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "final_captions.json").write_text(
        '{"captions": ['
        '{"index": 1, "start": 0.0, "end": 2.0, "text": "Hello there", "words": []},'
        '{"index": 2, "start": 1.5, "end": 3.0, "text": "How are you", "words": []}'
        "]}",
        encoding="utf-8",
    )
    path = srt_from_job(job, validate=True)
    text = path.read_text(encoding="utf-8")
    cues = parse_srt(text)
    assert cues[1].start >= cues[0].end - 1e-9
    report = (job / "timestamp_validation.json").read_text(encoding="utf-8")
    assert "overlap" in report


def test_validate_clamps_to_video_duration(tmp_path: Path) -> None:
    captions = [
        Caption(1, 0.0, 1.0, "Hi"),
        Caption(2, 1.2, 9.0, "This runs past the end"),
    ]
    repaired, issues = validate_captions(captions, video_duration=2.0)
    assert repaired[-1].end <= 2.0 + 1e-9
    assert any(issue.type == "outside_video" for issue in issues)

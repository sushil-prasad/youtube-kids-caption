from __future__ import annotations

from pathlib import Path

from app.pipeline.srt import format_srt_timestamp, render_srt, wrap_caption, write_srt
from app.transcript import Word


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
    assert "00:00:00,000 --> 00:00:00,800" in text
    assert (job / "final.srt").is_file()

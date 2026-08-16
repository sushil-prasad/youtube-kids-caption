from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from app.config import Settings, load_settings
from app.pipeline.audio import read_json, write_json
from app.pipeline.segmentation import (
    MAX_CHARS_PER_LINE,
    MAX_CUE_DURATION,
    MAX_LINES,
    PAUSE_GAP,
    build_captions,
    load_captions,
    wrap_caption,
    write_captions,
)
from app.pipeline.timestamps import job_transcript_path, load_transcript
from app.transcript import Caption, Word

__all__ = [
    "MAX_CHARS_PER_LINE",
    "MAX_CUE_DURATION",
    "MAX_LINES",
    "PAUSE_GAP",
    "format_srt_timestamp",
    "parse_srt_timestamp",
    "wrap_caption",
    "segment_words",
    "render_srt",
    "render_captions",
    "write_srt",
    "validate_captions",
    "srt_from_job",
]

_TS_RE = re.compile(
    r"(-?\d+):(\d{2}):(\d{2})[,.](\d{1,3})"
)
_ARROW_RE = re.compile(r"\s*-->\s*")


def format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_srt_timestamp(value: str) -> float:
    match = _TS_RE.search(value.strip())
    if not match:
        raise ValueError(f"Malformed SRT timestamp: {value!r}")
    hours, minutes, seconds, millis = (int(part) for part in match.groups())
    if len(match.group(4)) == 1:
        millis *= 100
    elif len(match.group(4)) == 2:
        millis *= 10
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def segment_words(
    words: list[Word],
    max_chars: int = MAX_CHARS_PER_LINE,
    max_duration: float = MAX_CUE_DURATION,
    pause_gap: float = PAUSE_GAP,
) -> list[list[Word]]:
    from app.pipeline.segmentation import segment_words as _segment

    return _segment(words, max_chars=max_chars, max_duration=max_duration, pause_gap=pause_gap)


@dataclass
class TimestampIssue:
    index: int
    type: str
    repaired: bool
    detail: str

    def to_dict(self) -> dict[str, str | int | bool]:
        return asdict(self)


def validate_captions(
    captions: list[Caption],
    video_duration: float | None = None,
    settings: Settings | None = None,
) -> tuple[list[Caption], list[TimestampIssue]]:
    """Repair simple timestamp errors and flag the rest."""
    settings = settings or load_settings()
    issues: list[TimestampIssue] = []
    repaired: list[Caption] = []

    for caption in captions:
        text = caption.text.strip()
        if not text:
            issues.append(TimestampIssue(caption.index, "empty", True, "Dropped empty caption"))
            continue

        start = float(caption.start)
        end = float(caption.end)
        flags = list(caption.flags)

        if start < 0 or end < 0:
            issues.append(
                TimestampIssue(caption.index, "negative", True, f"Clamped negative timestamp ({start:.3f}/{end:.3f})")
            )
            start = max(start, 0.0)
            end = max(end, 0.0)
            flags.append("timestamp_repaired")

        if end < start:
            issues.append(
                TimestampIssue(caption.index, "non_increasing", True, f"Swapped inverted times ({start:.3f}>{end:.3f})")
            )
            start, end = end, start
            if end < start:
                end = start
            flags.append("timestamp_repaired")

        duration = end - start
        if duration < settings.min_cue_duration:
            issues.append(
                TimestampIssue(
                    caption.index,
                    "short_duration",
                    True,
                    f"Extended cue shorter than {settings.min_cue_duration:.2f}s",
                )
            )
            end = start + settings.min_cue_duration
            flags.append("timestamp_repaired")

        if end - start > settings.max_cue_duration + 1e-6:
            issues.append(
                TimestampIssue(
                    caption.index,
                    "long_duration",
                    False,
                    f"Cue longer than {settings.max_cue_duration:.2f}s",
                )
            )
            flags.append("needs_review")

        repaired.append(
            Caption(
                index=caption.index,
                start=start,
                end=end,
                text=caption.text,
                words=caption.words,
                speaker=caption.speaker,
                cps=caption.cps,
                wps=caption.wps,
                reading_status=caption.reading_status,
                flags=_unique(flags),
                mean_confidence=caption.mean_confidence,
                confidence_band=caption.confidence_band,
            )
        )

    sequential: list[Caption] = []
    for caption in repaired:
        start, end = caption.start, caption.end
        flags = list(caption.flags)
        if sequential:
            prev = sequential[-1]
            if start < prev.end - 1e-6:
                issues.append(
                    TimestampIssue(
                        caption.index,
                        "overlap",
                        True,
                        f"Moved start from {start:.3f}s to {prev.end:.3f}s",
                    )
                )
                start = prev.end
                flags.append("timestamp_repaired")
            gap = start - prev.end
            if 0 < gap < settings.min_caption_gap:
                issues.append(
                    TimestampIssue(
                        caption.index,
                        "gap",
                        True,
                        f"Gap {gap:.3f}s below minimum {settings.min_caption_gap:.2f}s",
                    )
                )
                start = prev.end
                flags.append("timestamp_repaired")
        if end <= start:
            end = start + settings.min_cue_duration
            issues.append(
                TimestampIssue(caption.index, "non_increasing", True, "Extended end so the cue is increasing")
            )
            flags.append("timestamp_repaired")
        sequential.append(
            Caption(
                index=caption.index,
                start=start,
                end=end,
                text=caption.text,
                words=caption.words,
                speaker=caption.speaker,
                cps=caption.cps,
                wps=caption.wps,
                reading_status=caption.reading_status,
                flags=_unique(flags),
                mean_confidence=caption.mean_confidence,
                confidence_band=caption.confidence_band,
            )
        )

    clamped: list[Caption] = []
    for caption in sequential:
        start, end = caption.start, caption.end
        flags = list(caption.flags)
        if video_duration is not None:
            if start >= video_duration:
                issues.append(
                    TimestampIssue(caption.index, "outside_video", True, "Dropped caption starting after video end")
                )
                continue
            if end > video_duration + 1e-6:
                issues.append(
                    TimestampIssue(
                        caption.index,
                        "outside_video",
                        True,
                        f"Clamped end {end:.3f}s to video duration {video_duration:.3f}s",
                    )
                )
                end = video_duration
                flags.append("timestamp_repaired")
                if end <= start:
                    issues.append(
                        TimestampIssue(caption.index, "outside_video", False, "Caption collapsed at video end")
                    )
                    flags.append("needs_review")
                    continue
        clamped.append(
            Caption(
                index=caption.index,
                start=start,
                end=end,
                text=caption.text,
                words=caption.words,
                speaker=caption.speaker,
                cps=caption.cps,
                wps=caption.wps,
                reading_status=caption.reading_status,
                flags=_unique(flags),
                mean_confidence=caption.mean_confidence,
                confidence_band=caption.confidence_band,
            )
        )

    for index, caption in enumerate(clamped, start=1):
        caption.index = index
    return clamped, issues


def _unique(flags: list[str]) -> list[str]:
    seen: list[str] = []
    for flag in flags:
        if flag not in seen:
            seen.append(flag)
    return seen


def render_captions(captions: list[Caption]) -> str:
    blocks: list[str] = []
    for caption in captions:
        text = caption.text.strip()
        if not text:
            continue
        start = caption.start
        end = max(caption.end, start)
        blocks.append(
            f"{len(blocks) + 1}\n{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n{text}\n"
        )
    return "\n".join(blocks).rstrip() + ("\n" if blocks else "")


def parse_srt(text: str) -> list[Caption]:
    captions: list[Caption] = []
    chunks = re.split(r"\n\s*\n", text.strip()) if text.strip() else []
    for chunk in chunks:
        lines = chunk.splitlines()
        if len(lines) < 2:
            continue
        idx = 1
        time_line = lines[0]
        body_start = 1
        if lines[0].strip().isdigit() and len(lines) >= 3:
            idx = int(lines[0].strip())
            time_line = lines[1]
            body_start = 2
        if "-->" not in time_line:
            continue
        left, right = _ARROW_RE.split(time_line.strip(), maxsplit=1)
        start = parse_srt_timestamp(left)
        end = parse_srt_timestamp(right.split()[0])
        body = "\n".join(lines[body_start:]).strip()
        captions.append(Caption(index=idx, start=start, end=end, text=body))
    return captions


def render_srt(words: list[Word]) -> str:
    captions = build_captions(words)
    captions, _issues = validate_captions(captions)
    return render_captions(captions)


def write_srt(words: list[Word], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_srt(words), encoding="utf-8")
    return path


def _job_duration(job_dir: Path) -> float | None:
    job_path = job_dir / "job.json"
    if not job_path.is_file():
        return None
    duration = read_json(job_path).get("duration")
    try:
        value = float(duration)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def srt_from_job(
    job_dir: str | Path,
    output: str | Path | None = None,
    validate: bool = True,
) -> Path:
    job_dir = Path(job_dir)
    captions_path = job_dir / "final_captions.json"
    if captions_path.is_file():
        captions = load_captions(captions_path)
    else:
        source = job_transcript_path(job_dir, stage="segment")
        transcript = load_transcript(source)
        captions = build_captions(transcript.words, video_duration=_job_duration(job_dir))

    issues: list[TimestampIssue] = []
    if validate:
        captions, issues = validate_captions(captions, video_duration=_job_duration(job_dir))
        write_json(
            job_dir / "timestamp_validation.json",
            {
                "repaired": any(issue.repaired for issue in issues),
                "issue_count": len(issues),
                "issues": [issue.to_dict() for issue in issues],
            },
        )
        write_captions(job_dir / "final_captions.json", captions)

    dest = Path(output) if output else job_dir / "final.srt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render_captions(captions), encoding="utf-8")
    return dest


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate UTF-8 SRT and optionally validate timestamps.")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--validate", action="store_true", help="Repair/flag invalid or overlapping cues")
    args = parser.parse_args(argv)
    path = srt_from_job(args.job_dir, output=args.output, validate=True)
    print(path)
    validation = Path(args.job_dir) / "timestamp_validation.json"
    if args.validate and validation.is_file():
        print(validation)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from app.config import Settings, load_settings
from app.pipeline.audio import read_json, write_json
from app.pipeline.confidence import classify_confidence
from app.pipeline.timestamps import job_transcript_path, load_transcript
from app.transcript import Caption, Word

MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CUE_DURATION = 7.0
PAUSE_GAP = 0.6

WEAK_LINE_ENDINGS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "our",
    "their",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "and",
    "or",
    "but",
    "with",
    "from",
}

_TERMINALS = ".!?"
_SPLIT_CONJUNCTIONS = {"and", "but", "so", "then"}


def wrap_caption(text: str, max_chars: int = MAX_CHARS_PER_LINE, max_lines: int = MAX_LINES) -> str:
    """Wrap a cue into at most two lines, preferring phrase boundaries over orphans."""
    text = " ".join(text.split())
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    if max_lines < 2:
        return text[:max_chars].rstrip()

    words = text.split()
    best: tuple[float, str, str] | None = None
    for index in range(1, len(words)):
        first = " ".join(words[:index])
        second = " ".join(words[index:])
        if len(first) > max_chars:
            break
        if len(second) > max_chars:
            continue
        last = words[index - 1].rstrip(".!?,;:").lower()
        score = -abs(len(first) - len(second)) * 0.5
        if last in WEAK_LINE_ENDINGS:
            score -= 20.0
        if len(words) - index == 1 and len(first) > max_chars * 0.55:
            score -= 12.0
        if len(first) < max_chars * 0.35:
            score -= 6.0
        if best is None or score > best[0]:
            best = (score, first, second)
    if best is not None:
        return f"{best[1]}\n{best[2]}"

    split_at = text.rfind(" ", 0, max_chars + 1)
    if split_at <= 0:
        split_at = min(max_chars, len(text))
        first, rest = text[:split_at], text[split_at:].lstrip()
    else:
        first, rest = text[:split_at], text[split_at + 1 :]
    if len(rest) > max_chars:
        rest = rest[:max_chars].rstrip()
    return f"{first}\n{rest}".strip()


def compute_reading_speed(text: str, duration: float) -> tuple[float, float]:
    duration = max(float(duration), 0.001)
    compact = " ".join(text.split())
    if not compact:
        return 0.0, 0.0
    return len(compact) / duration, len(compact.split()) / duration


def _has_terminal(token: str) -> bool:
    stripped = token.rstrip("\"'”’")
    return bool(stripped) and stripped[-1] in _TERMINALS


def _speaker_of(word: Word) -> str | None:
    return word.speaker


def _group_speaker(words: list[Word]) -> str | None:
    counts: dict[str, int] = {}
    for word in words:
        if not word.speaker:
            continue
        counts[word.speaker] = counts.get(word.speaker, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def _plain_text(words: Iterable[Word]) -> str:
    return " ".join(word.word for word in words)


def segment_words(
    words: list[Word],
    max_chars: int = MAX_CHARS_PER_LINE,
    max_duration: float = MAX_CUE_DURATION,
    pause_gap: float = PAUSE_GAP,
    max_lines: int = MAX_LINES,
) -> list[list[Word]]:
    """Group words into cue-sized chunks using pauses, speakers, sentences, and length."""
    cues: list[list[Word]] = []
    current: list[Word] = []
    limit = max_chars * max_lines

    def flush() -> None:
        if current:
            cues.append(list(current))
            current.clear()

    for word in words:
        if not current:
            current.append(word)
            continue
        prev = current[-1]
        gap = word.start - prev.end
        duration = word.end - current[0].start
        text_len = len(_plain_text(current + [word]))
        speaker_change = bool(_speaker_of(word) and _speaker_of(prev) and _speaker_of(word) != _speaker_of(prev))
        sentence_break = _has_terminal(prev.word)
        if speaker_change or gap >= pause_gap or duration > max_duration or text_len > limit or sentence_break:
            flush()
        current.append(word)
    flush()
    return cues


def _merge_short_cues(groups: list[list[Word]], settings: Settings) -> list[list[Word]]:
    if not groups:
        return []
    merged: list[list[Word]] = [list(groups[0])]
    for group in groups[1:]:
        previous = merged[-1]
        prev_dur = previous[-1].end - previous[0].start
        combined = previous + group
        combined_text = _plain_text(combined)
        combined_dur = group[-1].end - previous[0].start
        same_speaker = _group_speaker(previous) == _group_speaker(group) or not (
            _group_speaker(previous) and _group_speaker(group)
        )
        if (
            prev_dur < settings.min_cue_duration
            and same_speaker
            and combined_dur <= settings.max_cue_duration
            and len(combined_text) <= settings.max_chars_per_line * settings.max_caption_lines
        ):
            merged[-1] = combined
        else:
            merged.append(list(group))
    return merged


def _mean_confidence(words: list[Word]) -> float | None:
    scores = [float(word.confidence) for word in words if word.confidence is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def caption_from_words(
    words: list[Word],
    index: int,
    settings: Settings | None = None,
    extra_flags: Iterable[str] | None = None,
) -> Caption:
    settings = settings or load_settings()
    text = wrap_caption(
        _plain_text(words),
        max_chars=settings.max_chars_per_line,
        max_lines=settings.max_caption_lines,
    )
    start = words[0].start if words else 0.0
    end = words[-1].end if words else 0.0
    if end < start:
        end = start
    duration = max(end - start, 0.0)
    cps, wps = compute_reading_speed(text, duration)
    too_fast = cps > settings.max_cps or wps > settings.max_wps
    flags = list(extra_flags or [])
    status = "too_fast" if too_fast else "OK"
    if too_fast:
        flags.append("reading_speed")
    mean = _mean_confidence(words)
    band = None
    if mean is not None:
        band = classify_confidence(mean, settings.confidence_high, settings.confidence_medium)
        if band == "low":
            flags.append("low_confidence")
    return Caption(
        index=index,
        start=start,
        end=end,
        text=text,
        words=list(words),
        speaker=_group_speaker(words),
        cps=cps,
        wps=wps,
        reading_status=status,
        flags=_unique_flags(flags),
        mean_confidence=None if mean is None else round(mean, 4),
        confidence_band=band,
    )


def _unique_flags(flags: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for flag in flags:
        if flag not in seen:
            seen.append(flag)
    return seen


def _split_dense_words(words: list[Word]) -> list[list[Word]] | None:
    if len(words) < 2:
        return None
    for index in range(1, len(words)):
        if _has_terminal(words[index - 1].word):
            left, right = words[:index], words[index:]
            if left and right:
                return [left, right]
    for index in range(1, len(words)):
        if words[index - 1].word.endswith(",") and index < len(words):
            left, right = words[:index], words[index:]
            if left and right:
                return [left, right]
    for index in range(1, len(words) - 1):
        if words[index].word.lower().strip(".,!?") in _SPLIT_CONJUNCTIONS:
            left, right = words[:index], words[index:]
            if left and right:
                return [left, right]
    if len(words) >= 6:
        mid = len(words) // 2
        return [words[:mid], words[mid:]]
    return None


def _is_dense(caption: Caption, settings: Settings) -> bool:
    cps, wps = compute_reading_speed(caption.text, caption.end - caption.start)
    return cps > settings.max_cps or wps > settings.max_wps


def _extend_caption(
    caption: Caption,
    settings: Settings,
    next_start: float | None,
    video_duration: float | None,
) -> Caption:
    chars = len(" ".join(caption.text.split()))
    needed = chars / max(settings.max_cps, 0.1)
    target_end = caption.start + needed
    limit = caption.start + settings.max_cue_duration
    if next_start is not None:
        limit = min(limit, next_start - settings.min_caption_gap)
    if video_duration is not None:
        limit = min(limit, video_duration)
    new_end = min(max(caption.end, target_end), limit)
    if new_end <= caption.end + 0.001:
        return caption
    cps, wps = compute_reading_speed(caption.text, new_end - caption.start)
    flags = list(caption.flags)
    flags.append("timing_extended")
    too_fast = cps > settings.max_cps or wps > settings.max_wps
    return Caption(
        index=caption.index,
        start=caption.start,
        end=new_end,
        text=caption.text,
        words=caption.words,
        speaker=caption.speaker,
        cps=cps,
        wps=wps,
        reading_status="too_fast" if too_fast else "OK",
        flags=_unique_flags(
            flags if too_fast else [flag for flag in flags if flag != "reading_speed"]
        ),
        mean_confidence=caption.mean_confidence,
        confidence_band=caption.confidence_band,
    )


def apply_reading_speed(
    captions: list[Caption],
    settings: Settings | None = None,
    video_duration: float | None = None,
) -> list[Caption]:
    """Split, extend, or flag captions that are too dense to read."""
    settings = settings or load_settings()
    if not captions:
        return []

    outgoing: list[Caption] = []
    for index, caption in enumerate(captions):
        next_start = captions[index + 1].start if index + 1 < len(captions) else None
        pieces = _fix_density(caption, settings, next_start, video_duration)
        outgoing.extend(pieces)

    return _reindex(outgoing)


def _fix_density(
    caption: Caption,
    settings: Settings,
    next_start: float | None,
    video_duration: float | None,
) -> list[Caption]:
    if not _is_dense(caption, settings):
        caption.reading_status = "OK"
        caption.flags = [flag for flag in caption.flags if flag != "reading_speed"]
        return [caption]

    # 1. Better split at a linguistic boundary.
    split = _split_dense_words(caption.words)
    if split and len(caption.words) >= 2:
        parts = [caption_from_words(group, 0, settings, extra_flags=caption.flags) for group in split]
        if any(not _is_dense(part, settings) for part in parts) or (
            len(parts) > 1 and len(caption.words) >= 6
        ):
            result: list[Caption] = []
            for part_index, part in enumerate(parts):
                part_next = parts[part_index + 1].start if part_index + 1 < len(parts) else next_start
                result.extend(_fix_density_after_split(part, settings, part_next, video_duration))
            return result

    # 2. Extend timing if there is room.
    extended = _extend_caption(caption, settings, next_start, video_duration)
    if not _is_dense(extended, settings):
        return [extended]

    # 3. Forced midpoint split only when there is enough material for two cues.
    if split and len(caption.words) >= 6:
        parts = [caption_from_words(group, 0, settings, extra_flags=caption.flags) for group in split]
        forced: list[Caption] = []
        for part_index, part in enumerate(parts):
            part_next = parts[part_index + 1].start if part_index + 1 < len(parts) else next_start
            stretched = _extend_caption(part, settings, part_next, video_duration)
            if _is_dense(stretched, settings):
                stretched.reading_status = "too_fast"
                stretched.flags = _unique_flags(list(stretched.flags) + ["reading_speed", "needs_review"])
            forced.append(stretched)
        return forced

    # 4. Flag for review.
    extended.reading_status = "too_fast"
    extended.flags = _unique_flags(list(extended.flags) + ["reading_speed", "needs_review"])
    return [extended]


def _fix_density_after_split(
    caption: Caption,
    settings: Settings,
    next_start: float | None,
    video_duration: float | None,
) -> list[Caption]:
    if not _is_dense(caption, settings):
        return [caption]
    extended = _extend_caption(caption, settings, next_start, video_duration)
    if _is_dense(extended, settings):
        extended.reading_status = "too_fast"
        extended.flags = _unique_flags(list(extended.flags) + ["reading_speed", "needs_review"])
    return [extended]


def _reindex(captions: list[Caption]) -> list[Caption]:
    for index, caption in enumerate(captions, start=1):
        caption.index = index
    return captions


def _annotate_from_confidence(captions: list[Caption], confidence_payload: dict[str, Any] | None) -> None:
    if not confidence_payload:
        return
    lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in confidence_payload.get("words") or []:
        lookup[(round(float(item["start"]), 3), round(float(item["end"]), 3), item.get("word"))] = item
    for caption in captions:
        overalls: list[float] = []
        for word in caption.words:
            record = lookup.get((round(word.start, 3), round(word.end, 3), word.word))
            if record and record.get("overall_confidence") is not None:
                overalls.append(float(record["overall_confidence"]))
                if record.get("band") == "low":
                    caption.flags = _unique_flags(list(caption.flags) + ["low_confidence"])
        if overalls:
            caption.mean_confidence = round(sum(overalls) / len(overalls), 4)


def build_captions(
    words: list[Word],
    settings: Settings | None = None,
    video_duration: float | None = None,
    confidence_payload: dict[str, Any] | None = None,
) -> list[Caption]:
    settings = settings or load_settings()
    groups = segment_words(
        words,
        max_chars=settings.max_chars_per_line,
        max_duration=settings.max_cue_duration,
        pause_gap=settings.pause_gap,
        max_lines=settings.max_caption_lines,
    )
    groups = _merge_short_cues(groups, settings)
    captions = [caption_from_words(group, index, settings) for index, group in enumerate(groups, start=1)]
    _annotate_from_confidence(captions, confidence_payload)
    captions = apply_reading_speed(captions, settings=settings, video_duration=video_duration)
    return captions


def captions_payload(captions: list[Caption]) -> dict[str, Any]:
    warnings = sum(1 for caption in captions if "reading_speed" in caption.flags or caption.reading_status != "OK")
    mean_cps = (sum(caption.cps for caption in captions) / len(captions)) if captions else 0.0
    return {
        "captions": [caption.to_dict() for caption in captions],
        "summary": {
            "caption_count": len(captions),
            "reading_speed_warnings": warnings,
            "mean_cps": round(mean_cps, 2),
            "needs_review": sum(1 for caption in captions if "needs_review" in caption.flags),
        },
    }


def load_captions(path: str | Path) -> list[Caption]:
    data = read_json(Path(path))
    return [Caption.from_dict(item) for item in data.get("captions") or []]


def write_captions(path: str | Path, captions: list[Caption]) -> Path:
    dest = Path(path)
    write_json(dest, captions_payload(captions))
    return dest


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


def _load_job_words(job_dir: Path) -> list[Word]:
    return load_transcript(job_transcript_path(job_dir, stage="segment")).words


def segment_job(job_dir: str | Path, check_reading_speed: bool = True) -> Path:
    job_dir = Path(job_dir)
    words = _load_job_words(job_dir)
    confidence_path = job_dir / "confidence.json"
    confidence_payload = read_json(confidence_path) if confidence_path.is_file() else None
    captions = build_captions(
        words,
        video_duration=_job_duration(job_dir),
        confidence_payload=confidence_payload,
    )
    if not check_reading_speed:
        captions = [caption_from_words(cap.words, cap.index) for cap in captions]
    dest = write_captions(job_dir / "final_captions.json", captions)
    return dest


def reading_speed_summary(captions: list[Caption]) -> str:
    lines = []
    for caption in captions:
        status = caption.reading_status
        lines.append(
            f"#{caption.index} Reading speed: {caption.cps:.1f} CPS / {caption.wps:.1f} WPS  Status: {status}"
        )
    warnings = sum(1 for caption in captions if caption.reading_status != "OK")
    lines.append(f"Reading-speed warnings: {warnings}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Segment words into caption cues and check reading speed.")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--check-reading-speed", action="store_true")
    args = parser.parse_args(argv)
    path = segment_job(args.job_dir, check_reading_speed=True)
    if args.check_reading_speed:
        print(reading_speed_summary(load_captions(path)))
    print(path)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

from app.pipeline.timestamps import load_transcript
from app.transcript import Word

MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CUE_DURATION = 7.0
PAUSE_GAP = 0.6


def format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def wrap_caption(text: str, max_chars: int = MAX_CHARS_PER_LINE, max_lines: int = MAX_LINES) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    if max_lines < 2:
        return text[:max_chars].rstrip()
    target = min(max_chars, max(1, len(text) // 2))
    split_at = text.rfind(" ", 0, max_chars + 1)
    if split_at < target // 2:
        split_at = text.find(" ", target)
    if split_at <= 0:
        split_at = min(max_chars, len(text))
        first, rest = text[:split_at], text[split_at:].lstrip()
    else:
        first, rest = text[:split_at], text[split_at + 1 :]
    if len(rest) > max_chars:
        rest = rest[:max_chars].rstrip()
    return f"{first}\n{rest}".strip()


def segment_words(
    words: list[Word],
    max_chars: int = MAX_CHARS_PER_LINE,
    max_duration: float = MAX_CUE_DURATION,
    pause_gap: float = PAUSE_GAP,
) -> list[list[Word]]:
    cues: list[list[Word]] = []
    current: list[Word] = []
    limit = max_chars * MAX_LINES

    def flush() -> None:
        if current:
            cues.append(list(current))
            current.clear()

    for word in words:
        if not current:
            current.append(word)
            continue
        gap = word.start - current[-1].end
        duration = word.end - current[0].start
        text_len = len(" ".join(item.word for item in current + [word]))
        if gap >= pause_gap or duration > max_duration or text_len > limit:
            flush()
        current.append(word)
    flush()
    return cues


def render_srt(words: list[Word]) -> str:
    cues = segment_words(words)
    blocks: list[str] = []
    for index, cue_words in enumerate(cues, start=1):
        text = wrap_caption(" ".join(word.word for word in cue_words))
        if not text.strip():
            continue
        start = cue_words[0].start
        end = max(cue_words[-1].end, start + 0.5)
        blocks.append(
            f"{index}\n{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n{text}\n"
        )
    return "\n".join(blocks).rstrip() + ("\n" if blocks else "")


def write_srt(words: list[Word], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_srt(words), encoding="utf-8")
    return path


def srt_from_job(job_dir: str | Path, output: str | Path | None = None) -> Path:
    job_dir = Path(job_dir)
    timestamps_path = job_dir / "word_timestamps.json"
    if not timestamps_path.is_file():
        raise FileNotFoundError(f"Missing {timestamps_path}. Run timestamps first.")
    transcript = load_transcript(timestamps_path)
    dest = Path(output) if output else job_dir / "final.srt"
    return write_srt(transcript.words, dest)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate UTF-8 SRT from word timestamps.")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--validate", action="store_true", help="Reserved for Phase 2 timestamp validation")
    args = parser.parse_args(argv)
    path = srt_from_job(args.job_dir, output=args.output)
    print(path)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import shutil
import subprocess
import wave
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import ROOT, TARGET_CHANNELS, TARGET_SAMPLE_RATE, load_settings

FFMPEG_CANDIDATES = ("ffmpeg", "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")
FFPROBE_CANDIDATES = ("ffprobe", "/opt/homebrew/bin/ffprobe", "/usr/local/bin/ffprobe")


def find_binary(candidates: tuple[str, ...], name: str) -> str:
    for candidate in candidates:
        if os.path.sep in candidate:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        else:
            found = shutil.which(candidate)
            if found:
                return found
    raise RuntimeError(f"{name} not found. Install FFmpeg and ensure it is on PATH.")


def find_ffmpeg() -> str:
    return find_binary(FFMPEG_CANDIDATES, "FFmpeg")


def find_ffprobe() -> str:
    return find_binary(FFPROBE_CANDIDATES, "ffprobe")


def probe_video(video_path: str | Path) -> dict[str, Any]:
    ffprobe = find_ffprobe()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    fmt = data.get("format") or {}
    video_stream = next((s for s in data.get("streams") or [] if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in data.get("streams") or [] if s.get("codec_type") == "audio"), {})
    fps = _parse_frame_rate(video_stream.get("r_frame_rate") or video_stream.get("avg_frame_rate"))
    return {
        "original_filename": Path(video_path).name,
        "original_path": str(Path(video_path).resolve()),
        "duration": float(fmt.get("duration") or 0.0),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "frame_rate": fps,
        "file_size": int(fmt.get("size") or Path(video_path).stat().st_size),
        "audio": {
            "codec": audio_stream.get("codec_name"),
            "sample_rate": int(audio_stream["sample_rate"]) if audio_stream.get("sample_rate") else None,
            "channels": audio_stream.get("channels"),
        },
    }


def _parse_frame_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        denominator = float(den)
        if denominator == 0:
            return None
        return float(num) / denominator
    return float(value)


def validate_wav(path: str | Path) -> dict[str, Any]:
    wav_path = Path(path)
    if not wav_path.is_file() or wav_path.stat().st_size == 0:
        raise RuntimeError(f"Audio extraction produced an empty file: {wav_path}")
    with wave.open(str(wav_path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.getnframes()
    if channels != TARGET_CHANNELS:
        raise RuntimeError(f"Expected mono audio, got {channels} channels")
    if sample_rate != TARGET_SAMPLE_RATE:
        raise RuntimeError(f"Expected {TARGET_SAMPLE_RATE} Hz, got {sample_rate} Hz")
    if sample_width != 2:
        raise RuntimeError(f"Expected 16-bit PCM, got sample width {sample_width}")
    duration = frames / float(sample_rate)
    return {
        "path": str(wav_path),
        "channels": channels,
        "sample_rate": sample_rate,
        "sample_width": sample_width,
        "duration": duration,
    }


def extract_audio(
    video_path: str | Path,
    job_dir: str | Path,
    normalize: bool | None = None,
) -> Path:
    """Extract 16 kHz mono PCM WAV. Does not modify the original video."""
    video_path = Path(video_path)
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ffmpeg = find_ffmpeg()
    settings = load_settings()
    if normalize is None:
        normalize = settings.enable_audio_normalization

    source_wav = job_dir / "source_audio.wav"
    audio_wav = job_dir / "audio.wav"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(source_wav if normalize else audio_wav),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed:\n{completed.stderr}")

    if normalize:
        norm_cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(source_wav),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(audio_wav),
        ]
        completed = subprocess.run(norm_cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"FFmpeg loudnorm failed:\n{completed.stderr}")
    else:
        # Keep a separately named copy only when we already wrote source_audio.wav.
        pass

    validate_wav(audio_wav)
    return audio_wav


def new_job_id(outputs_dir: Path | None = None) -> str:
    outputs_dir = outputs_dir or (ROOT / "outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    existing = list(outputs_dir.glob(f"job_{stamp}_*"))
    return f"job_{stamp}_{len(existing) + 1:03d}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract 16 kHz mono PCM WAV from a video.")
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--job-dir", required=True, help="Job output directory")
    parser.add_argument("--normalize", action="store_true", help="Apply loudnorm")
    args = parser.parse_args(argv)
    path = extract_audio(args.video, args.job_dir, normalize=args.normalize or None)
    print(path)


if __name__ == "__main__":
    main()

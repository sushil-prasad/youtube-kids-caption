from __future__ import annotations

import struct
import subprocess
import wave
from pathlib import Path

import pytest

from app.config import TARGET_CHANNELS, TARGET_SAMPLE_RATE
from app.pipeline.audio import extract_audio, find_ffmpeg, validate_wav


def _write_pcm_wav(path: Path, *, rate: int = 16000, channels: int = 1, seconds: float = 0.1) -> Path:
    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        silence = struct.pack("<h", 0)
        handle.writeframes(silence * frames * channels)
    return path


def test_validate_wav_accepts_16khz_mono(tmp_path: Path) -> None:
    path = _write_pcm_wav(tmp_path / "audio.wav")
    info = validate_wav(path)
    assert info["sample_rate"] == TARGET_SAMPLE_RATE
    assert info["channels"] == TARGET_CHANNELS
    assert info["sample_width"] == 2
    assert info["duration"] == pytest.approx(0.1, abs=0.01)


def test_validate_wav_rejects_wrong_rate(tmp_path: Path) -> None:
    path = _write_pcm_wav(tmp_path / "audio.wav", rate=44100)
    with pytest.raises(RuntimeError, match="16000"):
        validate_wav(path)


def test_validate_wav_rejects_stereo(tmp_path: Path) -> None:
    path = _write_pcm_wav(tmp_path / "audio.wav", channels=2)
    with pytest.raises(RuntimeError, match="mono"):
        validate_wav(path)


def ffmpeg_available() -> bool:
    try:
        find_ffmpeg()
        return True
    except RuntimeError:
        return False


def _make_test_video(path: Path, duration: float = 1.0) -> Path:
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration}",
        "-f",
        "lavfi",
        "-i",
        f"color=c=black:s=320x240:d={duration}",
        "-shortest",
        "-c:v",
        "mpeg4",
        "-c:a",
        "aac",
        str(path),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        pytest.skip(f"Could not synthesize a test video: {completed.stderr}")
    return path


@pytest.mark.skipif(not ffmpeg_available(), reason="FFmpeg is required for audio extraction tests")
def test_extract_audio_is_16khz_mono_pcm(tmp_path: Path) -> None:
    video = _make_test_video(tmp_path / "clip.mp4")
    original_mtime = video.stat().st_mtime
    original_size = video.stat().st_size
    job_dir = tmp_path / "job"
    wav_path = extract_audio(video, job_dir, normalize=False)
    info = validate_wav(wav_path)
    assert wav_path.name == "audio.wav"
    assert info["sample_rate"] == TARGET_SAMPLE_RATE
    assert info["channels"] == TARGET_CHANNELS
    assert info["sample_width"] == 2
    assert info["duration"] == pytest.approx(1.0, abs=0.15)
    assert video.stat().st_mtime == original_mtime
    assert video.stat().st_size == original_size


def test_extract_audio_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_audio(tmp_path / "missing.mp4", tmp_path / "job")

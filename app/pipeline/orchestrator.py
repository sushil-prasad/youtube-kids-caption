from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from app.config import ROOT, load_settings, print_device_banner, resolve_device
from app.pipeline.audio import extract_audio, new_job_id, probe_video, read_json, write_json
from app.pipeline.asr import transcribe_job
from app.pipeline.srt import srt_from_job
from app.pipeline.timestamps import build_word_timestamps

STOP_AFTER = ("extract-audio", "asr", "timestamps", "srt")


def create_job(video: str | Path, job_dir: str | Path | None = None) -> Path:
    video = Path(video)
    if job_dir:
        dest = Path(job_dir)
    else:
        job_id = new_job_id()
        dest = ROOT / "outputs" / job_id
    dest.mkdir(parents=True, exist_ok=True)
    meta = {
        "job_id": dest.name,
        "status": "QUEUED",
        "original_filename": video.name,
        "original_path": str(video.resolve()),
    }
    try:
        meta.update(probe_video(video))
        meta["status"] = "QUEUED"
    except Exception as exc:
        meta["probe_error"] = str(exc)
    write_json(dest / "job.json", meta)
    return dest


def _update_status(job_dir: Path, status: str, extra: dict | None = None) -> None:
    path = job_dir / "job.json"
    data = read_json(path) if path.is_file() else {}
    data["status"] = status
    if extra:
        data.update(extra)
    write_json(path, data)


def run_pipeline(
    video: str | Path | None = None,
    output: str | Path | None = None,
    device: str | None = None,
    skip_asr: bool = False,
    job_dir: str | Path | None = None,
    stop_after: str | None = None,
) -> Path:
    settings = load_settings()
    backend = resolve_device(device or settings.device)
    print_device_banner(backend, settings)

    if skip_asr:
        if not job_dir:
            raise ValueError("--skip-asr requires --job-dir with existing artifacts")
        dest = Path(job_dir)
        if not (dest / "word_timestamps.json").is_file() and not (dest / "raw_transcript.json").is_file():
            raise FileNotFoundError(f"No transcript artifacts in {dest}")
    else:
        if video is None:
            raise ValueError("video path is required unless --skip-asr is set")
        dest = create_job(video, job_dir)
        _update_status(dest, "EXTRACTING_AUDIO")
        extract_audio(video, dest)
        if stop_after == "extract-audio":
            _update_status(dest, "EXTRACTING_AUDIO")
            return dest / "audio.wav"

        _update_status(dest, "TRANSCRIBING", {"device": backend, "asr_model": settings.asr_model})
        transcribe_job(dest, device=backend, model_name=settings.asr_model)
        if stop_after == "asr":
            return dest / "raw_transcript.json"

    _update_status(dest, "ALIGNING")
    timestamps_path = build_word_timestamps(dest)
    if stop_after == "timestamps":
        return timestamps_path

    _update_status(dest, "GENERATING_SRT")
    srt_path = srt_from_job(dest, output=None)
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(srt_path, output)
        srt_path = output
    else:
        # Default CLI convenience: also write <video>.srt next to the input when we have it.
        if video:
            sidecar = Path(video).with_suffix(".srt")
            if sidecar.resolve() != srt_path.resolve():
                shutil.copyfile(srt_path, sidecar)
    _update_status(dest, "READY_FOR_REVIEW", {"srt": str(srt_path)})
    print(srt_path)
    return Path(srt_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.pipeline",
        description="Child-focused captioning: video → 16 kHz WAV → ASR → word timestamps → SRT.",
    )
    parser.add_argument("video", nargs="?", help="Input video (.mp4, .mov, .mkv, …)")
    parser.add_argument("--output", "-o", help="Output .srt path")
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default=None)
    parser.add_argument("--skip-asr", action="store_true", help="Reuse artifacts in --job-dir")
    parser.add_argument("--job-dir", help="Existing or destination job directory under outputs/")
    parser.add_argument("--stop-after", choices=STOP_AFTER, help="Run only through this stage")
    parser.add_argument("--safety-mode", help="Accepted for later phases; ignored in Phase 1")
    parser.add_argument("--enable-diarization", action="store_true", help="Ignored in Phase 1")
    parser.add_argument("--enable-sound-events", action="store_true", help="Ignored in Phase 1")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.video is None and not args.skip_asr:
        parser.print_help()
        raise SystemExit(2)
    run_pipeline(
        video=args.video,
        output=args.output,
        device=args.device,
        skip_asr=args.skip_asr,
        job_dir=args.job_dir,
        stop_after=args.stop_after,
    )


if __name__ == "__main__":
    main()

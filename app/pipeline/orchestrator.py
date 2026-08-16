from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from app.config import ROOT, load_settings, print_device_banner, resolve_device
from app.pipeline.audio import extract_audio, new_job_id, probe_video, read_json, write_json
from app.pipeline.asr import transcribe_job
from app.pipeline.confidence import confidence_from_job
from app.pipeline.correction import correction_from_job
from app.pipeline.diarization import diarization_from_job
from app.pipeline.punctuation import punctuate_job
from app.pipeline.profanity import profanity_from_job
from app.pipeline.segmentation import segment_job
from app.pipeline.sound_events import sound_events_from_job
from app.pipeline.srt import srt_from_job
from app.pipeline.timestamps import build_word_timestamps, job_transcript_path, load_transcript, write_word_timestamps
from app.pipeline.vocabulary import apply_vocabulary, load_vocabulary, vocabulary_from_job
from app.transcript import Transcript

STOP_AFTER = (
    "extract-audio",
    "asr",
    "timestamps",
    "confidence",
    "punctuation",
    "diarization",
    "sound-events",
    "vocabulary",
    "profanity",
    "correction",
    "segmentation",
    "srt",
)


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


def _apply_vocabulary_transcript(job_dir: Path, settings) -> None:
    vocab = load_vocabulary(job_dir, settings)
    source = job_transcript_path(job_dir, stage="annotate")
    transcript = load_transcript(source)
    words = apply_vocabulary(transcript.words, vocab)
    if words == transcript.words and not (job_dir / "annotated_transcript.json").is_file():
        return
    write_word_timestamps(
        Transcript(
            text=" ".join(word.word for word in words),
            words=words,
            language=transcript.language,
            model=transcript.model,
            device=transcript.device,
        ),
        job_dir / "annotated_transcript.json",
    )


def run_pipeline(
    video: str | Path | None = None,
    output: str | Path | None = None,
    device: str | None = None,
    skip_asr: bool = False,
    job_dir: str | Path | None = None,
    stop_after: str | None = None,
    safety_mode: str | None = None,
    enable_diarization: bool = False,
    enable_sound_events: bool = False,
) -> Path:
    settings = load_settings()
    backend = resolve_device(device or settings.device)
    print_device_banner(backend, settings)
    run_diarization = enable_diarization or settings.enable_diarization
    run_sound_events = enable_sound_events or settings.enable_sound_events

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

    confidence_from_job(dest)
    if stop_after == "confidence":
        return dest / "confidence.json"

    _update_status(dest, "PUNCTUATING")
    punctuate_job(dest)
    if stop_after == "punctuation":
        return dest / "punctuated_transcript.json"

    vocabulary_from_job(dest)
    if stop_after == "vocabulary":
        return dest / "vocabulary.json"

    if run_diarization or stop_after == "diarization":
        _update_status(dest, "DETECTING_SPEAKERS")
        diarization_from_job(dest, settings)
    if stop_after == "diarization":
        return dest / "speaker_segments.json"

    if run_sound_events or stop_after == "sound-events":
        _update_status(dest, "DETECTING_SOUNDS")
        sound_events_from_job(dest, settings)
    if stop_after == "sound-events":
        return dest / "sound_events.json"

    _apply_vocabulary_transcript(dest, settings)

    _update_status(dest, "SAFETY_ANALYSIS")
    analysis_path = profanity_from_job(dest)
    if stop_after == "profanity":
        return analysis_path

    _update_status(dest, "CORRECTING", {"safety_mode": safety_mode or settings.safety_mode})
    correction_path = correction_from_job(dest, safety_mode=safety_mode or settings.safety_mode)
    if stop_after == "correction":
        return correction_path

    _update_status(dest, "SEGMENTING")
    segment_job(dest)
    if stop_after == "segmentation":
        return dest / "final_captions.json"

    _update_status(dest, "GENERATING_SRT")
    srt_path = srt_from_job(dest, output=None, validate=True)
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
        description="Child-focused captioning: video → WAV → ASR → timestamps → confidence → punctuation → safety → captions → SRT.",
    )
    parser.add_argument("video", nargs="?", help="Input video or audio (.mp4, .mov, .mkv, .wav, .mp3, …)")
    parser.add_argument("--output", "-o", help="Output .srt path")
    parser.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default=None)
    parser.add_argument("--skip-asr", action="store_true", help="Reuse artifacts in --job-dir")
    parser.add_argument("--job-dir", help="Existing or destination job directory under outputs/")
    parser.add_argument("--stop-after", choices=STOP_AFTER, help="Run only through this stage")
    parser.add_argument("--safety-mode", help="strict, standard, or review-only (literal)")
    parser.add_argument("--enable-diarization", action="store_true", help="Run optional speaker-change detection")
    parser.add_argument("--enable-sound-events", action="store_true", help="Run optional [sound event] detection")
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
        safety_mode=args.safety_mode,
        enable_diarization=args.enable_diarization,
        enable_sound_events=args.enable_sound_events,
    )


if __name__ == "__main__":
    main()

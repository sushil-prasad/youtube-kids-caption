from __future__ import annotations

import json
from pathlib import Path

from app.models.registry import get_asr_model
from app.pipeline.timestamps import write_word_timestamps
from app.transcript import Transcript


def transcribe_job(job_dir: str | Path, device: str | None = None, model_name: str | None = None) -> Transcript:
    job_dir = Path(job_dir)
    audio_path = job_dir / "audio.wav"
    if not audio_path.is_file():
        raise FileNotFoundError(f"Missing {audio_path}. Extract audio first.")
    model = get_asr_model(name=model_name, device=device)
    from app.pipeline.vocabulary import load_vocabulary

    vocabulary = load_vocabulary(job_dir).asr_hints()
    transcript = model.transcribe(audio_path, vocabulary=vocabulary or None)
    raw_path = job_dir / "raw_transcript.json"
    raw_path.write_text(json.dumps(transcript.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_word_timestamps(transcript, job_dir / "word_timestamps.json")
    return transcript


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run child-focused ASR on a job directory.")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    transcript = transcribe_job(args.job_dir, device=args.device)
    print(transcript.text)


if __name__ == "__main__":
    main()

"""Job listing, status, quality report, and one-at-a-time pipeline worker."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.database import database as db
from app.pipeline.audio import new_job_id, read_json, write_json
from app.pipeline.orchestrator import run_pipeline
from app.pipeline.profanity import DISCLAIMER

router = APIRouter()

PROCESS_IN_BACKGROUND = True
_worker_lock = threading.Lock()

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv")
AUDIO_SUFFIXES = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg")


class ReviewBody(BaseModel):
    notes: str | None = None


class SpeakerMapBody(BaseModel):
    mapping: dict[str, str]


def outputs_dir() -> Path:
    return db.OUTPUTS


def job_path(job_id: str) -> Path:
    return db.job_dir(job_id)


def load_job_json(job_dir: Path) -> dict[str, Any]:
    path = job_dir / "job.json"
    if not path.is_file():
        return {}
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def write_job_json(job_dir: Path, payload: dict[str, Any]) -> None:
    write_json(job_dir / "job.json", payload)


def sync_sqlite(job_id: str, meta: dict[str, Any] | None = None, error: str | None = None) -> None:
    meta = meta or {}
    db.upsert_job(
        job_id,
        original_filename=meta.get("original_filename"),
        original_path=meta.get("original_path"),
        status=meta.get("status"),
        error=error if error is not None else meta.get("error"),
    )


def find_job_video(job_dir: Path) -> Path | None:
    for suffix in VIDEO_SUFFIXES:
        candidate = job_dir / f"original_video{suffix}"
        if candidate.is_file():
            return candidate
    matches = sorted(job_dir.glob("original_video.*"))
    if matches:
        return matches[0]
    meta = load_job_json(job_dir)
    original = meta.get("original_path")
    if original and Path(original).is_file():
        return Path(original)
    return None


def require_job_dir(job_id: str) -> Path:
    dest = job_path(job_id)
    if not dest.is_dir() or not (dest / "job.json").is_file():
        raise HTTPException(status_code=404, detail=f"Unknown job {job_id}")
    return dest


def quality_report(job_dir: Path) -> dict[str, Any]:
    confidence = _read(job_dir / "confidence.json")
    correction = _read(job_dir / "correction_log.json")
    captions = _read(job_dir / "reviewed_captions.json") or _read(job_dir / "final_captions.json")
    speakers = _read(job_dir / "speaker_segments.json")
    sounds = _read(job_dir / "sound_events.json")
    summary = (confidence or {}).get("summary") or {}
    corr_summary = (correction or {}).get("summary") or {}
    cap_summary = (captions or {}).get("summary") or {}
    word_count = int(summary.get("word_count") or 0)
    mean = float(summary.get("mean_confidence") or summary.get("overall_caption_confidence") or 0.0)
    return {
        "overall_confidence": round(mean * 100) if mean <= 1 else round(mean),
        "words": word_count,
        "low_confidence_words": int(summary.get("low") or 0),
        "potential_profanity": int(corr_summary.get("flagged") or 0)
        + int(corr_summary.get("replaced") or 0)
        + int(corr_summary.get("censored") or 0),
        "automatically_corrected": int(corr_summary.get("replaced") or 0),
        "censored": int(corr_summary.get("censored") or 0),
        "speaker_changes": len((speakers or {}).get("segments") or []),
        "sound_events": len((sounds or {}).get("events") or []),
        "reading_speed_warnings": int(cap_summary.get("reading_speed_warnings") or 0),
        "disclaimer": DISCLAIMER,
    }


def job_payload(job_id: str) -> dict[str, Any]:
    dest = require_job_dir(job_id)
    meta = load_job_json(dest)
    row = db.get_job_row(job_id) or {}
    status = meta.get("status") or row.get("status") or "QUEUED"
    payload = {
        "id": job_id,
        "job_id": job_id,
        "status": status,
        "original_filename": meta.get("original_filename") or row.get("original_filename"),
        "original_path": meta.get("original_path") or row.get("original_path"),
        "error": meta.get("error") or row.get("error"),
        "duration": meta.get("duration"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "has_video": find_job_video(dest) is not None,
        "has_captions": (dest / "reviewed_captions.json").is_file() or (dest / "final_captions.json").is_file(),
        "quality": quality_report(dest) if (dest / "final_captions.json").is_file() else None,
        "disclaimer": DISCLAIMER,
    }
    return payload


def discover_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    root = outputs_dir()
    if not root.is_dir():
        return jobs
    for path in sorted(root.glob("job_*/job.json"), reverse=True):
        job_id = path.parent.name
        meta = load_job_json(path.parent)
        sync_sqlite(job_id, meta)
        try:
            jobs.append(job_payload(job_id))
        except HTTPException:
            continue
    return jobs


def create_job_dir(original_name: str) -> Path:
    root = outputs_dir()
    root.mkdir(parents=True, exist_ok=True)
    job_id = new_job_id(root)
    dest = root / job_id
    dest.mkdir(parents=True, exist_ok=True)
    meta = {
        "job_id": job_id,
        "status": "UPLOAD",
        "original_filename": original_name,
        "original_path": "",
    }
    write_job_json(dest, meta)
    sync_sqlite(job_id, meta)
    return dest


def enqueue(job_id: str) -> None:
    dest = job_path(job_id)
    meta = load_job_json(dest)
    meta["status"] = "QUEUED"
    write_job_json(dest, meta)
    sync_sqlite(job_id, meta)
    if PROCESS_IN_BACKGROUND:
        threading.Thread(target=process_job, args=(job_id,), daemon=True).start()
    else:
        process_job(job_id)


def process_job(job_id: str) -> None:
    """Run the existing CLI pipeline for one job. Tests monkeypatch this."""
    with _worker_lock:
        dest = job_path(job_id)
        video = find_job_video(dest)
        if video is None:
            _fail(job_id, "No video found for this job")
            return
        from app.api.settings import load_creator_settings

        creator = load_creator_settings()
        previous = {
            key: os.environ.get(key)
            for key in ("SAFETY_MODE", "UNKNOWN_PROFANITY", "ENABLE_DIARIZATION", "ENABLE_SOUND_EVENTS")
        }
        try:
            os.environ["SAFETY_MODE"] = str(creator["safety_mode"])
            os.environ["UNKNOWN_PROFANITY"] = str(creator["unknown_profanity"])
            os.environ["ENABLE_DIARIZATION"] = "1" if creator.get("enable_speaker_labels") else "0"
            os.environ["ENABLE_SOUND_EVENTS"] = "1" if creator.get("enable_sound_events") else "0"
            run_pipeline(
                video=video,
                job_dir=dest,
                safety_mode=str(creator["safety_mode"]),
                enable_diarization=bool(creator.get("enable_speaker_labels")),
                enable_sound_events=bool(creator.get("enable_sound_events")),
            )
        except Exception as exc:  # noqa: BLE001 — surface any pipeline failure to the dashboard
            _fail(job_id, str(exc))
            return
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        meta = load_job_json(dest)
        if meta.get("status") not in {"READY_FOR_REVIEW", "REVIEWED", "EXPORTED"}:
            meta["status"] = "READY_FOR_REVIEW"
            write_job_json(dest, meta)
        sync_sqlite(job_id, meta, error="")


def _fail(job_id: str, message: str) -> None:
    dest = job_path(job_id)
    meta = load_job_json(dest)
    meta["status"] = "FAILED"
    meta["error"] = message
    write_job_json(dest, meta)
    sync_sqlite(job_id, meta, error=message)


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


@router.get("/jobs")
def list_jobs() -> dict[str, Any]:
    jobs = discover_jobs()
    return {"jobs": jobs, "busy_job_id": db.busy_job_id()}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    return job_payload(job_id)


@router.post("/jobs/{job_id}/review")
def mark_reviewed(job_id: str, body: ReviewBody | None = None) -> dict[str, Any]:
    dest = require_job_dir(job_id)
    meta = load_job_json(dest)
    status = meta.get("status")
    if status not in {"READY_FOR_REVIEW", "REVIEWED", "EXPORTED"}:
        raise HTTPException(status_code=409, detail=f"Job is not ready for review (status={status})")
    meta["status"] = "REVIEWED"
    if body and body.notes:
        meta["review_notes"] = body.notes
    write_job_json(dest, meta)
    sync_sqlite(job_id, meta)
    return job_payload(job_id)


@router.put("/jobs/{job_id}/speakers")
def rename_speakers(job_id: str, body: SpeakerMapBody) -> dict[str, Any]:
    dest = require_job_dir(job_id)
    write_json(dest / "speaker_map.json", {"mapping": body.mapping})
    return {"job_id": job_id, "mapping": body.mapping}

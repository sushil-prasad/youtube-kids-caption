"""Video upload. Copies the file into the job directory; never modifies the original."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.database import database as db
from app.api.jobs import (
    AUDIO_SUFFIXES,
    VIDEO_SUFFIXES,
    create_job_dir,
    enqueue,
    job_payload,
    load_job_json,
    sync_sqlite,
    write_job_json,
)
from app.pipeline.audio import probe_video, wrap_audio_as_video

router = APIRouter()

ALLOWED = {suffix.lower() for suffix in VIDEO_SUFFIXES + AUDIO_SUFFIXES}
AUDIO = {suffix.lower() for suffix in AUDIO_SUFFIXES}


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)) -> dict:
    filename = Path(file.filename or "video.mp4").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type {suffix or '(none)'}. Upload .mp4, .mov, .mkv, or audio (.wav, .mp3, .m4a).",
        )
    busy = db.busy_job_id()
    if busy:
        raise HTTPException(
            status_code=409,
            detail={
                "message": f"A job is already running ({busy}). The Mac prototype processes one video at a time.",
                "busy_job_id": busy,
            },
        )

    dest = create_job_dir(filename)
    if suffix in AUDIO:
        stored_audio = dest / f"original_audio{suffix}"
        try:
            with stored_audio.open("wb") as handle:
                shutil.copyfileobj(file.file, handle)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not save upload: {exc}") from exc
        stored = dest / "original_video.mp4"
        try:
            wrap_audio_as_video(stored_audio, stored)
        except Exception:
            stored = stored_audio
    else:
        stored = dest / f"original_video{suffix}"
        try:
            with stored.open("wb") as handle:
                shutil.copyfileobj(file.file, handle)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not save upload: {exc}") from exc

    meta = load_job_json(dest)
    meta["original_path"] = str(stored.resolve())
    meta["original_filename"] = filename
    try:
        meta.update(probe_video(stored))
        meta["original_path"] = str(stored.resolve())
        meta["original_filename"] = filename
    except Exception as exc:  # noqa: BLE001 — probe is optional for a saved copy
        meta["probe_error"] = str(exc)
    write_job_json(dest, meta)
    sync_sqlite(dest.name, meta)
    enqueue(dest.name)
    payload = job_payload(dest.name)
    payload["copied_to"] = str(stored)
    return payload

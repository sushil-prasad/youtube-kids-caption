"""Caption editing, safety review actions, and SRT export.

The original ASR transcript (raw_transcript.json) is never overwritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.api.jobs import (
    find_job_video,
    job_payload,
    load_job_json,
    require_job_dir,
    sync_sqlite,
    write_job_json,
)
from app.pipeline.audio import extract_thumbnail, read_json, write_json
from app.pipeline.correction import CENSOR_MARK
from app.pipeline.profanity import DISCLAIMER
from app.pipeline.segmentation import captions_payload, load_captions, write_captions
from app.pipeline.srt import render_captions, render_srt, validate_captions
from app.pipeline.timestamps import load_transcript
from app.transcript import Caption

router = APIRouter()


class CaptionEdit(BaseModel):
    index: int | None = None
    start: float
    end: float
    text: str
    speaker: str | None = None
    flags: list[str] | None = None
    confidence_band: str | None = None
    mean_confidence: float | None = None


class CaptionsBody(BaseModel):
    captions: list[CaptionEdit]


class SafetyActionBody(BaseModel):
    index: int = Field(ge=1)
    action: str
    text: str | None = None


def captions_file(job_dir: Path) -> Path:
    reviewed = job_dir / "reviewed_captions.json"
    if reviewed.is_file():
        return reviewed
    return job_dir / "final_captions.json"


def load_job_captions(job_dir: Path) -> list[Caption]:
    path = captions_file(job_dir)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Captions are not ready yet")
    captions = load_captions(path)
    mapping = _speaker_map(job_dir)
    if mapping:
        for caption in captions:
            if caption.speaker and caption.speaker in mapping:
                caption.speaker = mapping[caption.speaker]
    return captions


def persist_reviewed(job_dir: Path, captions: list[Caption]) -> None:
    for index, caption in enumerate(captions, start=1):
        caption.index = index
    write_captions(job_dir / "reviewed_captions.json", captions)
    (job_dir / "reviewed.srt").write_text(render_captions(captions), encoding="utf-8")


def _speaker_map(job_dir: Path) -> dict[str, str]:
    path = job_dir / "speaker_map.json"
    if not path.is_file():
        return {}
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return {}
    mapping = data.get("mapping") if isinstance(data, dict) else None
    return {str(key): str(value) for key, value in (mapping or {}).items()}


def _decisions(job_dir: Path) -> list[dict[str, Any]]:
    path = job_dir / "correction_log.json"
    if not path.is_file():
        return []
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return []
    return list(data.get("decisions") or [])


def _decision_times(job_dir: Path) -> list[tuple[dict[str, Any], float, float]]:
    decisions = _decisions(job_dir)
    words: list[Any] = []
    for name in ("corrected_transcript.json", "punctuated_transcript.json", "word_timestamps.json"):
        path = job_dir / name
        if path.is_file():
            try:
                words = load_transcript(path).words
            except (OSError, ValueError, KeyError):
                words = []
            if words:
                break
    timed: list[tuple[dict[str, Any], float, float]] = []
    for decision in decisions:
        start = end = None
        index = int(decision.get("index") or 0)
        end_index = int(decision.get("end_index") or index + 1)
        if 0 <= index < len(words):
            start = float(words[index].start)
            last = min(len(words), max(end_index, index + 1)) - 1
            end = float(words[last].end)
        timed.append((decision, start if start is not None else -1.0, end if end is not None else -1.0))
    return timed


def _caption_safety(caption: Caption, timed: list[tuple[dict[str, Any], float, float]]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    text = caption.text
    for decision, start, end in timed:
        overlaps = start >= 0 and not (end <= caption.start or start >= caption.end)
        mentioned = decision.get("original", "") and str(decision["original"]).lower() in text.lower()
        censored = CENSOR_MARK in text and decision.get("action") == "censor"
        if not (overlaps or mentioned or censored):
            continue
        if decision.get("action") not in {"censor", "flag", "replace"} and not decision.get("needs_review"):
            continue
        suggested = decision.get("replacement") or ""
        flags.append(
            {
                "original": decision.get("original"),
                "replacement": suggested,
                "case": decision.get("case"),
                "action": decision.get("action"),
                "reason": decision.get("reason"),
                "needs_review": bool(decision.get("needs_review") or decision.get("action") in {"censor", "flag"}),
                "suggested_text": _apply_token(text, str(decision.get("original") or ""), str(suggested)),
            }
        )
    return flags


def _apply_token(text: str, original: str, replacement: str) -> str:
    if not original:
        return text.replace(CENSOR_MARK, replacement) if replacement else text
    lowered = text.lower()
    key = original.lower()
    index = lowered.find(key)
    if index < 0:
        if CENSOR_MARK in text and replacement:
            return text.replace(CENSOR_MARK, replacement, 1)
        return text
    return text[:index] + replacement + text[index + len(original) :]


def export_job(job_dir: str | Path) -> dict[str, Path]:
    job_dir = Path(job_dir)
    if not job_dir.is_dir():
        raise FileNotFoundError(f"Job directory not found: {job_dir}")
    captions_path = captions_file(job_dir)
    if captions_path.is_file():
        captions = load_captions(captions_path)
        captions, _issues = validate_captions(captions)
        reviewed = job_dir / "reviewed.srt"
        reviewed.write_text(render_captions(captions), encoding="utf-8")
    else:
        reviewed = job_dir / "reviewed.srt"
        final = job_dir / "final.srt"
        if final.is_file() and not reviewed.is_file():
            reviewed.write_text(final.read_text(encoding="utf-8"), encoding="utf-8")

    raw_path = job_dir / "raw.srt"
    raw_source = job_dir / "raw_transcript.json"
    if not raw_source.is_file():
        raw_source = job_dir / "word_timestamps.json"
    if raw_source.is_file():
        words = load_transcript(raw_source).words
        raw_path.write_text(render_srt(words), encoding="utf-8")

    report = job_dir / "correction_report.md"
    report.write_text(_correction_report(job_dir), encoding="utf-8")

    meta = load_job_json(job_dir)
    if meta.get("status") in {"READY_FOR_REVIEW", "REVIEWED", "EXPORTED"}:
        meta["status"] = "EXPORTED"
        write_job_json(job_dir, meta)
        sync_sqlite(job_dir.name, meta)
    return {
        "reviewed": reviewed if reviewed.is_file() else job_dir / "reviewed.srt",
        "raw": raw_path,
        "report": report,
    }


def _correction_report(job_dir: Path) -> str:
    log = {}
    path = job_dir / "correction_log.json"
    if path.is_file():
        try:
            log = read_json(path)
        except (OSError, ValueError):
            log = {}
    lines = [
        "# Correction report",
        "",
        DISCLAIMER,
        "",
        f"Job: `{job_dir.name}`",
        f"Mode: {log.get('mode', 'unknown')}",
        f"Unknown profanity: {log.get('unknown_profanity', 'unknown')}",
        "",
        "| Original | Replacement | Case | Action | Needs review | Reason |",
        "|---|---|---|---|---|---|",
    ]
    for decision in log.get("decisions") or []:
        lines.append(
            "| {original} | {replacement} | {case} | {action} | {needs} | {reason} |".format(
                original=str(decision.get("original") or "").replace("|", "/"),
                replacement=str(decision.get("replacement") or "").replace("|", "/"),
                case=decision.get("case") or "",
                action=decision.get("action") or "",
                needs="yes" if decision.get("needs_review") else "no",
                reason=str(decision.get("reason") or "").replace("|", "/"),
            )
        )
    if not (log.get("decisions") or []):
        lines.append("| — | — | — | — | — | No safety decisions recorded |")
    lines.extend(["", "The original ASR transcript remains in `raw_transcript.json` and `raw.srt`.", ""])
    return "\n".join(lines)


def _caption_dict(caption: Caption, safety: list[dict[str, Any]]) -> dict[str, Any]:
    payload = caption.to_dict()
    payload["safety"] = safety
    return payload


@router.get("/jobs/{job_id}/captions")
def get_captions(job_id: str) -> dict[str, Any]:
    dest = require_job_dir(job_id)
    captions = load_job_captions(dest)
    timed = _decision_times(dest)
    return {
        "job_id": job_id,
        "captions": [_caption_dict(caption, _caption_safety(caption, timed)) for caption in captions],
        "summary": captions_payload(captions).get("summary"),
        "disclaimer": DISCLAIMER,
        "reviewed": (dest / "reviewed_captions.json").is_file(),
    }


@router.put("/jobs/{job_id}/captions")
def put_captions(job_id: str, body: CaptionsBody) -> dict[str, Any]:
    dest = require_job_dir(job_id)
    existing = []
    if captions_file(dest).is_file():
        existing = load_job_captions(dest)
    by_index = {caption.index: caption for caption in existing}
    updated: list[Caption] = []
    for item in body.captions:
        current = by_index.get(item.index or 0, Caption(index=item.index or 0, start=0, end=0, text=""))
        current.start = item.start
        current.end = item.end
        current.text = item.text
        current.speaker = item.speaker
        if item.flags is not None:
            current.flags = list(item.flags)
        if item.confidence_band is not None:
            current.confidence_band = item.confidence_band
        if item.mean_confidence is not None:
            current.mean_confidence = item.mean_confidence
        updated.append(current)
    persist_reviewed(dest, updated)
    return get_captions(job_id)


@router.post("/jobs/{job_id}/safety")
def safety_action(job_id: str, body: SafetyActionBody) -> dict[str, Any]:
    dest = require_job_dir(job_id)
    captions = load_job_captions(dest)
    caption = next((item for item in captions if item.index == body.index), None)
    if caption is None:
        raise HTTPException(status_code=404, detail="Caption not found")
    timed = _decision_times(dest)
    flags = _caption_safety(caption, timed)
    action = body.action.strip().lower().replace(" ", "_")
    if action == "accept":
        if flags:
            caption.text = flags[0].get("suggested_text") or caption.text
        if "needs_review" in caption.flags:
            caption.flags = [flag for flag in caption.flags if flag != "needs_review"]
    elif action in {"keep_censored", "keep"}:
        if flags and flags[0].get("original") and CENSOR_MARK not in caption.text:
            caption.text = _apply_token(caption.text, str(flags[0]["original"]), CENSOR_MARK)
    elif action == "edit":
        if body.text is None:
            raise HTTPException(status_code=400, detail="Edit action requires text")
        caption.text = body.text
        if "needs_review" in caption.flags:
            caption.flags = [flag for flag in caption.flags if flag != "needs_review"]
    else:
        raise HTTPException(status_code=400, detail="action must be accept, keep_censored, or edit")
    persist_reviewed(dest, captions)
    log_path = dest / "review_actions.json"
    history = []
    if log_path.is_file():
        try:
            history = list(read_json(log_path).get("actions") or [])
        except (OSError, ValueError):
            history = []
    history.append({"index": body.index, "action": action, "text": caption.text})
    write_json(log_path, {"actions": history, "disclaimer": DISCLAIMER})
    return get_captions(job_id)


@router.get("/jobs/{job_id}/video", response_model=None)
def get_video(job_id: str) -> FileResponse:
    dest = require_job_dir(job_id)
    video = find_job_video(dest)
    if video is None or not video.is_file():
        raise HTTPException(status_code=404, detail="Original video is not available")
    media = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
    }.get(video.suffix.lower(), "application/octet-stream")
    return FileResponse(
        video,
        media_type=media,
        filename=video.name,
        headers={"Cache-Control": "no-store"},
    )


@router.get("/jobs/{job_id}/thumbnail", response_model=None)
def get_thumbnail(job_id: str) -> FileResponse:
    dest = require_job_dir(job_id)
    thumb = dest / "thumbnail.jpg"
    if not thumb.is_file() or thumb.stat().st_size == 0:
        video = find_job_video(dest)
        if video is None or not video.is_file():
            raise HTTPException(status_code=404, detail="Thumbnail is not available")
        try:
            extract_thumbnail(video, thumb)
        except Exception as exc:
            raise HTTPException(status_code=404, detail="Thumbnail is not available") from exc
    return FileResponse(
        thumb,
        media_type="image/jpeg",
        filename="thumbnail.jpg",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/jobs/{job_id}/export/{kind}", response_model=None)
def download_export(job_id: str, kind: str) -> FileResponse:
    dest = require_job_dir(job_id)
    paths = export_job(dest)
    mapping = {
        "reviewed.srt": paths["reviewed"],
        "reviewed": paths["reviewed"],
        "raw.srt": paths["raw"],
        "raw": paths["raw"],
        "correction_report.md": paths["report"],
        "report": paths["report"],
        "correction-report": paths["report"],
    }
    path = mapping.get(kind)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail=f"Export {kind} is not available")
    media = "text/plain; charset=utf-8" if path.suffix == ".srt" else "text/markdown; charset=utf-8"
    return FileResponse(path, media_type=media, filename=path.name)


@router.post("/jobs/{job_id}/export")
def post_export(job_id: str) -> dict[str, Any]:
    dest = require_job_dir(job_id)
    paths = export_job(dest)
    return {
        "job_id": job_id,
        "status": job_payload(job_id)["status"],
        "files": {name: str(path) for name, path in paths.items()},
        "disclaimer": DISCLAIMER,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.api.captions",
        description="Export reviewed SRT, raw ASR SRT, and a correction report. Does not overwrite raw_transcript.json.",
    )
    parser.add_argument("--export", action="store_true", help="Write export files into --job-dir")
    parser.add_argument("--job-dir", default=None, help="Job directory under outputs/")
    args = parser.parse_args(argv)
    if not args.export:
        parser.print_help()
        return
    if not args.job_dir:
        parser.error("--export requires --job-dir")
    paths = export_job(args.job_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

"""Creator safety settings and vocabulary API. Persisted under data/, not .env."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import ROOT, normalize_safety_mode, normalize_unknown_profanity
from app.pipeline.audio import read_json, write_json
from app.pipeline.vocabulary import (
    CATEGORIES,
    Vocabulary,
    load_creator_vocabulary,
    load_vocabulary,
    save_creator_vocabulary,
)

CREATOR_SETTINGS_PATH = ROOT / "data" / "creator_settings.json"

DEFAULT_SETTINGS = {
    "safety_mode": "strict",
    "unknown_profanity": "censor",
    "enable_sound_events": False,
    "enable_speaker_labels": True,
}

router = APIRouter()


class CreatorSettings(BaseModel):
    safety_mode: str = "strict"
    unknown_profanity: str = "censor"
    enable_sound_events: bool = False
    enable_speaker_labels: bool = True


class VocabTerm(BaseModel):
    term: str = Field(min_length=1)
    category: str = "other"


def settings_path() -> Path:
    return CREATOR_SETTINGS_PATH


def load_creator_settings() -> dict[str, Any]:
    path = settings_path()
    data = dict(DEFAULT_SETTINGS)
    if path.is_file():
        try:
            stored = read_json(path)
        except (OSError, ValueError):
            stored = {}
        if isinstance(stored, dict):
            data.update(stored)
    data["safety_mode"] = normalize_safety_mode(str(data.get("safety_mode") or "strict"), "strict")
    data["unknown_profanity"] = normalize_unknown_profanity(
        str(data.get("unknown_profanity") or "censor"), "censor"
    )
    data["enable_sound_events"] = bool(data.get("enable_sound_events"))
    data["enable_speaker_labels"] = bool(data.get("enable_speaker_labels", True))
    return data


def save_creator_settings(payload: dict[str, Any]) -> dict[str, Any]:
    current = load_creator_settings()
    current.update(payload)
    current["safety_mode"] = normalize_safety_mode(str(current.get("safety_mode") or "strict"), "strict")
    current["unknown_profanity"] = normalize_unknown_profanity(
        str(current.get("unknown_profanity") or "censor"), "censor"
    )
    current["enable_sound_events"] = bool(current.get("enable_sound_events"))
    current["enable_speaker_labels"] = bool(current.get("enable_speaker_labels", True))
    dest = settings_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_json(dest, current)
    return current


@router.get("/settings")
def get_settings() -> dict[str, Any]:
    return load_creator_settings()


@router.put("/settings")
def put_settings(body: CreatorSettings) -> dict[str, Any]:
    return save_creator_settings(body.model_dump())


def _vocab_payload() -> dict[str, Any]:
    merged = load_vocabulary()
    creator = load_creator_vocabulary()
    return {
        "categories": list(CATEGORIES),
        "grouped": merged.grouped(),
        "creator": [entry.to_dict() for entry in creator.entries],
        "terms": merged.terms(),
    }


@router.get("/vocabulary")
def get_vocabulary() -> dict[str, Any]:
    return _vocab_payload()


@router.post("/vocabulary")
def add_vocabulary(body: VocabTerm) -> dict[str, Any]:
    vocab = load_creator_vocabulary()
    vocab.add(body.term, body.category)
    save_creator_vocabulary(vocab)
    return _vocab_payload()


@router.get("/vocabulary/export")
def export_vocabulary() -> dict[str, Any]:
    return load_creator_vocabulary().to_dict()


@router.post("/vocabulary/import")
def import_vocabulary(payload: dict[str, Any]) -> dict[str, Any]:
    incoming = Vocabulary.from_dict(payload)
    current = load_creator_vocabulary()
    for entry in incoming.entries:
        current.add(entry.term, entry.category)
    save_creator_vocabulary(current)
    return _vocab_payload()


@router.delete("/vocabulary/{term}")
def delete_vocabulary(term: str) -> dict[str, Any]:
    cleaned = " ".join((term or "").split())
    if not cleaned:
        raise HTTPException(status_code=400, detail="Term is required")
    vocab = load_creator_vocabulary()
    vocab.remove(cleaned)
    save_creator_vocabulary(vocab)
    return _vocab_payload()

from __future__ import annotations

from pathlib import Path

import pytest

from app.pipeline.audio import write_json
from app.pipeline.segmentation import write_captions
from app.transcript import Caption, Transcript, Word

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


def _seed_artifacts(job_dir: Path) -> None:
    words = [
        Word("hello", 0.0, 0.4, 0.99),
        Word("world", 0.4, 0.9, 0.40),
    ]
    write_json(
        job_dir / "raw_transcript.json",
        Transcript(text="hello world", words=words, language="en", model="test").to_dict(),
    )
    write_json(job_dir / "word_timestamps.json", Transcript(text="hello world", words=words).to_dict())
    write_json(
        job_dir / "confidence.json",
        {
            "summary": {
                "word_count": 2,
                "mean_confidence": 0.695,
                "overall_caption_confidence": 0.695,
                "high": 1,
                "medium": 0,
                "low": 1,
            }
        },
    )
    write_json(
        job_dir / "correction_log.json",
        {
            "disclaimer": "Captions are not guaranteed safe. Creator review is required.",
            "mode": "strict",
            "unknown_profanity": "censor",
            "decisions": [
                {
                    "index": 1,
                    "end_index": 2,
                    "original": "world",
                    "replacement": "there",
                    "reason": "unresolved",
                    "confidence": 0.4,
                    "case": "C",
                    "action": "censor",
                    "needs_review": True,
                }
            ],
            "summary": {"kept": 0, "replaced": 0, "censored": 1, "flagged": 1},
        },
    )
    write_captions(
        job_dir / "final_captions.json",
        [
            Caption(
                index=1,
                start=0.0,
                end=0.9,
                text="hello _",
                words=words,
                confidence_band="low",
                mean_confidence=0.4,
                flags=["needs_review"],
            )
        ],
    )
    meta = {
        "job_id": job_dir.name,
        "status": "READY_FOR_REVIEW",
        "original_filename": "clip.mp4",
        "original_path": str(job_dir / "original_video.mp4"),
        "duration": 0.9,
    }
    write_json(job_dir / "job.json", meta)


def _fake_processor(outputs: Path):
    def process_job(job_id: str) -> None:
        dest = outputs / job_id
        _seed_artifacts(dest)
        from app.api.jobs import load_job_json, sync_sqlite, write_job_json

        meta = load_job_json(dest)
        meta["status"] = "READY_FOR_REVIEW"
        write_job_json(dest, meta)
        sync_sqlite(job_id, meta, error="")

    return process_job


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.setattr("app.database.database.DB_PATH", tmp_path / "jobs.sqlite")
    monkeypatch.setattr("app.database.database.OUTPUTS", outputs)
    monkeypatch.setattr("app.api.settings.CREATOR_SETTINGS_PATH", tmp_path / "creator_settings.json")
    monkeypatch.setattr("app.pipeline.vocabulary.CREATOR_VOCAB_PATH", tmp_path / "creator_vocabulary.json")
    monkeypatch.setattr("app.api.jobs.PROCESS_IN_BACKGROUND", False)
    monkeypatch.setattr("app.api.jobs.process_job", _fake_processor(outputs))
    from app.database.database import reset_connection

    reset_connection()
    from app.main import create_app

    client = TestClient(create_app())
    yield client, tmp_path, outputs
    reset_connection()


def test_dashboard_is_served(api) -> None:
    client, _tmp, _outputs = api
    response = client.get("/")
    assert response.status_code == 200
    assert "Custom vocabulary" in response.text
    assert client.get("/api/health").json()["status"] == "ok"


def test_upload_rejects_non_video(api) -> None:
    client, _tmp, _outputs = api
    response = client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_upload_creates_job_and_copies_file(api, tmp_path: Path) -> None:
    client, _tmp, outputs = api
    original = tmp_path / "source.mp4"
    original.write_bytes(b"fake-mp4-bytes")
    response = client.post(
        "/api/upload",
        files={"file": ("kids.mp4", original.read_bytes(), "video/mp4")},
    )
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    copied = outputs / job_id / "original_video.mp4"
    assert copied.is_file()
    assert copied.read_bytes() == b"fake-mp4-bytes"
    assert original.read_bytes() == b"fake-mp4-bytes"
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "READY_FOR_REVIEW"
    assert job["quality"]["low_confidence_words"] == 1


def test_one_job_at_a_time_returns_409(api, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _tmp, outputs = api

    def noop(_job_id: str) -> None:
        return None

    monkeypatch.setattr("app.api.jobs.process_job", noop)
    first = client.post("/api/upload", files={"file": ("a.mp4", b"one", "video/mp4")})
    assert first.status_code == 200
    second = client.post("/api/upload", files={"file": ("b.mp4", b"two", "video/mp4")})
    assert second.status_code == 409


def test_captions_get_put_and_safety(api) -> None:
    client, _tmp, outputs = api
    job_id = client.post("/api/upload", files={"file": ("kids.mp4", b"x", "video/mp4")}).json()["job_id"]
    raw_before = (outputs / job_id / "raw_transcript.json").read_text(encoding="utf-8")
    captions = client.get(f"/api/jobs/{job_id}/captions").json()
    assert captions["captions"][0]["text"] == "hello _"
    assert captions["captions"][0]["safety"]

    edited = captions["captions"][0]
    payload = {
        "captions": [
            {
                "index": edited["index"],
                "start": edited["start"],
                "end": edited["end"],
                "text": "hello friends",
                "speaker": None,
            }
        ]
    }
    updated = client.put(f"/api/jobs/{job_id}/captions", json=payload).json()
    assert updated["captions"][0]["text"] == "hello friends"
    assert (outputs / job_id / "reviewed_captions.json").is_file()
    assert (outputs / job_id / "raw_transcript.json").read_text(encoding="utf-8") == raw_before

    client.put(
        f"/api/jobs/{job_id}/captions",
        json={"captions": [{**payload["captions"][0], "text": "hello _"}]},
    )
    accepted = client.post(
        f"/api/jobs/{job_id}/safety",
        json={"index": 1, "action": "accept"},
    ).json()
    assert accepted["captions"][0]["text"] == "hello there"


def test_export_keeps_raw_asr_recoverable(api) -> None:
    client, _tmp, outputs = api
    job_id = client.post("/api/upload", files={"file": ("kids.mp4", b"x", "video/mp4")}).json()["job_id"]
    client.put(
        f"/api/jobs/{job_id}/captions",
        json={"captions": [{"index": 1, "start": 0.0, "end": 0.9, "text": "Hello friends"}]},
    )
    exported = client.post(f"/api/jobs/{job_id}/export").json()
    assert exported["status"] == "EXPORTED"
    reviewed = client.get(f"/api/jobs/{job_id}/export/reviewed.srt")
    raw = client.get(f"/api/jobs/{job_id}/export/raw.srt")
    report = client.get(f"/api/jobs/{job_id}/export/correction_report.md")
    assert reviewed.status_code == 200
    assert "Hello friends" in reviewed.text
    assert "hello" in raw.text.lower()
    assert "world" in raw.text.lower()
    assert "not guaranteed safe" in report.text.lower()
    raw_json = (outputs / job_id / "raw_transcript.json").read_text(encoding="utf-8")
    assert "hello world" in raw_json
    assert "Hello friends" not in raw_json


def test_settings_and_vocabulary_persist(api) -> None:
    client, tmp_path, _outputs = api
    saved = client.put(
        "/api/settings",
        json={
            "safety_mode": "review-only",
            "unknown_profanity": "flag",
            "enable_sound_events": True,
            "enable_speaker_labels": False,
        },
    ).json()
    assert saved["safety_mode"] == "review-only"
    assert client.get("/api/settings").json()["unknown_profanity"] == "flag"
    assert (tmp_path / "creator_settings.json").is_file()

    added = client.post("/api/vocabulary", json={"term": "Bingo", "category": "character_names"}).json()
    assert "Bingo" in added["terms"]
    exported = client.get("/api/vocabulary/export").json()
    assert any(item["term"] == "Bingo" for item in exported["entries"])
    client.post("/api/vocabulary/import", json={"entries": [{"term": "Fluffernoodle", "category": "fictional"}]})
    terms = client.get("/api/vocabulary").json()["terms"]
    assert "Bingo" in terms
    assert "Fluffernoodle" in terms

    deleted = client.delete("/api/vocabulary/Bluey").json()
    assert "Bluey" not in deleted["terms"]
    assert "Bingo" in deleted["terms"]
    restored = client.post("/api/vocabulary", json={"term": "Bluey", "category": "character_names"}).json()
    assert "Bluey" in restored["terms"]


def test_export_cli_writes_files(api) -> None:
    client, _tmp, outputs = api
    job_id = client.post("/api/upload", files={"file": ("kids.mp4", b"x", "video/mp4")}).json()["job_id"]
    from app.api.captions import main

    main(["--export", "--job-dir", str(outputs / job_id)])
    assert (outputs / job_id / "reviewed.srt").is_file()
    assert (outputs / job_id / "raw.srt").is_file()
    assert (outputs / job_id / "correction_report.md").is_file()


def test_main_help_does_not_serve() -> None:
    from app.main import main

    main([])

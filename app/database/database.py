"""SQLite job index. Job directories under outputs/ remain the source of truth for artifacts."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import ROOT

DB_PATH = ROOT / "data" / "jobs.sqlite"
OUTPUTS = ROOT / "outputs"

_local = threading.local()

BUSY_STATUSES = {
    "UPLOAD",
    "QUEUED",
    "EXTRACTING_AUDIO",
    "TRANSCRIBING",
    "ALIGNING",
    "PUNCTUATING",
    "DETECTING_SPEAKERS",
    "DETECTING_SOUNDS",
    "SAFETY_ANALYSIS",
    "CORRECTING",
    "SEGMENTING",
    "GENERATING_SRT",
}

REVIEW_STATUSES = {"READY_FOR_REVIEW", "REVIEWED", "EXPORTED"}


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def db_path() -> Path:
    return DB_PATH


def connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        _local.conn = conn
        init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection | None = None) -> None:
    handle = conn or connect()
    handle.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            original_filename TEXT,
            original_path TEXT,
            status TEXT NOT NULL,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    handle.commit()


def reset_connection() -> None:
    conn = getattr(_local, "conn", None)
    if conn is not None:
        conn.close()
        _local.conn = None


def job_dir(job_id: str) -> Path:
    return OUTPUTS / job_id


def upsert_job(
    job_id: str,
    *,
    original_filename: str | None = None,
    original_path: str | None = None,
    status: str | None = None,
    error: str | None = None,
) -> None:
    now = utcnow()
    existing = get_job_row(job_id)
    if existing is None:
        connect().execute(
            "INSERT INTO jobs (id, original_filename, original_path, status, error, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                original_filename or "",
                original_path or "",
                status or "QUEUED",
                error,
                now,
                now,
            ),
        )
    else:
        connect().execute(
            """
            UPDATE jobs SET
                original_filename = COALESCE(?, original_filename),
                original_path = COALESCE(?, original_path),
                status = COALESCE(?, status),
                error = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                original_filename,
                original_path,
                status,
                error if error is not None else existing["error"],
                now,
                job_id,
            ),
        )
    connect().commit()


def get_job_row(job_id: str) -> dict[str, Any] | None:
    row = connect().execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def busy_job_id() -> str | None:
    rows = connect().execute(
        f"SELECT id FROM jobs WHERE status IN ({','.join('?' * len(BUSY_STATUSES))}) ORDER BY created_at",
        tuple(BUSY_STATUSES),
    ).fetchall()
    if rows:
        return str(rows[0]["id"])
    # Also honor CLI jobs that never hit SQLite.
    for path in sorted(OUTPUTS.glob("job_*/job.json")):
        try:
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("status") in BUSY_STATUSES:
            return path.parent.name
    return None

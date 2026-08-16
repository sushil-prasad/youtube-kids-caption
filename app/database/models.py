"""Job record helpers."""

from __future__ import annotations

from typing import Any

from app.database.database import get_job_row, upsert_job


def record_job(job_id: str, **fields: Any) -> None:
    upsert_job(job_id, **fields)


def fetch_job(job_id: str) -> dict[str, Any] | None:
    return get_job_row(job_id)

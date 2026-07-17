"""Pure training status plus explicit immutable job enqueue operations."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from . import training_jobs


def _public(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": job["state"],
        "job_id": job["job_id"],
        "engine": job["engine"],
        "profile": job["profile"],
        "generation": job["generation"],
        "attempts": job["attempts"],
        "promotable": job["promotable"],
        "fixture": job["fixture"],
        "deadline_epoch": job["deadline_epoch"],
        "detail": job["detail"],
    }


def status(
    root: Path,
    readiness_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read persisted job state without writing files, SQLite, or worker state."""

    latest = training_jobs.JobStore(root).latest()
    if latest is not None:
        return _public(latest)
    if readiness_report is not None and readiness_report.get("ready") is True:
        return {
            "state": "ready",
            "job_id": None,
            "detail": "corpus is ready; enqueue requires an explicit POST or enabled scheduler",
        }
    return {
        "state": "idle",
        "job_id": None,
        "detail": "corpus is not ready",
    }


def read_state(root: Path) -> dict[str, Any]:
    """Compatibility name retained as a pure read."""

    return status(root)


def enqueue(
    root: Path,
    readiness_report: dict[str, Any],
    *,
    engine: str,
    profile: str,
    now_iso: str,
    now_epoch: int | None = None,
    fixture: bool = False,
    promotable: bool = True,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Build, seal, and queue one job; never invoke a shell or mutate a model."""

    current_epoch = int(time.time()) if now_epoch is None else now_epoch
    if idempotency_key is None:
        corpus_sha256 = readiness_report.get("corpus_sha256")
        if not isinstance(corpus_sha256, str) or len(corpus_sha256) != 64:
            raise training_jobs.JobContractError("readiness corpus hash is absent")
        idempotency_key = (
            f"{engine}:{profile}:{corpus_sha256[:32]}:"
            f"{'fixture' if fixture else 'production'}"
        )
    bundle = training_jobs.build_bundle(
        root,
        readiness_report=readiness_report,
        engine=engine,
        profile=profile,
        idempotency_key=idempotency_key,
        promotable=promotable,
        fixture=fixture,
        now_iso=now_iso,
    )
    deadline_seconds = (
        4 * 60 * 60
        if (engine, profile) == ("cosyvoice3", "expressive-zero-shot")
        else training_jobs.MAX_JOB_SECONDS
    )
    job = training_jobs.JobStore(root).enqueue(
        bundle,
        now_epoch=current_epoch,
        deadline_epoch=current_epoch + deadline_seconds,
    )
    return _public(job)


def maybe_enqueue(
    root: Path,
    readiness_report: dict[str, Any],
    *,
    auto: bool,
    now_iso: str,
    now_epoch: int | None = None,
) -> dict[str, Any]:
    """Enqueue once only from a mutation/scheduler path, never from GET."""

    current = status(root, readiness_report)
    if current["state"] in training_jobs.ACTIVE_STATES:
        return current
    if readiness_report.get("ready") is not True or not auto:
        return current
    return enqueue(
        root,
        readiness_report,
        engine="cosyvoice3",
        profile="expressive-zero-shot",
        now_iso=now_iso,
        now_epoch=now_epoch,
        fixture=False,
        promotable=True,
    )

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.training_helpers import build_training_corpus, fixture_policy, healthy_worker
from webui import readiness, training_jobs


def _ready(tmp_path: Path) -> tuple[list[dict], dict]:
    sections = build_training_corpus(tmp_path)
    report = readiness.evaluate(
        tmp_path,
        sections,
        policy=fixture_policy(),
        trainer_health=healthy_worker(),
    )
    assert report["ready"] is True
    return sections, report


@pytest.mark.parametrize(
    ("engine", "profile", "purpose", "sample_rate"),
    (
        ("cosyvoice3", "expressive-zero-shot", "serve-24k", 24_000),
        ("piper", "medium", "piper-22050", 22_050),
    ),
)
def test_builder_emits_valid_ready_last_bundle_with_exact_audio_contract(
    tmp_path: Path,
    engine: str,
    profile: str,
    purpose: str,
    sample_rate: int,
) -> None:
    _sections, report = _ready(tmp_path)
    result = training_jobs.build_bundle(
        tmp_path,
        readiness_report=report,
        engine=engine,
        profile=profile,
        idempotency_key=f"fixture-{engine}",
        promotable=True,
        now_iso="2026-07-16T21:00:00Z",
    )
    validated = training_jobs.validate_bundle(result["path"])
    assert validated["job_id"] == result["job_id"]
    assert validated["target_audio"]["purpose"] == purpose
    assert validated["target_audio"]["sample_rate"] == sample_rate
    assert (result["path"] / "READY").is_file()
    rows = (result["path"] / "metadata.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 7
    splits = json.loads((result["path"] / "splits.json").read_text(encoding="utf-8"))
    for group in ("sec13_001", "sec13_002", "sec13_003"):
        memberships = [name for name in ("train", "validation", "test") if group in splits[name]]
        assert len(memberships) == 1


def test_validator_rejects_missing_ready_corruption_and_executable_extra(tmp_path: Path) -> None:
    _sections, report = _ready(tmp_path)
    result = training_jobs.build_bundle(
        tmp_path,
        readiness_report=report,
        engine="cosyvoice3",
        profile="expressive-zero-shot",
        idempotency_key="fixture-contract-rejections",
        promotable=True,
        now_iso="2026-07-16T21:00:00Z",
    )
    ready = result["path"] / "READY"
    ready.unlink()
    with pytest.raises(training_jobs.JobContractError, match="READY"):
        training_jobs.validate_bundle(result["path"])
    ready.write_text(result["manifest_sha256"] + "\n", encoding="ascii")
    extra = result["path"] / "run.sh"
    extra.write_text("#!/bin/sh\n", encoding="utf-8")
    with pytest.raises(training_jobs.JobContractError, match="unexpected|executable"):
        training_jobs.validate_bundle(result["path"])
    extra.unlink()
    audio = next((result["path"] / "audio").glob("*.wav"))
    audio.write_bytes(b"corrupt")
    with pytest.raises(training_jobs.JobContractError, match="checksum|audio"):
        training_jobs.validate_bundle(result["path"])


def test_job_state_machine_is_idempotent_single_active_and_compare_and_swap(tmp_path: Path) -> None:
    _sections, report = _ready(tmp_path)
    bundle = training_jobs.build_bundle(
        tmp_path,
        readiness_report=report,
        engine="cosyvoice3",
        profile="expressive-zero-shot",
        idempotency_key="fixture-state-machine",
        promotable=True,
        now_iso="2026-07-16T21:00:00Z",
    )
    jobs = training_jobs.JobStore(tmp_path)
    first = jobs.enqueue(bundle, deadline_epoch=2_000, now_epoch=1_000)
    assert jobs.enqueue(bundle, deadline_epoch=2_000, now_epoch=1_001) == first
    other = {**bundle, "idempotency_key": "another-key", "job_id": "job-" + "f" * 48}
    with pytest.raises(training_jobs.JobConflict, match="active"):
        jobs.enqueue(other, deadline_epoch=2_000, now_epoch=1_001)

    current = first
    for state in ("validating", "staging", "running", "evaluating", "packaging", "succeeded"):
        current = jobs.transition(
            first["job_id"],
            expected_generation=current["generation"],
            new_state=state,
            now_epoch=1_100 + current["generation"],
        )
    assert current["state"] == "succeeded"
    with pytest.raises(training_jobs.JobConflict, match="generation|transition"):
        jobs.transition(
            first["job_id"],
            expected_generation=0,
            new_state="failed",
            now_epoch=1_200,
        )


def test_fixture_jobs_can_never_be_promotable_and_deadlines_fail_boundedly(tmp_path: Path) -> None:
    _sections, report = _ready(tmp_path)
    with pytest.raises(training_jobs.JobContractError, match="fixture"):
        training_jobs.build_bundle(
            tmp_path,
            readiness_report=report,
            engine="cosyvoice3",
            profile="expressive-zero-shot",
            idempotency_key="fixture-promotable",
            promotable=True,
            fixture=True,
            now_iso="2026-07-16T21:00:00Z",
        )
    bundle = training_jobs.build_bundle(
        tmp_path,
        readiness_report=report,
        engine="cosyvoice3",
        profile="expressive-zero-shot",
        idempotency_key="fixture-deadline",
        promotable=False,
        fixture=True,
        now_iso="2026-07-16T21:00:00Z",
    )
    jobs = training_jobs.JobStore(tmp_path)
    job = jobs.enqueue(bundle, deadline_epoch=1_010, now_epoch=1_000)
    recovered = jobs.fail_expired(now_epoch=1_011)
    assert recovered == 1
    assert jobs.get(job["job_id"])["state"] == "failed"


def test_retry_cancel_and_restart_recovery_have_fixed_ceilings(tmp_path: Path) -> None:
    _sections, report = _ready(tmp_path)
    bundle = training_jobs.build_bundle(
        tmp_path,
        readiness_report=report,
        engine="piper",
        profile="medium",
        idempotency_key="fixture-retry-ceiling",
        promotable=False,
        fixture=True,
        now_iso="2026-07-16T21:00:00Z",
    )
    jobs = training_jobs.JobStore(tmp_path)
    current = jobs.enqueue(bundle, deadline_epoch=2_000, now_epoch=1_000)
    with pytest.raises(training_jobs.JobConflict, match="illegal transition"):
        jobs.transition(
            current["job_id"],
            expected_generation=0,
            new_state="running",
            now_epoch=1_001,
        )
    current = jobs.transition(
        current["job_id"],
        expected_generation=current["generation"],
        new_state="failed",
        now_epoch=1_002,
        detail="fixture failure",
    )
    for attempt in range(1, training_jobs.MAX_RETRIES + 1):
        current = jobs.retry(
            current["job_id"],
            expected_generation=current["generation"],
            deadline_epoch=2_100 + attempt,
            now_epoch=1_100 + attempt * 10,
        )
        assert current["attempts"] == attempt
        current = jobs.transition(
            current["job_id"],
            expected_generation=current["generation"],
            new_state="failed",
            now_epoch=1_101 + attempt * 10,
        )
    with pytest.raises(training_jobs.JobConflict, match="not retryable"):
        jobs.retry(
            current["job_id"],
            expected_generation=current["generation"],
            deadline_epoch=3_000,
            now_epoch=1_100,
        )
    restarted = training_jobs.JobStore(tmp_path).get(current["job_id"])
    assert restarted is not None
    assert restarted["state"] == "failed"
    assert restarted["attempts"] == training_jobs.MAX_RETRIES


def test_piper_job_pins_the_exact_medium_base_checkpoint_and_license(tmp_path: Path) -> None:
    _sections, report = _ready(tmp_path)
    result = training_jobs.build_bundle(
        tmp_path,
        readiness_report=report,
        engine="piper",
        profile="medium",
        idempotency_key="piper-base-checkpoint-pin",
        promotable=True,
        now_iso="2026-07-16T21:00:00Z",
    )
    manifest = json.loads((result["path"] / "job.json").read_text(encoding="utf-8"))
    assert manifest["source_pins"] == {
        "code": "OHF-Voice/piper1-gpl@d6975e21a440c0d8b6e5fb7c41027409af13d44d",
        "release": "1.4.2",
        "license": "GPL-3.0-only",
        "base_checkpoint_repository": (
            "rhasspy/piper-checkpoints@95a4b650bd38716c97caf16d07b2a1734859f91a"
        ),
        "base_checkpoint_path": "en/en_US/ljspeech/medium/lj-med_1000.ckpt",
        "base_checkpoint_sha256": (
            "dcf2449bdbdaad09256a08dfac211c59f6b36ce8d3f244fd844a9eb1d7384c7c"
        ),
        "base_checkpoint_license": "MIT",
        "base_training_data_license": "Public-Domain",
    }

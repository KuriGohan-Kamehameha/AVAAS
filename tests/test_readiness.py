from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.training_helpers import (
    build_training_corpus,
    fixture_policy,
    healthy_worker,
    tree_receipt,
)
from webui import readiness, server
from webui.store import Store


def test_ready_report_has_stable_grouped_splits_references_and_licenses(tmp_path: Path) -> None:
    sections = build_training_corpus(tmp_path)
    first = readiness.evaluate(
        tmp_path,
        sections,
        policy=fixture_policy(),
        trainer_health=healthy_worker(),
    )
    second = readiness.evaluate(
        tmp_path,
        sections,
        policy=fixture_policy(),
        trainer_health=healthy_worker(),
    )
    assert first == second
    assert first["ready"] is True
    assert first["blockers"] == []
    assert len(first["references"]) >= 2
    reference_groups = {item["group_id"] for item in first["references"]}
    assert reference_groups <= set(first["splits"]["train"])
    assert reference_groups.isdisjoint(first["splits"]["validation"])
    assert reference_groups.isdisjoint(first["splits"]["test"])
    assert {item["id"] for item in first["licenses"]} >= {
        "harvard_100",
        "cmu_arctic_400",
        "avaas_local",
        "cosyvoice3-code",
        "cosyvoice3-model",
        "piper-code",
    }
    assert len(first["corpus_sha256"]) == 64
    assert len(first["split_sha256"]) == 64


def test_identity_gap_corruption_and_worker_health_are_machine_blockers(tmp_path: Path) -> None:
    sections = build_training_corpus(tmp_path)
    store = Store(tmp_path)
    accepted = store.accepted("sec13_003__piranesi", "2026.07.16")
    assert accepted is not None
    store.tombstone(
        accepted["take_id"],
        reason="fixture identity gap",
        expected_generation=accepted["generation"],
    )
    other = store.accepted("sec13_001__satraj", "2026.07.16")
    assert other is not None
    review = store.take_for_review(other["take_id"])
    assert review is not None
    (tmp_path / review["path"]).write_bytes(b"corrupt")
    unhealthy = {**healthy_worker(), "healthy": False}
    report = readiness.evaluate(
        tmp_path,
        sections,
        policy=fixture_policy(),
        trainer_health=unhealthy,
    )
    codes = {item["code"] for item in report["blockers"]}
    assert "identity-group-incomplete" in codes
    assert "accepted-content-invalid" in codes
    assert "trainer-unhealthy" in codes
    assert report["ready"] is False


def test_progress_get_is_a_pure_read(tmp_path: Path, monkeypatch) -> None:
    sections = build_training_corpus(tmp_path)
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "SETTINGS", tmp_path / "data/studio_settings.json")
    monkeypatch.setattr(server, "_SECTIONS", sections)
    monkeypatch.setattr(server.readiness, "DEFAULT_POLICY", fixture_policy())
    monkeypatch.setattr(server, "_trainer_health", lambda: healthy_worker())
    client = TestClient(server.app)
    before = tree_receipt(tmp_path / "data")
    first = client.get("/api/progress")
    second = client.get("/api/progress")
    after = tree_receipt(tmp_path / "data")
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert before == after


def test_missing_mandatory_prompt_source_fails_closed(tmp_path: Path) -> None:
    sections = build_training_corpus(tmp_path)
    (tmp_path / "prompts/corpora/cmu_arctic_400.txt").unlink()
    report = readiness.evaluate(
        tmp_path,
        sections,
        policy=fixture_policy(),
        trainer_health=healthy_worker(),
    )
    assert report["ready"] is False
    assert "mandatory-source-invalid" in {item["code"] for item in report["blockers"]}

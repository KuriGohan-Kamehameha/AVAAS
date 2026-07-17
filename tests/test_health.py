from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.training_helpers import build_training_corpus
from webui import server


def test_liveness_and_service_readiness_are_distinct(tmp_path: Path, monkeypatch) -> None:
    sections = build_training_corpus(tmp_path)
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "SETTINGS", tmp_path / "data/studio_settings.json")
    monkeypatch.setattr(server, "_SECTIONS", sections)
    client = TestClient(server.app)

    live = client.get("/healthz")
    assert live.status_code == 200
    assert live.json() == {"schema": "avaas/liveness@v1", "status": "ok"}
    assert live.headers["cache-control"] == "no-store"

    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["schema"] == "avaas/service-readiness@v1"
    assert ready.json()["ready"] is True
    assert ready.headers["cache-control"] == "no-store"

    monkeypatch.setattr(server, "_SECTIONS", [])
    unavailable = client.get("/readyz")
    assert unavailable.status_code == 503
    assert unavailable.json()["ready"] is False

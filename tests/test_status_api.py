from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.training_helpers import build_training_corpus
from webui import server
from webui.store import Store


TOKEN = "operator_test_token_0123456789abcdef"


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    sections = build_training_corpus(tmp_path)
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "SETTINGS", tmp_path / "data/studio_settings.json")
    monkeypatch.setattr(server, "_SECTIONS", sections)
    token = tmp_path / "data/status-api-token"
    token.write_text(TOKEN, encoding="ascii")
    token.chmod(0o600)
    return TestClient(server.app)


def test_operator_status_requires_private_bearer_and_is_read_only(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    assert client.get("/api/status/job").status_code == 401
    assert (
        client.get(
            "/api/status/job", headers={"Authorization": "Bearer incorrect_token_0123456789abcdef"}
        ).status_code
        == 401
    )
    before = (tmp_path / "data/corpus.sqlite3").stat().st_mtime_ns
    response = client.get(
        "/api/status/job", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert response.status_code == 200
    assert response.json() == {"schema": "avaas/operator-job-status@v1", "job": None}
    assert response.headers["cache-control"] == "no-store"
    assert (tmp_path / "data/corpus.sqlite3").stat().st_mtime_ns == before


def test_artifact_status_hides_paths_and_has_no_activation_endpoint(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    with Store(tmp_path).transaction() as connection:
        connection.execute(
            "INSERT INTO artifacts(artifact_id,job_id,schema_name,root_path,manifest_sha256,"
            "promotable,created_at) VALUES (?,?,?,?,?,?,?)",
            (
                "fixture-profile-v1",
                None,
                "avaas/voice-profile@v1",
                "data/artifacts/private/path",
                "a" * 64,
                0,
                "2026-07-16T20:00:00Z",
            ),
        )
    headers = {"Authorization": f"Bearer {TOKEN}"}
    response = client.get("/api/status/artifacts", headers=headers)
    assert response.status_code == 200
    artifact = response.json()["artifacts"][0]
    assert artifact["artifact_id"] == "fixture-profile-v1"
    assert artifact["promotable"] is False
    assert "root_path" not in artifact
    assert client.post("/api/status/artifacts/fixture-profile-v1/activate", headers=headers).status_code == 404


def test_status_secret_fails_closed_for_loose_permissions_or_symlink(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    token = tmp_path / "data/status-api-token"
    token.chmod(0o644)
    assert (
        client.get(
            "/api/status/artifacts", headers={"Authorization": f"Bearer {TOKEN}"}
        ).status_code
        == 503
    )
    token.unlink()
    outside = tmp_path / "outside-token"
    outside.write_text(TOKEN, encoding="ascii")
    outside.chmod(0o600)
    token.symlink_to(outside)
    assert (
        client.get(
            "/api/status/artifacts", headers={"Authorization": f"Bearer {TOKEN}"}
        ).status_code
        == 503
    )

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.backup import BackupContractError, create_backup, verify_backup_bundle
from scripts.restore_verify import restore_backup
from tests.training_helpers import build_training_corpus
from webui.store import Store


def _rewrite_manifest(bundle: Path, manifest: dict) -> None:
    data = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    (bundle / "manifest.json").write_bytes(data)
    lines = (bundle / "checksums.sha256").read_text(encoding="ascii").splitlines()
    rewritten = [
        f"{hashlib.sha256(data).hexdigest()}  manifest.json"
        if line.endswith("  manifest.json")
        else line
        for line in lines
    ]
    (bundle / "checksums.sha256").write_text("\n".join(rewritten) + "\n", encoding="ascii")
    (bundle / "READY").touch()


def test_backup_copies_database_and_content_and_restores_atomically(tmp_path: Path) -> None:
    build_training_corpus(tmp_path)
    secret = tmp_path / "data/status-api-token"
    secret.write_text("operator_secret_must_not_leave_source", encoding="ascii")
    secret.chmod(0o600)

    bundle = create_backup(tmp_path, tmp_path / "backups", backup_id="test-backup")
    report = verify_backup_bundle(bundle)
    assert report["schema"] == "avaas/backup@v1"
    assert report["integrity"] == "ok"
    assert report["takes"] > 0
    assert report["files"] > 0
    assert (bundle / "READY").read_bytes() == b""
    assert not (bundle / "data/status-api-token").exists()

    restored = tmp_path / "restored"
    restored_report = restore_backup(bundle, restored)
    assert restored_report == report
    assert Store.verify_backup(
        restored / "data/corpus.sqlite3", content_root=restored
    )["integrity"] == "ok"
    assert not (restored / "data/status-api-token").exists()


def test_backup_is_append_only_and_detects_content_tampering(tmp_path: Path) -> None:
    build_training_corpus(tmp_path)
    destination = tmp_path / "backups"
    bundle = create_backup(tmp_path, destination, backup_id="immutable-backup")
    with pytest.raises(BackupContractError, match="already exists"):
        create_backup(tmp_path, destination, backup_id="immutable-backup")

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    content = next(item for item in manifest["files"] if item["path"].startswith("data/"))
    (bundle / content["path"]).write_bytes(b"corrupt")
    with pytest.raises(BackupContractError, match="checksum|size"):
        verify_backup_bundle(bundle)


def test_restore_refuses_existing_destination_and_symlinked_bundle(tmp_path: Path) -> None:
    build_training_corpus(tmp_path)
    bundle = create_backup(tmp_path, tmp_path / "backups", backup_id="safe-backup")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(BackupContractError, match="destination already exists"):
        restore_backup(bundle, existing)

    link = tmp_path / "backup-link"
    link.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(BackupContractError, match="non-symlink directory"):
        verify_backup_bundle(link)


def test_backup_manifest_rejects_noncanonical_timestamp_type(tmp_path: Path) -> None:
    build_training_corpus(tmp_path)
    bundle = create_backup(tmp_path, tmp_path / "backups", backup_id="typed-backup")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["created_at"] = 7
    _rewrite_manifest(bundle, manifest)

    with pytest.raises(BackupContractError, match="created_at"):
        verify_backup_bundle(bundle)


def test_backup_verification_reads_bundle_files_through_safe_descriptors(
    tmp_path: Path, monkeypatch
) -> None:
    build_training_corpus(tmp_path)
    bundle = create_backup(tmp_path, tmp_path / "backups", backup_id="descriptor-backup")
    original = Path.read_bytes

    def reject_followable_read(path: Path) -> bytes:
        if path.is_relative_to(bundle):
            raise AssertionError("bundle verification used a followable path read")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", reject_followable_read)
    assert verify_backup_bundle(bundle)["integrity"] == "ok"

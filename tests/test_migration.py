from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from webui.store import MAX_LEGACY_MANIFEST_BYTES, Store, StoreContractError

PROMPT = {
    "id": "sec13_001__satraj",
    "corpus_version": "2026.07.16",
    "speaker_id": "satraj",
    "voice_model_id": "satraj-piranesi",
    "identity": "satraj",
    "text": "This is Satraj.",
    "section": "sec13",
    "kind": "identity",
    "source": "avaas_local",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _legacy_record() -> dict:
    return {
        "id": PROMPT["id"],
        "section": PROMPT["section"],
        "idx": 1,
        "kind": PROMPT["kind"],
        "prompt_text": PROMPT["text"],
        "raw_path": "data/raw/legacy.wav",
        "processed_path": "data/processed/legacy.wav",
        "transcript": PROMPT["text"],
        "asr_hypothesis": PROMPT["text"],
        "qc": {"duration": 1.0, "flags": []},
        "source": "browser-mic",
        "ts": "2026-07-16T18:00:00-0400",
    }


def test_bounded_legacy_import_is_deterministic_idempotent_and_quarantines(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.migrate()
    store.register_prompts([PROMPT])
    for relative, data in (
        ("data/raw/legacy.wav", b"raw"),
        ("data/processed/legacy.wav", b"processed"),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    invalid_text = _legacy_record()
    invalid_text["prompt_text"] = "This is not the registered prompt."
    escaping = _legacy_record()
    escaping["raw_path"] = "../outside.wav"
    invalid_qc = _legacy_record()
    invalid_qc["qc"] = {"duration": "not-a-number", "flags": []}
    manifest = tmp_path / "data/processed/manifest.jsonl"
    manifest.write_text(
        "\n".join(
            (
                json.dumps(_legacy_record(), sort_keys=True),
                json.dumps(invalid_text, sort_keys=True),
                json.dumps(escaping, sort_keys=True),
                json.dumps(invalid_qc, sort_keys=True),
                "{malformed",
            )
        )
        + "\n"
    )

    report = store.import_legacy_manifest(manifest)
    assert report["schema"] == "avaas/legacy-import-report@v1"
    assert report["manifest_sha256"] == _sha(manifest.read_bytes())
    assert report["imported"] == 1
    assert report["quarantined"] == 4
    assert [item["line"] for item in report["quarantine"]] == [2, 3, 4, 5]
    assert len(store.load_accepted_records()) == 1
    assert (tmp_path / "data/migration/legacy-manifest-report.json").exists()

    repeated = store.import_legacy_manifest(manifest)
    assert repeated == report
    assert len(store.load_accepted_records()) == 1


def test_legacy_import_rejects_oversize_before_parsing(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.migrate()
    store.register_prompts([PROMPT])
    manifest = tmp_path / "manifest.jsonl"
    with manifest.open("wb") as handle:
        handle.seek(MAX_LEGACY_MANIFEST_BYTES)
        handle.write(b"x")
    with pytest.raises(StoreContractError, match="size"):
        store.import_legacy_manifest(manifest)


def test_legacy_import_quarantines_invalid_qc_without_partial_rows(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.migrate()
    store.register_prompts([PROMPT])
    invalid = _legacy_record()
    invalid["qc"] = {"duration": "not-a-number", "flags": []}
    for relative in ("data/raw/legacy.wav", "data/processed/legacy.wav"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(invalid) + "\n")
    report = store.import_legacy_manifest(manifest)
    assert report["imported"] == 0
    assert report["quarantined"] == 1
    assert "qc" in report["quarantine"][0]["reason"]
    assert store.load_accepted_records() == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record.update({"id": ["not", "a", "string"]}),
        lambda record: record.update({"unexpected_metric": float("nan")}),
    ],
)
def test_legacy_import_quarantines_malicious_json_shapes(
    tmp_path: Path, mutation
) -> None:
    store = Store(tmp_path)
    store.migrate()
    store.register_prompts([PROMPT])
    record = _legacy_record()
    mutation(record)
    for relative in ("data/raw/legacy.wav", "data/processed/legacy.wav"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(record) + "\n")
    report = store.import_legacy_manifest(manifest)
    assert report["imported"] == 0
    assert report["quarantined"] == 1
    assert store.load_accepted_records() == []


def test_legacy_import_refuses_non_new_database(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.migrate()
    store.register_prompts([PROMPT])
    raw = tmp_path / "data/raw/existing.wav"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"existing")
    store.create_take(
        take_id="existing",
        prompt_id=PROMPT["id"],
        corpus_version=PROMPT["corpus_version"],
        raw_path=str(raw.relative_to(tmp_path)),
        raw_sha256=_sha(b"existing"),
        source="test",
        metadata={},
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(_legacy_record()) + "\n")
    with pytest.raises(StoreContractError, match="new database"):
        store.import_legacy_manifest(manifest)


def test_legacy_report_parent_symlink_fails_without_external_write(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.migrate()
    store.register_prompts([PROMPT])
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "data/migration").symlink_to(outside, target_is_directory=True)
    manifest = tmp_path / "manifest.jsonl"
    invalid = _legacy_record()
    invalid["id"] = "unknown"
    manifest.write_text(json.dumps(invalid) + "\n")
    with pytest.raises(StoreContractError, match="symlink"):
        store.import_legacy_manifest(manifest)
    assert not (outside / "legacy-manifest-report.json").exists()

from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from webui.store import Store, StoreConflict, StoreContractError


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


def _file(root: Path, relative: str, data: bytes) -> tuple[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return relative, _sha(data)


def _store(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    store.migrate()
    store.register_prompts([PROMPT])
    return store


def _take(store: Store, root: Path, take_id: str, payload: bytes = b"raw") -> str:
    path, digest = _file(root, f"data/raw/{take_id}.wav", payload)
    store.create_take(
        take_id=take_id,
        prompt_id=PROMPT["id"],
        corpus_version=PROMPT["corpus_version"],
        raw_path=path,
        raw_sha256=digest,
        source="test",
        metadata={"idx": 1},
    )
    return digest


def test_migrations_and_connection_pragmas_are_ordered_and_idempotent(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.migrate()
    first = store.migration_rows()
    store.migrate()
    assert store.migration_rows() == first
    assert [(row["version"], row["name"]) for row in first] == [
        (1, "initial"),
        (2, "import_legacy"),
        (3, "capture_sessions"),
        (4, "voice_training_jobs"),
    ]
    assert all(len(row["sha256"]) == 64 for row in first)
    assert store.diagnostics() == {
        "journal_mode": "wal",
        "synchronous": 2,
        "foreign_keys": 1,
        "busy_timeout": 5000,
    }


def test_unknown_applied_migration_fails_closed(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.migrate()
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "INSERT INTO schema_migrations(version,name,sha256,applied_at) VALUES (99,?,?,?)",
            ("unknown", "0" * 64, "now"),
        )
        connection.commit()
    with pytest.raises(StoreContractError, match="unknown applied migration"):
        store.migrate()


def test_prompts_takes_derivatives_qc_and_history_are_immutable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _take(store, tmp_path, "take-a")
    derivative_path, derivative_sha = _file(
        tmp_path, "data/derivatives/take-a-24k.wav", b"processed"
    )
    store.add_derivative(
        derivative_id="derivative-a",
        take_id="take-a",
        purpose="serve-24k",
        path=derivative_path,
        sha256=derivative_sha,
        sample_rate=24_000,
        channels=1,
        sample_format="pcm_s16le",
        frames=9,
        duration_ms=1,
        parameters={"tool": "fixture"},
    )
    store.add_qc(
        qc_id="qc-a",
        take_id="take-a",
        status="pass",
        flags=[],
        metrics={"duration": 1.0},
        transcript="This is Satraj.",
    )
    assert store.accept(
        prompt_id=PROMPT["id"],
        corpus_version=PROMPT["corpus_version"],
        take_id="take-a",
        expected_generation=0,
    ) == 1

    immutable_updates = (
        "UPDATE prompts SET prompt_text='changed'",
        "UPDATE takes SET source='changed'",
        "UPDATE derivatives SET purpose='changed'",
        "UPDATE qc_results SET status='fail'",
        "UPDATE acceptance_history SET reason='changed'",
    )
    for statement in immutable_updates:
        with sqlite3.connect(store.database_path) as connection:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(statement)
                connection.commit()


def test_acceptance_compare_and_swap_allows_one_concurrent_retake(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _take(store, tmp_path, "take-a", b"a")
    _take(store, tmp_path, "take-b", b"b")

    def accept(take_id: str) -> int | str:
        try:
            return store.accept(
                prompt_id=PROMPT["id"],
                corpus_version=PROMPT["corpus_version"],
                take_id=take_id,
                expected_generation=0,
            )
        except StoreConflict:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(accept, ("take-a", "take-b")))
    assert sorted(map(str, outcomes)) == ["1", "conflict"]
    accepted = store.accepted(PROMPT["id"], PROMPT["corpus_version"])
    assert accepted is not None and accepted["take_id"] in {"take-a", "take-b"}
    assert accepted["generation"] == 1
    assert len(store.acceptance_history(PROMPT["id"], PROMPT["corpus_version"])) == 1


def test_transaction_rolls_back_on_base_exception(tmp_path: Path) -> None:
    store = _store(tmp_path)

    class SimulatedCrash(BaseException):
        pass

    with pytest.raises(SimulatedCrash):
        with store.transaction() as connection:
            connection.execute(
                "INSERT INTO training_jobs(job_id,state,manifest_json,created_at,updated_at) "
                "VALUES ('crash','queued','{}','now','now')"
            )
            raise SimulatedCrash()
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM training_jobs").fetchone()[0] == 0


def test_tombstone_clears_pointer_but_preserves_files_and_audit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _take(store, tmp_path, "take-a")
    store.accept(
        prompt_id=PROMPT["id"],
        corpus_version=PROMPT["corpus_version"],
        take_id="take-a",
        expected_generation=0,
    )
    raw = tmp_path / "data/raw/take-a.wav"
    assert store.tombstone("take-a", reason="operator requested retake", expected_generation=1) == 2
    assert raw.exists()
    assert store.accepted(PROMPT["id"], PROMPT["corpus_version"]) is None
    history = store.acceptance_history(PROMPT["id"], PROMPT["corpus_version"])
    assert [event["action"] for event in history] == ["accept", "tombstone"]
    with pytest.raises(StoreContractError, match="tombstoned"):
        store.accept(
            prompt_id=PROMPT["id"],
            corpus_version=PROMPT["corpus_version"],
            take_id="take-a",
            expected_generation=2,
        )


def test_checksums_jsonl_export_and_backup_restore_are_verified(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _take(store, tmp_path, "take-a", b"immutable raw")
    derivative_path, derivative_sha = _file(
        tmp_path, "data/processed/take-a.wav", b"immutable processed"
    )
    store.add_derivative(
        derivative_id="derivative-a",
        take_id="take-a",
        purpose="serve-24k",
        path=derivative_path,
        sha256=derivative_sha,
        sample_rate=24_000,
        channels=1,
        sample_format="pcm_s16le",
        frames=19,
        duration_ms=1,
        parameters={},
    )
    store.add_qc(
        qc_id="qc-a",
        take_id="take-a",
        status="pass",
        flags=[],
        metrics={"duration": 1.25},
        transcript=PROMPT["text"],
    )
    store.accept(
        prompt_id=PROMPT["id"],
        corpus_version=PROMPT["corpus_version"],
        take_id="take-a",
        expected_generation=0,
    )
    assert store.validate_accepted_checksums() == {"takes": 1, "files": 2}

    exported = tmp_path / "exports/manifest.jsonl"
    store.export_jsonl(exported)
    rows = [json.loads(line) for line in exported.read_text().splitlines()]
    assert [row["id"] for row in rows] == [PROMPT["id"]]
    assert rows[0]["processed_path"] == derivative_path

    backup = tmp_path / "backups/corpus.sqlite3"
    store.backup(backup)
    assert Store.verify_backup(backup, content_root=tmp_path) == {
        "integrity": "ok",
        "takes": 1,
        "files": 2,
    }

    (tmp_path / derivative_path).write_bytes(b"corrupt")
    with pytest.raises(StoreContractError, match="checksum"):
        store.validate_accepted_checksums()


@pytest.mark.parametrize("path", ["../escape.wav", "/absolute.wav", "bad\\path.wav"])
def test_content_paths_fail_closed(path: str, tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(StoreContractError, match="path"):
        store.create_take(
            take_id="take-a",
            prompt_id=PROMPT["id"],
            corpus_version=PROMPT["corpus_version"],
            raw_path=path,
            raw_sha256=_sha(b"x"),
            source="test",
            metadata={},
        )


def test_database_parent_symlink_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(StoreContractError, match="symlink"):
        Store(root).migrate()
    assert not (outside / "corpus.sqlite3").exists()

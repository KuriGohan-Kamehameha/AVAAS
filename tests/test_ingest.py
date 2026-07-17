from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from webui import ingest
from webui.store import Store


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


def test_stage_stream_is_bounded_hashed_and_cleans_failures(tmp_path: Path) -> None:
    staged = ingest.stage_stream(tmp_path, [b"abc", b"def"], max_bytes=6)
    assert staged.size == 6
    assert staged.sha256 == hashlib.sha256(b"abcdef").hexdigest()
    assert staged.path.read_bytes() == b"abcdef"
    assert staged.path.parent == tmp_path / "data" / "staging"
    staged.cleanup()
    assert not staged.path.exists()

    with pytest.raises(ingest.IngestError, match="empty"):
        ingest.stage_stream(tmp_path, [], max_bytes=6)
    with pytest.raises(ingest.IngestError, match="exceeds"):
        ingest.stage_stream(tmp_path, [b"1234", b"5678"], max_bytes=7)
    assert list((tmp_path / "data" / "staging").iterdir()) == []


def test_stage_stream_enforces_chunk_count_size_and_deadline(tmp_path: Path) -> None:
    with pytest.raises(ingest.IngestError, match="chunk"):
        ingest.stage_stream(
            tmp_path,
            [b"x" * (ingest.UPLOAD_CHUNK_BYTES + 1)],
            max_bytes=ingest.MAX_UPLOAD_BYTES,
        )

    times = iter((0.0, 0.4, 2.0))
    with pytest.raises(ingest.IngestError, match="deadline"):
        ingest.stage_stream(
            tmp_path,
            [b"a", b"b"],
            max_bytes=2,
            timeout_seconds=1.0,
            clock=lambda: next(times),
        )

    def disconnected():
        yield b"partial"
        raise ConnectionError("client disconnected")

    with pytest.raises(ConnectionError, match="disconnected"):
        ingest.stage_stream(tmp_path, disconnected(), max_bytes=32)
    assert list((tmp_path / "data" / "staging").iterdir()) == []


def test_publish_content_is_immutable_content_addressed_and_idempotent(tmp_path: Path) -> None:
    first = ingest.stage_stream(tmp_path, [b"audio"], max_bytes=16)
    published = ingest.publish_content(tmp_path, first, label="master-48k", extension=".wav")
    assert published.sha256 == hashlib.sha256(b"audio").hexdigest()
    assert published.relative_path.startswith("data/audio/sha256/")
    assert published.relative_path.endswith(".master-48k.wav")
    assert (tmp_path / published.relative_path).read_bytes() == b"audio"
    assert not first.path.exists()

    second = ingest.stage_stream(tmp_path, [b"audio"], max_bytes=16)
    repeated = ingest.publish_content(tmp_path, second, label="master-48k", extension=".wav")
    assert repeated == published
    assert not second.path.exists()


def test_publish_rejects_escape_symlink_and_untrusted_label(tmp_path: Path) -> None:
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"audio")
    escaped = ingest.StagedUpload(
        path=outside,
        sha256=hashlib.sha256(b"audio").hexdigest(),
        size=5,
    )
    with pytest.raises(ingest.IngestError, match="staging"):
        ingest.publish_content(tmp_path, escaped, label="master-48k", extension=".wav")

    staged = ingest.stage_stream(tmp_path, [b"audio"], max_bytes=16)
    target = staged.path.with_name("link")
    target.symlink_to(staged.path)
    linked = ingest.StagedUpload(target, staged.sha256, staged.size)
    with pytest.raises(ingest.IngestError, match="symlink"):
        ingest.publish_content(tmp_path, linked, label="master-48k", extension=".wav")
    target.unlink()
    staged.cleanup()

    staged = ingest.stage_stream(tmp_path, [b"audio"], max_bytes=16)
    with pytest.raises(ingest.IngestError, match="label"):
        ingest.publish_content(tmp_path, staged, label="../../escape", extension=".wav")
    staged.cleanup()


@pytest.mark.parametrize("crash_method", ["_insert_take", "_insert_derivative", "_insert_qc", "_accept_tx"])
def test_capture_database_commit_rolls_back_every_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crash_method: str
) -> None:
    store = Store(tmp_path)
    store.migrate()
    store.register_prompts([PROMPT])
    master = tmp_path / "data/audio/master.wav"
    derivative = tmp_path / "data/audio/serve.wav"
    master.parent.mkdir(parents=True)
    master.write_bytes(b"master")
    derivative.write_bytes(b"serve")

    original = getattr(store, crash_method)

    def crash_after_write(*args, **kwargs):
        original(*args, **kwargs)
        raise KeyboardInterrupt(crash_method)

    monkeypatch.setattr(store, crash_method, crash_after_write)
    with pytest.raises(KeyboardInterrupt, match=crash_method):
        store.commit_capture(
            take_id="take-atomic",
            prompt_id=PROMPT["id"],
            corpus_version=PROMPT["corpus_version"],
            raw_path="data/audio/master.wav",
            raw_sha256=hashlib.sha256(b"master").hexdigest(),
            source="browser-mic",
            metadata={"capture": "atomic"},
            derivatives=[
                {
                    "derivative_id": "derivative-atomic",
                    "purpose": "serve-24k",
                    "path": "data/audio/serve.wav",
                    "sha256": hashlib.sha256(b"serve").hexdigest(),
                    "sample_rate": 24_000,
                    "channels": 1,
                    "sample_format": "pcm_s16le",
                    "frames": 24_000,
                    "duration_ms": 1_000,
                    "parameters": {"dsp": "fixture"},
                }
            ],
            qc={
                "qc_id": "qc-atomic",
                "status": "pass",
                "flags": [],
                "metrics": {"duration": 1.0},
                "transcript": "This is Satraj.",
            },
            expected_generation=0,
        )

    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM takes").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM derivatives").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM qc_results").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM acceptances").fetchone()[0] == 0


def test_review_capture_requires_audited_override_before_acceptance(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.migrate()
    store.register_prompts([PROMPT])
    for name, payload in (("master.wav", b"master"), ("serve.wav", b"serve")):
        path = tmp_path / "data/audio" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    common = {
        "take_id": "take-review",
        "prompt_id": PROMPT["id"],
        "corpus_version": PROMPT["corpus_version"],
        "raw_path": "data/audio/master.wav",
        "raw_sha256": hashlib.sha256(b"master").hexdigest(),
        "source": "browser-mic",
        "metadata": {"capture": "review"},
        "derivatives": [
            {
                "derivative_id": "derivative-review",
                "purpose": "serve-24k",
                "path": "data/audio/serve.wav",
                "sha256": hashlib.sha256(b"serve").hexdigest(),
                "sample_rate": 24_000,
                "channels": 1,
                "sample_format": "pcm_s16le",
                "frames": 24_000,
                "duration_ms": 1_000,
                "parameters": {"dsp": "fixture"},
            }
        ],
        "qc": {
            "qc_id": "qc-review",
            "status": "review_required",
            "flags": ["asr-unavailable"],
            "metrics": {"duration": 1.0},
            "transcript": None,
        },
        "expected_generation": 0,
    }
    result = store.commit_capture(**common)
    assert result == {"accepted": False, "generation": 0, "take_id": "take-review"}
    assert store.accepted(PROMPT["id"], PROMPT["corpus_version"]) is None

    overridden = store.commit_capture(**common, override_reason="Manual transcript review passed.")
    assert overridden["accepted"] is True
    accepted = store.accepted(PROMPT["id"], PROMPT["corpus_version"])
    assert accepted is not None
    assert accepted["override_reason"] == "Manual transcript review passed."

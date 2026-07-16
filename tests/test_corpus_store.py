from __future__ import annotations

import hashlib
from pathlib import Path

from webui import corpus
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
    "idx": 1,
}


def _sections() -> list[dict]:
    return [
        {
            "section": "sec13",
            "title": "Identity",
            "target_min": 1.0,
            "prompts": [PROMPT],
        }
    ]


def _record(root: Path, suffix: str, payload: bytes) -> dict:
    raw = root / f"data/raw/{suffix}.wav"
    processed = root / f"data/processed/{suffix}.wav"
    raw.parent.mkdir(parents=True, exist_ok=True)
    processed.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"raw-" + payload)
    processed.write_bytes(b"processed-" + payload)
    return {
        **PROMPT,
        "prompt_text": PROMPT["text"],
        "raw_path": str(raw.relative_to(root)),
        "processed_path": str(processed.relative_to(root)),
        "transcript": PROMPT["text"],
        "asr_hypothesis": PROMPT["text"],
        "qc": {"duration": 1.0, "flags": [], "peak_dbfs": -6.0},
        "prompt_source": PROMPT["source"],
        "source": "browser-mic",
        "ts": "2026-07-16T18:00:00-0400",
    }


def test_compatibility_facade_preserves_retake_then_tombstones_pointer(tmp_path: Path) -> None:
    corpus.initialize(tmp_path, _sections())
    first = _record(tmp_path, "first", b"one")
    second = _record(tmp_path, "second", b"two")
    corpus.append_record(tmp_path, first)
    first_take = corpus.load_manifest(tmp_path)[0]["take_id"]
    corpus.append_record(tmp_path, second)

    records = corpus.load_manifest(tmp_path)
    assert len(records) == 1
    assert records[0]["raw_path"] == second["raw_path"]
    assert records[0]["take_id"] != first_take
    assert (tmp_path / first["raw_path"]).exists()
    assert (tmp_path / first["processed_path"]).exists()
    history = Store(tmp_path).acceptance_history(PROMPT["id"], PROMPT["corpus_version"])
    assert [item["action"] for item in history] == ["accept", "repoint"]

    corpus.tombstone_record(tmp_path, PROMPT["id"], reason="record again")
    assert corpus.load_manifest(tmp_path) == []
    assert (tmp_path / second["raw_path"]).exists()
    assert (tmp_path / second["processed_path"]).exists()


def test_progress_is_a_pure_store_read(tmp_path: Path) -> None:
    corpus.initialize(tmp_path, _sections())
    corpus.append_record(tmp_path, _record(tmp_path, "one", b"one"))
    database = tmp_path / "data/corpus.sqlite3"
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    first = corpus.progress(tmp_path, _sections())
    second = corpus.progress(tmp_path, _sections())
    after = hashlib.sha256(database.read_bytes()).hexdigest()
    assert first == second
    assert first["recorded_count"] == 1
    assert before == after

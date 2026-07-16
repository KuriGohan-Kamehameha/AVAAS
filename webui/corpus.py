#!/usr/bin/env python3
"""Corpus compatibility façade over the transactional immutable store.

Legacy callers still append/load record-shaped dictionaries, but SQLite is the
source of truth. Retakes create new immutable takes and atomically repoint the
single accepted pointer; deletion creates a tombstone and never unlinks audio.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from .store import Store, StoreContractError

# Readiness thresholds — from RECORDING-SCRIPT.md "Quality checklist".
TARGET_MIN = 60.0           # minimum clean minutes before training may start
SECTION_COVERAGE = 0.80     # each section must reach 80% of its target minutes
WER_FLAG = 0.25             # transcript disagreement above this flags a misread
SNR_FLAG_DB = 12.0          # below this the clip is too noisy to keep

def _paths(root: Path) -> dict:
    return {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "manifest": root / "data" / "processed" / "manifest.jsonl",
        "database": root / "data" / "corpus.sqlite3",
        "room_tone": root / "data" / "room_tone_48k.wav",
    }


def qc_flags(qc: dict, wer_val: float | None) -> list[str]:
    """Derive human-readable QC flags. Empty list == clean."""
    flags: list[str] = []
    if qc.get("clipping"):
        flags.append("clipping")
    if qc.get("snr_db") is not None and qc["snr_db"] < SNR_FLAG_DB:
        flags.append("noisy")
    if qc.get("peak_dbfs") is not None and qc["peak_dbfs"] < -30.0:
        flags.append("too-quiet")
    if wer_val is not None and wer_val > WER_FLAG:
        flags.append("misread")
    if qc.get("duration", 0.0) < 0.4:
        flags.append("too-short")
    return flags


def _prompt_snapshot(record: dict) -> dict:
    return {
        "id": record["id"],
        "corpus_version": record.get("corpus_version", "legacy-v1"),
        "speaker_id": record.get("speaker_id", "satraj"),
        "voice_model_id": record.get("voice_model_id", "satraj-piranesi"),
        "identity": record.get("identity", "shared"),
        "text": record["prompt_text"],
        "section": record["section"],
        "kind": record["kind"],
        "source": record.get("prompt_source", record.get("source", "legacy")),
    }


def initialize(root: Path, sections: list[dict]) -> Store:
    """Migrate, register exact prompt snapshots, then import legacy JSONL once."""
    root = Path(root)
    prompts = [prompt for section in sections for prompt in section.get("prompts", [])]
    store = Store(root)
    store.migrate()
    store.register_prompts(prompts)
    manifest = _paths(root)["manifest"]
    if manifest.exists():
        store.import_legacy_manifest(manifest)
    return store


def append_record(root: Path, record: dict) -> None:
    """Persist a fully processed record, accepting it only after every insert succeeds."""
    if not isinstance(record, dict) or "id" not in record:
        raise StoreContractError("record needs an id")
    root = Path(root)
    store = Store(root)
    if not store.database_path.exists():
        store.migrate()
    snapshot = _prompt_snapshot(record)
    store.register_prompts([snapshot])
    raw_path, raw_sha, _ = store.hash_content(record.get("raw_path"), "raw")
    processed_path, processed_sha, _ = store.hash_content(
        record.get("processed_path"), "processed"
    )
    take_id = "take-" + hashlib.sha256(
        f"{record['id']}\0{snapshot['corpus_version']}\0{raw_sha}\0{processed_sha}".encode()
    ).hexdigest()[:48]
    store.create_take(
        take_id=take_id,
        prompt_id=record["id"],
        corpus_version=snapshot["corpus_version"],
        raw_path=raw_path,
        raw_sha256=raw_sha,
        source=record.get("source", "unknown"),
        metadata=record,
    )
    duration = float(record.get("qc", {}).get("duration", 0.0))
    if not math.isfinite(duration) or duration < 0:
        raise StoreContractError("record duration must be finite and non-negative")
    store.add_derivative(
        derivative_id="derivative-" + hashlib.sha256(
            f"{take_id}\0{processed_sha}".encode()
        ).hexdigest()[:48],
        take_id=take_id,
        purpose="serve-24k",
        path=processed_path,
        sha256=processed_sha,
        sample_rate=24_000,
        channels=1,
        sample_format="pcm_s16le",
        frames=max(1, min(200_000_000, int(duration * 24_000))),
        duration_ms=max(0, min(7_200_000, int(duration * 1_000))),
        parameters={"compatibility_facade": "v1"},
    )
    qc = dict(record.get("qc", {}))
    flags = qc.pop("flags", [])
    store.add_qc(
        qc_id="qc-" + hashlib.sha256(take_id.encode()).hexdigest()[:48],
        take_id=take_id,
        status="pass" if not flags else "review_required",
        flags=flags,
        metrics=qc,
        transcript=record.get("transcript", ""),
    )
    state = store.acceptance_state(record["id"], snapshot["corpus_version"])
    if state is not None and state["take_id"] == take_id:
        return
    store.accept(
        prompt_id=record["id"],
        corpus_version=snapshot["corpus_version"],
        take_id=take_id,
        expected_generation=state["generation"] if state is not None else 0,
    )


def load_manifest(root: Path) -> list[dict]:
    store = Store(root)
    if not store.database_path.exists():
        return []
    return store.load_accepted_records()


def tombstone_record(root: Path, prompt_id: str, *, reason: str) -> None:
    store = Store(root)
    accepted = store.accepted_for_prompt(prompt_id)
    if accepted is None:
        raise StoreContractError("no accepted take for prompt")
    store.tombstone(
        accepted["take_id"],
        reason=reason,
        expected_generation=accepted["generation"],
    )


def _is_clean(rec: dict) -> bool:
    return not rec.get("qc", {}).get("flags")


def progress(root: Path, sections: list[dict]) -> dict:
    """Per-section coverage + global clean minutes + readiness verdict."""
    records = load_manifest(root)
    by_id = {r["id"]: r for r in records}
    clean_sec: dict[str, float] = {}
    total_clean = 0.0

    for r in records:
        if not _is_clean(r):
            continue
        dur_min = r.get("qc", {}).get("duration", 0.0) / 60.0
        total_clean += dur_min
        clean_sec[r["section"]] = clean_sec.get(r["section"], 0.0) + dur_min

    sec_rows = []
    for s in sections:
        recorded = sum(1 for p in s["prompts"] if p["id"] in by_id)
        mins = clean_sec.get(s["section"], 0.0)
        tgt = s["target_min"] or 0.0
        cov = (mins / tgt) if tgt > 0 else (1.0 if recorded else 0.0)
        sec_rows.append({
            "section": s["section"], "title": s["title"],
            "recorded": recorded, "total": len(s["prompts"]),
            "minutes": round(mins, 2), "target_min": tgt,
            "coverage": round(min(cov, 1.0), 3),
        })

    # A section with no prompts (e.g. a corpus fetch failed) can never be
    # covered — exclude it from the gate so auto-train can't deadlock on it.
    under = [row for row in sec_rows
             if row["target_min"] > 0 and row["total"] > 0
             and row["coverage"] < SECTION_COVERAGE]
    flagged = [{"id": r["id"], "flags": r["qc"]["flags"],
                "section": r["section"], "text": r.get("prompt_text", "")}
               for r in records if not _is_clean(r)]

    ready = total_clean >= TARGET_MIN and not under
    return {
        "total_clean_min": round(total_clean, 2),
        "target_min": TARGET_MIN,
        "ready": ready,
        "sections": sec_rows,
        "under_represented": [u["section"] for u in under],
        "flagged": flagged,
        "recorded_count": len(records),
    }

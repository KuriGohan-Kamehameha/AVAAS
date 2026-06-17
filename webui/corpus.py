#!/usr/bin/env python3
"""Corpus state: save/label/organize clips, manifest, progress, readiness.

The manifest (data/processed/manifest.jsonl) is the single source of truth for
what has been recorded. One JSON line per accepted clip. A clip is "clean" if it
carries no QC flags; only clean minutes count toward the training-readiness gate.

Thread-safe append (the FastAPI threadpool can process concurrent uploads).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

# Readiness thresholds — from RECORDING-SCRIPT.md "Quality checklist".
TARGET_MIN = 60.0           # minimum clean minutes before training may start
SECTION_COVERAGE = 0.80     # each section must reach 80% of its target minutes
WER_FLAG = 0.25             # transcript disagreement above this flags a misread
SNR_FLAG_DB = 12.0          # below this the clip is too noisy to keep

_LOCK = threading.Lock()


def _paths(root: Path) -> dict:
    return {
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "manifest": root / "data" / "processed" / "manifest.jsonl",
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


def append_record(root: Path, record: dict) -> None:
    """Append one clip record to the manifest, replacing any prior same-id line."""
    assert "id" in record, "record needs an id"
    mpath = _paths(root)["manifest"]
    mpath.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        existing = load_manifest(root)
        existing = [r for r in existing if r.get("id") != record["id"]]
        existing.append(record)
        tmp = mpath.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as fp:
            for r in existing:
                fp.write(json.dumps(r) + "\n")
        tmp.replace(mpath)       # atomic swap


def load_manifest(root: Path) -> list[dict]:
    mpath = _paths(root)["manifest"]
    if not mpath.exists():
        return []
    out: list[dict] = []
    for line in mpath.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue             # skip a corrupt line, never crash the server
    return out


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

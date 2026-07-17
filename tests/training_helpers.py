from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path

import numpy as np

from webui import audio_contracts, corpus, prompts


REPO = Path(__file__).resolve().parents[1]
IDENTITY_PROMPT_IDS = (
    "sec13_001__satraj",
    "sec13_001__piranesi",
    "sec13_002__satraj",
    "sec13_002__piranesi",
    "sec13_003__satraj",
    "sec13_003__piranesi",
)


def _tone(sample_rate: int, seconds: float, frequency: float) -> np.ndarray:
    frames = int(sample_rate * seconds)
    timeline = np.arange(frames, dtype=np.float64) / sample_rate
    return (0.08 * np.sin(2.0 * math.pi * frequency * timeline)).astype(np.float32)


def _write_audio(path: Path, purpose: str, *, seconds: float, frequency: float) -> dict:
    spec = audio_contracts.AUDIO_SPECS[purpose]
    audio_contracts.write_pcm16_wav(
        path,
        _tone(spec.sample_rate, seconds, frequency),
        spec,
        min_seconds=0.4,
        max_seconds=10.0,
    )
    return audio_contracts.inspect_pcm16_wav(
        path,
        spec,
        min_seconds=0.4,
        max_seconds=10.0,
    )


def build_training_corpus(
    root: Path,
    *,
    prompt_ids: tuple[str, ...] = IDENTITY_PROMPT_IDS,
    seconds: float = 0.6,
) -> list[dict]:
    shutil.copytree(REPO / "prompts", root / "prompts")
    sections = prompts.compile_sections(root)
    corpus.initialize(root, sections)
    by_id = {
        prompt["id"]: prompt
        for section in sections
        for prompt in section["prompts"]
    }
    for index, prompt_id in enumerate(prompt_ids):
        prompt = by_id[prompt_id]
        base = root / "data" / "fixture-audio" / prompt_id
        master_path = base.with_name(base.name + "-master-48k.wav")
        master = _write_audio(
            master_path,
            "master-48k",
            seconds=seconds,
            frequency=180.0 + index * 11.0,
        )
        derivatives = []
        for purpose in ("serve-24k", "piper-22050", "wake-16k"):
            path = base.with_name(base.name + f"-{purpose}.wav")
            observed = _write_audio(
                path,
                purpose,
                seconds=seconds,
                frequency=180.0 + index * 11.0,
            )
            derivatives.append(
                {
                    "purpose": purpose,
                    "path": str(path.relative_to(root)),
                    "sha256": observed["sha256"],
                    "sample_rate": observed["sample_rate"],
                    "channels": observed["channels"],
                    "sample_format": observed["sample_format"],
                    "frames": observed["frames"],
                    "duration_ms": observed["duration_ms"],
                    "parameters": {"schema": "avaas/test-audio@v1"},
                }
            )
        record = {
            **prompt,
            "prompt_text": prompt["text"],
            "prompt_source": prompt["source"],
            "raw_path": str(master_path.relative_to(root)),
            "raw_sha256": master["sha256"],
            "processed_path": derivatives[0]["path"],
            "derivatives": derivatives,
            "transcript": prompt["text"],
            "asr_hypothesis": prompt["text"],
            "asr": {"status": "ok", "model": "fixture", "reason": None},
            "qc": {
                "duration": seconds,
                "flags": [],
                "status": "pass",
                "peak_dbfs": -21.9,
                "snr_db": 40.0,
                "wer": 0.0,
                "asr_status": "ok",
            },
            "source": "fixture",
            "ts": "2026-07-16T20:30:00Z",
        }
        corpus.append_record(root, record)
    return sections


def fixture_policy():
    from webui.readiness import ReadinessPolicy

    return ReadinessPolicy(
        minimum_clean_seconds=3.0,
        section_coverage_ratio=0.0,
        kind_minimum_seconds=(("identity", 3.0),),
        required_identity_groups=("sec13_001", "sec13_002", "sec13_003"),
        minimum_reference_count=2,
        minimum_validation_groups=1,
        minimum_test_groups=1,
    )


def healthy_worker() -> dict:
    return {
        "schema": "avaas/trainer-health@v1",
        "healthy": True,
        "worker_id": "fixture-worker",
        "profiles": ["cosyvoice3/expressive-zero-shot", "piper/medium"],
        "checked_at": "2026-07-16T20:30:00Z",
    }


def tree_receipt(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and not item.name.endswith(("-shm", "-wal"))
    ):
        relative = str(path.relative_to(root)).encode()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()

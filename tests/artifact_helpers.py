from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from webui import audio_contracts


PRESENTATIONS = [
    {
        "id": "satraj.en-ca.neutral",
        "alias": "satraj",
        "speaker_id": 0,
        "language_tag": "en-CA",
        "phonemizer_voice": "en-us",
        "accent": "general-north-american",
        "delivery": "neutral",
    },
    {
        "id": "piranesi.en-gb.neutral",
        "alias": "piranesi",
        "speaker_id": 1,
        "language_tag": "en-GB",
        "phonemizer_voice": "en-gb",
        "accent": "received-pronunciation",
        "delivery": "neutral",
    },
]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seal(bundle: Path) -> None:
    ready = bundle / "READY"
    ready.unlink(missing_ok=True)
    checksum = bundle / "checksums.sha256"
    checksum.unlink(missing_ok=True)
    files = sorted(
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    checksum.write_text(
        "".join(f"{sha256(bundle / relative)}  {relative}\n" for relative in files),
        encoding="ascii",
    )
    ready.write_bytes(b"")
    os.utime(ready, None)


def refresh_declarations(bundle: Path) -> None:
    manifest = load_json(bundle / "manifest.json")
    for entry in manifest["files"].values():
        entry["sha256"] = sha256(bundle / entry["path"])
    if manifest["schema"] == "avaas/voice-model@v1":
        model_hash = manifest["files"]["model_onnx"]["sha256"]
        for alias in manifest["aliases"]:
            alias["model_sha256"] = model_hash
        manifest["provenance"]["corpus_manifest_sha256"] = manifest["files"][
            "corpus_manifest"
        ]["sha256"]
        manifest["evaluation"]["report_sha256"] = manifest["files"][
            "evaluation_report"
        ]["sha256"]
    else:
        manifest["provenance"]["references_manifest_sha256"] = manifest["files"][
            "references_manifest"
        ]["sha256"]
        manifest["evaluation"]["report_sha256"] = manifest["files"][
            "evaluation_report"
        ]["sha256"]
    write_json(bundle / "manifest.json", manifest)
    seal(bundle)


def write_tone(path: Path, sample_rate: int = 24_000, seconds: float = 0.8) -> dict:
    frames = int(sample_rate * seconds)
    timeline = np.arange(frames, dtype=np.float32) / sample_rate
    samples = np.asarray(0.12 * np.sin(2.0 * np.pi * 180.0 * timeline), dtype=np.float32)
    purpose = {24_000: "serve-24k", 16_000: "wake-16k"}[sample_rate]
    return audio_contracts.write_pcm16_wav(
        path,
        samples,
        audio_contracts.AUDIO_SPECS[purpose],
        min_seconds=0.25,
        max_seconds=5.0,
    )


def _license_entries(components: tuple[str, ...]) -> list[dict]:
    return [
        {
            "component": component,
            "name": f"Fixture {component}",
            "spdx": "LicenseRef-Fixture",
            "source_url": f"https://example.invalid/{component}",
            "license_url": f"https://example.invalid/{component}/license",
            "distribution_restrictions": "Fixture only; do not promote.",
        }
        for component in components
    ]


def build_model_bundle(root: Path, name: str = "satraj-piranesi-medium-v1") -> Path:
    bundle = root / name
    model_path = "model/satraj-piranesi-medium.onnx"
    config_path = model_path + ".json"
    for directory in ("model", "pronunciation", "corpus", "evaluation", "provenance"):
        (bundle / directory).mkdir(parents=True, exist_ok=True)
    (bundle / model_path).write_bytes(b"\x08\x01fixture-onnx\x00")
    write_json(
        bundle / config_path,
        {
            "audio": {"sample_rate": 22_050},
            "num_speakers": 2,
            "speaker_id_map": {item["id"]: item["speaker_id"] for item in PRESENTATIONS},
            "phoneme_type": "espeak",
        },
    )
    write_json(
        bundle / "pronunciation/lexicon.json",
        {
            "schema": "avaas/pronunciation-lexicon@v1",
            "entries": [
                {"name": "Satraj", "phonemes": ["s", "ah", "t", "r", "ah", "j"]},
                {
                    "name": "Piranesi",
                    "phonemes": ["p", "ih", "r", "ah", "n", "eh", "s", "ee"],
                },
            ],
        },
    )
    split_hash, dsp_hash = "1" * 64, "2" * 64
    write_json(
        bundle / "corpus/manifest.json",
        {
            "schema": "avaas/corpus-manifest@v1",
            "corpus_version": "2026.07.16",
            "split_sha256": split_hash,
            "dsp_sha256": dsp_hash,
        },
    )
    thresholds = {
        "max_wer": 0.12,
        "min_speaker_similarity": 0.72,
        "max_realtime_factor": 0.5,
    }
    measured = {"wer": 0.04, "speaker_similarity": 0.81, "realtime_factor": 0.09}
    write_json(
        bundle / "evaluation/report.json",
        {
            "schema": "avaas/voice-evaluation@v1",
            "passed": True,
            "thresholds": thresholds,
            "measured": measured,
            "checks": {
                "cpu_inference": True,
                "cuda_inference": True,
                "finite_audio": True,
                "non_silent_audio": True,
                "satraj_pronunciation": True,
                "piranesi_pronunciation": True,
            },
        },
    )
    write_json(
        bundle / "provenance/licenses.json",
        {
            "schema": "avaas/license-ledger@v1",
            "entries": _license_entries(("piper", "base-checkpoint", "training-corpus")),
        },
    )
    base_hash, trainer_hash = "3" * 64, "4" * 64
    commits = {"avaas": "5" * 40, "piranesi_workspace": "6" * 40}
    write_json(
        bundle / "provenance/training.json",
        {
            "schema": "avaas/training-provenance@v1",
            "base_checkpoint_sha256": base_hash,
            "trainer_sha256": trainer_hash,
            "source_commits": commits,
            "synthetic_material": {
                "used": False,
                "fraction": 0.0,
                "teacher_schema": None,
                "teacher_artifact_id": None,
                "teacher_manifest_sha256": None,
                "teacher_presentations": [],
                "license_component": None,
            },
        },
    )
    (bundle / "model-card.md").write_text("# Fixture voice model\n", encoding="utf-8")
    file_paths = {
        "model_onnx": model_path,
        "model_config": config_path,
        "pronunciation_lexicon": "pronunciation/lexicon.json",
        "corpus_manifest": "corpus/manifest.json",
        "evaluation_report": "evaluation/report.json",
        "license_ledger": "provenance/licenses.json",
        "training_provenance": "provenance/training.json",
        "model_card": "model-card.md",
    }
    files = {
        key: {"path": path, "sha256": sha256(bundle / path)}
        for key, path in file_paths.items()
    }
    model_hash = files["model_onnx"]["sha256"]
    write_json(
        bundle / "manifest.json",
        {
            "schema": "avaas/voice-model@v1",
            "artifact_id": name,
            "model_id": "satraj-piranesi",
            "version": "1.0.0",
            "speaker_id": "satraj",
            "aliases": [
                {
                    "id": "satraj",
                    "model_sha256": model_hash,
                    "default_presentation": "satraj.en-ca.neutral",
                },
                {
                    "id": "piranesi",
                    "model_sha256": model_hash,
                    "default_presentation": "piranesi.en-gb.neutral",
                },
            ],
            "presentations": PRESENTATIONS,
            "engine": {
                "name": "piper",
                "version": "1.4.2",
                "commit": "d6975e21a440c0d8b6e5fb7c41027409af13d44d",
                "architecture": "medium",
            },
            "audio": {
                "native_sample_rate_hz": 22_050,
                "serving_sample_rate_hz": 24_000,
                "channels": 1,
                "sample_format": "pcm_s16le",
            },
            "files": files,
            "provenance": {
                "corpus_manifest_sha256": files["corpus_manifest"]["sha256"],
                "split_sha256": split_hash,
                "dsp_sha256": dsp_hash,
                "base_checkpoint_sha256": base_hash,
                "trainer_sha256": trainer_hash,
                "source_commits": commits,
            },
            "validation": {"cpu": {"passed": True}, "cuda": {"passed": True}},
            "evaluation": {
                "report_sha256": files["evaluation_report"]["sha256"],
                "passed": True,
                "thresholds": thresholds,
                "measured": measured,
            },
            "target_compatibility": [
                "kudzu-vox/legacy-tts@v1",
                "openai/audio-speech@v1",
                "astroclaw/pcm24@v1",
            ],
            "promotable": False,
            "created_at": "2026-07-16T20:00:00Z",
            "author": "CPCS",
        },
    )
    seal(bundle)
    return bundle


def build_profile_bundle(root: Path, name: str = "satraj-piranesi-profile-v1") -> Path:
    bundle = root / name
    for directory in ("references/audio", "presentation", "evaluation", "provenance"):
        (bundle / directory).mkdir(parents=True, exist_ok=True)
    observed = write_tone(bundle / "references/audio/source.wav")
    audio_path = f"references/audio/{observed['sha256']}.wav"
    (bundle / "references/audio/source.wav").rename(bundle / audio_path)
    write_json(
        bundle / "references/manifest.json",
        {
            "schema": "avaas/reference-manifest@v1",
            "speaker_id": "satraj",
            "sample_rate_hz": 24_000,
            "channels": 1,
            "sample_format": "pcm_s16le",
            "references": [
                {
                    "path": audio_path,
                    "sha256": observed["sha256"],
                    "transcript": "My name is Satraj, and this voice may also present as Piranesi.",
                    "language_tag": "en-CA",
                    "recording_profile_id": "satraj.en-ca.natural",
                    "source_prompt_id": "sec01_001__satraj",
                    "source_take_id": "take-" + "7" * 32,
                    "frames": observed["frames"],
                    "duration_ms": observed["duration_ms"],
                }
            ],
        },
    )
    styles = [
        {"id": "neutral", "instruction": "Use an even, natural delivery."},
        {"id": "warm", "instruction": "Use a warm, conversational delivery."},
        {"id": "authoritative", "instruction": "Use a measured, authoritative delivery."},
        {"id": "urgent", "instruction": "Use an urgent but intelligible delivery."},
        {"id": "whisper", "instruction": "Use a quiet whispered delivery."},
        {"id": "projected", "instruction": "Use a strongly projected delivery."},
    ]
    write_json(
        bundle / "presentation/policies.json",
        {
            "schema": "avaas/presentation-policies@v1",
            "identity_mode": "shared-speaker",
            "controls": {
                "speed": {"minimum": 0.75, "maximum": 1.25, "default": 1.0},
                "volume": {"minimum": 0.5, "maximum": 1.5, "default": 1.0},
                "pitch_semitones": {"minimum": -6.0, "maximum": 6.0, "default": 0.0},
            },
            "styles": styles,
            "presentations": [
                {
                    "id": "satraj.en-ca.neutral",
                    "alias": "satraj",
                    "language_tag": "en-CA",
                    "accent": "general-north-american",
                    "default_style": "neutral",
                },
                {
                    "id": "piranesi.en-gb.neutral",
                    "alias": "piranesi",
                    "language_tag": "en-GB",
                    "accent": "received-pronunciation",
                    "default_style": "neutral",
                },
            ],
        },
    )
    thresholds = {"min_speaker_similarity": 0.72, "max_realtime_factor": 1.0}
    measured = {"speaker_similarity": 0.82, "realtime_factor": 0.18}
    write_json(
        bundle / "evaluation/report.json",
        {
            "schema": "avaas/voice-profile-evaluation@v1",
            "passed": True,
            "thresholds": thresholds,
            "measured": measured,
            "checks": {
                "cpu_worker_contract": True,
                "cuda_inference": True,
                "finite_audio": True,
                "reference_integrity": True,
                "speaker_similarity": True,
                "accent_control": True,
            },
        },
    )
    write_json(
        bundle / "provenance/licenses.json",
        {
            "schema": "avaas/license-ledger@v1",
            "entries": _license_entries(
                ("cosyvoice-source", "cosyvoice-model", "training-corpus")
            ),
        },
    )
    split_hash, dsp_hash = "8" * 64, "9" * 64
    commits = {"avaas": "a" * 40, "piranesi_workspace": "b" * 40}
    write_json(
        bundle / "provenance/profile.json",
        {
            "schema": "avaas/profile-provenance@v1",
            "corpus_manifest_sha256": "c" * 64,
            "split_sha256": split_hash,
            "dsp_sha256": dsp_hash,
            "source_commits": commits,
        },
    )
    (bundle / "model-card.md").write_text("# Fixture expressive profile\n", encoding="utf-8")
    file_paths = {
        "references_manifest": "references/manifest.json",
        "presentation_policies": "presentation/policies.json",
        "evaluation_report": "evaluation/report.json",
        "license_ledger": "provenance/licenses.json",
        "profile_provenance": "provenance/profile.json",
        "model_card": "model-card.md",
    }
    files = {
        key: {"path": path, "sha256": sha256(bundle / path)}
        for key, path in file_paths.items()
    }
    write_json(
        bundle / "manifest.json",
        {
            "schema": "avaas/voice-profile@v1",
            "artifact_id": name,
            "profile_id": "satraj-piranesi",
            "version": "1.0.0",
            "speaker_id": "satraj",
            "aliases": [
                {
                    "id": "satraj",
                    "speaker_id": "satraj",
                    "default_presentation": "satraj.en-ca.neutral",
                },
                {
                    "id": "piranesi",
                    "speaker_id": "satraj",
                    "default_presentation": "piranesi.en-gb.neutral",
                },
            ],
            "engine": {
                "name": "cosyvoice3",
                "code_repository": "FunAudioLLM/CosyVoice",
                "code_commit": "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc",
                "model_repository": "FunAudioLLM/Fun-CosyVoice3-0.5B",
                "model_snapshot": "29e01c4e8d000f4bcd70751be16fa94bf3d85a18",
                "model_license": "Apache-2.0",
                "mode": "zero-shot",
            },
            "audio": {
                "sample_rate_hz": 24_000,
                "channels": 1,
                "sample_format": "pcm_s16le",
            },
            "files": files,
            "provenance": {
                "references_manifest_sha256": files["references_manifest"]["sha256"],
                "corpus_manifest_sha256": "c" * 64,
                "split_sha256": split_hash,
                "dsp_sha256": dsp_hash,
                "source_commits": commits,
            },
            "validation": {"cpu": {"passed": True}, "cuda": {"passed": True}},
            "evaluation": {
                "report_sha256": files["evaluation_report"]["sha256"],
                "passed": True,
                "thresholds": thresholds,
                "measured": measured,
            },
            "target_compatibility": [
                "kudzu-vox/expressive@v1",
                "openai/audio-speech@v1",
                "astroclaw/pcm24@v1",
            ],
            "promotable": False,
            "fixture": True,
            "created_at": "2026-07-16T20:00:00Z",
            "author": "CPCS",
        },
    )
    seal(bundle)
    return bundle

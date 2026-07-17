from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.artifact_helpers import load_json, seal, write_json, write_tone
from webui import audio_contracts, wakeword_export


def _source_artifact() -> dict:
    return {
        "schema": "avaas/voice-profile@v1",
        "artifact_id": "satraj-piranesi-profile-v1",
        "manifest_sha256": "a" * 64,
        "alias": "piranesi",
        "presentation_id": "piranesi.en-gb.neutral",
        "engine": "cosyvoice3",
        "promotable": False,
    }


def _licenses() -> list[dict]:
    return [
        {
            "component": "source-voice-artifact",
            "name": "Fixture source profile",
            "spdx": "LicenseRef-Fixture",
            "source_url": "https://example.invalid/profile",
            "license_url": "https://example.invalid/profile/license",
            "distribution_restrictions": "Fixture only; do not promote.",
        },
        {
            "component": "wakeword-corpus",
            "name": "AVAAS wakeword export",
            "spdx": "LicenseRef-AVAAS-Wakeword",
            "source_url": "https://github.com/KuriGohan-Kamehameha/AVAAS",
            "license_url": "https://github.com/KuriGohan-Kamehameha/AVAAS/blob/main/LICENSE",
            "distribution_restrictions": "Use only with the declared speaker consent.",
        },
    ]


def _clips(tmp_path: Path) -> list[dict]:
    source = tmp_path / "rendered-24k.wav"
    write_tone(source, 24_000, 0.8)
    return [
        {
            "path": source,
            "style": "neutral",
            "speed": 1.0,
            "pitch_semitones": 0.0,
            "volume": 1.0,
            "seed": 20_260_716,
        }
    ]


def _build(tmp_path: Path, *, export_id: str = "hey-piranesi-fixture-v1") -> dict:
    return wakeword_export.build_export(
        tmp_path / "exports",
        export_id=export_id,
        version="1.0.0",
        phrase="Hey Piranesi",
        source_artifact=_source_artifact(),
        rendered_clips=_clips(tmp_path),
        license_entries=_licenses(),
        promotable=False,
        fixture=True,
        created_at="2026-07-16T20:00:00Z",
    )


def test_export_normalizes_deterministically_to_16k_pcm16_and_seals_last(
    tmp_path: Path,
) -> None:
    first = _build(tmp_path)
    validated = wakeword_export.validate_export(first["path"])
    assert validated["schema"] == "avaas/wakeword-export@v1"
    assert validated["phrase"] == "Hey Piranesi"
    assert validated["alias"] == "piranesi"
    assert validated["clip_count"] == 1
    clip = next((first["path"] / "clips").glob("*.wav"))
    observed = audio_contracts.inspect_pcm16_wav(
        clip,
        audio_contracts.AUDIO_SPECS["wake-16k"],
        min_seconds=0.25,
        max_seconds=5.0,
    )
    assert clip.name == f"{observed['sha256']}.wav"
    assert (first["path"] / "READY").read_bytes() == b""

    second = _build(tmp_path, export_id="hey-piranesi-fixture-v2")
    second_clip = next((second["path"] / "clips").glob("*.wav"))
    assert second_clip.name == clip.name
    assert second_clip.read_bytes() == clip.read_bytes()


def test_export_rejects_unapproved_phrase_and_fixture_promotion(tmp_path: Path) -> None:
    with pytest.raises(wakeword_export.WakewordExportError, match="phrase"):
        wakeword_export.build_export(
            tmp_path / "exports",
            export_id="bad-phrase-v1",
            version="1.0.0",
            phrase="Okay Piranesi",
            source_artifact=_source_artifact(),
            rendered_clips=_clips(tmp_path),
            license_entries=_licenses(),
            promotable=False,
            fixture=True,
            created_at="2026-07-16T20:00:00Z",
        )
    with pytest.raises(wakeword_export.WakewordExportError, match="fixture|promot"):
        wakeword_export.build_export(
            tmp_path / "exports",
            export_id="lying-fixture-v1",
            version="1.0.0",
            phrase="Hey Piranesi",
            source_artifact=_source_artifact(),
            rendered_clips=_clips(tmp_path),
            license_entries=_licenses(),
            promotable=True,
            fixture=True,
            created_at="2026-07-16T20:00:00Z",
        )


def test_validator_rejects_tamper_symlink_bad_prosody_and_wrong_source_alias(
    tmp_path: Path,
) -> None:
    result = _build(tmp_path)
    clip = next((result["path"] / "clips").glob("*.wav"))
    clip.write_bytes(clip.read_bytes() + b"tamper")
    os.utime(result["path"] / "READY", None)
    with pytest.raises(wakeword_export.WakewordExportError, match="checksum|audio"):
        wakeword_export.validate_export(result["path"])

    result = _build(tmp_path, export_id="symlink-export-v1")
    clip = next((result["path"] / "clips").glob("*.wav"))
    outside = tmp_path / "outside.wav"
    outside.write_bytes(clip.read_bytes())
    clip.unlink()
    clip.symlink_to(outside)
    os.utime(result["path"] / "READY", None)
    with pytest.raises(wakeword_export.WakewordExportError, match="symlink"):
        wakeword_export.validate_export(result["path"])

    result = _build(tmp_path, export_id="prosody-export-v1")
    clips_manifest = load_json(result["path"] / "clips/manifest.json")
    clips_manifest["clips"][0]["speed"] = 9.0
    write_json(result["path"] / "clips/manifest.json", clips_manifest)
    manifest = load_json(result["path"] / "manifest.json")
    manifest["clips"]["manifest_sha256"] = wakeword_export.sha256_path(
        result["path"] / "clips/manifest.json"
    )
    write_json(result["path"] / "manifest.json", manifest)
    seal(result["path"])
    with pytest.raises(wakeword_export.WakewordExportError, match="speed|prosody"):
        wakeword_export.validate_export(result["path"])

    result = _build(tmp_path, export_id="alias-export-v1")
    manifest = load_json(result["path"] / "manifest.json")
    manifest["source_artifact"]["alias"] = "satraj"
    write_json(result["path"] / "manifest.json", manifest)
    seal(result["path"])
    with pytest.raises(wakeword_export.WakewordExportError, match="alias|Piranesi"):
        wakeword_export.validate_export(result["path"])


def test_validator_rejects_duplicate_clip_identity_and_license_gap(tmp_path: Path) -> None:
    result = _build(tmp_path)
    clip_manifest = load_json(result["path"] / "clips/manifest.json")
    clip_manifest["clips"].append(dict(clip_manifest["clips"][0]))
    write_json(result["path"] / "clips/manifest.json", clip_manifest)
    manifest = load_json(result["path"] / "manifest.json")
    manifest["clips"]["count"] = 2
    manifest["clips"]["total_frames"] *= 2
    manifest["clips"]["manifest_sha256"] = wakeword_export.sha256_path(
        result["path"] / "clips/manifest.json"
    )
    write_json(result["path"] / "manifest.json", manifest)
    seal(result["path"])
    with pytest.raises(wakeword_export.WakewordExportError, match="duplicate"):
        wakeword_export.validate_export(result["path"])

    result = _build(tmp_path, export_id="license-export-v1")
    ledger = load_json(result["path"] / "provenance/licenses.json")
    ledger["entries"] = ledger["entries"][:1]
    write_json(result["path"] / "provenance/licenses.json", ledger)
    seal(result["path"])
    with pytest.raises(wakeword_export.WakewordExportError, match="license"):
        wakeword_export.validate_export(result["path"])

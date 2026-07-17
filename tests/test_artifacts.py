from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.artifact_helpers import (
    build_model_bundle,
    build_profile_bundle,
    load_json,
    refresh_declarations,
    seal,
    write_json,
)
from webui import artifacts


def test_voice_model_matches_canonical_shared_identity_contract(tmp_path: Path) -> None:
    bundle = build_model_bundle(tmp_path)
    result = artifacts.validate_voice_model(bundle)
    assert result["schema"] == "avaas/voice-model@v1"
    assert result["aliases"] == ("satraj", "piranesi")
    assert result["presentation_ids"] == (
        "satraj.en-ca.neutral",
        "piranesi.en-gb.neutral",
    )
    assert result["model_sha256"] == load_json(bundle / "manifest.json")["aliases"][0][
        "model_sha256"
    ]
    assert result["promotable"] is False


def test_voice_profile_is_one_speaker_with_native_accent_and_style_controls(
    tmp_path: Path,
) -> None:
    bundle = build_profile_bundle(tmp_path)
    result = artifacts.validate_voice_profile(bundle)
    assert result["schema"] == "avaas/voice-profile@v1"
    assert result["speaker_id"] == "satraj"
    assert result["aliases"] == ("satraj", "piranesi")
    assert result["reference_count"] == 1
    assert result["styles"] == (
        "neutral",
        "warm",
        "authoritative",
        "urgent",
        "whisper",
        "projected",
    )
    assert result["controls"] == {
        "speed": {"minimum": 0.75, "maximum": 1.25, "default": 1.0},
        "volume": {"minimum": 0.5, "maximum": 1.5, "default": 1.0},
        "pitch_semitones": {"minimum": -6.0, "maximum": 6.0, "default": 0.0},
    }


@pytest.mark.parametrize("builder", (build_model_bundle, build_profile_bundle))
def test_artifacts_reject_unknown_schema_symlinks_extras_and_untrusted_payloads(
    tmp_path: Path, builder
) -> None:
    bundle = builder(tmp_path)
    manifest = load_json(bundle / "manifest.json")
    manifest["schema"] = "avaas/voice-artifact@v999"
    write_json(bundle / "manifest.json", manifest)
    seal(bundle)
    with pytest.raises(artifacts.ArtifactContractError, match="schema"):
        artifacts.validate_artifact(bundle)

    bundle = builder(tmp_path, "symlink-bundle")
    outside = tmp_path / "outside"
    outside.write_text("outside", encoding="utf-8")
    (bundle / "escape").symlink_to(outside)
    os.utime(bundle / "READY", None)
    with pytest.raises(artifacts.ArtifactContractError, match="symlink"):
        artifacts.validate_artifact(bundle)

    bundle = builder(tmp_path, "pickle-bundle")
    (bundle / "prompt.pt").write_bytes(b"pickle")
    seal(bundle)
    with pytest.raises(artifacts.ArtifactContractError, match="unexpected|untrusted"):
        artifacts.validate_artifact(bundle)


def test_model_rejects_alias_hash_presentation_config_and_validation_failures(
    tmp_path: Path,
) -> None:
    bundle = build_model_bundle(tmp_path)
    manifest = load_json(bundle / "manifest.json")
    manifest["aliases"][1]["model_sha256"] = "f" * 64
    write_json(bundle / "manifest.json", manifest)
    seal(bundle)
    with pytest.raises(artifacts.ArtifactContractError, match="alias|model"):
        artifacts.validate_voice_model(bundle)

    bundle = build_model_bundle(tmp_path, "bad-presentation")
    manifest = load_json(bundle / "manifest.json")
    manifest["presentations"][1]["accent"] = "general-north-american"
    write_json(bundle / "manifest.json", manifest)
    seal(bundle)
    with pytest.raises(artifacts.ArtifactContractError, match="presentation"):
        artifacts.validate_voice_model(bundle)

    bundle = build_model_bundle(tmp_path, "bad-speaker-map")
    config = load_json(bundle / "model/satraj-piranesi-medium.onnx.json")
    config["speaker_id_map"]["piranesi.en-gb.neutral"] = 0
    write_json(bundle / "model/satraj-piranesi-medium.onnx.json", config)
    refresh_declarations(bundle)
    with pytest.raises(artifacts.ArtifactContractError, match="speaker map"):
        artifacts.validate_voice_model(bundle)

    bundle = build_model_bundle(tmp_path, "failed-cuda")
    manifest = load_json(bundle / "manifest.json")
    manifest["validation"]["cuda"]["passed"] = False
    write_json(bundle / "manifest.json", manifest)
    seal(bundle)
    with pytest.raises(artifacts.ArtifactContractError, match="cuda"):
        artifacts.validate_voice_model(bundle)


def test_profile_rejects_wrong_reference_audio_transcript_and_license_gap(
    tmp_path: Path,
) -> None:
    bundle = build_profile_bundle(tmp_path)
    reference = next((bundle / "references/audio").glob("*.wav"))
    reference.write_bytes(b"not-a-wave")
    seal(bundle)
    with pytest.raises(artifacts.ArtifactContractError, match="reference|audio|checksum"):
        artifacts.validate_voice_profile(bundle)

    bundle = build_profile_bundle(tmp_path, "blank-transcript")
    references = load_json(bundle / "references/manifest.json")
    references["references"][0]["transcript"] = ""
    write_json(bundle / "references/manifest.json", references)
    refresh_declarations(bundle)
    with pytest.raises(artifacts.ArtifactContractError, match="transcript"):
        artifacts.validate_voice_profile(bundle)

    bundle = build_profile_bundle(tmp_path, "license-gap")
    licenses = load_json(bundle / "provenance/licenses.json")
    licenses["entries"] = licenses["entries"][:-1]
    write_json(bundle / "provenance/licenses.json", licenses)
    refresh_declarations(bundle)
    with pytest.raises(artifacts.ArtifactContractError, match="license"):
        artifacts.validate_voice_profile(bundle)


def test_fixture_artifacts_cannot_claim_promotability(tmp_path: Path) -> None:
    bundle = build_profile_bundle(tmp_path)
    manifest = load_json(bundle / "manifest.json")
    manifest["promotable"] = True
    write_json(bundle / "manifest.json", manifest)
    seal(bundle)
    with pytest.raises(artifacts.ArtifactContractError, match="fixture|promot"):
        artifacts.validate_voice_profile(bundle)


def test_model_requires_machine_readable_teacher_provenance_for_synthetic_audio(
    tmp_path: Path,
) -> None:
    bundle = build_model_bundle(tmp_path)
    training = load_json(bundle / "provenance/training.json")
    training["synthetic_material"] = {
        "used": True,
        "fraction": 0.2,
        "teacher_schema": None,
        "teacher_artifact_id": None,
        "teacher_manifest_sha256": None,
        "teacher_presentations": [],
        "license_component": None,
    }
    write_json(bundle / "provenance/training.json", training)
    refresh_declarations(bundle)
    with pytest.raises(artifacts.ArtifactContractError, match="synthetic|teacher"):
        artifacts.validate_voice_model(bundle)

"""Fail-closed validators for immutable AVAAS voice artifacts.

The validators inspect data only.  They never import model code, load ONNX, or
deserialize framework checkpoints/pickle state.
"""
from __future__ import annotations

import json
import os
import re
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

from . import audio_contracts, contracts


VOICE_MODEL_SCHEMA = "avaas/voice-model@v1"
VOICE_PROFILE_SCHEMA = "avaas/voice-profile@v1"
PIPER_VERSION = "1.4.2"
PIPER_COMMIT = "d6975e21a440c0d8b6e5fb7c41027409af13d44d"
COSYVOICE_COMMIT = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
COSYVOICE_MODEL_SNAPSHOT = "29e01c4e8d000f4bcd70751be16fa94bf3d85a18"
MODEL_PATH = "model/satraj-piranesi-medium.onnx"
CONFIG_PATH = MODEL_PATH + ".json"
MAX_REFERENCES = 16
MAX_PRESENTATIONS = 16
MAX_STYLES = 16
MAX_MODEL_BYTES = 512 * 1024 * 1024
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024

_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]{0,126}[a-z0-9])?$")
_SEMVER_RE = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_PRESENTATION_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]{0,126}[a-z0-9])?$")
_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$")
_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(?:-[A-Z]{2})?$")
_CREATED_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_FORBIDDEN_SUFFIXES = (
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".pyc",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".sh",
)

MODEL_FILE_PATHS = {
    "model_onnx": MODEL_PATH,
    "model_config": CONFIG_PATH,
    "pronunciation_lexicon": "pronunciation/lexicon.json",
    "corpus_manifest": "corpus/manifest.json",
    "evaluation_report": "evaluation/report.json",
    "license_ledger": "provenance/licenses.json",
    "training_provenance": "provenance/training.json",
    "model_card": "model-card.md",
}
MODEL_FILES = {
    "manifest.json",
    *MODEL_FILE_PATHS.values(),
    "checksums.sha256",
    "READY",
}
PROFILE_FILE_PATHS = {
    "references_manifest": "references/manifest.json",
    "presentation_policies": "presentation/policies.json",
    "evaluation_report": "evaluation/report.json",
    "license_ledger": "provenance/licenses.json",
    "profile_provenance": "provenance/profile.json",
    "model_card": "model-card.md",
}
PROFILE_FIXED_FILES = {
    "manifest.json",
    *PROFILE_FILE_PATHS.values(),
    "checksums.sha256",
    "READY",
}

REQUIRED_PRESENTATIONS = {
    "satraj.en-ca.neutral": {
        "alias": "satraj",
        "language_tag": "en-CA",
        "accent": "general-north-american",
    },
    "piranesi.en-gb.neutral": {
        "alias": "piranesi",
        "language_tag": "en-GB",
        "accent": "received-pronunciation",
    },
}
STYLE_POLICIES = (
    ("neutral", "Use an even, natural delivery."),
    ("warm", "Use a warm, conversational delivery."),
    ("authoritative", "Use a measured, authoritative delivery."),
    ("urgent", "Use an urgent but intelligible delivery."),
    ("whisper", "Use a quiet whispered delivery."),
    ("projected", "Use a strongly projected delivery."),
)
CONTROL_POLICIES = {
    "speed": {"minimum": 0.75, "maximum": 1.25, "default": 1.0},
    "volume": {"minimum": 0.5, "maximum": 1.5, "default": 1.0},
    "pitch_semitones": {"minimum": -6.0, "maximum": 6.0, "default": 0.0},
}


class ArtifactContractError(contracts.ContractError):
    """An immutable voice artifact failed validation."""


class _DuplicateKey(ValueError):
    pass


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _bound_json(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    for _index in range(100_000):
        if not stack:
            return
        current, depth = stack.pop()
        if depth > 32:
            raise ArtifactContractError("JSON structure exceeds bounds")
        if isinstance(current, dict):
            if len(current) > 1_024:
                raise ArtifactContractError("JSON object exceeds bounds")
            for key, item in current.items():
                if not isinstance(key, str) or len(key) > 1_024:
                    raise ArtifactContractError("JSON key exceeds bounds")
                stack.append((item, depth + 1))
        elif isinstance(current, list):
            if len(current) > 20_000:
                raise ArtifactContractError("JSON array exceeds bounds")
            for item in current:
                stack.append((item, depth + 1))
        elif isinstance(current, str) and len(current) > 1_000_000:
            raise ArtifactContractError("JSON string exceeds bounds")
        elif current is not None and not isinstance(current, (str, int, float, bool)):
            raise ArtifactContractError("unsupported JSON value")
    raise ArtifactContractError("JSON node count exceeds bounds")


def _json(root: Path, relative: str) -> dict[str, Any]:
    try:
        raw = contracts.read_regular(root, relative, maximum=contracts.MAX_JSON_BYTES)
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_reject_constant,
        )
    except ArtifactContractError:
        raise
    except (contracts.ContractError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactContractError(f"invalid JSON file: {relative}") from exc
    _bound_json(value)
    if not isinstance(value, dict):
        raise ArtifactContractError(f"JSON root must be an object: {relative}")
    return value


def _keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ArtifactContractError(f"{label} keys mismatch")
    return value


def _string(value: Any, label: str, *, minimum: int = 1, maximum: int = 4_096) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        raise ArtifactContractError(f"invalid {label}")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ArtifactContractError(f"invalid {label}")
    return value


def _created(value: Any) -> str:
    value = _string(value, "created_at", maximum=32)
    if _CREATED_RE.fullmatch(value) is None:
        raise ArtifactContractError("created_at must be UTC RFC3339 seconds")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ArtifactContractError("created_at is not a real timestamp") from exc
    return value


def _metrics(value: Any, label: str) -> dict[str, int | float]:
    if not isinstance(value, dict) or not 1 <= len(value) <= 32:
        raise ArtifactContractError(f"invalid {label}")
    result: dict[str, int | float] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not 1 <= len(key) <= 128
            or not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not -1_000_000 <= item <= 1_000_000
        ):
            raise ArtifactContractError(f"invalid {label}")
        result[key] = item
    return result


def _source_commits(value: Any) -> dict[str, str]:
    value = _keys(value, {"avaas", "piranesi_workspace"}, "source commits")
    if any(not isinstance(item, str) or _COMMIT_RE.fullmatch(item) is None for item in value.values()):
        raise ArtifactContractError("source commits must be full lowercase Git hashes")
    return value


def _checksum_inventory(root: Path, files: list[str]) -> dict[str, str]:
    try:
        raw = contracts.read_regular(root, "checksums.sha256", maximum=4 * 1024 * 1024)
        text = raw.decode("ascii")
    except (contracts.ContractError, UnicodeDecodeError) as exc:
        raise ArtifactContractError("invalid checksum ledger") from exc
    expected_paths = sorted(set(files) - {"checksums.sha256", "READY"})
    lines = text.splitlines()
    if not text.endswith("\n") or len(lines) != len(expected_paths):
        raise ArtifactContractError("checksum inventory mismatch")
    result: dict[str, str] = {}
    ordered: list[str] = []
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise ArtifactContractError("invalid checksum ledger line")
        digest = _sha(line[:64], "checksum")
        try:
            relative = contracts.safe_relative(line[66:], "checksum path")
        except contracts.ContractError as exc:
            raise ArtifactContractError(str(exc)) from exc
        if relative in result or relative in {"checksums.sha256", "READY"}:
            raise ArtifactContractError("duplicate or recursive checksum path")
        ordered.append(relative)
        result[relative] = digest
    if ordered != expected_paths:
        raise ArtifactContractError("checksum inventory is not in canonical order")
    for relative, expected in result.items():
        try:
            actual = contracts.sha256_file(
                root,
                relative,
                maximum=MAX_MODEL_BYTES if relative == MODEL_PATH else contracts.MAX_BUNDLE_FILE_BYTES,
            )
        except contracts.ContractError as exc:
            raise ArtifactContractError(str(exc)) from exc
        if actual != expected:
            raise ArtifactContractError(f"checksum mismatch: {relative}")
    return result


def _envelope(root: Path) -> tuple[list[str], dict[str, str]]:
    root = Path(root)
    try:
        files = contracts.inventory(root)
    except contracts.ContractError as exc:
        raise ArtifactContractError(str(exc)) from exc
    if "manifest.json" not in files or "checksums.sha256" not in files or "READY" not in files:
        raise ArtifactContractError("artifact inventory lacks manifest, checksums, or READY")
    for relative in files:
        suffix = Path(relative).suffix.lower()
        if suffix in _FORBIDDEN_SUFFIXES:
            raise ArtifactContractError(f"untrusted executable or serialized payload: {relative}")
        try:
            mode = os.lstat(root / relative).st_mode
        except OSError as exc:
            raise ArtifactContractError(f"cannot inspect artifact file: {relative}") from exc
        if mode & 0o111:
            raise ArtifactContractError(f"unexpected executable artifact file: {relative}")
    try:
        ready = os.lstat(root / "READY")
    except OSError as exc:
        raise ArtifactContractError("READY marker is unavailable") from exc
    if not stat.S_ISREG(ready.st_mode) or ready.st_size != 0:
        raise ArtifactContractError("READY marker must be an empty regular file")
    latest_payload = max(
        os.lstat(root / relative).st_mtime_ns for relative in files if relative != "READY"
    )
    if ready.st_mtime_ns < latest_payload:
        raise ArtifactContractError("READY marker was not written last")
    return files, _checksum_inventory(root, files)


def _file_declarations(
    manifest: dict[str, Any],
    expected: dict[str, str],
    checksums: dict[str, str],
) -> dict[str, dict[str, str]]:
    files = _keys(manifest.get("files"), set(expected), "manifest files")
    result: dict[str, dict[str, str]] = {}
    for key, path in expected.items():
        entry = _keys(files[key], {"path", "sha256"}, f"files.{key}")
        if entry.get("path") != path:
            raise ArtifactContractError(f"files.{key} path mismatch or escape")
        digest = _sha(entry.get("sha256"), f"files.{key}.sha256")
        if checksums.get(path) != digest:
            raise ArtifactContractError(f"files.{key} checksum mismatch")
        result[key] = {"path": path, "sha256": digest}
    return result


def _licenses(root: Path, required: set[str]) -> set[str]:
    value = _keys(_json(root, "provenance/licenses.json"), {"schema", "entries"}, "license ledger")
    if value.get("schema") != "avaas/license-ledger@v1":
        raise ArtifactContractError("license ledger schema mismatch")
    entries = value.get("entries")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 64:
        raise ArtifactContractError("license entry count outside bounds")
    components: set[str] = set()
    expected_keys = {
        "component",
        "name",
        "spdx",
        "source_url",
        "license_url",
        "distribution_restrictions",
    }
    for raw in entries:
        entry = _keys(raw, expected_keys, "license entry")
        component = _string(entry.get("component"), "license component", maximum=128)
        if component in components:
            raise ArtifactContractError("duplicate license component")
        components.add(component)
        _string(entry.get("name"), "license name")
        _string(entry.get("spdx"), "license SPDX", maximum=128)
        for key in ("source_url", "license_url"):
            url = _string(entry.get(key), f"license {key}")
            if not url.startswith("https://"):
                raise ArtifactContractError(f"license {key} must use HTTPS")
        _string(entry.get("distribution_restrictions"), "license restrictions")
    if not required <= components:
        raise ArtifactContractError(f"license gap: missing {sorted(required - components)!r}")
    return components


def _synthetic_material(value: Any, license_components: set[str]) -> dict[str, Any]:
    value = _keys(
        value,
        {
            "used",
            "fraction",
            "teacher_schema",
            "teacher_artifact_id",
            "teacher_manifest_sha256",
            "teacher_presentations",
            "license_component",
        },
        "synthetic material provenance",
    )
    used = value.get("used")
    fraction = value.get("fraction")
    if (
        not isinstance(used, bool)
        or not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or not 0.0 <= fraction <= 0.35
    ):
        raise ArtifactContractError("synthetic material fraction is outside bounds")
    if not used:
        if value != {
            "used": False,
            "fraction": 0.0,
            "teacher_schema": None,
            "teacher_artifact_id": None,
            "teacher_manifest_sha256": None,
            "teacher_presentations": [],
            "license_component": None,
        }:
            raise ArtifactContractError("unused synthetic teacher provenance must be empty")
        return dict(value)
    if fraction <= 0.0 or value.get("teacher_schema") != VOICE_PROFILE_SCHEMA:
        raise ArtifactContractError("synthetic teacher schema or fraction is invalid")
    artifact_id = _string(
        value.get("teacher_artifact_id"), "synthetic teacher artifact id", maximum=128
    )
    if _SLUG_RE.fullmatch(artifact_id) is None:
        raise ArtifactContractError("synthetic teacher artifact id is unsafe")
    _sha(value.get("teacher_manifest_sha256"), "synthetic teacher manifest hash")
    presentations = value.get("teacher_presentations")
    if (
        not isinstance(presentations, list)
        or not 1 <= len(presentations) <= MAX_PRESENTATIONS
        or len(set(presentations)) != len(presentations)
        or any(
            not isinstance(item, str) or _PRESENTATION_RE.fullmatch(item) is None
            for item in presentations
        )
    ):
        raise ArtifactContractError("synthetic teacher presentations are invalid")
    license_component = _string(
        value.get("license_component"), "synthetic teacher license component", maximum=128
    )
    if license_component not in license_components:
        raise ArtifactContractError("synthetic teacher license provenance is missing")
    return dict(value)


def _validate_model_presentations(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not 2 <= len(value) <= MAX_PRESENTATIONS:
        raise ArtifactContractError("presentations count outside bounds")
    expected_keys = {
        "id",
        "alias",
        "speaker_id",
        "language_tag",
        "phonemizer_voice",
        "accent",
        "delivery",
    }
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for expected_speaker_id, raw in enumerate(value):
        entry = _keys(raw, expected_keys, "presentation")
        presentation_id = _string(entry.get("id"), "presentation id", maximum=128)
        if _PRESENTATION_RE.fullmatch(presentation_id) is None or presentation_id in ids:
            raise ArtifactContractError("presentation id is unsafe or duplicated")
        ids.add(presentation_id)
        alias = entry.get("alias")
        if alias not in {"satraj", "piranesi"}:
            raise ArtifactContractError("presentation alias is unsupported")
        if entry.get("speaker_id") != expected_speaker_id:
            raise ArtifactContractError("presentation speaker ids must be contiguous")
        language = entry.get("language_tag")
        if not isinstance(language, str) or _LANGUAGE_RE.fullmatch(language) is None:
            raise ArtifactContractError("presentation language tag is invalid")
        phonemizer = entry.get("phonemizer_voice")
        if not isinstance(phonemizer, str) or _LABEL_RE.fullmatch(phonemizer) is None:
            raise ArtifactContractError("presentation phonemizer is invalid")
        for key in ("accent", "delivery"):
            label = entry.get(key)
            if not isinstance(label, str) or _LABEL_RE.fullmatch(label) is None:
                raise ArtifactContractError(f"presentation {key} is invalid")
        result.append(dict(entry))
    by_id = {item["id"]: item for item in result}
    for presentation_id, required in REQUIRED_PRESENTATIONS.items():
        item = by_id.get(presentation_id)
        if item is None or any(item.get(key) != value for key, value in required.items()):
            raise ArtifactContractError(f"required presentation mismatch: {presentation_id}")
        if item.get("delivery") != "neutral":
            raise ArtifactContractError(f"required presentation mismatch: {presentation_id}")
    return tuple(result)


def _evaluation(
    root: Path,
    *,
    schema: str,
    required_checks: set[str],
) -> tuple[dict[str, int | float], dict[str, int | float]]:
    value = _keys(
        _json(root, "evaluation/report.json"),
        {"schema", "passed", "thresholds", "measured", "checks"},
        "evaluation report",
    )
    if value.get("schema") != schema or value.get("passed") is not True:
        raise ArtifactContractError("evaluation report did not pass")
    checks = _keys(value.get("checks"), required_checks, "evaluation checks")
    if any(item is not True for item in checks.values()):
        raise ArtifactContractError("evaluation checks did not all pass")
    return _metrics(value.get("thresholds"), "evaluation thresholds"), _metrics(
        value.get("measured"), "evaluation measured"
    )


def validate_voice_model(
    root: Path | str, *, require_promotable: bool = False
) -> dict[str, Any]:
    """Validate one Piper bundle without loading or executing its ONNX payload."""

    root = Path(root)
    files, checksums = _envelope(root)
    if set(files) != MODEL_FILES:
        raise ArtifactContractError("voice model has an unexpected or missing inventory")
    manifest = _keys(
        _json(root, "manifest.json"),
        {
            "schema",
            "artifact_id",
            "model_id",
            "version",
            "speaker_id",
            "aliases",
            "presentations",
            "engine",
            "audio",
            "files",
            "provenance",
            "validation",
            "evaluation",
            "target_compatibility",
            "promotable",
            "created_at",
            "author",
        },
        "voice model manifest",
    )
    if manifest.get("schema") != VOICE_MODEL_SCHEMA:
        raise ArtifactContractError("unknown voice model schema")
    artifact_id = _string(manifest.get("artifact_id"), "artifact_id", maximum=128)
    if _SLUG_RE.fullmatch(artifact_id) is None or artifact_id != root.name:
        raise ArtifactContractError("artifact_id must be a safe slug matching its directory")
    version = _string(manifest.get("version"), "version", maximum=32)
    if _SEMVER_RE.fullmatch(version) is None:
        raise ArtifactContractError("version must be numeric semantic version")
    if (
        manifest.get("model_id") != "satraj-piranesi"
        or manifest.get("speaker_id") != "satraj"
        or manifest.get("author") != "CPCS"
    ):
        raise ArtifactContractError("voice model identity or author mismatch")
    _created(manifest.get("created_at"))
    if manifest.get("engine") != {
        "name": "piper",
        "version": PIPER_VERSION,
        "commit": PIPER_COMMIT,
        "architecture": "medium",
    }:
        raise ArtifactContractError("Piper engine pin mismatch")
    if manifest.get("audio") != {
        "native_sample_rate_hz": 22_050,
        "serving_sample_rate_hz": 24_000,
        "channels": 1,
        "sample_format": "pcm_s16le",
    }:
        raise ArtifactContractError("voice model audio contract mismatch")
    declarations = _file_declarations(manifest, MODEL_FILE_PATHS, checksums)
    model_sha = declarations["model_onnx"]["sha256"]
    presentations = _validate_model_presentations(manifest.get("presentations"))
    by_id = {item["id"]: item for item in presentations}
    aliases = manifest.get("aliases")
    if not isinstance(aliases, list) or len(aliases) != 2:
        raise ArtifactContractError("aliases must contain exactly satraj and piranesi")
    alias_ids: list[str] = []
    for raw in aliases:
        alias = _keys(raw, {"id", "model_sha256", "default_presentation"}, "alias")
        alias_id = _string(alias.get("id"), "alias id", maximum=32)
        alias_ids.append(alias_id)
        if _sha(alias.get("model_sha256"), "alias model hash") != model_sha:
            raise ArtifactContractError("alias model hash does not match the shared ONNX")
        default = _string(alias.get("default_presentation"), "default presentation", maximum=128)
        if default not in by_id or by_id[default]["alias"] != alias_id:
            raise ArtifactContractError("alias default presentation mismatch")
    if alias_ids != ["satraj", "piranesi"]:
        raise ArtifactContractError("aliases must be ordered satraj, piranesi")

    provenance = _keys(
        manifest.get("provenance"),
        {
            "corpus_manifest_sha256",
            "split_sha256",
            "dsp_sha256",
            "base_checkpoint_sha256",
            "trainer_sha256",
            "source_commits",
        },
        "model provenance",
    )
    if _sha(provenance.get("corpus_manifest_sha256"), "corpus manifest hash") != declarations[
        "corpus_manifest"
    ]["sha256"]:
        raise ArtifactContractError("corpus manifest hash mismatch")
    for key in ("split_sha256", "dsp_sha256", "base_checkpoint_sha256", "trainer_sha256"):
        _sha(provenance.get(key), key)
    source_commits = _source_commits(provenance.get("source_commits"))
    validation = _keys(manifest.get("validation"), {"cpu", "cuda"}, "validation")
    for backend in ("cpu", "cuda"):
        result = _keys(validation.get(backend), {"passed"}, f"validation.{backend}")
        if result.get("passed") is not True:
            raise ArtifactContractError(f"{backend} validation did not pass")
    evaluation = _keys(
        manifest.get("evaluation"),
        {"report_sha256", "passed", "thresholds", "measured"},
        "manifest evaluation",
    )
    if (
        _sha(evaluation.get("report_sha256"), "evaluation report hash")
        != declarations["evaluation_report"]["sha256"]
        or evaluation.get("passed") is not True
    ):
        raise ArtifactContractError("manifest evaluation did not pass")
    manifest_thresholds = _metrics(evaluation.get("thresholds"), "manifest thresholds")
    manifest_measured = _metrics(evaluation.get("measured"), "manifest measured")
    if manifest.get("target_compatibility") != [
        "kudzu-vox/legacy-tts@v1",
        "openai/audio-speech@v1",
        "astroclaw/pcm24@v1",
    ]:
        raise ArtifactContractError("voice model target compatibility mismatch")
    promotable = manifest.get("promotable")
    if not isinstance(promotable, bool) or (require_promotable and not promotable):
        raise ArtifactContractError("voice model is not promotable")

    config = _json(root, CONFIG_PATH)
    expected_map = {item["id"]: item["speaker_id"] for item in presentations}
    if (
        not isinstance(config.get("audio"), dict)
        or config["audio"].get("sample_rate") != 22_050
        or config.get("num_speakers") != len(presentations)
        or config.get("speaker_id_map") != expected_map
        or config.get("phoneme_type") != "espeak"
    ):
        raise ArtifactContractError("Piper speaker map or audio config mismatch")
    lexicon = _keys(
        _json(root, "pronunciation/lexicon.json"),
        {"schema", "entries"},
        "pronunciation lexicon",
    )
    entries = lexicon.get("entries")
    if (
        lexicon.get("schema") != "avaas/pronunciation-lexicon@v1"
        or not isinstance(entries, list)
        or len(entries) != 2
        or [entry.get("name") if isinstance(entry, dict) else None for entry in entries]
        != ["Satraj", "Piranesi"]
    ):
        raise ArtifactContractError("pronunciation lexicon identity mismatch")
    corpus_manifest = _keys(
        _json(root, "corpus/manifest.json"),
        {"schema", "corpus_version", "split_sha256", "dsp_sha256"},
        "corpus manifest",
    )
    if (
        corpus_manifest.get("schema") != "avaas/corpus-manifest@v1"
        or _sha(corpus_manifest.get("split_sha256"), "corpus split hash")
        != provenance["split_sha256"]
        or _sha(corpus_manifest.get("dsp_sha256"), "corpus DSP hash")
        != provenance["dsp_sha256"]
    ):
        raise ArtifactContractError("corpus provenance mismatch")
    _string(corpus_manifest.get("corpus_version"), "corpus version", maximum=64)
    report_thresholds, report_measured = _evaluation(
        root,
        schema="avaas/voice-evaluation@v1",
        required_checks={
            "cpu_inference",
            "cuda_inference",
            "finite_audio",
            "non_silent_audio",
            "satraj_pronunciation",
            "piranesi_pronunciation",
        },
    )
    if report_thresholds != manifest_thresholds or report_measured != manifest_measured:
        raise ArtifactContractError("evaluation report and manifest disagree")
    license_components = _licenses(
        root, {"piper", "base-checkpoint", "training-corpus"}
    )
    training = _keys(
        _json(root, "provenance/training.json"),
        {
            "schema",
            "base_checkpoint_sha256",
            "trainer_sha256",
            "source_commits",
            "synthetic_material",
        },
        "training provenance",
    )
    if (
        training.get("schema") != "avaas/training-provenance@v1"
        or _sha(training.get("base_checkpoint_sha256"), "base checkpoint hash")
        != provenance["base_checkpoint_sha256"]
        or _sha(training.get("trainer_sha256"), "trainer hash") != provenance["trainer_sha256"]
        or _source_commits(training.get("source_commits")) != source_commits
    ):
        raise ArtifactContractError("training provenance and manifest disagree")
    _synthetic_material(training.get("synthetic_material"), license_components)
    try:
        card = contracts.read_regular(
            root, "model-card.md", maximum=MAX_DOCUMENT_BYTES
        ).decode("utf-8")
    except (contracts.ContractError, UnicodeDecodeError) as exc:
        raise ArtifactContractError("model card is invalid") from exc
    if not card.strip():
        raise ArtifactContractError("model card is empty")
    return {
        "schema": VOICE_MODEL_SCHEMA,
        "artifact_id": artifact_id,
        "model_id": "satraj-piranesi",
        "version": version,
        "aliases": tuple(alias_ids),
        "presentation_ids": tuple(item["id"] for item in presentations),
        "model_sha256": model_sha,
        "promotable": promotable,
        "path": root,
    }


def _validate_profile_policies(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    policies = _keys(
        _json(root, "presentation/policies.json"),
        {"schema", "identity_mode", "controls", "styles", "presentations"},
        "presentation policies",
    )
    if (
        policies.get("schema") != "avaas/presentation-policies@v1"
        or policies.get("identity_mode") != "shared-speaker"
    ):
        raise ArtifactContractError("presentation policy identity mode mismatch")
    if policies.get("controls") != CONTROL_POLICIES:
        raise ArtifactContractError("presentation scalar controls mismatch")
    styles = policies.get("styles")
    if not isinstance(styles, list) or not 1 <= len(styles) <= MAX_STYLES:
        raise ArtifactContractError("style policy count outside bounds")
    observed_styles: list[tuple[str, str]] = []
    for raw in styles:
        style = _keys(raw, {"id", "instruction"}, "style policy")
        style_id = _string(style.get("id"), "style id", maximum=64)
        if _LABEL_RE.fullmatch(style_id) is None:
            raise ArtifactContractError("style id is unsafe")
        instruction = _string(style.get("instruction"), "style instruction", maximum=256)
        observed_styles.append((style_id, instruction))
    if tuple(observed_styles) != STYLE_POLICIES:
        raise ArtifactContractError("style policies do not match the bounded static contract")
    presentations = policies.get("presentations")
    if not isinstance(presentations, list) or not 2 <= len(presentations) <= MAX_PRESENTATIONS:
        raise ArtifactContractError("profile presentation count outside bounds")
    expected_keys = {"id", "alias", "language_tag", "accent", "default_style"}
    ids: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for raw in presentations:
        item = _keys(raw, expected_keys, "profile presentation")
        presentation_id = _string(item.get("id"), "presentation id", maximum=128)
        if _PRESENTATION_RE.fullmatch(presentation_id) is None or presentation_id in by_id:
            raise ArtifactContractError("profile presentation id is unsafe or duplicated")
        if item.get("alias") not in {"satraj", "piranesi"}:
            raise ArtifactContractError("profile presentation alias is unsupported")
        language = item.get("language_tag")
        accent = item.get("accent")
        if (
            not isinstance(language, str)
            or _LANGUAGE_RE.fullmatch(language) is None
            or not isinstance(accent, str)
            or _LABEL_RE.fullmatch(accent) is None
            or item.get("default_style") not in {style[0] for style in STYLE_POLICIES}
        ):
            raise ArtifactContractError("profile presentation controls are invalid")
        ids.append(presentation_id)
        by_id[presentation_id] = item
    for presentation_id, required in REQUIRED_PRESENTATIONS.items():
        item = by_id.get(presentation_id)
        if item is None or any(item.get(key) != value for key, value in required.items()):
            raise ArtifactContractError(f"required profile presentation mismatch: {presentation_id}")
        if item.get("default_style") != "neutral":
            raise ArtifactContractError(f"required profile presentation mismatch: {presentation_id}")
    return tuple(ids), tuple(item[0] for item in STYLE_POLICIES)


def validate_voice_profile(
    root: Path | str, *, require_promotable: bool = False
) -> dict[str, Any]:
    """Validate a bounded CosyVoice reference profile without loading prompt state."""

    root = Path(root)
    files, checksums = _envelope(root)
    audio_files = [item for item in files if item.startswith("references/audio/")]
    if (
        not 1 <= len(audio_files) <= MAX_REFERENCES
        or set(files) != PROFILE_FIXED_FILES | set(audio_files)
        or any(not item.endswith(".wav") for item in audio_files)
    ):
        raise ArtifactContractError("voice profile has an unexpected or missing inventory")
    manifest = _keys(
        _json(root, "manifest.json"),
        {
            "schema",
            "artifact_id",
            "profile_id",
            "version",
            "speaker_id",
            "aliases",
            "engine",
            "audio",
            "files",
            "provenance",
            "validation",
            "evaluation",
            "target_compatibility",
            "promotable",
            "fixture",
            "created_at",
            "author",
        },
        "voice profile manifest",
    )
    if manifest.get("schema") != VOICE_PROFILE_SCHEMA:
        raise ArtifactContractError("unknown voice profile schema")
    artifact_id = _string(manifest.get("artifact_id"), "artifact_id", maximum=128)
    version = _string(manifest.get("version"), "version", maximum=32)
    if (
        _SLUG_RE.fullmatch(artifact_id) is None
        or artifact_id != root.name
        or _SEMVER_RE.fullmatch(version) is None
    ):
        raise ArtifactContractError("profile artifact id or version is invalid")
    if (
        manifest.get("profile_id") != "satraj-piranesi"
        or manifest.get("speaker_id") != "satraj"
        or manifest.get("author") != "CPCS"
    ):
        raise ArtifactContractError("voice profile identity or author mismatch")
    _created(manifest.get("created_at"))
    if manifest.get("engine") != {
        "name": "cosyvoice3",
        "code_repository": "FunAudioLLM/CosyVoice",
        "code_commit": COSYVOICE_COMMIT,
        "model_repository": "FunAudioLLM/Fun-CosyVoice3-0.5B",
        "model_snapshot": COSYVOICE_MODEL_SNAPSHOT,
        "model_license": "Apache-2.0",
        "mode": "zero-shot",
    }:
        raise ArtifactContractError("CosyVoice source or model pin mismatch")
    if manifest.get("audio") != {
        "sample_rate_hz": 24_000,
        "channels": 1,
        "sample_format": "pcm_s16le",
    }:
        raise ArtifactContractError("voice profile audio contract mismatch")
    declarations = _file_declarations(manifest, PROFILE_FILE_PATHS, checksums)
    presentation_ids, styles = _validate_profile_policies(root)
    aliases = manifest.get("aliases")
    if not isinstance(aliases, list) or len(aliases) != 2:
        raise ArtifactContractError("profile aliases must contain satraj and piranesi")
    alias_ids: list[str] = []
    policies_by_id = {
        item["id"]: item
        for item in _json(root, "presentation/policies.json")["presentations"]
    }
    for raw in aliases:
        alias = _keys(raw, {"id", "speaker_id", "default_presentation"}, "profile alias")
        alias_id = _string(alias.get("id"), "profile alias id", maximum=32)
        default = _string(alias.get("default_presentation"), "default presentation", maximum=128)
        if (
            alias.get("speaker_id") != "satraj"
            or default not in policies_by_id
            or policies_by_id[default].get("alias") != alias_id
        ):
            raise ArtifactContractError("profile aliases must share Satraj's speaker identity")
        alias_ids.append(alias_id)
    if alias_ids != ["satraj", "piranesi"]:
        raise ArtifactContractError("profile aliases must be ordered satraj, piranesi")
    provenance = _keys(
        manifest.get("provenance"),
        {
            "references_manifest_sha256",
            "corpus_manifest_sha256",
            "split_sha256",
            "dsp_sha256",
            "source_commits",
        },
        "profile provenance",
    )
    if _sha(provenance.get("references_manifest_sha256"), "references manifest hash") != declarations[
        "references_manifest"
    ]["sha256"]:
        raise ArtifactContractError("references manifest hash mismatch")
    for key in ("corpus_manifest_sha256", "split_sha256", "dsp_sha256"):
        _sha(provenance.get(key), key)
    source_commits = _source_commits(provenance.get("source_commits"))
    validation = _keys(manifest.get("validation"), {"cpu", "cuda"}, "profile validation")
    for backend in ("cpu", "cuda"):
        result = _keys(validation.get(backend), {"passed"}, f"validation.{backend}")
        if result.get("passed") is not True:
            raise ArtifactContractError(f"{backend} profile validation did not pass")
    evaluation = _keys(
        manifest.get("evaluation"),
        {"report_sha256", "passed", "thresholds", "measured"},
        "profile evaluation",
    )
    if (
        _sha(evaluation.get("report_sha256"), "evaluation report hash")
        != declarations["evaluation_report"]["sha256"]
        or evaluation.get("passed") is not True
    ):
        raise ArtifactContractError("profile evaluation did not pass")
    manifest_thresholds = _metrics(evaluation.get("thresholds"), "profile thresholds")
    manifest_measured = _metrics(evaluation.get("measured"), "profile measured")
    if manifest.get("target_compatibility") != [
        "kudzu-vox/expressive@v1",
        "openai/audio-speech@v1",
        "astroclaw/pcm24@v1",
    ]:
        raise ArtifactContractError("voice profile target compatibility mismatch")
    promotable, fixture = manifest.get("promotable"), manifest.get("fixture")
    if (
        not isinstance(promotable, bool)
        or not isinstance(fixture, bool)
        or (fixture and promotable)
        or (require_promotable and not promotable)
    ):
        raise ArtifactContractError("fixture/promotability contract mismatch")

    references = _keys(
        _json(root, "references/manifest.json"),
        {"schema", "speaker_id", "sample_rate_hz", "channels", "sample_format", "references"},
        "reference manifest",
    )
    if (
        references.get("schema") != "avaas/reference-manifest@v1"
        or references.get("speaker_id") != "satraj"
        or references.get("sample_rate_hz") != 24_000
        or references.get("channels") != 1
        or references.get("sample_format") != "pcm_s16le"
    ):
        raise ArtifactContractError("reference manifest audio or identity mismatch")
    entries = references.get("references")
    if not isinstance(entries, list) or len(entries) != len(audio_files):
        raise ArtifactContractError("reference count does not match audio inventory")
    expected_entry_keys = {
        "path",
        "sha256",
        "transcript",
        "language_tag",
        "recording_profile_id",
        "source_prompt_id",
        "source_take_id",
        "frames",
        "duration_ms",
    }
    seen_paths: set[str] = set()
    reference_languages: set[str] = set()
    for raw in entries:
        entry = _keys(raw, expected_entry_keys, "reference entry")
        try:
            relative = contracts.safe_relative(entry.get("path"), "reference path")
        except contracts.ContractError as exc:
            raise ArtifactContractError(str(exc)) from exc
        digest = _sha(entry.get("sha256"), "reference sha256")
        if (
            relative in seen_paths
            or relative not in audio_files
            or relative != f"references/audio/{digest}.wav"
            or checksums.get(relative) != digest
        ):
            raise ArtifactContractError("reference path, checksum, or duplicate mismatch")
        seen_paths.add(relative)
        _string(entry.get("transcript"), "reference transcript", maximum=2_000)
        language_tag = entry.get("language_tag")
        if not isinstance(language_tag, str) or _LANGUAGE_RE.fullmatch(language_tag) is None:
            raise ArtifactContractError("reference language tag is invalid")
        reference_languages.add(language_tag)
        expected_recording_profile = f"satraj.{language_tag.lower()}.natural"
        if entry.get("recording_profile_id") != expected_recording_profile:
            raise ArtifactContractError("reference recording profile mismatch")
        _string(entry.get("source_prompt_id"), "source prompt id", maximum=128)
        take_id = _string(entry.get("source_take_id"), "source take id", maximum=64)
        if not take_id.startswith("take-"):
            raise ArtifactContractError("source take id mismatch")
        try:
            observed = audio_contracts.inspect_pcm16_wav(
                root / relative,
                audio_contracts.AUDIO_SPECS["serve-24k"],
                min_seconds=0.4,
                max_seconds=30.0,
            )
        except (OSError, audio_contracts.AudioContractError) as exc:
            raise ArtifactContractError("reference audio contract failed") from exc
        if observed["sha256"] != digest or entry.get("frames") != observed["frames"] or entry.get(
            "duration_ms"
        ) != observed["duration_ms"]:
            raise ArtifactContractError("reference audio metadata mismatch")
    if seen_paths != set(audio_files):
        raise ArtifactContractError("reference audio inventory mismatch")
    if "en-CA" not in reference_languages:
        raise ArtifactContractError("profile requires a natural en-CA reference")
    report_thresholds, report_measured = _evaluation(
        root,
        schema="avaas/voice-profile-evaluation@v1",
        required_checks={
            "cpu_worker_contract",
            "cuda_inference",
            "finite_audio",
            "reference_integrity",
            "speaker_similarity",
            "accent_control",
        },
    )
    if report_thresholds != manifest_thresholds or report_measured != manifest_measured:
        raise ArtifactContractError("profile evaluation report and manifest disagree")
    _licenses(root, {"cosyvoice-source", "cosyvoice-model", "training-corpus"})
    profile_provenance = _keys(
        _json(root, "provenance/profile.json"),
        {"schema", "corpus_manifest_sha256", "split_sha256", "dsp_sha256", "source_commits"},
        "profile provenance file",
    )
    if (
        profile_provenance.get("schema") != "avaas/profile-provenance@v1"
        or _sha(profile_provenance.get("corpus_manifest_sha256"), "profile corpus hash")
        != provenance["corpus_manifest_sha256"]
        or _sha(profile_provenance.get("split_sha256"), "profile split hash")
        != provenance["split_sha256"]
        or _sha(profile_provenance.get("dsp_sha256"), "profile DSP hash")
        != provenance["dsp_sha256"]
        or _source_commits(profile_provenance.get("source_commits")) != source_commits
    ):
        raise ArtifactContractError("profile provenance and manifest disagree")
    try:
        card = contracts.read_regular(
            root, "model-card.md", maximum=MAX_DOCUMENT_BYTES
        ).decode("utf-8")
    except (contracts.ContractError, UnicodeDecodeError) as exc:
        raise ArtifactContractError("profile model card is invalid") from exc
    if not card.strip():
        raise ArtifactContractError("profile model card is empty")
    return {
        "schema": VOICE_PROFILE_SCHEMA,
        "artifact_id": artifact_id,
        "profile_id": "satraj-piranesi",
        "version": version,
        "speaker_id": "satraj",
        "aliases": tuple(alias_ids),
        "presentation_ids": presentation_ids,
        "styles": styles,
        "controls": CONTROL_POLICIES,
        "reference_count": len(entries),
        "promotable": promotable,
        "fixture": fixture,
        "path": root,
    }


def validate_artifact(
    root: Path | str, *, require_promotable: bool = False
) -> dict[str, Any]:
    """Dispatch to the exact validator named by a safely-read manifest."""

    root = Path(root)
    try:
        schema = _json(root, "manifest.json").get("schema")
    except (ArtifactContractError, contracts.ContractError) as exc:
        raise ArtifactContractError(str(exc)) from exc
    if schema == VOICE_MODEL_SCHEMA:
        return validate_voice_model(root, require_promotable=require_promotable)
    if schema == VOICE_PROFILE_SCHEMA:
        return validate_voice_profile(root, require_promotable=require_promotable)
    raise ArtifactContractError(f"unknown voice artifact schema: {schema!r}")

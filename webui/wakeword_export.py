"""Immutable, deterministic AVAAS exports for personalized wakeword positives."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from . import artifacts, audio_contracts, contracts, processing


WAKEWORD_EXPORT_SCHEMA = "avaas/wakeword-export@v1"
CLIP_MANIFEST_SCHEMA = "avaas/wakeword-clips@v1"
SYNTHESIS_SCHEMA = "avaas/wakeword-synthesis@v1"
ALLOWED_PHRASES = {"Hey Piranesi"}
ALLOWED_STYLES = {item[0] for item in artifacts.STYLE_POLICIES}
MAX_CLIPS = 512
MAX_TOTAL_FRAMES = 16_000 * 60 * 30
MIN_CLIP_SECONDS = 0.25
MAX_CLIP_SECONDS = 5.0
MAX_SOURCE_BYTES = 24 * 1024 * 1024
ALLOWED_SOURCE_RATES = {16_000, 22_050, 24_000, 48_000}
_SEMVER_RE = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]{0,126}[a-z0-9])?$")
_CLIP_RE = re.compile(r"^clips/([0-9a-f]{64})\.wav$")


class WakewordExportError(artifacts.ArtifactContractError):
    """A wakeword export input or immutable bundle failed closed."""


def sha256_path(path: Path) -> str:
    """Hash one bounded, non-symlink regular file."""

    try:
        digest, _size = audio_contracts.sha256_file(Path(path), maximum=MAX_SOURCE_BYTES)
        return digest
    except audio_contracts.AudioContractError as exc:
        raise WakewordExportError("source audio cannot be hashed") from exc


def _canonical_json(value: Any) -> bytes:
    try:
        return contracts.canonical_json(value) + b"\n"
    except contracts.ContractError as exc:
        raise WakewordExportError(str(exc)) from exc


def _write_json(path: Path, value: Any) -> None:
    try:
        contracts.write_durable(path, _canonical_json(value))
    except contracts.ContractError as exc:
        raise WakewordExportError(str(exc)) from exc


def _validate_source_artifact(value: Any, *, export_promotable: bool) -> dict[str, Any]:
    value = artifacts._keys(
        value,
        {
            "schema",
            "artifact_id",
            "manifest_sha256",
            "alias",
            "presentation_id",
            "engine",
            "promotable",
        },
        "source artifact",
    )
    schema = value.get("schema")
    expected_engine = {
        artifacts.VOICE_PROFILE_SCHEMA: "cosyvoice3",
        artifacts.VOICE_MODEL_SCHEMA: "piper",
    }.get(schema)
    artifact_id = artifacts._string(value.get("artifact_id"), "source artifact id", maximum=128)
    if expected_engine is None or _SLUG_RE.fullmatch(artifact_id) is None:
        raise WakewordExportError("source artifact schema or id is invalid")
    artifacts._sha(value.get("manifest_sha256"), "source artifact manifest hash")
    if (
        value.get("alias") != "piranesi"
        or value.get("presentation_id") != "piranesi.en-gb.neutral"
        or value.get("engine") != expected_engine
    ):
        raise WakewordExportError("wakeword source must use Piranesi's declared presentation alias")
    source_promotable = value.get("promotable")
    if not isinstance(source_promotable, bool) or (export_promotable and not source_promotable):
        raise WakewordExportError("promotable export requires a promotable source artifact")
    return dict(value)


def _validate_license_entries(entries: Any) -> list[dict[str, Any]]:
    if not isinstance(entries, list) or not 2 <= len(entries) <= 64:
        raise WakewordExportError("license ledger entry count outside bounds")
    expected = {
        "component",
        "name",
        "spdx",
        "source_url",
        "license_url",
        "distribution_restrictions",
    }
    result: list[dict[str, Any]] = []
    components: set[str] = set()
    for raw in entries:
        item = artifacts._keys(raw, expected, "license entry")
        component = artifacts._string(item.get("component"), "license component", maximum=128)
        if component in components:
            raise WakewordExportError("duplicate license component")
        components.add(component)
        for key in ("name", "spdx", "distribution_restrictions"):
            artifacts._string(item.get(key), f"license {key}")
        for key in ("source_url", "license_url"):
            url = artifacts._string(item.get(key), f"license {key}")
            if not url.startswith("https://"):
                raise WakewordExportError(f"license {key} must use HTTPS")
        result.append(dict(item))
    required = {"source-voice-artifact", "wakeword-corpus"}
    if not required <= components:
        raise WakewordExportError(f"license gap: missing {sorted(required - components)!r}")
    return result


def _bounded_source(path: Path) -> tuple[np.ndarray, int, str]:
    path = Path(path)
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise WakewordExportError("rendered source audio is missing") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise WakewordExportError("rendered source audio must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= MAX_SOURCE_BYTES:
        raise WakewordExportError("rendered source audio size outside bounds")
    try:
        info = sf.info(path)
    except (RuntimeError, TypeError) as exc:
        raise WakewordExportError("rendered source is not readable audio") from exc
    if (
        info.format != "WAV"
        or info.subtype != "PCM_16"
        or info.channels != 1
        or info.samplerate not in ALLOWED_SOURCE_RATES
        or not int(MIN_CLIP_SECONDS * info.samplerate)
        <= info.frames
        <= int(MAX_CLIP_SECONDS * info.samplerate)
    ):
        raise WakewordExportError("rendered source audio contract mismatch")
    source_sha = sha256_path(path)
    try:
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except (RuntimeError, TypeError, ValueError, OSError) as exc:
        raise WakewordExportError("rendered source audio decode failed") from exc
    try:
        samples = audio_contracts.validate_samples(
            samples,
            sample_rate,
            min_seconds=MIN_CLIP_SECONDS,
            max_seconds=MAX_CLIP_SECONDS,
        )
    except audio_contracts.AudioContractError as exc:
        raise WakewordExportError("rendered source samples failed validation") from exc
    return samples, sample_rate, source_sha


def _prosody(raw: Any) -> dict[str, Any]:
    raw = artifacts._keys(
        raw,
        {"path", "style", "speed", "pitch_semitones", "volume", "seed"},
        "rendered clip",
    )
    if not isinstance(raw.get("path"), (str, os.PathLike, Path)):
        raise WakewordExportError("rendered clip path is invalid")
    style = raw.get("style")
    speed = raw.get("speed")
    pitch = raw.get("pitch_semitones")
    volume = raw.get("volume")
    seed = raw.get("seed")
    if style not in ALLOWED_STYLES:
        raise WakewordExportError("rendered clip style is unsupported")
    if (
        not isinstance(speed, (int, float))
        or isinstance(speed, bool)
        or not 0.75 <= speed <= 1.25
    ):
        raise WakewordExportError("rendered clip speed is outside bounds")
    if (
        not isinstance(pitch, (int, float))
        or isinstance(pitch, bool)
        or not -6.0 <= pitch <= 6.0
    ):
        raise WakewordExportError("rendered clip pitch is outside bounds")
    if (
        not isinstance(volume, (int, float))
        or isinstance(volume, bool)
        or not 0.5 <= volume <= 1.5
    ):
        raise WakewordExportError("rendered clip volume is outside bounds")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**63 - 1:
        raise WakewordExportError("rendered clip seed is outside bounds")
    return {
        "path": Path(raw["path"]),
        "style": style,
        "speed": float(speed),
        "pitch_semitones": float(pitch),
        "volume": float(volume),
        "seed": seed,
    }


def _ensure_output_root(path: Path) -> Path:
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o750)
        metadata = os.lstat(path)
    except OSError as exc:
        raise WakewordExportError("cannot create wakeword export root") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise WakewordExportError("wakeword export root must be a non-symlink directory")
    return path


def _write_checksums(staging: Path) -> None:
    try:
        payloads = contracts.inventory(staging)
        lines = [
            f"{contracts.sha256_file(staging, relative)}  {relative}\n"
            for relative in payloads
        ]
        contracts.write_durable(
            staging / "checksums.sha256", "".join(lines).encode("ascii")
        )
    except contracts.ContractError as exc:
        raise WakewordExportError(str(exc)) from exc


def _result(path: Path, validated: dict[str, Any]) -> dict[str, Any]:
    try:
        manifest_sha = contracts.sha256_file(path, "manifest.json")
    except contracts.ContractError as exc:
        raise WakewordExportError(str(exc)) from exc
    return {
        "export_id": validated["export_id"],
        "path": path,
        "manifest_sha256": manifest_sha,
        "clip_count": validated["clip_count"],
        "promotable": validated["promotable"],
        "fixture": validated["fixture"],
    }


def build_export(
    output_root: Path,
    *,
    export_id: str,
    version: str,
    phrase: str,
    source_artifact: dict[str, Any],
    rendered_clips: list[dict[str, Any]],
    license_entries: list[dict[str, Any]],
    promotable: bool,
    fixture: bool,
    created_at: str,
) -> dict[str, Any]:
    """Normalize rendered positives and publish one immutable export atomically."""

    if phrase not in ALLOWED_PHRASES:
        raise WakewordExportError("wake phrase is not in the explicit allow-list")
    if not isinstance(export_id, str) or _SLUG_RE.fullmatch(export_id) is None:
        raise WakewordExportError("export id is invalid")
    if not isinstance(version, str) or _SEMVER_RE.fullmatch(version) is None:
        raise WakewordExportError("export version is invalid")
    if (
        not isinstance(promotable, bool)
        or not isinstance(fixture, bool)
        or (fixture and promotable)
    ):
        raise WakewordExportError("fixture exports can never be promotable")
    artifacts._created(created_at)
    source = _validate_source_artifact(source_artifact, export_promotable=promotable)
    licenses = _validate_license_entries(license_entries)
    if not isinstance(rendered_clips, list) or not 1 <= len(rendered_clips) <= MAX_CLIPS:
        raise WakewordExportError("rendered clip count outside bounds")
    parsed = [_prosody(item) for item in rendered_clips]
    output_root = _ensure_output_root(output_root)
    final = output_root / export_id
    if final.exists() or final.is_symlink():
        raise WakewordExportError("immutable wakeword export already exists")
    staging = Path(tempfile.mkdtemp(prefix=f".{export_id}.", dir=output_root))
    try:
        for directory in ("clips", "provenance"):
            (staging / directory).mkdir(mode=0o750)
        entries: list[dict[str, Any]] = []
        digests: set[str] = set()
        total_frames = 0
        source_rates: set[int] = set()
        for item in parsed:
            samples, source_rate, source_sha = _bounded_source(item["path"])
            source_rates.add(source_rate)
            try:
                normalized = processing.resample_audio(samples, source_rate, 16_000)
                temporary = staging / "clips" / f".render-{len(entries):04d}.wav"
                observed = audio_contracts.write_pcm16_wav(
                    temporary,
                    normalized,
                    audio_contracts.AUDIO_SPECS["wake-16k"],
                    min_seconds=MIN_CLIP_SECONDS,
                    max_seconds=MAX_CLIP_SECONDS,
                )
            except (processing.ProcessingError, audio_contracts.AudioContractError) as exc:
                raise WakewordExportError("wakeword normalization failed") from exc
            digest = str(observed["sha256"])
            if digest in digests:
                raise WakewordExportError("duplicate normalized wakeword clip")
            digests.add(digest)
            relative = f"clips/{digest}.wav"
            temporary.rename(staging / relative)
            total_frames += int(observed["frames"])
            if total_frames > MAX_TOTAL_FRAMES:
                raise WakewordExportError("wakeword export duration exceeds bound")
            entries.append(
                {
                    "path": relative,
                    "sha256": digest,
                    "source_sha256": source_sha,
                    "source_sample_rate_hz": source_rate,
                    "phrase": phrase,
                    "style": item["style"],
                    "speed": item["speed"],
                    "pitch_semitones": item["pitch_semitones"],
                    "volume": item["volume"],
                    "seed": item["seed"],
                    "frames": observed["frames"],
                    "duration_ms": observed["duration_ms"],
                }
            )
        clip_manifest = {
            "schema": CLIP_MANIFEST_SCHEMA,
            "phrase": phrase,
            "alias": "piranesi",
            "source_artifact_manifest_sha256": source["manifest_sha256"],
            "clips": entries,
        }
        _write_json(staging / "clips/manifest.json", clip_manifest)
        _write_json(
            staging / "provenance/licenses.json",
            {"schema": "avaas/license-ledger@v1", "entries": licenses},
        )
        (staging / "README.md").write_text(
            "# Hey Piranesi wakeword positives\n\n"
            "Immutable AVAAS export. Validate checksums before staging.\n",
            encoding="utf-8",
        )
        clip_manifest_sha = hashlib.sha256(
            (staging / "clips/manifest.json").read_bytes()
        ).hexdigest()
        manifest = {
            "schema": WAKEWORD_EXPORT_SCHEMA,
            "export_id": export_id,
            "version": version,
            "phrase": {"text": phrase, "normalized": "hey piranesi", "language_tag": "en"},
            "source_artifact": source,
            "synthesis": {
                "schema": SYNTHESIS_SCHEMA,
                "engine": source["engine"],
                "presentation_id": "piranesi.en-gb.neutral",
                "normalizer": "avaas-wake-normalize/1",
                "source_sample_rates_hz": sorted(source_rates),
            },
            "audio": {
                "sample_rate_hz": 16_000,
                "channels": 1,
                "sample_format": "pcm_s16le",
            },
            "clips": {
                "manifest_path": "clips/manifest.json",
                "manifest_sha256": clip_manifest_sha,
                "count": len(entries),
                "total_frames": total_frames,
            },
            "license_ledger": {"path": "provenance/licenses.json"},
            "promotable": promotable,
            "fixture": fixture,
            "created_at": created_at,
            "author": "CPCS",
        }
        _write_json(staging / "manifest.json", manifest)
        _write_checksums(staging)
        try:
            contracts.write_durable(staging / "READY", b"")
            for directory in (staging / "clips", staging / "provenance", staging):
                contracts.fsync_directory(directory)
        except contracts.ContractError as exc:
            raise WakewordExportError(str(exc)) from exc
        _validate_export(staging, expected_export_id=export_id)
        try:
            os.rename(staging, final)
            contracts.fsync_directory(output_root)
        except OSError as exc:
            raise WakewordExportError("cannot publish immutable wakeword export") from exc
        validated = validate_export(final)
        return _result(final, validated)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _validate_clip_entry(
    root: Path,
    raw: Any,
    *,
    phrase: str,
    checksums: dict[str, str],
    seen_paths: set[str],
) -> tuple[int, int]:
    entry = artifacts._keys(
        raw,
        {
            "path",
            "sha256",
            "source_sha256",
            "source_sample_rate_hz",
            "phrase",
            "style",
            "speed",
            "pitch_semitones",
            "volume",
            "seed",
            "frames",
            "duration_ms",
        },
        "wakeword clip",
    )
    try:
        relative = contracts.safe_relative(entry.get("path"), "wakeword clip path")
    except contracts.ContractError as exc:
        raise WakewordExportError(str(exc)) from exc
    match = _CLIP_RE.fullmatch(relative)
    digest = artifacts._sha(entry.get("sha256"), "wakeword clip hash")
    artifacts._sha(entry.get("source_sha256"), "wakeword source clip hash")
    if (
        match is None
        or match.group(1) != digest
        or checksums.get(relative) != digest
        or relative in seen_paths
    ):
        raise WakewordExportError("duplicate or mismatched wakeword clip identity")
    seen_paths.add(relative)
    if entry.get("phrase") != phrase or entry.get("style") not in ALLOWED_STYLES:
        raise WakewordExportError("wakeword phrase or style provenance mismatch")
    source_rate = entry.get("source_sample_rate_hz")
    if source_rate not in ALLOWED_SOURCE_RATES:
        raise WakewordExportError("wakeword source sample rate mismatch")
    speed, pitch, volume, seed = (
        entry.get("speed"),
        entry.get("pitch_semitones"),
        entry.get("volume"),
        entry.get("seed"),
    )
    if (
        not isinstance(speed, (int, float))
        or isinstance(speed, bool)
        or not 0.75 <= speed <= 1.25
        or not isinstance(pitch, (int, float))
        or isinstance(pitch, bool)
        or not -6.0 <= pitch <= 6.0
        or not isinstance(volume, (int, float))
        or isinstance(volume, bool)
        or not 0.5 <= volume <= 1.5
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed <= 2**63 - 1
    ):
        raise WakewordExportError("wakeword prosody provenance outside bounds")
    try:
        observed = audio_contracts.inspect_pcm16_wav(
            root / relative,
            audio_contracts.AUDIO_SPECS["wake-16k"],
            min_seconds=MIN_CLIP_SECONDS,
            max_seconds=MAX_CLIP_SECONDS,
        )
    except (OSError, audio_contracts.AudioContractError) as exc:
        raise WakewordExportError("wakeword clip audio contract failed") from exc
    if (
        observed["sha256"] != digest
        or entry.get("frames") != observed["frames"]
        or entry.get("duration_ms") != observed["duration_ms"]
    ):
        raise WakewordExportError("wakeword clip metadata mismatch")
    return int(observed["frames"]), int(source_rate)


def _validate_export(
    root: Path | str,
    *,
    require_promotable: bool = False,
    expected_export_id: str | None = None,
) -> dict[str, Any]:
    """Validate one export without executing any contained content."""

    root = Path(root)
    try:
        files, checksums = artifacts._envelope(root)
        clip_files = [item for item in files if _CLIP_RE.fullmatch(item)]
        expected = {
            "manifest.json",
            "clips/manifest.json",
            "provenance/licenses.json",
            "README.md",
            "checksums.sha256",
            "READY",
            *clip_files,
        }
        if not 1 <= len(clip_files) <= MAX_CLIPS or set(files) != expected:
            raise WakewordExportError("wakeword export inventory mismatch")
        manifest = artifacts._keys(
            artifacts._json(root, "manifest.json"),
            {
                "schema",
                "export_id",
                "version",
                "phrase",
                "source_artifact",
                "synthesis",
                "audio",
                "clips",
                "license_ledger",
                "promotable",
                "fixture",
                "created_at",
                "author",
            },
            "wakeword export manifest",
        )
        if manifest.get("schema") != WAKEWORD_EXPORT_SCHEMA:
            raise WakewordExportError("unknown wakeword export schema")
        export_id = artifacts._string(manifest.get("export_id"), "export id", maximum=128)
        version = artifacts._string(manifest.get("version"), "version", maximum=32)
        if (
            _SLUG_RE.fullmatch(export_id) is None
                or export_id != (expected_export_id if expected_export_id is not None else root.name)
            or _SEMVER_RE.fullmatch(version) is None
        ):
            raise WakewordExportError("wakeword export id or version mismatch")
        if manifest.get("author") != "CPCS":
            raise WakewordExportError("wakeword export author mismatch")
        artifacts._created(manifest.get("created_at"))
        phrase = artifacts._keys(
            manifest.get("phrase"), {"text", "normalized", "language_tag"}, "wake phrase"
        )
        if phrase != {"text": "Hey Piranesi", "normalized": "hey piranesi", "language_tag": "en"}:
            raise WakewordExportError("wake phrase is not in the declared allow-list")
        promotable, fixture = manifest.get("promotable"), manifest.get("fixture")
        if (
            not isinstance(promotable, bool)
            or not isinstance(fixture, bool)
            or (fixture and promotable)
            or (require_promotable and not promotable)
        ):
            raise WakewordExportError("fixture/promotability contract mismatch")
        source = _validate_source_artifact(
            manifest.get("source_artifact"), export_promotable=promotable
        )
        synthesis = artifacts._keys(
            manifest.get("synthesis"),
            {"schema", "engine", "presentation_id", "normalizer", "source_sample_rates_hz"},
            "wakeword synthesis",
        )
        source_rates = synthesis.get("source_sample_rates_hz")
        if (
            synthesis.get("schema") != SYNTHESIS_SCHEMA
            or synthesis.get("engine") != source["engine"]
            or synthesis.get("presentation_id") != "piranesi.en-gb.neutral"
            or synthesis.get("normalizer") != "avaas-wake-normalize/1"
            or not isinstance(source_rates, list)
            or source_rates != sorted(set(source_rates))
            or any(rate not in ALLOWED_SOURCE_RATES for rate in source_rates)
        ):
            raise WakewordExportError("wakeword synthesis provenance mismatch")
        if manifest.get("audio") != {
            "sample_rate_hz": 16_000,
            "channels": 1,
            "sample_format": "pcm_s16le",
        }:
            raise WakewordExportError("wakeword audio contract mismatch")
        clips_declaration = artifacts._keys(
            manifest.get("clips"),
            {"manifest_path", "manifest_sha256", "count", "total_frames"},
            "wakeword clips declaration",
        )
        if clips_declaration.get("manifest_path") != "clips/manifest.json":
            raise WakewordExportError("wakeword clip manifest path mismatch")
        manifest_sha = artifacts._sha(
            clips_declaration.get("manifest_sha256"), "wakeword clip manifest hash"
        )
        if checksums.get("clips/manifest.json") != manifest_sha:
            raise WakewordExportError("wakeword clip manifest checksum mismatch")
        clip_manifest = artifacts._keys(
            artifacts._json(root, "clips/manifest.json"),
            {"schema", "phrase", "alias", "source_artifact_manifest_sha256", "clips"},
            "wakeword clip manifest",
        )
        entries = clip_manifest.get("clips")
        if (
            clip_manifest.get("schema") != CLIP_MANIFEST_SCHEMA
            or clip_manifest.get("phrase") != "Hey Piranesi"
            or clip_manifest.get("alias") != "piranesi"
            or clip_manifest.get("source_artifact_manifest_sha256")
            != source["manifest_sha256"]
            or not isinstance(entries, list)
            or not 1 <= len(entries) <= MAX_CLIPS
            or clips_declaration.get("count") != len(entries)
        ):
            raise WakewordExportError("wakeword clip manifest identity mismatch")
        seen: set[str] = set()
        total_frames = 0
        observed_rates: set[int] = set()
        for entry in entries:
            frames, source_rate = _validate_clip_entry(
                root,
                entry,
                phrase="Hey Piranesi",
                checksums=checksums,
                seen_paths=seen,
            )
            total_frames += frames
            observed_rates.add(source_rate)
        if (
            seen != set(clip_files)
            or total_frames != clips_declaration.get("total_frames")
            or total_frames > MAX_TOTAL_FRAMES
            or sorted(observed_rates) != source_rates
        ):
            raise WakewordExportError("wakeword clip totals or source rates mismatch")
        if manifest.get("license_ledger") != {"path": "provenance/licenses.json"}:
            raise WakewordExportError("wakeword license ledger path mismatch")
        ledger = artifacts._keys(
            artifacts._json(root, "provenance/licenses.json"),
            {"schema", "entries"},
            "wakeword license ledger",
        )
        if ledger.get("schema") != "avaas/license-ledger@v1":
            raise WakewordExportError("wakeword license ledger schema mismatch")
        _validate_license_entries(ledger.get("entries"))
        try:
            readme = contracts.read_regular(root, "README.md", maximum=256 * 1024).decode(
                "utf-8"
            )
        except (contracts.ContractError, UnicodeDecodeError) as exc:
            raise WakewordExportError("wakeword export README is invalid") from exc
        if not readme.strip():
            raise WakewordExportError("wakeword export README is empty")
        return {
            "schema": WAKEWORD_EXPORT_SCHEMA,
            "export_id": export_id,
            "version": version,
            "phrase": "Hey Piranesi",
            "alias": "piranesi",
            "source_artifact_id": source["artifact_id"],
            "clip_count": len(entries),
            "total_frames": total_frames,
            "promotable": promotable,
            "fixture": fixture,
            "path": root,
        }
    except WakewordExportError:
        raise
    except (artifacts.ArtifactContractError, contracts.ContractError) as exc:
        raise WakewordExportError(str(exc)) from exc


def validate_export(
    root: Path | str, *, require_promotable: bool = False
) -> dict[str, Any]:
    """Validate one published export whose id matches its directory."""

    return _validate_export(Path(root), require_promotable=require_promotable)

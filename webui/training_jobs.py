"""Immutable voice-training bundles and a bounded persisted job state machine."""
from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import shutil
import sqlite3
import stat
import tempfile
from pathlib import Path
from typing import Any

from . import audio_contracts, contracts, corpus, prompts
from .store import Store


TRAINING_JOB_SCHEMA = "avaas/training-job@v1"
MAX_JOB_SECONDS = 7 * 24 * 60 * 60
MAX_JOB_ROWS = 10_000
MAX_AUDIO_BYTES = 64 * 1024 * 1024
MAX_RETRIES = 3
ACTIVE_STATES = (
    "queued",
    "validating",
    "staging",
    "running",
    "evaluating",
    "packaging",
)
TERMINAL_STATES = ("succeeded", "failed", "cancelled")
ALLOWED_TRANSITIONS = {
    "queued": {"validating", "failed", "cancelled"},
    "validating": {"staging", "failed", "cancelled"},
    "staging": {"running", "failed", "cancelled"},
    "running": {"evaluating", "failed", "cancelled"},
    "evaluating": {"packaging", "failed", "cancelled"},
    "packaging": {"succeeded", "failed", "cancelled"},
}
_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_JOB_ID_RE = re.compile(r"^job-[0-9a-f]{48}$")
_AUDIO_PATH_RE = re.compile(r"^audio/([0-9a-f]{64})\.wav$")

ENGINE_PROFILES = {
    ("cosyvoice3", "expressive-zero-shot"): {
        "target_audio": {
            "purpose": "serve-24k",
            "sample_rate": 24_000,
            "channels": 1,
            "encoding": "pcm_s16le",
        },
        "hyperparameters": {
            "mode": "zero-shot",
            "seed": 20_260_716,
            "max_references": 16,
        },
        "toolchain": {
            "python": "3.12",
            "worker": "cosyvoice3-isolated",
            "native_sample_rate": 24_000,
        },
        "source_pins": {
            "code": "FunAudioLLM/CosyVoice@074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc",
            "model": "FunAudioLLM/Fun-CosyVoice3-0.5B@29e01c4e8d000f4bcd70751be16fa94bf3d85a18",
            "license": "Apache-2.0",
        },
    },
    ("piper", "medium"): {
        "target_audio": {
            "purpose": "piper-22050",
            "sample_rate": 22_050,
            "channels": 1,
            "encoding": "pcm_s16le",
        },
        "hyperparameters": {
            "seed": 20_260_716,
            "batch_size": 16,
            "learning_rate": 0.0002,
            "max_steps": 200_000,
            "synthetic_max_fraction": 0.35,
        },
        "toolchain": {
            "python": "3.12",
            "worker": "piper-1.4.2-gpl-isolated",
            "native_sample_rate": 22_050,
        },
        "source_pins": {
            "code": "OHF-Voice/piper1-gpl@d6975e21a440c0d8b6e5fb7c41027409af13d44d",
            "release": "1.4.2",
            "license": "GPL-3.0-only",
            "base_checkpoint_repository": (
                "rhasspy/piper-checkpoints@95a4b650bd38716c97caf16d07b2a1734859f91a"
            ),
            "base_checkpoint_path": "en/en_US/ljspeech/medium/lj-med_1000.ckpt",
            "base_checkpoint_sha256": (
                "dcf2449bdbdaad09256a08dfac211c59f6b36ce8d3f244fd844a9eb1d7384c7c"
            ),
            "base_checkpoint_license": "MIT",
            "base_training_data_license": "Public-Domain",
        },
    },
}


class JobContractError(contracts.ContractError):
    """A training bundle or state-machine input failed closed."""


class JobConflict(JobContractError):
    """A job compare-and-swap, idempotency, or active-slot conflict."""


def _canonical(value: Any) -> bytes:
    try:
        return contracts.canonical_json(value)
    except contracts.ContractError as exc:
        raise JobContractError(str(exc)) from exc


def _idempotency_key(value: str) -> str:
    if not isinstance(value, str) or _IDEMPOTENCY_RE.fullmatch(value) is None:
        raise JobContractError("invalid idempotency key")
    return value


def _job_id(value: str) -> str:
    if not isinstance(value, str) or _JOB_ID_RE.fullmatch(value) is None:
        raise JobContractError("invalid job id")
    return value


def _group_id(record: dict[str, Any]) -> str:
    value = record.get("variant_of")
    if isinstance(value, str) and value:
        return value
    return str(record.get("id", "")).rsplit("__", 1)[0]


def _find_derivative(record: dict[str, Any], purpose: str) -> dict[str, Any]:
    derivatives = record.get("derivatives")
    if not isinstance(derivatives, list) or len(derivatives) > 8:
        raise JobContractError("accepted record lacks bounded derivatives")
    matches = [
        item
        for item in derivatives
        if isinstance(item, dict) and item.get("purpose") == purpose
    ]
    if len(matches) != 1:
        raise JobContractError(f"accepted record lacks exactly one {purpose} derivative")
    return matches[0]


def _corpus_sha256(records: list[dict[str, Any]]) -> str:
    inventory = []
    for record in records:
        qc = record.get("qc") if isinstance(record.get("qc"), dict) else {}
        derivative_inventory = []
        for purpose in ("serve-24k", "piper-22050", "wake-16k"):
            try:
                derivative = _find_derivative(record, purpose)
                sha256 = derivative.get("sha256")
            except JobContractError:
                sha256 = None
            derivative_inventory.append({"purpose": purpose, "sha256": sha256})
        inventory.append(
            {
                "prompt_id": record.get("id"),
                "take_id": record.get("take_id"),
                "generation": record.get("generation"),
                "group_id": _group_id(record),
                "identity": record.get("identity"),
                "transcript": record.get("transcript"),
                "qc": {
                    "duration": float(qc.get("duration", 0.0)),
                    "status": qc.get("status"),
                    "flags": qc.get("flags", []),
                },
                "derivatives": derivative_inventory,
            }
        )
    return hashlib.sha256(
        _canonical(sorted(inventory, key=lambda item: str(item["prompt_id"])))
    ).hexdigest()


def _validate_splits(report: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, list[str]]:
    splits = report.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise JobContractError("invalid frozen split")
    normalized: dict[str, list[str]] = {}
    union: set[str] = set()
    for name in ("train", "validation", "test"):
        values = splits.get(name)
        if (
            not isinstance(values, list)
            or len(values) > MAX_JOB_ROWS
            or any(not isinstance(item, str) or not item for item in values)
            or values != sorted(values)
            or len(set(values)) != len(values)
            or union.intersection(values)
        ):
            raise JobContractError("invalid or overlapping frozen split")
        normalized[name] = values
        union.update(values)
    groups = {_group_id(record) for record in records}
    if union != groups or not normalized["train"]:
        raise JobContractError("frozen split does not cover the corpus")
    expected = hashlib.sha256(_canonical(normalized)).hexdigest()
    if report.get("split_sha256") != expected:
        raise JobContractError("frozen split checksum mismatch")
    return normalized


def _ensure_output_root(root: Path) -> Path:
    root = Path(root)
    cursor = root
    for part in ("data", "training", "jobs"):
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            try:
                os.mkdir(cursor, 0o750)
                metadata = os.lstat(cursor)
            except OSError as exc:
                raise JobContractError("cannot create training job root") from exc
        except OSError as exc:
            raise JobContractError("cannot inspect training job root") from exc
        if stat_is_symlink_or_not_directory(metadata.st_mode):
            raise JobContractError("training job root contains a symlink or non-directory")
    return cursor


def stat_is_symlink_or_not_directory(mode: int) -> bool:
    return stat.S_ISLNK(mode) or not stat.S_ISDIR(mode)


def _write_json(path: Path, value: Any) -> None:
    contracts.write_durable(path, _canonical(value) + b"\n")


def _copy_audio(root: Path, relative: str, destination: Path, expected_sha256: str) -> None:
    try:
        raw = contracts.read_regular(root, relative, maximum=MAX_AUDIO_BYTES)
    except contracts.ContractError as exc:
        raise JobContractError(str(exc)) from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise JobContractError("accepted derivative checksum mismatch")
    contracts.write_durable(destination, raw)


def build_bundle(
    root: Path,
    *,
    readiness_report: dict[str, Any],
    engine: str,
    profile: str,
    idempotency_key: str,
    promotable: bool,
    now_iso: str,
    fixture: bool = False,
) -> dict[str, Any]:
    """Create and validate one immutable job directory, writing READY last."""

    root = Path(root)
    key = (engine, profile)
    if key not in ENGINE_PROFILES:
        raise JobContractError("unsupported engine/profile")
    idempotency_key = _idempotency_key(idempotency_key)
    if not isinstance(promotable, bool) or not isinstance(fixture, bool):
        raise JobContractError("invalid promotability class")
    if fixture and promotable:
        raise JobContractError("fixture jobs can never be promotable")
    if readiness_report.get("schema") != "avaas/readiness-report@v1":
        raise JobContractError("unknown readiness report schema")
    if promotable and readiness_report.get("ready") is not True:
        raise JobContractError("promotable job requires a ready corpus")
    if not isinstance(now_iso, str) or not 1 <= len(now_iso) <= 64:
        raise JobContractError("invalid creation time")

    records = corpus.load_manifest(root)
    if not 1 <= len(records) <= MAX_JOB_ROWS:
        raise JobContractError("accepted record count outside bounds")
    if _corpus_sha256(records) != readiness_report.get("corpus_sha256"):
        raise JobContractError("readiness corpus checksum mismatch")
    splits = _validate_splits(readiness_report, records)
    config = ENGINE_PROFILES[key]
    seed = {
        "idempotency_key": idempotency_key,
        "engine": engine,
        "profile": profile,
        "corpus_sha256": readiness_report["corpus_sha256"],
        "split_sha256": readiness_report["split_sha256"],
        "promotable": promotable,
        "fixture": fixture,
    }
    job_id = "job-" + hashlib.sha256(_canonical(seed)).hexdigest()[:48]
    job_manifest = {
        "schema": TRAINING_JOB_SCHEMA,
        "job_id": job_id,
        "idempotency_key": idempotency_key,
        "engine": engine,
        "profile": profile,
        "corpus": {
            "schema": "avaas/corpus-snapshot@v1",
            "version": str(records[0].get("corpus_version")),
            "sha256": readiness_report["corpus_sha256"],
            "split_sha256": readiness_report["split_sha256"],
            "record_count": len(records),
        },
        "target_audio": dict(config["target_audio"]),
        "hyperparameters": dict(config["hyperparameters"]),
        "toolchain": dict(config["toolchain"]),
        "source_pins": dict(config["source_pins"]),
        "output": {
            "schema": "avaas/training-result@v1",
            "path": f"results/{job_id}",
        },
        "promotable": promotable,
        "fixture": fixture,
        "author": "CPCS",
        "created_at": now_iso,
    }
    jobs_root = _ensure_output_root(root)
    final = jobs_root / job_id
    if final.exists():
        validated = validate_bundle(final)
        if validated["idempotency_key"] != idempotency_key:
            raise JobConflict("job id collision")
        return _bundle_result(final, validated)

    staging = Path(tempfile.mkdtemp(prefix=f".{job_id}.", dir=jobs_root))
    try:
        (staging / "audio").mkdir(mode=0o750)
        (staging / "pronunciation").mkdir(mode=0o750)
        (staging / "provenance").mkdir(mode=0o750)
        _write_json(staging / "job.json", job_manifest)
        _write_json(staging / "splits.json", splits)
        identities = prompts.load_identities(root)
        _write_json(
            staging / "pronunciation" / "lexicon.json",
            {
                "schema": "avaas/pronunciation-lexicon@v1",
                "entries": identities["pronunciation_lexicon"],
            },
        )
        licenses = readiness_report.get("licenses")
        if not isinstance(licenses, list) or not licenses:
            raise JobContractError("license ledger is absent")
        _write_json(
            staging / "provenance" / "licenses.json",
            {"schema": "avaas/license-ledger@v1", "licenses": licenses},
        )

        header = (
            "audio_path",
            "prompt_id",
            "group_id",
            "identity",
            "recording_profile_id",
            "split",
            "text",
            "sha256",
        )
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(header)
        copied: set[str] = set()
        split_for = {
            group: name
            for name, groups in splits.items()
            for group in groups
        }
        purpose = config["target_audio"]["purpose"]
        for record in sorted(records, key=lambda item: str(item.get("id"))):
            derivative = _find_derivative(record, purpose)
            sha256 = contracts.digest(derivative.get("sha256"), "audio sha256")
            relative = contracts.safe_relative(derivative.get("path"), "accepted audio path")
            path = root / relative
            try:
                observed = audio_contracts.inspect_pcm16_wav(
                    path,
                    audio_contracts.AUDIO_SPECS[purpose],
                    min_seconds=0.4,
                    max_seconds=120.0,
                )
            except (OSError, audio_contracts.AudioContractError) as exc:
                raise JobContractError("accepted audio contract failed") from exc
            if observed["sha256"] != sha256:
                raise JobContractError("accepted audio checksum mismatch")
            audio_relative = f"audio/{sha256}.wav"
            if sha256 not in copied:
                _copy_audio(root, relative, staging / audio_relative, sha256)
                copied.add(sha256)
            group = _group_id(record)
            writer.writerow(
                (
                    audio_relative,
                    record.get("id"),
                    group,
                    record.get("identity"),
                    record.get("recording_profile_id"),
                    split_for[group],
                    record.get("transcript"),
                    sha256,
                )
            )
        metadata = buffer.getvalue().encode("utf-8")
        if len(metadata) > 4 * 1024 * 1024:
            raise JobContractError("metadata exceeds bound")
        contracts.write_durable(staging / "metadata.csv", metadata)

        content = contracts.inventory(staging)
        checksum_lines = []
        for relative in content:
            checksum_lines.append(
                f"{contracts.sha256_file(staging, relative)}  {relative}\n"
            )
        contracts.write_durable(
            staging / "checksums.sha256",
            "".join(checksum_lines).encode("ascii"),
        )
        _validate_bundle(staging, require_ready=False)
        for directory in (
            staging / "audio",
            staging / "pronunciation",
            staging / "provenance",
            staging,
        ):
            contracts.fsync_directory(directory)
        manifest_sha256 = contracts.sha256_file(staging, "job.json")
        contracts.write_durable(
            staging / "READY",
            (manifest_sha256 + "\n").encode("ascii"),
        )
        contracts.fsync_directory(staging)
        try:
            os.rename(staging, final)
        except FileExistsError:
            validated = validate_bundle(final)
            if validated["idempotency_key"] != idempotency_key:
                raise JobConflict("concurrent job id collision")
            return _bundle_result(final, validated)
        contracts.fsync_directory(jobs_root)
        validated = validate_bundle(final)
        return _bundle_result(final, validated)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _bundle_result(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": manifest["job_id"],
        "idempotency_key": manifest["idempotency_key"],
        "engine": manifest["engine"],
        "profile": manifest["profile"],
        "promotable": manifest["promotable"],
        "fixture": manifest["fixture"],
        "manifest_sha256": contracts.sha256_file(path, "job.json"),
        "path": path,
    }


def _validate_job_manifest(value: Any) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "job_id",
        "idempotency_key",
        "engine",
        "profile",
        "corpus",
        "target_audio",
        "hyperparameters",
        "toolchain",
        "source_pins",
        "output",
        "promotable",
        "fixture",
        "author",
        "created_at",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise JobContractError("job manifest keys mismatch")
    if value.get("schema") != TRAINING_JOB_SCHEMA:
        raise JobContractError("unknown training job schema")
    _job_id(value.get("job_id"))
    _idempotency_key(value.get("idempotency_key"))
    key = (value.get("engine"), value.get("profile"))
    if key not in ENGINE_PROFILES:
        raise JobContractError("unsupported engine/profile")
    config = ENGINE_PROFILES[key]
    if (
        value.get("target_audio") != config["target_audio"]
        or value.get("hyperparameters") != config["hyperparameters"]
        or value.get("toolchain") != config["toolchain"]
        or value.get("source_pins") != config["source_pins"]
    ):
        raise JobContractError("engine profile does not match pinned contract")
    corpus_value = value.get("corpus")
    if not isinstance(corpus_value, dict) or set(corpus_value) != {
        "schema",
        "version",
        "sha256",
        "split_sha256",
        "record_count",
    }:
        raise JobContractError("corpus snapshot keys mismatch")
    if (
        corpus_value.get("schema") != "avaas/corpus-snapshot@v1"
        or not isinstance(corpus_value.get("version"), str)
        or not 1 <= len(corpus_value["version"]) <= 64
        or not 1 <= corpus_value.get("record_count", 0) <= MAX_JOB_ROWS
    ):
        raise JobContractError("invalid corpus snapshot")
    contracts.digest(corpus_value.get("sha256"), "corpus sha256")
    contracts.digest(corpus_value.get("split_sha256"), "split sha256")
    output = value.get("output")
    if output != {
        "schema": "avaas/training-result@v1",
        "path": f"results/{value['job_id']}",
    }:
        raise JobContractError("invalid result output contract")
    if (
        not isinstance(value.get("promotable"), bool)
        or not isinstance(value.get("fixture"), bool)
        or (value["fixture"] and value["promotable"])
        or value.get("author") != "CPCS"
        or not isinstance(value.get("created_at"), str)
        or not 1 <= len(value["created_at"]) <= 64
    ):
        raise JobContractError("invalid provenance or promotability")
    return value


def _validate_bundle(root: Path, *, require_ready: bool) -> dict[str, Any]:
    try:
        files = contracts.inventory(root)
        if require_ready and "READY" not in files:
            raise JobContractError("READY receipt is absent")
        checksums = contracts.parse_checksums(root)
        expected_files = set(checksums) | {"checksums.sha256"}
        if require_ready:
            expected_files.add("READY")
        if set(files) != expected_files:
            raise JobContractError("unexpected or missing bundle file")
        for relative, expected in checksums.items():
            if contracts.sha256_file(root, relative) != expected:
                raise JobContractError(f"checksum mismatch: {relative}")
            mode = os.lstat(root / relative).st_mode
            if mode & 0o111:
                raise JobContractError("unexpected executable bundle file")
        manifest = _validate_job_manifest(contracts.read_json(root, "job.json"))
        manifest_sha256 = contracts.sha256_file(root, "job.json")
        if require_ready:
            ready = contracts.read_regular(root, "READY", maximum=65)
            if ready != (manifest_sha256 + "\n").encode("ascii"):
                raise JobContractError("READY receipt mismatch")

        splits = contracts.read_json(root, "splits.json")
        if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
            raise JobContractError("invalid split file")
        union: set[str] = set()
        for name in ("train", "validation", "test"):
            values = splits[name]
            if (
                not isinstance(values, list)
                or values != sorted(values)
                or len(set(values)) != len(values)
                or union.intersection(values)
            ):
                raise JobContractError("invalid or overlapping split file")
            union.update(values)
        split_sha256 = hashlib.sha256(_canonical(splits)).hexdigest()
        if split_sha256 != manifest["corpus"]["split_sha256"]:
            raise JobContractError("split file checksum mismatch")

        metadata_raw = contracts.read_regular(
            root,
            "metadata.csv",
            maximum=4 * 1024 * 1024,
        )
        try:
            reader = csv.DictReader(io.StringIO(metadata_raw.decode("utf-8")))
            rows = list(reader)
        except (UnicodeDecodeError, csv.Error) as exc:
            raise JobContractError("invalid metadata CSV") from exc
        expected_header = [
            "audio_path",
            "prompt_id",
            "group_id",
            "identity",
            "recording_profile_id",
            "split",
            "text",
            "sha256",
        ]
        if (
            reader.fieldnames != expected_header
            or len(rows) != manifest["corpus"]["record_count"]
            or not 1 <= len(rows) <= MAX_JOB_ROWS
        ):
            raise JobContractError("metadata rows or header mismatch")
        prompt_ids: set[str] = set()
        referenced_audio: set[str] = set()
        purpose = manifest["target_audio"]["purpose"]
        spec = audio_contracts.AUDIO_SPECS[purpose]
        for row in rows:
            if set(row) != set(expected_header):
                raise JobContractError("metadata row keys mismatch")
            prompt_id = row["prompt_id"]
            group_id = row["group_id"]
            if (
                not prompt_id
                or len(prompt_id) > 128
                or prompt_id in prompt_ids
                or group_id not in union
                or row["split"] not in splits
                or group_id not in splits[row["split"]]
                or row["identity"] not in {"satraj", "piranesi", "shared"}
                or row["recording_profile_id"] != "satraj.en-ca.natural"
                or not 1 <= len(row["text"]) <= 2_000
            ):
                raise JobContractError("invalid or duplicate metadata row")
            prompt_ids.add(prompt_id)
            match = _AUDIO_PATH_RE.fullmatch(row["audio_path"])
            sha256 = contracts.digest(row["sha256"], "metadata audio sha256")
            if match is None or match.group(1) != sha256:
                raise JobContractError("invalid metadata audio path")
            referenced_audio.add(row["audio_path"])
        actual_audio = {item for item in files if item.startswith("audio/")}
        if actual_audio != referenced_audio:
            raise JobContractError("audio inventory does not match metadata")
        for relative in sorted(actual_audio):
            try:
                observed = audio_contracts.inspect_pcm16_wav(
                    root / relative,
                    spec,
                    min_seconds=0.4,
                    max_seconds=120.0,
                )
            except (OSError, audio_contracts.AudioContractError) as exc:
                raise JobContractError("audio contract validation failed") from exc
            if observed["sha256"] != _AUDIO_PATH_RE.fullmatch(relative).group(1):
                raise JobContractError("audio checksum/name mismatch")

        lexicon = contracts.read_json(root, "pronunciation/lexicon.json")
        if (
            not isinstance(lexicon, dict)
            or set(lexicon) != {"schema", "entries"}
            or lexicon.get("schema") != "avaas/pronunciation-lexicon@v1"
            or not isinstance(lexicon.get("entries"), list)
            or len(lexicon["entries"]) != 2
        ):
            raise JobContractError("pronunciation lexicon is invalid")
        licenses = contracts.read_json(root, "provenance/licenses.json")
        if (
            not isinstance(licenses, dict)
            or set(licenses) != {"schema", "licenses"}
            or licenses.get("schema") != "avaas/license-ledger@v1"
            or not isinstance(licenses.get("licenses"), list)
            or not 4 <= len(licenses["licenses"]) <= 32
        ):
            raise JobContractError("license ledger is invalid")
        license_ids = [
            item.get("id")
            for item in licenses["licenses"]
            if isinstance(item, dict)
        ]
        required_licenses = {
            "harvard_100",
            "cmu_arctic_400",
            "avaas_local",
            "cosyvoice3-code" if manifest["engine"] == "cosyvoice3" else "piper-code",
        }
        if manifest["engine"] == "piper":
            required_licenses.add("piper-ljspeech-medium-base")
        if (
            len(license_ids) != len(licenses["licenses"])
            or len(set(license_ids)) != len(license_ids)
            or not required_licenses <= set(license_ids)
        ):
            raise JobContractError("license coverage is incomplete")
        return manifest
    except JobContractError:
        raise
    except contracts.ContractError as exc:
        raise JobContractError(str(exc)) from exc


def validate_bundle(root: Path) -> dict[str, Any]:
    """Validate a sealed training input without executing any contained data."""

    return _validate_bundle(Path(root), require_ready=True)


class JobStore:
    """SQLite-backed compare-and-swap job lifecycle."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.store = Store(self.root)

    @staticmethod
    def _public(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["promotable"] = bool(value["promotable"])
        value["fixture"] = bool(value["fixture"])
        return value

    @staticmethod
    def _validate_time(value: int, field: str) -> int:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= 2**63 - 1
        ):
            raise JobContractError(f"invalid {field}")
        return value

    def _existing_by_idempotency(
        self,
        connection: sqlite3.Connection,
        key: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            "SELECT * FROM voice_training_jobs WHERE idempotency_key=?",
            (key,),
        ).fetchone()

    def enqueue(
        self,
        bundle: dict[str, Any],
        *,
        deadline_epoch: int,
        now_epoch: int,
    ) -> dict[str, Any]:
        if not isinstance(bundle, dict) or set(bundle) != {
            "job_id",
            "idempotency_key",
            "engine",
            "profile",
            "promotable",
            "fixture",
            "manifest_sha256",
            "path",
        }:
            raise JobContractError("bundle receipt keys mismatch")
        job_id = _job_id(bundle.get("job_id"))
        idempotency_key = _idempotency_key(bundle.get("idempotency_key"))
        now_epoch = self._validate_time(now_epoch, "enqueue time")
        deadline_epoch = self._validate_time(deadline_epoch, "deadline")
        if not now_epoch < deadline_epoch <= now_epoch + MAX_JOB_SECONDS:
            raise JobContractError("job deadline outside bounds")

        with self.store.transaction() as connection:
            existing = self._existing_by_idempotency(connection, idempotency_key)
            if existing is not None:
                if existing["job_id"] != job_id:
                    raise JobConflict("idempotency key is bound to another job")
                return self._public(existing)
            active = connection.execute(
                "SELECT job_id FROM voice_training_jobs WHERE state IN "
                "('queued','validating','staging','running','evaluating','packaging') LIMIT 1"
            ).fetchone()
            if active is not None:
                raise JobConflict(f"active job already exists: {active['job_id']}")

        path = Path(bundle["path"])
        manifest = validate_bundle(path)
        if (
            manifest["job_id"] != job_id
            or manifest["idempotency_key"] != idempotency_key
            or contracts.sha256_file(path, "job.json") != bundle["manifest_sha256"]
            or manifest["engine"] != bundle["engine"]
            or manifest["profile"] != bundle["profile"]
            or manifest["promotable"] is not bundle["promotable"]
            or manifest["fixture"] is not bundle["fixture"]
        ):
            raise JobContractError("bundle receipt does not match validated manifest")
        try:
            bundle_path = str(path.resolve(strict=True).relative_to(self.root.resolve(strict=True)))
        except (OSError, ValueError) as exc:
            raise JobContractError("bundle path escapes corpus root") from exc
        bundle_path = contracts.safe_relative(bundle_path, "bundle path")

        with self.store.transaction() as connection:
            existing = self._existing_by_idempotency(connection, idempotency_key)
            if existing is not None:
                if existing["job_id"] != job_id:
                    raise JobConflict("idempotency key is bound to another job")
                return self._public(existing)
            try:
                connection.execute(
                    "INSERT INTO voice_training_jobs(job_id,idempotency_key,engine,profile,"
                    "promotable,fixture,manifest_sha256,bundle_path,state,generation,attempts,"
                    "deadline_epoch,active_slot,detail,created_at_epoch,updated_at_epoch) "
                    "VALUES (?,?,?,?,?,?,?,?, 'queued',0,0,?,1,'queued',?,?)",
                    (
                        job_id,
                        idempotency_key,
                        manifest["engine"],
                        manifest["profile"],
                        int(manifest["promotable"]),
                        int(manifest["fixture"]),
                        bundle["manifest_sha256"],
                        bundle_path,
                        deadline_epoch,
                        now_epoch,
                        now_epoch,
                    ),
                )
                connection.execute(
                    "INSERT INTO voice_training_job_events(job_id,from_state,to_state,generation,"
                    "detail,event_at_epoch) VALUES (?,NULL,'queued',0,'queued',?)",
                    (job_id, now_epoch),
                )
            except sqlite3.IntegrityError as exc:
                raise JobConflict("active or idempotent job conflict") from exc
            row = connection.execute(
                "SELECT * FROM voice_training_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            return self._public(row)

    def get(self, job_id: str) -> dict[str, Any] | None:
        job_id = _job_id(job_id)
        connection = self.store._connect(read_only=True)
        try:
            row = connection.execute(
                "SELECT * FROM voice_training_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            return self._public(row) if row is not None else None
        finally:
            connection.close()

    def latest(self) -> dict[str, Any] | None:
        connection = self.store._connect(read_only=True)
        try:
            row = connection.execute(
                "SELECT * FROM voice_training_jobs ORDER BY created_at_epoch DESC,job_id DESC LIMIT 1"
            ).fetchone()
            return self._public(row) if row is not None else None
        finally:
            connection.close()

    def transition(
        self,
        job_id: str,
        *,
        expected_generation: int,
        new_state: str,
        now_epoch: int,
        detail: str = "",
    ) -> dict[str, Any]:
        job_id = _job_id(job_id)
        now_epoch = self._validate_time(now_epoch, "transition time")
        if (
            not isinstance(expected_generation, int)
            or isinstance(expected_generation, bool)
            or expected_generation < 0
            or not isinstance(detail, str)
            or len(detail) > 2_000
        ):
            raise JobContractError("invalid transition input")
        with self.store.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM voice_training_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise JobContractError("unknown training job")
            if row["generation"] != expected_generation:
                raise JobConflict(
                    f"job generation is {row['generation']}, expected {expected_generation}"
                )
            if new_state not in ALLOWED_TRANSITIONS.get(row["state"], set()):
                raise JobConflict(f"illegal transition {row['state']} -> {new_state}")
            if now_epoch > row["deadline_epoch"] and new_state not in {"failed", "cancelled"}:
                raise JobConflict("job deadline expired")
            generation = expected_generation + 1
            try:
                connection.execute(
                    "UPDATE voice_training_jobs SET state=?,generation=?,detail=?,updated_at_epoch=? "
                    "WHERE job_id=? AND generation=?",
                    (new_state, generation, detail[:2_000], now_epoch, job_id, expected_generation),
                )
                connection.execute(
                    "INSERT INTO voice_training_job_events(job_id,from_state,to_state,generation,"
                    "detail,event_at_epoch) VALUES (?,?,?,?,?,?)",
                    (job_id, row["state"], new_state, generation, detail[:2_000], now_epoch),
                )
            except sqlite3.IntegrityError as exc:
                raise JobConflict("illegal or concurrent training job transition") from exc
            updated = connection.execute(
                "SELECT * FROM voice_training_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            return self._public(updated)

    def retry(
        self,
        job_id: str,
        *,
        expected_generation: int,
        deadline_epoch: int,
        now_epoch: int,
    ) -> dict[str, Any]:
        job_id = _job_id(job_id)
        now_epoch = self._validate_time(now_epoch, "retry time")
        deadline_epoch = self._validate_time(deadline_epoch, "retry deadline")
        if not now_epoch < deadline_epoch <= now_epoch + MAX_JOB_SECONDS:
            raise JobContractError("retry deadline outside bounds")
        with self.store.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM voice_training_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise JobContractError("unknown training job")
            if row["generation"] != expected_generation:
                raise JobConflict("stale retry generation")
            if row["state"] != "failed" or row["attempts"] >= MAX_RETRIES:
                raise JobConflict("job is not retryable")
            generation = expected_generation + 1
            attempts = row["attempts"] + 1
            try:
                connection.execute(
                    "UPDATE voice_training_jobs SET state='queued',generation=?,attempts=?,"
                    "deadline_epoch=?,detail='retry queued',updated_at_epoch=? WHERE job_id=?",
                    (generation, attempts, deadline_epoch, now_epoch, job_id),
                )
                connection.execute(
                    "INSERT INTO voice_training_job_events(job_id,from_state,to_state,generation,"
                    "detail,event_at_epoch) VALUES (?,'failed','queued',?,'retry queued',?)",
                    (job_id, generation, now_epoch),
                )
            except sqlite3.IntegrityError as exc:
                raise JobConflict("active or illegal retry") from exc
            updated = connection.execute(
                "SELECT * FROM voice_training_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            return self._public(updated)

    def fail_expired(self, *, now_epoch: int) -> int:
        now_epoch = self._validate_time(now_epoch, "recovery time")
        with self.store.transaction() as connection:
            rows = connection.execute(
                "SELECT * FROM voice_training_jobs WHERE state IN "
                "('queued','validating','staging','running','evaluating','packaging') "
                "AND deadline_epoch < ? ORDER BY job_id LIMIT 100",
                (now_epoch,),
            ).fetchall()
            for row in rows:
                generation = row["generation"] + 1
                connection.execute(
                    "UPDATE voice_training_jobs SET state='failed',generation=?,"
                    "detail='deadline expired',updated_at_epoch=? WHERE job_id=?",
                    (generation, now_epoch, row["job_id"]),
                )
                connection.execute(
                    "INSERT INTO voice_training_job_events(job_id,from_state,to_state,generation,"
                    "detail,event_at_epoch) VALUES (?,?,'failed',?,'deadline expired',?)",
                    (row["job_id"], row["state"], generation, now_epoch),
                )
            return len(rows)

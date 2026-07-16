#!/usr/bin/env python3
"""Deterministic prompt compiler for the Satraj/Piranesi voice corpus.

Structured JSON and checked-in text corpora are authoritative.  The Markdown
recording script is generated for humans and is rejected when it drifts.

P10: every file, section, prompt, string, and hashing loop has a fixed ceiling.
The resulting Python objects are bounded by those application-level limits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any


IDENTITY_SCHEMA = "avaas/identities@v1"
CATALOG_SCHEMA = "avaas/prompt-catalog@v1"
PROVENANCE_SCHEMA = "avaas/prompt-provenance@v1"
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_CORPUS_BYTES = 2 * 1024 * 1024
MAX_SOURCE_LINES = 2_000
MAX_SECTIONS = 64
MAX_SOURCE_PROMPTS = 1_500
MAX_COMPILED_PROMPTS = 2_000
MAX_PROMPT_CHARS = 2_000
MAX_HASH_CHUNKS = 64
HASH_CHUNK_BYTES = 64 * 1024
PINNED_SOURCE_HASHES = {
    "2026.07.16": {
        "harvard_100": "eb79c0bfadc4b3988ee7caa343b84691c2ae1ded100f61fdb6966ef5b5bb9c72",
        "cmu_arctic_400": "58f6d9d3e2d7d8d836eb3a2d7eb36ebff824657e6ba5191099891aaa8f87de25",
        "avaas_local": "939172b760f9ba57f6e0b72afad2f40d6d285bb7863f29257f264b33d4898c71",
    }
}

_BASE_ID_RE = re.compile(r"^sec\d{2}[a-z]?_[0-9]{3}$")
_SECTION_ID_RE = re.compile(r"^sec\d{2}$")
# Case-sensitive by design: lowercase "sat" is an ordinary verb in the source
# corpora, while capitalized "Sat" is the unwanted personal form.
_FORBIDDEN_NAME_RE = re.compile(r"(?<![A-Za-z])Sat(?![A-Za-z])")
_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_PRESENTATION_ID_RE = re.compile(r"^[a-z0-9]+(?:[.-][a-z0-9]+){2,7}$")
_ALLOWED_KINDS = {"conversational", "identity", "read", "spell", "spontaneous", "take"}
_ALLOWED_DELIVERIES = {
    "neutral",
    "warm",
    "authoritative",
    "urgent",
    "whisper",
    "projected",
}
_REQUIRED_IDENTITY_FRAGMENTS = (
    "this is satraj",
    "this is piranesi",
    "s a t r a j",
    "p i r a n e s i",
    "voicemail",
    "call you back",
    "please hold",
    "transfer",
    "wrong person",
    "hey piranesi",
    "speaking as piranesi",
    "bequeathed this voice",
    "noisy room",
)


class PromptContractError(ValueError):
    """A checked-in prompt source violated its declared contract."""


def _require_regular(path: Path, max_bytes: int) -> int:
    try:
        if path.is_symlink() or not path.is_file():
            raise PromptContractError(f"not a regular source file: {path}")
        size = path.stat().st_size
    except OSError as exc:
        raise PromptContractError(f"cannot inspect source file: {path}") from exc
    if size <= 0 or size > max_bytes:
        raise PromptContractError(f"source size outside 1..{max_bytes}: {path}")
    return size


def _read_bounded_bytes(path: Path, max_bytes: int) -> bytes:
    """Read one immutable regular-file snapshot without allocating past its bound."""
    descriptor = -1
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise PromptContractError(f"not a regular source file: {path}")
        if metadata.st_size <= 0 or metadata.st_size > max_bytes:
            raise PromptContractError(f"source size outside 1..{max_bytes}: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            raw = handle.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise PromptContractError(f"source exceeds {max_bytes} bytes: {path}")
        if len(raw) != metadata.st_size:
            raise PromptContractError(f"source changed while reading: {path}")
        return raw
    except PromptContractError:
        raise
    except OSError as exc:
        raise PromptContractError(f"cannot read source file: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = _read_bounded_bytes(path, MAX_JSON_BYTES)
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromptContractError(f"invalid JSON source: {path}") from exc
    if not isinstance(value, dict):
        raise PromptContractError(f"JSON source must be an object: {path}")
    return value


def _resolve_prompt_source(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise PromptContractError("invalid prompt source path")
    candidate = Path(relative)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise PromptContractError(f"prompt source path escapes root: {relative!r}")
    source_root = (Path(root) / "prompts").resolve()
    resolved = (source_root / candidate).resolve()
    if resolved == source_root or source_root not in resolved.parents:
        raise PromptContractError(f"prompt source path escapes root: {relative!r}")
    return resolved


def sha256_file(path: Path) -> str:
    """Hash a bounded regular file without following a symlink."""
    size = _require_regular(path, MAX_CORPUS_BYTES)
    digest = hashlib.sha256()
    read_bytes = 0
    try:
        with path.open("rb") as handle:
            for _ in range(MAX_HASH_CHUNKS):
                chunk = handle.read(HASH_CHUNK_BYTES)
                if not chunk:
                    break
                read_bytes += len(chunk)
                digest.update(chunk)
            if handle.read(1):
                raise PromptContractError(f"source exceeds hash loop bound: {path}")
    except OSError as exc:
        raise PromptContractError(f"cannot hash source: {path}") from exc
    if read_bytes != size:
        raise PromptContractError(f"source changed while hashing: {path}")
    return digest.hexdigest()


def read_corpus_lines(path: Path) -> list[str]:
    """Read a bounded UTF-8 corpus and reject blanks or oversized prompts."""
    try:
        text = _read_bounded_bytes(path, MAX_CORPUS_BYTES).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PromptContractError(f"invalid UTF-8 corpus: {path}") from exc
    raw_lines = text.splitlines()
    if not raw_lines or len(raw_lines) > MAX_SOURCE_LINES:
        raise PromptContractError(f"corpus line count outside bounds: {path}")
    lines: list[str] = []
    for index, raw in enumerate(raw_lines, start=1):
        line = raw.strip()
        if not line or len(line) > MAX_PROMPT_CHARS:
            raise PromptContractError(f"invalid corpus line {index}: {path}")
        lines.append(line)
    return lines


def load_identities(root: Path) -> dict[str, Any]:
    value = _read_json(Path(root) / "prompts" / "identities.json")
    if set(value) != {
        "schema",
        "speaker_id",
        "voice_model_id",
        "recording_profile",
        "identities",
        "presentations",
        "calibration_sets",
        "additional_languages",
        "pronunciation_lexicon",
    }:
        raise PromptContractError("identity keys mismatch")
    if value.get("schema") != IDENTITY_SCHEMA:
        raise PromptContractError(f"unknown identity schema: {value.get('schema')!r}")
    if value.get("speaker_id") != "satraj" or value.get("voice_model_id") != "satraj-piranesi":
        raise PromptContractError("identity speaker/model contract mismatch")
    recording_profile = value.get("recording_profile")
    if not isinstance(recording_profile, dict) or set(recording_profile) != {
        "id",
        "language_tag",
        "accent",
        "default_delivery",
        "accent_imitation_required",
    }:
        raise PromptContractError("recording profile keys mismatch")
    if recording_profile.get("accent_imitation_required") is not False:
        raise PromptContractError("recording profile must never require accent imitation")
    if recording_profile != {
        "id": "satraj.en-ca.natural",
        "language_tag": "en-CA",
        "accent": "natural",
        "default_delivery": "neutral",
        "accent_imitation_required": False,
    }:
        raise PromptContractError("recording profile contract mismatch")

    entries = value.get("identities")
    if not isinstance(entries, list) or len(entries) != 2:
        raise PromptContractError("identity contract needs exactly two aliases")
    expected = [("satraj", "Satraj"), ("piranesi", "Piranesi")]
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "id",
            "display_name",
            "role",
            "default_presentation",
        }:
            raise PromptContractError("identity alias keys mismatch")
    actual = [(entry.get("id"), entry.get("display_name")) for entry in entries]
    if actual != expected:
        raise PromptContractError("identity aliases/order mismatch")
    if [entry["role"] for entry in entries] != ["creator", "bequeathed"]:
        raise PromptContractError("identity roles mismatch")

    presentations = value.get("presentations")
    if not isinstance(presentations, list) or len(presentations) != 2:
        raise PromptContractError("presentation contract needs exactly two defaults")
    expected_presentations = (
        {
            "id": "satraj.en-ca.neutral",
            "identity": "satraj",
            "language_tag": "en-CA",
            "accent": "general-north-american",
            "delivery": "neutral",
        },
        {
            "id": "piranesi.en-gb.neutral",
            "identity": "piranesi",
            "language_tag": "en-GB",
            "accent": "received-pronunciation",
            "delivery": "neutral",
        },
    )
    presentation_ids: set[str] = set()
    for presentation, expected_presentation in zip(
        presentations, expected_presentations, strict=True
    ):
        if not isinstance(presentation, dict) or set(presentation) != {
            "id",
            "identity",
            "language_tag",
            "accent",
            "delivery",
        }:
            raise PromptContractError("presentation keys mismatch")
        presentation_id = presentation.get("id")
        if (
            not isinstance(presentation_id, str)
            or len(presentation_id) > 64
            or not _PRESENTATION_ID_RE.fullmatch(presentation_id)
            or presentation_id in presentation_ids
            or presentation != expected_presentation
        ):
            raise PromptContractError("invalid or duplicate presentation")
        presentation_ids.add(presentation_id)
    for entry, expected_presentation in zip(entries, expected_presentations, strict=True):
        if entry.get("default_presentation") != expected_presentation["id"]:
            raise PromptContractError("identity default presentation mismatch")

    calibration_sets = value.get("calibration_sets")
    if not isinstance(calibration_sets, list) or len(calibration_sets) != len(
        _ALLOWED_DELIVERIES
    ):
        raise PromptContractError("calibration delivery set mismatch")
    seen_deliveries: set[str] = set()
    seen_calibration_prompts: set[str] = set()
    for calibration in calibration_sets:
        if not isinstance(calibration, dict) or set(calibration) != {
            "delivery",
            "optional",
            "prompt_ids",
            "safety_cue",
        }:
            raise PromptContractError("calibration keys mismatch")
        delivery = calibration.get("delivery")
        if delivery not in _ALLOWED_DELIVERIES or delivery in seen_deliveries:
            raise PromptContractError("invalid or duplicate calibration delivery")
        seen_deliveries.add(delivery)
        optional = calibration.get("optional")
        if not isinstance(optional, bool) or optional is (delivery == "neutral"):
            raise PromptContractError("calibration optionality mismatch")
        safety_cue = calibration.get("safety_cue")
        if not isinstance(safety_cue, str) or not 1 <= len(safety_cue) <= 256:
            raise PromptContractError("calibration safety cue outside bounds")
        prompt_ids = calibration.get("prompt_ids")
        if not isinstance(prompt_ids, list) or not 1 <= len(prompt_ids) <= 16:
            raise PromptContractError("calibration prompt count outside bounds")
        for prompt_id in prompt_ids:
            if (
                not isinstance(prompt_id, str)
                or not _BASE_ID_RE.fullmatch(prompt_id)
                or prompt_id in seen_calibration_prompts
            ):
                raise PromptContractError("invalid or duplicate calibration prompt")
            seen_calibration_prompts.add(prompt_id)
    if seen_deliveries != _ALLOWED_DELIVERIES:
        raise PromptContractError("calibration delivery set mismatch")

    additional_languages = value.get("additional_languages")
    if not isinstance(additional_languages, list) or additional_languages:
        raise PromptContractError("additional language packs must remain undeclared until supplied")

    lexicon = value.get("pronunciation_lexicon")
    if not isinstance(lexicon, list) or len(lexicon) != 2:
        raise PromptContractError("pronunciation lexicon needs both names")
    expected_names = (("Satraj", "satraj"), ("Piranesi", "piranesi"))
    validation_ids: set[str] = set()
    for entry, (expected_name, expected_identity) in zip(lexicon, expected_names, strict=True):
        if not isinstance(entry, dict) or set(entry) != {
            "name",
            "language",
            "status",
            "phonemes",
            "validation_prompts",
        }:
            raise PromptContractError("pronunciation lexicon keys mismatch")
        if entry.get("name") != expected_name or entry.get("language") != "en-CA":
            raise PromptContractError("pronunciation name/language mismatch")
        if entry.get("status") != "speaker-validation-required":
            raise PromptContractError("name pronunciation must require speaker validation")
        if entry.get("phonemes") is not None:
            raise PromptContractError("unvalidated name phonemes must not be guessed")
        prompts = entry.get("validation_prompts")
        if not isinstance(prompts, list) or not 1 <= len(prompts) <= 16:
            raise PromptContractError("validation prompt count outside bounds")
        for prompt_id in prompts:
            if (
                not isinstance(prompt_id, str)
                or not re.fullmatch(r"sec\d{2}[a-z]?_[0-9]{3}__(satraj|piranesi)", prompt_id)
                or not prompt_id.endswith(f"__{expected_identity}")
                or prompt_id in validation_ids
            ):
                raise PromptContractError(f"invalid or duplicate validation prompt: {prompt_id!r}")
            validation_ids.add(prompt_id)
    return value


def load_provenance(root: Path) -> dict[str, Any]:
    value = _read_json(Path(root) / "prompts" / "provenance.json")
    if set(value) != {"schema", "corpus_version", "sources"}:
        raise PromptContractError("provenance keys mismatch")
    if value.get("schema") != PROVENANCE_SCHEMA:
        raise PromptContractError(f"unknown provenance schema: {value.get('schema')!r}")
    if not isinstance(value.get("corpus_version"), str) or not re.fullmatch(
        r"\d{4}\.\d{2}\.\d{2}", value["corpus_version"]
    ):
        raise PromptContractError("invalid provenance corpus version")
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources or len(sources) > 16:
        raise PromptContractError("provenance source count outside bounds")
    ids: set[str] = set()
    common_keys = {
        "id",
        "path",
        "sha256",
        "source_url",
        "source_revision",
        "selection",
        "license_name",
        "license_url",
        "distribution_restrictions",
        "materialization",
        "redistribution_allowed",
        "required",
    }
    expected_extra_keys = {
        "harvard_100": {"expected_lines", "source_sha256"},
        "cmu_arctic_400": {"expected_lines", "source_sha256", "upstream_url"},
        "avaas_local": {"expected_records"},
    }
    for source in sources:
        if not isinstance(source, dict):
            raise PromptContractError("provenance source must be an object")
        source_id = source.get("id")
        if not isinstance(source_id, str) or source_id in ids or source_id not in expected_extra_keys:
            raise PromptContractError(f"duplicate or invalid provenance id: {source_id!r}")
        if set(source) != common_keys | expected_extra_keys[source_id]:
            raise PromptContractError(f"provenance source keys mismatch: {source_id}")
        ids.add(source_id)
        for field in ("path", "source_revision", "selection", "license_name", "distribution_restrictions"):
            if not isinstance(source.get(field), str) or not 1 <= len(source[field]) <= 4_096:
                raise PromptContractError(f"provenance {source_id} missing {field}")
        for field in ("source_url", "license_url"):
            if (
                not isinstance(source.get(field), str)
                or not source[field].startswith(("https://", "http://"))
                or len(source[field]) > 2_048
            ):
                raise PromptContractError(f"provenance {source_id} invalid {field}")
        if not isinstance(source.get("sha256"), str) or not _DIGEST_RE.fullmatch(source["sha256"]):
            raise PromptContractError(f"provenance {source_id} invalid sha256")
        if "source_sha256" in source and (
            not isinstance(source["source_sha256"], str)
            or not _DIGEST_RE.fullmatch(source["source_sha256"])
        ):
            raise PromptContractError(f"provenance {source_id} invalid source_sha256")
        if source.get("materialization") not in {"build-only", "checked-in"}:
            raise PromptContractError(f"provenance {source_id} invalid materialization")
        if not isinstance(source.get("redistribution_allowed"), bool):
            raise PromptContractError(f"provenance {source_id} invalid redistribution flag")
        if source_id == "harvard_100":
            if source["materialization"] != "build-only" or source["redistribution_allowed"] is not False:
                raise PromptContractError("Harvard source must remain build-only and non-redistributable")
        elif source["materialization"] != "checked-in" or source["redistribution_allowed"] is not True:
            raise PromptContractError(f"provenance {source_id} checked-in distribution mismatch")
        count_field = "expected_records" if source_id == "avaas_local" else "expected_lines"
        count = source.get(count_field)
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= MAX_SOURCE_PROMPTS:
            raise PromptContractError(f"provenance {source_id} invalid {count_field}")
        if "upstream_url" in source and (
            not isinstance(source["upstream_url"], str)
            or not source["upstream_url"].startswith(("https://", "http://"))
            or len(source["upstream_url"]) > 2_048
        ):
            raise PromptContractError(f"provenance {source_id} invalid upstream_url")
        if source.get("required") is not True:
            raise PromptContractError(f"provenance {source_id} is not mandatory")
    if ids != set(expected_extra_keys):
        raise PromptContractError("mandatory provenance source set mismatch")
    return value


def _load_catalog(root: Path) -> dict[str, Any]:
    value = _read_json(Path(root) / "prompts" / "catalog.json")
    if set(value) != {"schema", "corpus_version", "sections"}:
        raise PromptContractError("catalog keys mismatch")
    if value.get("schema") != CATALOG_SCHEMA:
        raise PromptContractError(f"unknown catalog schema: {value.get('schema')!r}")
    version = value.get("corpus_version")
    if not isinstance(version, str) or not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", version):
        raise PromptContractError("invalid corpus version")
    sections = value.get("sections")
    if not isinstance(sections, list) or not 1 <= len(sections) <= MAX_SECTIONS:
        raise PromptContractError("catalog section count outside bounds")
    return value


def _slug(text: str, limit: int = 40) -> str:
    value = _SLUG_RE.sub("-", text.lower()).strip("-")
    return value[:limit] or "line"


def _record(
    *,
    prompt_id: str,
    base_id: str,
    identity: str,
    text: str,
    kind: str,
    section: str,
    source: str,
    version: str,
) -> dict[str, Any]:
    if not isinstance(text, str) or not text or len(text) > MAX_PROMPT_CHARS:
        raise PromptContractError(f"prompt text outside bounds: {base_id}")
    if _FORBIDDEN_NAME_RE.search(text):
        raise PromptContractError(f"forbidden user-facing name in {base_id}")
    return {
        "id": prompt_id,
        "variant_of": base_id,
        "identity": identity,
        "speaker_id": "satraj",
        "voice_model_id": "satraj-piranesi",
        "text": text,
        "kind": kind,
        "section": section,
        "source": source,
        "corpus_version": version,
        "slug": _slug(text),
    }


def _compile_local_prompt(
    item: dict[str, Any], section_id: str, version: str, aliases: dict[str, str]
) -> list[dict[str, Any]]:
    common_keys = {"id", "kind", "source"}
    shapes = (
        common_keys | {"text"},
        common_keys | {"identity_variants", "template"},
        common_keys | {"variants"},
    )
    if set(item) not in shapes:
        raise PromptContractError(f"prompt declaration keys mismatch: {item.get('id')!r}")
    base_id = item.get("id")
    kind = item.get("kind")
    source = item.get("source")
    if not isinstance(base_id, str) or not _BASE_ID_RE.fullmatch(base_id):
        raise PromptContractError(f"invalid prompt base id: {base_id!r}")
    if kind not in _ALLOWED_KINDS or source != "avaas_local":
        raise PromptContractError(f"prompt {base_id} missing kind/source")
    if not re.fullmatch(rf"{re.escape(section_id)}[a-z]?_[0-9]{{3}}", base_id):
        raise PromptContractError(f"prompt {base_id} is in the wrong section")

    if item.get("identity_variants") is True:
        template = item.get("template")
        remainder = template.replace("{display_name}", "") if isinstance(template, str) else "{}"
        if (
            not isinstance(template, str)
            or template.count("{display_name}") < 1
            or "{" in remainder
            or "}" in remainder
        ):
            raise PromptContractError(f"identity template malformed: {base_id}")
        return [
            _record(
                prompt_id=f"{base_id}__{identity}",
                base_id=base_id,
                identity=identity,
                text=template.format(display_name=aliases[identity]),
                kind=kind,
                section=section_id,
                source=source,
                version=version,
            )
            for identity in ("satraj", "piranesi")
        ]

    variants = item.get("variants")
    if variants is not None:
        if not isinstance(variants, dict) or set(variants) != {"satraj", "piranesi"}:
            raise PromptContractError(f"identity variants malformed: {base_id}")
        if any(not isinstance(variants[identity], str) for identity in ("satraj", "piranesi")):
            raise PromptContractError(f"identity variant text malformed: {base_id}")
        return [
            _record(
                prompt_id=f"{base_id}__{identity}",
                base_id=base_id,
                identity=identity,
                text=variants[identity],
                kind=kind,
                section=section_id,
                source=source,
                version=version,
            )
            for identity in ("satraj", "piranesi")
        ]

    text = item.get("text")
    if not isinstance(text, str):
        raise PromptContractError(f"prompt {base_id} missing text")
    return [
        _record(
            prompt_id=base_id,
            base_id=base_id,
            identity="shared",
            text=text,
            kind=kind,
            section=section_id,
            source=source,
            version=version,
        )
    ]


def _validate_source(path: Path, source: dict[str, Any], expected_lines: int | None = None) -> None:
    if expected_lines is not None:
        lines = read_corpus_lines(path)
        if len(lines) != expected_lines or source.get("expected_lines") != expected_lines:
            raise PromptContractError(f"source line count mismatch: {source['id']}")
    actual_hash = sha256_file(path)
    if actual_hash != source.get("sha256"):
        raise PromptContractError(f"source sha256 mismatch: {source['id']}")


def _annotate_prompt_metadata(
    sections: list[dict[str, Any]], identities: dict[str, Any]
) -> None:
    """Attach observed-performance and desired-synthesis metadata separately.

    P10: both presentation and calibration inputs were bounded by
    ``load_identities``; this pass visits each already-bounded prompt once.
    """
    presentation_ids = [item["id"] for item in identities["presentations"]]
    default_presentations = {
        item["id"]: item["default_presentation"] for item in identities["identities"]
    }
    recording_profile = identities["recording_profile"]
    calibrations = {
        prompt_id: (item["delivery"], item["optional"], item["safety_cue"])
        for item in identities["calibration_sets"]
        for prompt_id in item["prompt_ids"]
    }
    base_ids = {
        record["variant_of"]
        for section in sections
        for record in section["prompts"]
    }
    missing = sorted(set(calibrations) - base_ids)
    if missing:
        raise PromptContractError(f"calibration prompt is not in corpus: {missing[0]}")

    for section in sections:
        for record in section["prompts"]:
            delivery, optional, safety_cue = calibrations.get(
                record["variant_of"],
                (
                    recording_profile["default_delivery"],
                    False,
                    "Use your natural, comfortable speaking voice.",
                ),
            )
            identity = record["identity"]
            synthesis_presentation = default_presentations.get(identity)
            record.update(
                {
                    "recording_profile_id": recording_profile["id"],
                    "recording_language_tag": recording_profile["language_tag"],
                    "recording_accent": recording_profile["accent"],
                    "recording_delivery": delivery,
                    "recording_safety_cue": safety_cue,
                    "calibration_optional": optional,
                    "synthesis_presentation": synthesis_presentation,
                    "eligible_presentations": (
                        [synthesis_presentation]
                        if synthesis_presentation is not None
                        else list(presentation_ids)
                    ),
                }
            )


def compile_sections(root: Path) -> list[dict[str, Any]]:
    """Validate all sources and return server-compatible compiled sections."""
    root = Path(root)
    identities = load_identities(root)
    catalog = _load_catalog(root)
    provenance = load_provenance(root)
    version = catalog["corpus_version"]
    if provenance.get("corpus_version") != version:
        raise PromptContractError("catalog/provenance version mismatch")

    aliases = {entry["id"]: entry["display_name"] for entry in identities["identities"]}
    sources = {source["id"]: source for source in provenance["sources"]}
    required_sources = {"harvard_100", "cmu_arctic_400", "avaas_local"}
    if set(sources) != required_sources:
        raise PromptContractError("mandatory prompt source set mismatch")
    pinned_hashes = PINNED_SOURCE_HASHES.get(version)
    if pinned_hashes is None or set(pinned_hashes) != required_sources:
        raise PromptContractError("corpus version has no immutable source pins")
    for source_id, expected_hash in pinned_hashes.items():
        if sources[source_id].get("sha256") != expected_hash:
            raise PromptContractError(
                f"source {source_id} changed without a corpus version bump"
            )
    # Authenticate the editable declaration source before interpreting any of it.
    local_source = sources["avaas_local"]
    catalog_path = _resolve_prompt_source(root, local_source["path"])
    _validate_source(catalog_path, local_source)

    compiled_sections: list[dict[str, Any]] = []
    seen_sections: set[str] = set()
    seen_ids: set[str] = set()
    local_declarations = 0

    for raw_section in catalog["sections"]:
        if not isinstance(raw_section, dict):
            raise PromptContractError("catalog section must be an object")
        base_section_keys = {"section", "num", "title", "target_min"}
        external_section_keys = base_section_keys | {"source_id", "kind", "id_prefix"}
        local_section_keys = base_section_keys | {"prompts"}
        if frozenset(raw_section) not in {
            frozenset(external_section_keys),
            frozenset(local_section_keys),
        }:
            raise PromptContractError(
                f"catalog section keys mismatch: {raw_section.get('section')!r}"
            )
        section_id = raw_section.get("section")
        num = raw_section.get("num")
        title = raw_section.get("title")
        target_min = raw_section.get("target_min")
        if (
            not isinstance(section_id, str)
            or not _SECTION_ID_RE.fullmatch(section_id)
            or section_id in seen_sections
            or not isinstance(num, int)
            or section_id != f"sec{num:02d}"
            or not isinstance(title, str)
            or not title
            or not isinstance(target_min, (int, float))
            or target_min <= 0
        ):
            raise PromptContractError(f"invalid or duplicate section: {section_id!r}")
        seen_sections.add(section_id)
        section_prompts: list[dict[str, Any]] = []

        source_id = raw_section.get("source_id")
        if source_id is not None:
            if source_id not in {"harvard_100", "cmu_arctic_400"}:
                raise PromptContractError(f"invalid external source for {section_id}")
            source = sources[source_id]
            corpus_path = _resolve_prompt_source(root, source["path"])
            expected = int(source["expected_lines"])
            _validate_source(corpus_path, source, expected)
            lines = read_corpus_lines(corpus_path)
            kind = raw_section.get("kind")
            prefix = raw_section.get("id_prefix")
            if kind != "read" or prefix != section_id or len(lines) > MAX_SOURCE_PROMPTS:
                raise PromptContractError(f"invalid external section contract: {section_id}")
            for index, text in enumerate(lines, start=1):
                base_id = f"{prefix}_{index:03d}"
                section_prompts.append(
                    _record(
                        prompt_id=base_id,
                        base_id=base_id,
                        identity="shared",
                        text=text,
                        kind=kind,
                        section=section_id,
                        source=source_id,
                        version=version,
                    )
                )
        else:
            declarations = raw_section.get("prompts")
            if not isinstance(declarations, list) or not declarations:
                raise PromptContractError(f"mandatory section is empty: {section_id}")
            if len(declarations) > MAX_SOURCE_PROMPTS:
                raise PromptContractError(f"section prompt count outside bounds: {section_id}")
            local_declarations += len(declarations)
            for item in declarations:
                if not isinstance(item, dict):
                    raise PromptContractError(f"prompt declaration must be an object: {section_id}")
                section_prompts.extend(_compile_local_prompt(item, section_id, version, aliases))

        for index, record in enumerate(section_prompts, start=1):
            prompt_id = record["id"]
            if prompt_id in seen_ids:
                raise PromptContractError(f"duplicate compiled prompt id: {prompt_id}")
            seen_ids.add(prompt_id)
            record["idx"] = index
            record["sub"] = ""
        compiled_sections.append(
            {
                "section": section_id,
                "num": num,
                "title": title,
                "target_min": float(target_min),
                "prompts": section_prompts,
            }
        )

    expected_sections = {f"sec{number:02d}" for number in range(1, 14)}
    if seen_sections != expected_sections:
        raise PromptContractError("mandatory section set mismatch")
    if len(seen_ids) > MAX_COMPILED_PROMPTS:
        raise PromptContractError("compiled prompt count outside bounds")
    if local_declarations != local_source.get("expected_records"):
        raise PromptContractError("local source record count mismatch")

    for lexicon_entry in identities["pronunciation_lexicon"]:
        for prompt_id in lexicon_entry["validation_prompts"]:
            if prompt_id not in seen_ids:
                raise PromptContractError(f"validation prompt is not in corpus: {prompt_id}")

    _annotate_prompt_metadata(compiled_sections, identities)

    pair_members: dict[str, set[str]] = {}
    pair_texts: dict[str, dict[str, str]] = {}
    source_counts = {"harvard_100": 0, "cmu_arctic_400": 0}
    for section in compiled_sections:
        for record in section["prompts"]:
            if record["source"] in source_counts:
                source_counts[record["source"]] += 1
            if record["identity"] in aliases:
                pair_members.setdefault(record["variant_of"], set()).add(record["identity"])
                pair_texts.setdefault(record["variant_of"], {})[record["identity"]] = record["text"]
    if any(members != {"satraj", "piranesi"} for members in pair_members.values()):
        raise PromptContractError("identity variant set is incomplete")
    if len(pair_members) < 35:
        raise PromptContractError("identity corpus is under minimum coverage")
    for base_id, variants in pair_texts.items():
        satraj_text = variants.get("satraj", "")
        piranesi_text = variants.get("piranesi", "")
        if satraj_text == piranesi_text:
            raise PromptContractError(f"identity pair text is duplicated: {base_id}")
        normalized_satraj = re.sub(r"[^A-Za-z]", "", satraj_text).lower()
        normalized_piranesi = re.sub(r"[^A-Za-z]", "", piranesi_text).lower()
        if "satraj" not in normalized_satraj or "piranesi" not in normalized_piranesi:
            raise PromptContractError(f"identity pair does not name both variants: {base_id}")
    identity_text = "\n".join(text for variants in pair_texts.values() for text in variants.values()).lower()
    missing_fragments = [fragment for fragment in _REQUIRED_IDENTITY_FRAGMENTS if fragment not in identity_text]
    if missing_fragments:
        raise PromptContractError(f"identity coverage missing: {', '.join(missing_fragments)}")
    if source_counts != {"harvard_100": 100, "cmu_arctic_400": 400}:
        raise PromptContractError("external source compilation count mismatch")
    return compiled_sections


def compile_prompts(root: Path) -> list[dict[str, Any]]:
    return [prompt for section in compile_sections(root) for prompt in section["prompts"]]


def render_markdown(root: Path) -> str:
    identities = load_identities(root)
    sections = compile_sections(root)
    lines = [
        "# AVAAS Recording Script — Satraj/Piranesi",
        "",
        "<!-- GENERATED by `python -m webui.prompts --write`; do not edit by hand. -->",
        "",
        "One human speaker records every line. The `satraj` and `piranesi` labels",
        "are identity variants for the same `satraj-piranesi` voice model.",
        "",
        "Record in your natural accent. A Piranesi label names the intended synthesis",
        "presentation; it does not ask you to imitate King's English. Expressive",
        "calibration cues marked optional may be skipped, and projected means clear",
        "speech without shouting.",
        "",
        f"Speaker: **{identities['identities'][0]['display_name']}**",
        "",
    ]
    for section in sections:
        target = section["target_min"]
        lines.extend(
            [
                f"## Section {section['num']}: {section['title']} — ~{target:g} min",
                "",
                "```text",
            ]
        )
        for index, prompt in enumerate(section["prompts"], start=1):
            label = "" if prompt["identity"] == "shared" else f" [{prompt['identity']}]"
            if prompt["recording_delivery"] != "neutral":
                optional = "; optional" if prompt["calibration_optional"] else ""
                label += f" [record {prompt['recording_delivery']}{optional}]"
            lines.append(f"{index}. {prompt['text']}{label}")
        lines.extend(["```", ""])
    return "\n".join(lines).rstrip() + "\n"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compile the AVAAS recording corpus")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="fail if RECORDING-SCRIPT.md drifted")
    group.add_argument("--write", action="store_true", help="regenerate RECORDING-SCRIPT.md")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    rendered = render_markdown(root)
    destination = root / "RECORDING-SCRIPT.md"
    if args.check:
        try:
            current = destination.read_text(encoding="utf-8")
        except OSError:
            current = ""
        if current != rendered:
            print("RECORDING-SCRIPT.md is stale; run: python -m webui.prompts --write")
            return 1
        return 0
    _write_atomic(destination, rendered)
    print(f"wrote {destination} ({len(compile_prompts(root))} prompts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

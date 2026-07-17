"""Pure, machine-readable readiness evaluation for immutable voice releases."""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import audio_contracts, corpus, prompts
from .prompts import PromptContractError
from .store import Store, StoreContractError


MAX_BLOCKERS = 256
MAX_HEALTH_BYTES = 32 * 1024
MAX_REFERENCES = 16
HARD_QC_FLAGS = frozenset({"clipping", "too-short"})

ENGINE_LICENSES = (
    {
        "id": "cosyvoice3-code",
        "license_name": "Apache-2.0",
        "license_url": "https://github.com/FunAudioLLM/CosyVoice/blob/074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc/LICENSE",
        "source_revision": "FunAudioLLM/CosyVoice@074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc",
        "distribution_restrictions": "Retain Apache-2.0 license and NOTICE obligations.",
    },
    {
        "id": "cosyvoice3-model",
        "license_name": "Apache-2.0",
        "license_url": "https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B",
        "source_revision": "huggingface@29e01c4e8d000f4bcd70751be16fa94bf3d85a18",
        "distribution_restrictions": "Private profile references are never redistributed with base weights.",
    },
    {
        "id": "piper-code",
        "license_name": "GPL-3.0-only",
        "license_url": "https://github.com/OHF-Voice/piper1-gpl/blob/d6975e21a440c0d8b6e5fb7c41027409af13d44d/LICENSE.md",
        "source_revision": "OHF-Voice/piper1-gpl@d6975e21a440c0d8b6e5fb7c41027409af13d44d",
        "distribution_restrictions": "Piper remains an independently deployed GPL service boundary.",
    },
)


@dataclass(frozen=True, slots=True)
class ReadinessPolicy:
    minimum_clean_seconds: float = 3_600.0
    section_coverage_ratio: float = 0.80
    kind_minimum_seconds: tuple[tuple[str, float], ...] = (
        ("read", 1_800.0),
        ("identity", 60.0),
        ("conversational", 300.0),
        ("spontaneous", 600.0),
    )
    required_identity_groups: tuple[str, ...] | None = None
    minimum_reference_count: int = 6
    minimum_validation_groups: int = 1
    minimum_test_groups: int = 1
    required_profiles: tuple[str, ...] = (
        "cosyvoice3/expressive-zero-shot",
        "piper/medium",
    )

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.minimum_clean_seconds)
            or self.minimum_clean_seconds < 0
            or not math.isfinite(self.section_coverage_ratio)
            or not 0 <= self.section_coverage_ratio <= 1
            or not 1 <= self.minimum_reference_count <= MAX_REFERENCES
            or not 0 <= self.minimum_validation_groups <= 128
            or not 0 <= self.minimum_test_groups <= 128
            or len(self.kind_minimum_seconds) > 32
            or len(self.required_profiles) > 16
        ):
            raise ValueError("readiness policy outside bounds")
        seen: set[str] = set()
        for kind, seconds in self.kind_minimum_seconds:
            if (
                not isinstance(kind, str)
                or not kind
                or kind in seen
                or not isinstance(seconds, (int, float))
                or isinstance(seconds, bool)
                or not math.isfinite(float(seconds))
                or seconds < 0
            ):
                raise ValueError("invalid readiness kind threshold")
            seen.add(kind)
        if self.required_identity_groups is not None and (
            len(self.required_identity_groups) > 256
            or len(set(self.required_identity_groups)) != len(self.required_identity_groups)
        ):
            raise ValueError("invalid required identity groups")


DEFAULT_POLICY = ReadinessPolicy()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _block(blockers: list[dict[str, str]], code: str, detail: str) -> None:
    if len(blockers) < MAX_BLOCKERS:
        blockers.append({"code": code, "detail": detail[:512]})


def _read_health(root: Path) -> dict[str, Any] | None:
    path = root / "data" / "trainer_health.json"
    try:
        metadata = os.lstat(path)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or not 1 <= metadata.st_size <= MAX_HEALTH_BYTES
        ):
            return None
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            raw = os.read(descriptor, MAX_HEALTH_BYTES + 1)
        finally:
            os.close(descriptor)
        if len(raw) != metadata.st_size or len(raw) > MAX_HEALTH_BYTES:
            return None
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def load_trainer_health(root: Path) -> dict[str, Any] | None:
    """Read a bounded health receipt without mutating or probing the worker."""

    return _read_health(Path(root))


def _validate_health(
    health: dict[str, Any] | None,
    policy: ReadinessPolicy,
    blockers: list[dict[str, str]],
) -> dict[str, Any] | None:
    if not isinstance(health, dict) or set(health) != {
        "schema",
        "healthy",
        "worker_id",
        "profiles",
        "checked_at",
    }:
        _block(blockers, "trainer-unhealthy", "compatible trainer health receipt is absent")
        return None
    profiles = health.get("profiles")
    if (
        health.get("schema") != "avaas/trainer-health@v1"
        or health.get("healthy") is not True
        or not isinstance(health.get("worker_id"), str)
        or not 1 <= len(health["worker_id"]) <= 128
        or not isinstance(health.get("checked_at"), str)
        or not 1 <= len(health["checked_at"]) <= 64
        or not isinstance(profiles, list)
        or not 1 <= len(profiles) <= 16
        or any(not isinstance(item, str) or not item for item in profiles)
        or not set(policy.required_profiles) <= set(profiles)
    ):
        _block(blockers, "trainer-unhealthy", "trainer is unhealthy or lacks a required profile")
        return None
    return health


def _group_id(record: dict[str, Any]) -> str:
    value = record.get("variant_of")
    if isinstance(value, str) and value:
        return value
    prompt_id = str(record.get("id", ""))
    return prompt_id.rsplit("__", 1)[0]


def _frozen_splits(
    groups: list[str],
    *,
    validation_count: int,
    test_count: int,
) -> dict[str, list[str]]:
    ordered = sorted(groups, key=lambda item: (hashlib.sha256(item.encode()).hexdigest(), item))
    validation = ordered[:validation_count]
    test = ordered[validation_count : validation_count + test_count]
    train = ordered[validation_count + test_count :]
    return {"train": sorted(train), "validation": sorted(validation), "test": sorted(test)}


def _find_derivative(record: dict[str, Any], purpose: str) -> dict[str, Any] | None:
    declared = record.get("derivatives")
    if not isinstance(declared, list) or len(declared) > 8:
        return None
    matches = [
        item
        for item in declared
        if isinstance(item, dict) and item.get("purpose") == purpose
    ]
    return matches[0] if len(matches) == 1 else None


def evaluate(
    root: Path,
    sections: list[dict[str, Any]],
    *,
    policy: ReadinessPolicy | None = None,
    trainer_health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate readiness without mutating files, SQLite, jobs, or workers."""

    root = Path(root)
    policy = policy or DEFAULT_POLICY
    blockers: list[dict[str, str]] = []
    canonical_prompts: list[dict[str, Any]] = []
    provenance_sources: list[dict[str, Any]] = []
    try:
        compiled_sections = prompts.compile_sections(root)
        provenance = prompts.load_provenance(root)
        canonical_prompts = [
            prompt for section in compiled_sections for prompt in section["prompts"]
        ]
        provenance_sources = provenance["sources"]
        supplied_ids = [
            prompt.get("id")
            for section in sections
            for prompt in section.get("prompts", [])
        ]
        compiled_ids = [prompt["id"] for prompt in canonical_prompts]
        if supplied_ids != compiled_ids:
            _block(
                blockers,
                "mandatory-source-invalid",
                "server prompt snapshot differs from canonical corpus",
            )
        for source in provenance_sources:
            unique = {
                prompt.get("variant_of", prompt["id"])
                for prompt in canonical_prompts
                if prompt["source"] == source["id"]
            }
            expected = source.get("expected_records", source.get("expected_lines"))
            if len(unique) != expected:
                _block(
                    blockers,
                    "mandatory-source-invalid",
                    f"source {source['id']} count is {len(unique)}, expected {expected}",
                )
    except (PromptContractError, OSError, ValueError):
        _block(
            blockers,
            "mandatory-source-invalid",
            "mandatory prompt source validation failed",
        )

    records = corpus.load_manifest(root)
    store = Store(root)
    if store.database_path.exists():
        try:
            store.validate_accepted_checksums()
        except StoreContractError:
            _block(
                blockers,
                "accepted-content-invalid",
                "an accepted file is missing or corrupt",
            )

    durations_by_section: dict[str, float] = {}
    durations_by_kind: dict[str, float] = {}
    total_seconds = 0.0
    flagged: list[dict[str, Any]] = []
    accepted_by_group: dict[str, list[dict[str, Any]]] = {}
    corpus_inventory: list[dict[str, Any]] = []
    for record in records[:10_000]:
        qc = record.get("qc") if isinstance(record.get("qc"), dict) else {}
        duration = qc.get("duration", 0.0)
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(float(duration))
            or duration < 0
        ):
            _block(
                blockers,
                "qc-invalid",
                f"prompt {record.get('id')} has invalid duration",
            )
            duration = 0.0
        duration = float(duration)
        flags = qc.get("flags") if isinstance(qc.get("flags"), list) else []
        if any(flag in HARD_QC_FLAGS for flag in flags):
            _block(
                blockers,
                "hard-qc-failed",
                f"prompt {record.get('id')} has a hard QC failure",
            )
        if flags and not record.get("override_reason"):
            _block(
                blockers,
                "override-audit-missing",
                f"prompt {record.get('id')} lacks an override reason",
            )
        if flags:
            flagged.append(
                {
                    "id": record.get("id"),
                    "flags": flags,
                    "section": record.get("section"),
                    "text": record.get("prompt_text", ""),
                }
            )
        if (
            record.get("kind") != "spontaneous"
            and record.get("transcript") != record.get("prompt_text")
        ):
            _block(
                blockers,
                "transcript-mismatch",
                f"prompt {record.get('id')} transcript differs",
            )
        total_seconds += duration
        section = str(record.get("section", ""))
        kind = str(record.get("kind", ""))
        durations_by_section[section] = durations_by_section.get(section, 0.0) + duration
        durations_by_kind[kind] = durations_by_kind.get(kind, 0.0) + duration
        group = _group_id(record)
        accepted_by_group.setdefault(group, []).append(record)
        derivative_inventory = []
        for purpose in ("serve-24k", "piper-22050", "wake-16k"):
            derivative = _find_derivative(record, purpose)
            derivative_inventory.append(
                {
                    "purpose": purpose,
                    "sha256": derivative.get("sha256") if derivative else None,
                }
            )
        corpus_inventory.append(
            {
                "prompt_id": record.get("id"),
                "take_id": record.get("take_id"),
                "generation": record.get("generation"),
                "group_id": group,
                "identity": record.get("identity"),
                "transcript": record.get("transcript"),
                "qc": {
                    "duration": duration,
                    "status": qc.get("status"),
                    "flags": flags,
                },
                "derivatives": derivative_inventory,
            }
        )

    if total_seconds < policy.minimum_clean_seconds:
        _block(
            blockers,
            "clean-duration-insufficient",
            f"clean duration {total_seconds:.3f}s is below {policy.minimum_clean_seconds:.3f}s",
        )

    section_rows = []
    under_represented = []
    accepted_ids = {record.get("id") for record in records}
    for section in sections[:64]:
        section_id = str(section.get("section", ""))
        target_minutes = float(section.get("target_min", 0.0) or 0.0)
        seconds = durations_by_section.get(section_id, 0.0)
        coverage = seconds / (target_minutes * 60.0) if target_minutes > 0 else 1.0
        recorded = sum(
            1
            for prompt in section.get("prompts", [])
            if prompt.get("id") in accepted_ids
        )
        row = {
            "section": section_id,
            "title": section.get("title", ""),
            "recorded": recorded,
            "total": len(section.get("prompts", [])),
            "minutes": round(seconds / 60.0, 2),
            "target_min": target_minutes,
            "coverage": round(min(coverage, 1.0), 3),
        }
        section_rows.append(row)
        if target_minutes > 0 and coverage < policy.section_coverage_ratio:
            under_represented.append(section_id)
            _block(
                blockers,
                "section-coverage-insufficient",
                f"section {section_id} is under target",
            )

    for kind, minimum in policy.kind_minimum_seconds:
        if durations_by_kind.get(kind, 0.0) < minimum:
            _block(
                blockers,
                "kind-duration-insufficient",
                f"kind {kind} is under target",
            )

    canonical_groups: dict[str, set[str]] = {}
    for prompt in canonical_prompts:
        group = str(prompt.get("variant_of", prompt["id"]))
        canonical_groups.setdefault(group, set()).add(prompt.get("identity"))
    required_groups = (
        policy.required_identity_groups
        if policy.required_identity_groups is not None
        else tuple(
            sorted(
                group
                for group, identities in canonical_groups.items()
                if identities == {"satraj", "piranesi"}
            )
        )
    )
    for group in required_groups:
        identities = {
            record.get("identity") for record in accepted_by_group.get(group, [])
        }
        if identities != {"satraj", "piranesi"}:
            _block(
                blockers,
                "identity-group-incomplete",
                f"identity group {group} lacks both variants",
            )

    groups = sorted(accepted_by_group)
    splits = _frozen_splits(
        groups,
        validation_count=policy.minimum_validation_groups,
        test_count=policy.minimum_test_groups,
    )
    if len(splits["validation"]) < policy.minimum_validation_groups:
        _block(blockers, "validation-split-insufficient", "validation split lacks groups")
    if len(splits["test"]) < policy.minimum_test_groups:
        _block(blockers, "test-split-insufficient", "test split lacks groups")
    if not splits["train"]:
        _block(blockers, "train-split-empty", "training split has no groups")

    references: list[dict[str, Any]] = []
    for group in splits["train"]:
        for record in sorted(
            accepted_by_group[group],
            key=lambda item: str(item.get("id")),
        ):
            qc = record.get("qc", {})
            if qc.get("status") != "pass" or qc.get("flags"):
                continue
            derivative = _find_derivative(record, "serve-24k")
            if derivative is None:
                continue
            path = root / str(derivative.get("path"))
            try:
                observed = audio_contracts.inspect_pcm16_wav(
                    path,
                    audio_contracts.AUDIO_SPECS["serve-24k"],
                    min_seconds=0.4,
                    max_seconds=120.0,
                )
            except (OSError, audio_contracts.AudioContractError):
                continue
            if observed["sha256"] != derivative.get("sha256"):
                continue
            references.append(
                {
                    "prompt_id": record["id"],
                    "group_id": group,
                    "identity": record.get("identity"),
                    "transcript": record.get("transcript"),
                    "path": derivative["path"],
                    "sha256": derivative["sha256"],
                }
            )
            if len(references) >= MAX_REFERENCES:
                break
        if len(references) >= MAX_REFERENCES:
            break
    if len(references) < policy.minimum_reference_count:
        _block(
            blockers,
            "reference-set-insufficient",
            "clean transcript-matched references are insufficient",
        )

    licenses = [
        {
            "id": source["id"],
            "license_name": source["license_name"],
            "license_url": source["license_url"],
            "source_revision": source["source_revision"],
            "distribution_restrictions": source["distribution_restrictions"],
        }
        for source in provenance_sources
    ] + [dict(item) for item in ENGINE_LICENSES]
    if len(licenses) < 6:
        _block(
            blockers,
            "license-ledger-incomplete",
            "source and engine license ledger is incomplete",
        )

    validated_health = _validate_health(
        trainer_health if trainer_health is not None else _read_health(root),
        policy,
        blockers,
    )
    corpus_sha256 = hashlib.sha256(
        _canonical(
            sorted(
                corpus_inventory,
                key=lambda item: str(item["prompt_id"]),
            )
        )
    ).hexdigest()
    split_sha256 = hashlib.sha256(_canonical(splits)).hexdigest()
    blockers = sorted(
        blockers,
        key=lambda item: (item["code"], item["detail"]),
    )[:MAX_BLOCKERS]
    return {
        "schema": "avaas/readiness-report@v1",
        "ready": not blockers,
        "blockers": blockers,
        "corpus_sha256": corpus_sha256,
        "split_sha256": split_sha256,
        "splits": splits,
        "references": references,
        "licenses": licenses,
        "trainer_health": validated_health,
        "total_clean_min": round(total_seconds / 60.0, 2),
        "target_min": round(policy.minimum_clean_seconds / 60.0, 2),
        "sections": section_rows,
        "under_represented": under_represented,
        "kind_seconds": {
            key: round(value, 3)
            for key, value in sorted(durations_by_kind.items())
        },
        "flagged": flagged,
        "recorded_count": len(records),
    }

#!/usr/bin/env python3
"""Verify and atomically restore an immutable AVAAS backup bundle."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

_APPLICATION_ROOT = Path(__file__).resolve().parents[1]
if str(_APPLICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(_APPLICATION_ROOT))

from scripts.backup import (  # noqa: E402
    MAX_CONTENT_BYTES,
    MAX_DATABASE_BYTES,
    BackupContractError,
    _copy_verified,
    _read_manifest,
    _remove_staging,
    verify_backup_bundle,
)
from webui.store import Store, StoreContractError  # noqa: E402


def _materialize_restore(source: Path, staging: Path, report: dict) -> None:
    manifest = _read_manifest(source)
    database = manifest["database"]
    _copy_verified(
        source,
        database["path"],
        staging,
        database["sha256"],
        MAX_DATABASE_BYTES,
    )
    restored_database = staging / "data/corpus.sqlite3"
    restored_database.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging / database["path"], restored_database)
    for item in manifest["files"]:
        _copy_verified(
            source, item["path"], staging, item["sha256"], MAX_CONTENT_BYTES
        )
    try:
        restored = Store.verify_backup(restored_database, content_root=staging)
    except StoreContractError as exc:
        raise BackupContractError("restored corpus failed verification") from exc
    if restored["takes"] != report["takes"] or restored["files"] != report["files"]:
        raise BackupContractError("restored corpus counts differ from backup")
    (staging / "database").rmdir()


def restore_backup(bundle: Path, destination: Path) -> dict:
    source = Path(bundle)
    report = verify_backup_bundle(source)
    destination = Path(destination)
    if os.path.lexists(destination):
        raise BackupContractError("restore destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise BackupContractError("restore parent must be a non-symlink directory")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.restore-", dir=destination.parent)
    )
    try:
        _materialize_restore(source, staging, report)
        try:
            os.rename(staging, destination)
        except OSError as exc:
            raise BackupContractError("cannot publish restored corpus") from exc
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        Store.verify_backup(destination / "data/corpus.sqlite3", content_root=destination)
        return report
    finally:
        if staging.exists():
            _remove_staging(staging, destination.parent)


def main(argv: Sequence[str] | None = None) -> int:
    if argv is not None and isinstance(argv, (str, bytes)):
        raise TypeError("argv must be a sequence of arguments")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    report = (
        verify_backup_bundle(args.bundle)
        if args.verify_only
        else restore_backup(args.bundle, args.destination)
    )
    if not isinstance(report, dict):
        raise AssertionError("restore operation returned an invalid receipt")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

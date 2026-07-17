#!/usr/bin/env python3
"""Create and verify immutable AVAAS corpus backup bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Sequence
from urllib.parse import quote

_APPLICATION_ROOT = Path(__file__).resolve().parents[1]
if str(_APPLICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(_APPLICATION_ROOT))

from webui.store import (  # noqa: E402
    HASH_CHUNK_BYTES,
    MAX_CONTENT_BYTES,
    MAX_DATABASE_BYTES,
    MAX_DERIVATIVES_PER_TAKE,
    MAX_EXPORT_ROWS,
    Store,
    StoreContractError,
)


SCHEMA = "avaas/backup@v1"
MAX_BACKUP_FILES = MAX_EXPORT_ROWS * (MAX_DERIVATIVES_PER_TAKE + 1)
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_CHECKSUM_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024 * 1024
MAX_DIRECTORIES = MAX_BACKUP_FILES * 8 + 32
MAX_JSON_NODES = 500_000
MAX_JSON_DEPTH = 24
_BACKUP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,127}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class BackupContractError(ValueError):
    """A backup source, bundle, or append-only publication failed closed."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _directory(path: Path, label: str) -> Path:
    path = Path(path)
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise BackupContractError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise BackupContractError(f"{label} must be a non-symlink directory")
    return path


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 1_024 or "\\" in value:
        raise BackupContractError("invalid backup path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise BackupContractError("invalid backup path")
    return path.as_posix()


def _source_path(root: Path, relative: str, maximum: int) -> tuple[Path, os.stat_result]:
    cursor = root
    parts = PurePosixPath(_safe_relative(relative)).parts
    for index, part in enumerate(parts):
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise BackupContractError(f"backup source is unavailable: {relative}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise BackupContractError(f"backup source contains a symlink: {relative}")
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise BackupContractError(f"backup source parent is not a directory: {relative}")
    minimum = 0 if maximum == 0 else 1
    if not stat.S_ISREG(metadata.st_mode) or not minimum <= metadata.st_size <= maximum:
        raise BackupContractError(f"backup source size outside bounds: {relative}")
    return cursor, metadata


def _copy_verified(
    source_root: Path,
    relative: str,
    target_root: Path,
    expected_sha256: str,
    maximum: int,
) -> int:
    if _SHA_RE.fullmatch(expected_sha256) is None:
        raise BackupContractError("invalid expected checksum")
    source, before = _source_path(source_root, relative, maximum)
    target = target_root.joinpath(*PurePosixPath(relative).parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    source_fd = -1
    target_fd = -1
    digest = hashlib.sha256()
    total = 0
    try:
        source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        observed = os.fstat(source_fd)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_ino != before.st_ino
            or observed.st_dev != before.st_dev
            or observed.st_size != before.st_size
        ):
            raise BackupContractError(f"backup source changed: {relative}")
        target_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(target_fd, "wb") as output:
            target_fd = -1
            for _index in range((maximum // HASH_CHUNK_BYTES) + 2):
                chunk = os.read(source_fd, HASH_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum:
                    raise BackupContractError(f"backup source exceeds bound: {relative}")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if total != observed.st_size or os.read(source_fd, 1):
            raise BackupContractError(f"backup source changed while copying: {relative}")
        if digest.hexdigest() != expected_sha256:
            raise BackupContractError(f"backup source checksum mismatch: {relative}")
        return total
    except OSError as exc:
        raise BackupContractError(f"cannot copy backup source: {relative}") from exc
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if target_fd >= 0:
            os.close(target_fd)
        if target.exists() and digest.hexdigest() != expected_sha256:
            target.unlink(missing_ok=True)


def _hash_regular(root: Path, relative: str, maximum: int) -> tuple[str, int]:
    path, before = _source_path(root, relative, maximum)
    descriptor = -1
    digest = hashlib.sha256()
    total = 0
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        observed = os.fstat(descriptor)
        if observed.st_ino != before.st_ino or observed.st_dev != before.st_dev:
            raise BackupContractError(f"backup file changed: {relative}")
        for _index in range((maximum // HASH_CHUNK_BYTES) + 2):
            chunk = os.read(descriptor, HASH_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise BackupContractError(f"backup file exceeds bound: {relative}")
            digest.update(chunk)
        if total != observed.st_size or os.read(descriptor, 1):
            raise BackupContractError(f"backup file changed while hashing: {relative}")
        return digest.hexdigest(), total
    except OSError as exc:
        raise BackupContractError(f"cannot hash backup file: {relative}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _read_regular(
    root: Path, relative: str, maximum: int
) -> tuple[bytes, os.stat_result]:
    path, before = _source_path(root, relative, maximum)
    descriptor = -1
    chunks: list[bytes] = []
    total = 0
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        observed = os.fstat(descriptor)
        if (
            observed.st_ino != before.st_ino
            or observed.st_dev != before.st_dev
            or observed.st_size != before.st_size
        ):
            raise BackupContractError(f"backup file changed: {relative}")
        for _index in range((maximum // HASH_CHUNK_BYTES) + 2):
            chunk = os.read(descriptor, HASH_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise BackupContractError(f"backup file exceeds bound: {relative}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            total != observed.st_size
            or os.read(descriptor, 1)
            or after.st_size != observed.st_size
            or after.st_mtime_ns != observed.st_mtime_ns
            or after.st_ctime_ns != observed.st_ctime_ns
        ):
            raise BackupContractError(f"backup file changed while reading: {relative}")
        return b"".join(chunks), observed
    except OSError as exc:
        raise BackupContractError(f"cannot read backup file: {relative}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_exclusive(path: Path, data: bytes, maximum: int) -> None:
    if not 0 <= len(data) <= maximum:
        raise BackupContractError("backup metadata exceeds bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise BackupContractError(f"cannot write backup metadata: {path.name}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _database_inventory(
    database: Path, *, content_root: Path
) -> tuple[list[dict[str, str]], dict[str, int | str]]:
    try:
        report = Store.verify_backup(database, content_root=content_root)
        uri = f"file:{quote(str(database.resolve()))}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT t.raw_path AS path,t.raw_sha256 AS sha256 FROM acceptances a "
            "JOIN takes t ON t.take_id=a.take_id WHERE a.take_id IS NOT NULL UNION ALL "
            "SELECT d.path AS path,d.sha256 AS sha256 FROM derivatives d "
            "JOIN acceptances a ON a.take_id=d.take_id WHERE a.take_id IS NOT NULL "
            "ORDER BY path LIMIT ?",
            (MAX_BACKUP_FILES + 1,),
        ).fetchall()
        connection.close()
    except (sqlite3.Error, StoreContractError, OSError) as exc:
        raise BackupContractError("cannot inventory corpus backup database") from exc
    if len(rows) > MAX_BACKUP_FILES:
        raise BackupContractError("backup content file count exceeds bound")
    by_path: dict[str, str] = {}
    for row in rows:
        relative = _safe_relative(row["path"])
        if not relative.startswith("data/") or relative == "data/corpus.sqlite3":
            raise BackupContractError("backup content path is outside data storage")
        digest = row["sha256"]
        if not isinstance(digest, str) or _SHA_RE.fullmatch(digest) is None:
            raise BackupContractError("backup database contains an invalid checksum")
        if relative in by_path and by_path[relative] != digest:
            raise BackupContractError("backup database reuses a path with different content")
        by_path[relative] = digest
    return ([{"path": path, "sha256": by_path[path]} for path in sorted(by_path)], report)


def _bound_json(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    for _index in range(MAX_JSON_NODES):
        if not stack:
            return
        current, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise BackupContractError("backup manifest depth exceeds bound")
        if isinstance(current, dict):
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
    raise BackupContractError("backup manifest node count exceeds bound")


def _read_manifest(root: Path) -> dict[str, Any]:
    raw, metadata = _read_regular(root, "manifest.json", MAX_MANIFEST_BYTES)
    if metadata.st_size > MAX_MANIFEST_BYTES:
        raise BackupContractError("backup manifest exceeds bound")
    try:
        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate key")
                result[key] = value
            return result

        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BackupContractError("invalid backup manifest JSON") from exc
    if not isinstance(value, dict):
        raise BackupContractError("backup manifest must be an object")
    _bound_json(value)
    return value


def _bounded_entries(directory: Path) -> list[os.DirEntry[str]]:
    entries: list[os.DirEntry[str]] = []
    try:
        with os.scandir(directory) as iterator:
            for _index in range(MAX_BACKUP_FILES + 17):
                try:
                    entry = next(iterator)
                except StopIteration:
                    break
                entries.append(entry)
                if len(entries) > MAX_BACKUP_FILES + 16:
                    raise BackupContractError("backup directory entry count exceeds bound")
    except BackupContractError:
        raise
    except OSError as exc:
        raise BackupContractError("cannot enumerate backup bundle") from exc
    return sorted(entries, key=lambda item: item.name)


def _walk_files(root: Path) -> list[str]:
    directories = [root]
    files: list[str] = []
    for _index in range(MAX_DIRECTORIES):
        if not directories:
            break
        directory = directories.pop()
        for entry in _bounded_entries(directory):
            relative = Path(entry.path).relative_to(root).as_posix()
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise BackupContractError(f"backup bundle contains symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                directories.append(Path(entry.path))
            elif stat.S_ISREG(metadata.st_mode):
                files.append(_safe_relative(relative))
                if len(files) > MAX_BACKUP_FILES + 4:
                    raise BackupContractError("backup file count exceeds bound")
            else:
                raise BackupContractError(f"backup bundle contains special file: {relative}")
    if directories:
        raise BackupContractError("backup directory count exceeds bound")
    return sorted(files)


def _validate_manifest(
    value: dict[str, Any], root: Path, *, expected_backup_id: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(value) != {"schema", "backup_id", "created_at", "database", "files", "counts"}:
        raise BackupContractError("backup manifest fields are invalid")
    if value.get("schema") != SCHEMA or _BACKUP_ID_RE.fullmatch(value.get("backup_id", "")) is None:
        raise BackupContractError("backup identity is invalid")
    created_at = value.get("created_at")
    if not isinstance(created_at, str) or len(created_at) != 20:
        raise BackupContractError("backup created_at is invalid")
    try:
        datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise BackupContractError("backup created_at is invalid") from exc
    expected_backup_id = root.name if expected_backup_id is None else expected_backup_id
    if value["backup_id"] != expected_backup_id:
        raise BackupContractError("backup directory and manifest identity differ")
    database = value.get("database")
    if not isinstance(database, dict) or set(database) != {"path", "sha256", "bytes"}:
        raise BackupContractError("backup database record is invalid")
    if database.get("path") != "database/corpus.sqlite3":
        raise BackupContractError("backup database path is invalid")
    files = value.get("files")
    if not isinstance(files, list) or len(files) > MAX_BACKUP_FILES:
        raise BackupContractError("backup content inventory is invalid")
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
            raise BackupContractError("backup content record is invalid")
        relative = _safe_relative(item.get("path"))
        if not relative.startswith("data/") or relative in seen:
            raise BackupContractError("backup content path is invalid or duplicated")
        seen.add(relative)
    counts = value.get("counts")
    if not isinstance(counts, dict) or set(counts) != {"takes", "files", "stored_files"}:
        raise BackupContractError("backup counts are invalid")
    count_bounds = {
        "takes": MAX_EXPORT_ROWS,
        "files": MAX_BACKUP_FILES,
        "stored_files": MAX_BACKUP_FILES,
    }
    for key, maximum in count_bounds.items():
        count = counts.get(key)
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or not 0 <= count <= maximum
        ):
            raise BackupContractError("backup counts are invalid")
    return files, database


def _validate_file_record(root: Path, item: dict[str, Any], maximum: int) -> tuple[str, int]:
    digest = item.get("sha256")
    size = item.get("bytes")
    if _SHA_RE.fullmatch(digest if isinstance(digest, str) else "") is None:
        raise BackupContractError("backup checksum is invalid")
    if not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= maximum:
        raise BackupContractError("backup size is invalid")
    observed_digest, observed_size = _hash_regular(root, item["path"], maximum)
    if observed_size != size:
        raise BackupContractError(f"backup size mismatch: {item['path']}")
    if observed_digest != digest:
        raise BackupContractError(f"backup checksum mismatch: {item['path']}")
    return observed_digest, observed_size


def verify_backup_bundle(
    bundle: Path, *, expected_backup_id: str | None = None
) -> dict[str, Any]:
    root = _directory(Path(bundle), "backup bundle")
    value = _read_manifest(root)
    files, database = _validate_manifest(
        value, root, expected_backup_id=expected_backup_id
    )
    expected = {"manifest.json", "checksums.sha256", "READY", database["path"]}
    expected.update(item["path"] for item in files)
    if set(_walk_files(root)) != expected:
        raise BackupContractError("backup inventory differs from manifest")
    checksums: dict[str, str] = {}
    for item in [database, *files]:
        digest, _size = _validate_file_record(
            root,
            item,
            MAX_DATABASE_BYTES if item["path"] == database["path"] else MAX_CONTENT_BYTES,
        )
        checksums[item["path"]] = digest
    manifest_sha, manifest_bytes = _hash_regular(root, "manifest.json", MAX_MANIFEST_BYTES)
    checksums["manifest.json"] = manifest_sha
    expected_checksum_data = "".join(
        f"{digest}  {path}\n" for path, digest in sorted(checksums.items())
    ).encode("ascii")
    checksum_data, checksum_metadata = _read_regular(
        root, "checksums.sha256", MAX_CHECKSUM_BYTES
    )
    if checksum_data != expected_checksum_data:
        raise BackupContractError("backup checksum ledger is invalid")
    ready_data, ready_metadata = _read_regular(root, "READY", 0)
    if ready_metadata.st_size != 0 or ready_data != b"":
        raise BackupContractError("backup READY marker is invalid")
    if ready_metadata.st_mtime_ns < max((root / item).stat().st_mtime_ns for item in expected - {"READY"}):
        raise BackupContractError("backup READY marker was not written last")
    try:
        database_report = Store.verify_backup(root / database["path"], content_root=root)
    except StoreContractError as exc:
        raise BackupContractError("backup database/content verification failed") from exc
    counts = value["counts"]
    if (
        counts["takes"] != database_report["takes"]
        or counts["files"] != database_report["files"]
        or counts["stored_files"] != len(files)
    ):
        raise BackupContractError("backup count receipt differs from database")
    total_bytes = manifest_bytes + checksum_metadata.st_size + sum(item["bytes"] for item in [database, *files])
    if total_bytes > MAX_TOTAL_BYTES:
        raise BackupContractError("backup total size exceeds bound")
    return {
        "schema": SCHEMA,
        "backup_id": value["backup_id"],
        "integrity": "ok",
        "takes": database_report["takes"],
        "files": database_report["files"],
        "stored_files": len(files),
        "total_bytes": total_bytes,
        "manifest_sha256": manifest_sha,
    }


def _remove_staging(path: Path, parent: Path) -> None:
    if path.parent == parent and path.name.startswith(".") and path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)


def _copy_backup_content(
    source_root: Path, staging: Path
) -> tuple[list[dict[str, Any]], dict[str, int | str], str, int]:
    database = staging / "database/corpus.sqlite3"
    Store(source_root).backup(database)
    inventory, database_report = _database_inventory(
        database, content_root=source_root
    )
    copied: list[dict[str, Any]] = []
    total_bytes = 0
    for item in inventory:
        size = _copy_verified(
            source_root, item["path"], staging, item["sha256"], MAX_CONTENT_BYTES
        )
        total_bytes += size
        if total_bytes > MAX_TOTAL_BYTES:
            raise BackupContractError("backup total size exceeds bound")
        copied.append({**item, "bytes": size})
    if len(copied) != len(inventory):
        raise BackupContractError("backup content inventory was not copied completely")
    database_sha, database_bytes = _hash_regular(
        staging, "database/corpus.sqlite3", MAX_DATABASE_BYTES
    )
    return copied, database_report, database_sha, database_bytes


def _write_bundle_metadata(
    staging: Path,
    backup_id: str,
    copied: list[dict[str, Any]],
    database_report: dict[str, int | str],
    database_sha: str,
    database_bytes: int,
) -> None:
    if _BACKUP_ID_RE.fullmatch(backup_id) is None or _SHA_RE.fullmatch(database_sha) is None:
        raise BackupContractError("backup metadata identity/checksum is invalid")
    if not 1 <= database_bytes <= MAX_DATABASE_BYTES or len(copied) > MAX_BACKUP_FILES:
        raise BackupContractError("backup metadata counts are outside bounds")
    manifest = {
        "schema": SCHEMA,
        "backup_id": backup_id,
        "created_at": _now(),
        "database": {
            "path": "database/corpus.sqlite3",
            "sha256": database_sha,
            "bytes": database_bytes,
        },
        "files": copied,
        "counts": {
            "takes": database_report["takes"],
            "files": database_report["files"],
            "stored_files": len(copied),
        },
    }
    manifest_data = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    _write_exclusive(staging / "manifest.json", manifest_data, MAX_MANIFEST_BYTES)
    checksums = {
        "database/corpus.sqlite3": database_sha,
        "manifest.json": hashlib.sha256(manifest_data).hexdigest(),
        **{item["path"]: item["sha256"] for item in copied},
    }
    checksum_data = "".join(
        f"{digest}  {path}\n" for path, digest in sorted(checksums.items())
    ).encode("ascii")
    _write_exclusive(staging / "checksums.sha256", checksum_data, MAX_CHECKSUM_BYTES)
    _write_exclusive(staging / "READY", b"", 0)


def _publish_staging(staging: Path, final: Path, destination: Path) -> None:
    try:
        os.rename(staging, final)
    except OSError as exc:
        raise BackupContractError("cannot publish append-only backup") from exc
    directory_fd = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def create_backup(root: Path, destination: Path, *, backup_id: str | None = None) -> Path:
    source_root = _directory(Path(root), "AVAAS root")
    if backup_id is None:
        backup_id = datetime.now(timezone.utc).strftime("backup-%Y%m%dT%H%M%SZ")
    if not isinstance(backup_id, str) or _BACKUP_ID_RE.fullmatch(backup_id) is None:
        raise BackupContractError("invalid backup id")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = _directory(destination, "backup destination")
    final = destination / backup_id
    if os.path.lexists(final):
        raise BackupContractError("backup already exists")
    staging = Path(tempfile.mkdtemp(prefix=f".{backup_id}.staging-", dir=destination))
    try:
        copied, report, database_sha, database_bytes = _copy_backup_content(
            source_root, staging
        )
        _write_bundle_metadata(
            staging, backup_id, copied, report, database_sha, database_bytes
        )
        verify_backup_bundle(staging, expected_backup_id=backup_id)
        _publish_staging(staging, final, destination)
        verify_backup_bundle(final)
        return final
    finally:
        if staging.exists():
            _remove_staging(staging, destination)


def main(argv: Sequence[str] | None = None) -> int:
    if argv is not None and isinstance(argv, (str, bytes)):
        raise TypeError("argv must be a sequence of arguments")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="AVAAS application root")
    parser.add_argument("destination", type=Path, help="append-only backup directory")
    parser.add_argument("--backup-id")
    parser.add_argument("--verify", type=Path, help="verify an existing bundle instead")
    args = parser.parse_args(argv)
    if args.verify is not None:
        result = verify_backup_bundle(args.verify)
    else:
        result = verify_backup_bundle(
            create_backup(args.root, args.destination, backup_id=args.backup_id)
        )
    if not isinstance(result, dict):
        raise AssertionError("backup operation returned an invalid receipt")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

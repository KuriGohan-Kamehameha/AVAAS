"""Bounded filesystem and canonical serialization helpers for private bundles."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any


MAX_BUNDLE_FILES = 20_000
MAX_BUNDLE_FILE_BYTES = 512 * 1024 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024
HASH_CHUNK_BYTES = 64 * 1024
MAX_HASH_CHUNKS = MAX_BUNDLE_FILE_BYTES // HASH_CHUNK_BYTES
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """A bounded immutable-bundle contract failed closed."""


def canonical_json(value: Any) -> bytes:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError("value is not canonical JSON") from exc
    if len(raw) > MAX_JSON_BYTES:
        raise ContractError("canonical JSON exceeds bound")
    return raw


def safe_relative(value: str, field: str = "path") -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1_024
        or "\\" in value
        or "\x00" in value
    ):
        raise ContractError(f"invalid {field}")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ContractError(f"invalid {field}")
    return candidate.as_posix()


def digest(value: str, field: str = "sha256") -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ContractError(f"invalid {field}")
    return value


def _root(root: Path) -> Path:
    root = Path(root)
    try:
        metadata = os.lstat(root)
    except OSError as exc:
        raise ContractError("bundle root is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ContractError("bundle root must be a non-symlink directory")
    return root


def regular_path(root: Path, relative: str, *, maximum: int = MAX_BUNDLE_FILE_BYTES) -> Path:
    root = _root(root)
    relative = safe_relative(relative)
    cursor = root
    parts = PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        cursor = cursor / part
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise ContractError(f"missing bundle file: {relative}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ContractError(f"bundle path is a symlink: {relative}")
        if index < len(parts) - 1:
            if not stat.S_ISDIR(metadata.st_mode):
                raise ContractError(f"bundle parent is not a directory: {relative}")
        elif (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size < 0
            or metadata.st_size > maximum
        ):
            raise ContractError(f"bundle file size outside bounds: {relative}")
    return cursor


def read_regular(root: Path, relative: str, *, maximum: int) -> bytes:
    path = regular_path(root, relative, maximum=maximum)
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
            raise ContractError(f"bundle file changed: {relative}")
        chunks = []
        total = 0
        for _index in range((maximum // HASH_CHUNK_BYTES) + 2):
            chunk = os.read(descriptor, min(HASH_CHUNK_BYTES, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise ContractError(f"bundle file exceeds bound: {relative}")
        data = b"".join(chunks)
        if len(data) != metadata.st_size:
            raise ContractError(f"bundle file changed: {relative}")
        return data
    except ContractError:
        raise
    except OSError as exc:
        raise ContractError(f"cannot read bundle file: {relative}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def read_json(root: Path, relative: str, *, maximum: int = MAX_JSON_BYTES) -> Any:
    try:
        return json.loads(read_regular(root, relative, maximum=maximum))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid JSON file: {relative}") from exc


def sha256_file(root: Path, relative: str, *, maximum: int = MAX_BUNDLE_FILE_BYTES) -> str:
    path = regular_path(root, relative, maximum=maximum)
    descriptor = -1
    sha256 = hashlib.sha256()
    total = 0
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        metadata = os.fstat(descriptor)
        for _index in range(MAX_HASH_CHUNKS + 1):
            chunk = os.read(descriptor, HASH_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise ContractError(f"bundle file exceeds bound: {relative}")
            sha256.update(chunk)
        if os.read(descriptor, 1):
            raise ContractError(f"bundle file exceeds hash bound: {relative}")
        if total != metadata.st_size:
            raise ContractError(f"bundle file changed: {relative}")
        return sha256.hexdigest()
    except ContractError:
        raise
    except OSError as exc:
        raise ContractError(f"cannot hash bundle file: {relative}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def inventory(root: Path) -> list[str]:
    root = _root(root)
    files: list[str] = []
    directories = [root]
    for _index in range(MAX_BUNDLE_FILES):
        if not directories:
            break
        directory = directories.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise ContractError("cannot enumerate bundle") from exc
        for entry in entries:
            relative = entry.path.removeprefix(str(root) + os.sep).replace(os.sep, "/")
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ContractError(f"cannot inspect bundle path: {relative}") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise ContractError(f"bundle contains symlink: {relative}")
            if stat.S_ISDIR(metadata.st_mode):
                directories.append(Path(entry.path))
            elif stat.S_ISREG(metadata.st_mode):
                files.append(safe_relative(relative))
                if len(files) > MAX_BUNDLE_FILES:
                    raise ContractError("bundle file count exceeds bound")
            else:
                raise ContractError(f"bundle contains special file: {relative}")
    if directories:
        raise ContractError("bundle directory count exceeds bound")
    return sorted(files)


def write_durable(path: Path, data: bytes, *, mode: int = 0o640) -> None:
    if not isinstance(data, bytes) or len(data) > MAX_BUNDLE_FILE_BYTES:
        raise ContractError("bundle output exceeds bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ContractError(f"bundle output already exists: {path.name}")
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
        offset = 0
        for _index in range((len(data) // HASH_CHUNK_BYTES) + 2):
            if offset == len(data):
                break
            written = os.write(descriptor, data[offset : offset + HASH_CHUNK_BYTES])
            if written <= 0:
                raise ContractError("short bundle write")
            offset += written
        if offset != len(data):
            raise ContractError("short bundle write")
        os.fsync(descriptor)
    except ContractError:
        raise
    except OSError as exc:
        raise ContractError(f"cannot write bundle file: {path.name}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def fsync_directory(path: Path) -> None:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError as exc:
        raise ContractError(f"cannot fsync bundle directory: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def parse_checksums(root: Path) -> dict[str, str]:
    raw = read_regular(root, "checksums.sha256", maximum=4 * 1024 * 1024)
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ContractError("checksums are not ASCII") from exc
    if not lines or len(lines) > MAX_BUNDLE_FILES:
        raise ContractError("checksum line count outside bounds")
    result: dict[str, str] = {}
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise ContractError("invalid checksum line")
        checksum = digest(line[:64], "checksum")
        relative = safe_relative(line[66:], "checksum path")
        if relative in result or relative in {"checksums.sha256", "READY"}:
            raise ContractError("duplicate or recursive checksum path")
        result[relative] = checksum
    return result

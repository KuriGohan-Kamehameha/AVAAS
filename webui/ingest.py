#!/usr/bin/env python3
"""Bounded streaming upload staging and immutable content publication."""
from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterable


MAX_UPLOAD_BYTES = 64 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 64 * 1024
MAX_UPLOAD_CHUNKS = MAX_UPLOAD_BYTES // UPLOAD_CHUNK_BYTES + 1
UPLOAD_TIMEOUT_SECONDS = 45.0
_LABEL_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+){0,3}$")
_EXTENSION_RE = re.compile(r"^\.[a-z0-9]{1,8}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class IngestError(ValueError):
    """An upload exceeded a bound or an immutable path contract."""


@dataclass(frozen=True)
class StagedUpload:
    path: Path
    sha256: str
    size: int

    def cleanup(self) -> None:
        self.path.unlink(missing_ok=True)


@dataclass(frozen=True)
class PublishedContent:
    relative_path: str
    sha256: str
    size: int


def _safe_directory(root: Path, relative: tuple[str, ...]) -> Path:
    root = Path(root).resolve()
    cursor = root
    for part in relative:
        cursor /= part
        try:
            metadata = os.lstat(cursor)
        except FileNotFoundError:
            try:
                os.mkdir(cursor, 0o750)
                metadata = os.lstat(cursor)
            except OSError as exc:
                raise IngestError("cannot create ingest directory") from exc
        except OSError as exc:
            raise IngestError("cannot inspect ingest directory") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise IngestError("ingest directory must not traverse a symlink")
    return cursor


def stage_stream(
    root: Path,
    chunks: Iterable[bytes],
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
    timeout_seconds: float = UPLOAD_TIMEOUT_SECONDS,
    clock: Callable[[], float] = time.monotonic,
) -> StagedUpload:
    """Stage a bounded iterable without trusting a caller-provided filename."""
    if (
        not isinstance(max_bytes, int)
        or isinstance(max_bytes, bool)
        or not 1 <= max_bytes <= MAX_UPLOAD_BYTES
        or not 0.0 < timeout_seconds <= UPLOAD_TIMEOUT_SECONDS
    ):
        raise IngestError("upload limits are invalid")
    staging = _safe_directory(root, ("data", "staging"))
    descriptor, name = tempfile.mkstemp(prefix="upload-", suffix=".part", dir=staging)
    path = Path(name)
    digest = hashlib.sha256()
    total = 0
    start = clock()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            for index, chunk in enumerate(chunks, start=1):
                if index > MAX_UPLOAD_CHUNKS:
                    raise IngestError("upload chunk count exceeds bound")
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise IngestError("upload chunk must be bytes")
                if not 1 <= len(chunk) <= UPLOAD_CHUNK_BYTES:
                    raise IngestError("upload chunk size outside bound")
                total += len(chunk)
                if total > max_bytes:
                    raise IngestError("upload exceeds byte limit")
                if clock() - start > timeout_seconds:
                    raise IngestError("upload deadline exceeded")
                handle.write(chunk)
                digest.update(chunk)
            if total == 0:
                raise IngestError("empty upload")
            handle.flush()
            os.fsync(handle.fileno())
        return StagedUpload(path=path, sha256=digest.hexdigest(), size=total)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def stage_file(
    root: Path,
    file_object: BinaryIO,
    *,
    max_bytes: int = MAX_UPLOAD_BYTES,
    timeout_seconds: float = UPLOAD_TIMEOUT_SECONDS,
) -> StagedUpload:
    def chunks() -> Iterable[bytes]:
        for _ in range(MAX_UPLOAD_CHUNKS):
            chunk = file_object.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                return
            yield chunk
        raise IngestError("upload chunk count exceeds bound")

    return stage_stream(
        root,
        chunks(),
        max_bytes=max_bytes,
        timeout_seconds=timeout_seconds,
    )


def _verify_staged(root: Path, staged: StagedUpload) -> None:
    if (
        not isinstance(staged, StagedUpload)
        or not _DIGEST_RE.fullmatch(staged.sha256)
        or not 1 <= staged.size <= MAX_UPLOAD_BYTES
    ):
        raise IngestError("invalid staged upload receipt")
    staging = _safe_directory(root, ("data", "staging"))
    try:
        metadata = os.lstat(staged.path)
    except OSError as exc:
        raise IngestError("staged upload is missing") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise IngestError("staged upload must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode) or staged.path.parent.resolve() != staging:
        raise IngestError("upload is outside the staging directory")
    if metadata.st_size != staged.size:
        raise IngestError("staged upload size changed")
    digest, read_bytes = _hash_bounded(staged.path)
    if read_bytes != staged.size or digest != staged.sha256:
        raise IngestError("staged upload checksum changed")


def _hash_bounded(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    read_bytes = 0
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= MAX_UPLOAD_BYTES:
            raise IngestError("staged content size outside bounds")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            for _ in range(MAX_UPLOAD_CHUNKS):
                chunk = handle.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                read_bytes += len(chunk)
                digest.update(chunk)
            if handle.read(1):
                raise IngestError("staged verification loop bound exceeded")
    except IngestError:
        raise
    except OSError as exc:
        raise IngestError("cannot hash staged content") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest(), read_bytes


def receipt_for_staged_path(root: Path, path: Path) -> StagedUpload:
    """Create a bounded receipt for a DSP output already in the staging directory."""
    staging = _safe_directory(root, ("data", "staging"))
    path = Path(path)
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise IngestError("staged output is missing") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise IngestError("staged output must not be a symlink")
    if (
        not stat.S_ISREG(metadata.st_mode)
        or path.parent.resolve() != staging
        or not 1 <= metadata.st_size <= MAX_UPLOAD_BYTES
    ):
        raise IngestError("staged output contract mismatch")
    digest, size = _hash_bounded(path)
    return StagedUpload(path=path, sha256=digest, size=size)


def publish_content(
    root: Path,
    staged: StagedUpload,
    *,
    label: str,
    extension: str,
) -> PublishedContent:
    """Atomically move one verified staged file to a content-addressed path."""
    if not isinstance(label, str) or not _LABEL_RE.fullmatch(label):
        raise IngestError("invalid content label")
    if not isinstance(extension, str) or not _EXTENSION_RE.fullmatch(extension):
        raise IngestError("invalid content extension")
    _verify_staged(root, staged)
    leaf = f"{staged.sha256}.{label}{extension}"
    destination_dir = _safe_directory(
        root,
        ("data", "audio", "sha256", staged.sha256[:2]),
    )
    destination = destination_dir / leaf
    relative = destination.relative_to(Path(root).resolve()).as_posix()
    try:
        existing = os.lstat(destination)
    except FileNotFoundError:
        existing = None
    except OSError as exc:
        raise IngestError("cannot inspect immutable destination") from exc
    if existing is not None:
        if stat.S_ISLNK(existing.st_mode) or not stat.S_ISREG(existing.st_mode):
            raise IngestError("immutable destination is not a regular file")
        duplicate = StagedUpload(destination, staged.sha256, staged.size)
        # Verify the existing file using the same bounded receipt logic without
        # pretending it lives in staging.
        if existing.st_size != staged.size:
            raise IngestError("immutable destination collision")
        digest, size = _hash_bounded(destination)
        if size != staged.size or digest != staged.sha256:
            raise IngestError("immutable destination collision")
        staged.cleanup()
        return PublishedContent(relative, duplicate.sha256, duplicate.size)
    try:
        os.chmod(staged.path, 0o440)
        os.replace(staged.path, destination)
        with destination.open("rb") as handle:
            os.fsync(handle.fileno())
        directory = os.open(destination_dir, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise IngestError("cannot publish immutable content") from exc
    return PublishedContent(relative, staged.sha256, staged.size)

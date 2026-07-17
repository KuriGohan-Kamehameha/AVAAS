#!/usr/bin/env python3
"""Transactional, immutable AVAAS corpus state.

One SQLite connection is opened per operation.  Writes use ``BEGIN IMMEDIATE``
and roll back on ``BaseException``.  P10: migrations, JSON, legacy input,
content files, rows, paths, hashes, and backup size all have fixed ceilings.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator
from urllib.parse import quote


MAX_MIGRATIONS = 32
MAX_MIGRATION_BYTES = 256 * 1024
MAX_JSON_BYTES = 64 * 1024
MAX_CONTENT_BYTES = 512 * 1024 * 1024
MAX_DATABASE_BYTES = 4 * 1024 * 1024 * 1024
MAX_LEGACY_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_LEGACY_LINES = 10_000
MAX_LEGACY_LINE_BYTES = 64 * 1024
MAX_PROMPTS = 2_000
MAX_DERIVATIVES_PER_TAKE = 8
MAX_QC_FLAGS = 64
MAX_EXPORT_ROWS = 10_000
MAX_EXPORT_BYTES = 64 * 1024 * 1024
MAX_CAPTURE_SESSIONS = 100_000
HASH_CHUNK_BYTES = 64 * 1024
MAX_HASH_CHUNKS = MAX_CONTENT_BYTES // HASH_CHUNK_BYTES
BUSY_TIMEOUT_MS = 5_000

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_MIGRATION_RE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")


class StoreContractError(ValueError):
    """Persistent input or on-disk state violated a fail-closed contract."""


class StoreConflict(StoreContractError):
    """A compare-and-swap generation was stale."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _digest(value: str, field: str = "sha256") -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise StoreContractError(f"invalid {field}")
    return value


def _identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise StoreContractError(f"invalid {field}")
    return value


def _bounded_text(value: Any, field: str, maximum: int = 4_096, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value):
        raise StoreContractError(f"invalid {field}")
    return value


def _json(value: Any, field: str) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise StoreContractError(f"invalid {field} JSON") from exc
    if len(encoded.encode("utf-8")) > MAX_JSON_BYTES:
        raise StoreContractError(f"{field} JSON exceeds bound")
    return encoded


def _read_bounded(path: Path, maximum: int) -> bytes:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= maximum:
            raise StoreContractError(f"file size outside bounds: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            data = handle.read(maximum + 1)
        if len(data) > maximum or len(data) != metadata.st_size:
            raise StoreContractError(f"file changed or size outside bounds: {path}")
        return data
    except StoreContractError:
        raise
    except OSError as exc:
        raise StoreContractError(f"cannot read regular file: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_atomic(path: Path, data: bytes, maximum: int) -> None:
    if not 0 <= len(data) <= maximum:
        raise StoreContractError(f"atomic output exceeds bound: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise StoreContractError(f"output path is a symlink: {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


class Store:
    """One corpus database rooted beside immutable content-addressed files."""

    def __init__(self, root: Path, database_path: Path | None = None) -> None:
        self.root = Path(root).resolve()
        self.database_path = (
            Path(database_path).resolve()
            if database_path is not None
            else self.root / "data" / "corpus.sqlite3"
        )
        self._migrations = Path(__file__).resolve().parent / "migrations"

    def _ensure_directory(self, directory: Path) -> None:
        """Create a directory below the content root without traversing symlinks."""
        try:
            parts = Path(directory).relative_to(self.root).parts
        except ValueError as exc:
            raise StoreContractError("writable directory escapes content root") from exc
        cursor = self.root
        for part in parts:
            candidate = cursor / part
            try:
                metadata = os.lstat(candidate)
            except FileNotFoundError:
                try:
                    os.mkdir(candidate, 0o750)
                    metadata = os.lstat(candidate)
                except OSError as exc:
                    raise StoreContractError(f"cannot create writable directory: {candidate}") from exc
            except OSError as exc:
                raise StoreContractError(f"cannot inspect writable directory: {candidate}") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise StoreContractError(f"writable directory is a symlink: {candidate}")
            if not stat.S_ISDIR(metadata.st_mode):
                raise StoreContractError(f"writable path is not a directory: {candidate}")
            cursor = candidate

    def _connect(self, *, read_only: bool = False) -> sqlite3.Connection:
        if not read_only:
            self._ensure_directory(self.database_path.parent)
            if self.database_path.is_symlink():
                raise StoreContractError("database path must not be a symlink")
            connection = sqlite3.connect(
                self.database_path,
                timeout=BUSY_TIMEOUT_MS / 1_000,
                isolation_level=None,
            )
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
        else:
            uri = f"file:{quote(str(self.database_path))}?mode=ro"
            connection = sqlite3.connect(
                uri,
                uri=True,
                timeout=BUSY_TIMEOUT_MS / 1_000,
                isolation_level=None,
            )
            connection.execute("PRAGMA query_only=ON")
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def migrate(self) -> None:
        migration_paths = sorted(self._migrations.glob("*.sql"))
        if not 1 <= len(migration_paths) <= MAX_MIGRATIONS:
            raise StoreContractError("migration count outside bounds")
        parsed: list[tuple[int, str, str, str]] = []
        previous = 0
        for path in migration_paths:
            match = _MIGRATION_RE.fullmatch(path.name)
            if path.is_symlink() or match is None:
                raise StoreContractError(f"invalid migration file: {path.name}")
            version = int(match.group(1))
            if version != previous + 1:
                raise StoreContractError("migrations must be contiguous and ordered")
            raw = _read_bounded(path, MAX_MIGRATION_BYTES)
            try:
                sql = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise StoreContractError(f"migration is not UTF-8: {path.name}") from exc
            parsed.append((version, match.group(2), hashlib.sha256(raw).hexdigest(), sql))
            previous = version

        connection = self._connect()
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, "
                "sha256 TEXT NOT NULL, applied_at TEXT NOT NULL)"
            )
            existing = {
                row["version"]: (row["name"], row["sha256"])
                for row in connection.execute(
                    "SELECT version,name,sha256 FROM schema_migrations ORDER BY version"
                )
            }
            expected_versions = {item[0] for item in parsed}
            if set(existing) - expected_versions:
                raise StoreContractError("database contains an unknown applied migration")
            for version, name, digest, sql in parsed:
                if version in existing:
                    if existing[version] != (name, digest):
                        raise StoreContractError(f"applied migration {version} changed")
                    continue
                values = tuple(item.replace("'", "''") for item in (name, digest, _now()))
                script = (
                    "BEGIN IMMEDIATE;\n"
                    + sql
                    + "\nINSERT INTO schema_migrations(version,name,sha256,applied_at) "
                    + f"VALUES ({version},'{values[0]}','{values[1]}','{values[2]}');\nCOMMIT;"
                )
                try:
                    connection.executescript(script)
                except sqlite3.DatabaseError as exc:
                    try:
                        connection.execute("ROLLBACK")
                    except sqlite3.DatabaseError:
                        pass
                    raise StoreContractError(f"migration {version} failed") from exc
        finally:
            connection.close()

    def diagnostics(self) -> dict[str, int | str]:
        connection = self._connect()
        try:
            return {
                "journal_mode": str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower(),
                "synchronous": int(connection.execute("PRAGMA synchronous").fetchone()[0]),
                "foreign_keys": int(connection.execute("PRAGMA foreign_keys").fetchone()[0]),
                "busy_timeout": int(connection.execute("PRAGMA busy_timeout").fetchone()[0]),
            }
        finally:
            connection.close()

    def migration_rows(self) -> list[dict[str, Any]]:
        connection = self._connect(read_only=True)
        try:
            return [dict(row) for row in connection.execute(
                "SELECT version,name,sha256,applied_at FROM schema_migrations ORDER BY version"
            )]
        finally:
            connection.close()

    def register_prompts(self, prompts: list[dict[str, Any]]) -> None:
        if not isinstance(prompts, list) or not 1 <= len(prompts) <= MAX_PROMPTS:
            raise StoreContractError("prompt registration count outside bounds")
        prepared: list[tuple[Any, ...]] = []
        for prompt in prompts:
            if not isinstance(prompt, dict):
                raise StoreContractError("prompt snapshot must be an object")
            fields = {
                "prompt_id": _identifier(prompt.get("id"), "prompt id"),
                "corpus_version": _bounded_text(prompt.get("corpus_version"), "corpus version", 64),
                "speaker_id": _identifier(prompt.get("speaker_id"), "speaker id"),
                "voice_model_id": _identifier(prompt.get("voice_model_id"), "voice model id"),
                "identity": _identifier(prompt.get("identity"), "identity"),
                "prompt_text": _bounded_text(prompt.get("text"), "prompt text", 2_000),
                "section": _identifier(prompt.get("section"), "section"),
                "kind": _identifier(prompt.get("kind"), "kind"),
                "source": _identifier(prompt.get("source"), "source"),
            }
            snapshot = hashlib.sha256(_json(fields, "prompt snapshot").encode("utf-8")).hexdigest()
            prepared.append(tuple(fields.values()) + (snapshot,))
        with self.transaction() as connection:
            for values in prepared:
                key = values[0:2]
                existing = connection.execute(
                    "SELECT prompt_id,corpus_version,speaker_id,voice_model_id,identity,"
                    "prompt_text,section,kind,source,snapshot_sha256 FROM prompts "
                    "WHERE prompt_id=? AND corpus_version=?",
                    key,
                ).fetchone()
                if existing is not None:
                    if tuple(existing) != values:
                        raise StoreContractError(f"prompt snapshot changed: {values[0]}")
                    continue
                connection.execute(
                    "INSERT INTO prompts(prompt_id,corpus_version,speaker_id,voice_model_id,"
                    "identity,prompt_text,section,kind,source,snapshot_sha256,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    values + (_now(),),
                )

    def _relative(self, value: str, field: str) -> tuple[str, Path]:
        if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 1_024:
            raise StoreContractError(f"invalid {field} path")
        if "\\" in value or "\x00" in value:
            raise StoreContractError(f"invalid {field} path")
        relative = PurePosixPath(value)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise StoreContractError(f"invalid {field} path")
        candidate = self.root.joinpath(*relative.parts)
        cursor = self.root
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise StoreContractError(f"{field} path contains a symlink")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise StoreContractError(f"{field} path is missing") from exc
        if self.root not in resolved.parents:
            raise StoreContractError(f"{field} path escapes content root")
        return relative.as_posix(), resolved

    def hash_content(self, relative: str, field: str = "content") -> tuple[str, str, int]:
        """Return canonical relative path, SHA-256, and size for bounded content."""
        canonical, path = self._relative(relative, field)
        descriptor = -1
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= MAX_CONTENT_BYTES:
                raise StoreContractError(f"{field} file size outside bounds")
            digest = hashlib.sha256()
            read_bytes = 0
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                for _ in range(MAX_HASH_CHUNKS):
                    chunk = handle.read(HASH_CHUNK_BYTES)
                    if not chunk:
                        break
                    read_bytes += len(chunk)
                    digest.update(chunk)
                if handle.read(1):
                    raise StoreContractError(f"{field} file exceeds hash bound")
            if read_bytes != metadata.st_size:
                raise StoreContractError(f"{field} file changed while hashing: {canonical}")
            return canonical, digest.hexdigest(), read_bytes
        except StoreContractError:
            raise
        except OSError as exc:
            raise StoreContractError(f"cannot verify {field} file") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _verify_content(self, relative: str, expected: str, field: str) -> int:
        expected = _digest(expected, f"{field} sha256")
        canonical, actual, size = self.hash_content(relative, field)
        if actual != expected:
            raise StoreContractError(f"{field} checksum mismatch: {canonical}")
        return size

    def _insert_take(
        self,
        connection: sqlite3.Connection,
        *,
        take_id: str,
        prompt_id: str,
        corpus_version: str,
        raw_path: str,
        raw_sha256: str,
        source: str,
        metadata_json: str,
    ) -> None:
        values = (take_id, prompt_id, corpus_version, source, raw_sha256, raw_path, metadata_json)
        existing = connection.execute(
            "SELECT take_id,prompt_id,corpus_version,source,raw_sha256,raw_path,metadata_json "
            "FROM takes WHERE take_id=?", (take_id,)
        ).fetchone()
        if existing is not None:
            if tuple(existing) != values:
                raise StoreContractError(f"take id collision: {take_id}")
            return
        connection.execute(
            "INSERT INTO takes(take_id,prompt_id,corpus_version,source,raw_sha256,raw_path,"
            "metadata_json,created_at) VALUES (?,?,?,?,?,?,?,?)",
            values + (_now(),),
        )

    def create_take(
        self,
        *,
        take_id: str,
        prompt_id: str,
        corpus_version: str,
        raw_path: str,
        raw_sha256: str,
        source: str,
        metadata: dict[str, Any],
    ) -> None:
        take_id = _identifier(take_id, "take id")
        prompt_id = _identifier(prompt_id, "prompt id")
        corpus_version = _bounded_text(corpus_version, "corpus version", 64)
        source = _identifier(source, "take source")
        raw_sha256 = _digest(raw_sha256, "raw sha256")
        raw_path, _ = self._relative(raw_path, "raw")
        self._verify_content(raw_path, raw_sha256, "raw")
        metadata_json = _json(metadata, "take metadata")
        with self.transaction() as connection:
            self._insert_take(
                connection,
                take_id=take_id,
                prompt_id=prompt_id,
                corpus_version=corpus_version,
                raw_path=raw_path,
                raw_sha256=raw_sha256,
                source=source,
                metadata_json=metadata_json,
            )

    def _insert_derivative(self, connection: sqlite3.Connection, values: tuple[Any, ...]) -> None:
        existing = connection.execute(
            "SELECT derivative_id,take_id,purpose,sha256,path,sample_rate,channels,"
            "sample_format,frames,duration_ms,parameters_json FROM derivatives "
            "WHERE derivative_id=?", (values[0],)
        ).fetchone()
        if existing is not None:
            if tuple(existing) != values:
                raise StoreContractError(f"derivative id collision: {values[0]}")
            return
        connection.execute(
            "INSERT INTO derivatives(derivative_id,take_id,purpose,sha256,path,sample_rate,"
            "channels,sample_format,frames,duration_ms,parameters_json,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            values + (_now(),),
        )

    def add_derivative(
        self,
        *,
        derivative_id: str,
        take_id: str,
        purpose: str,
        path: str,
        sha256: str,
        sample_rate: int,
        channels: int,
        sample_format: str,
        frames: int,
        duration_ms: int,
        parameters: dict[str, Any],
    ) -> None:
        path, _ = self._relative(path, "derivative")
        sha256 = _digest(sha256, "derivative sha256")
        self._verify_content(path, sha256, "derivative")
        if (
            not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or not 1 <= sample_rate <= 384_000
            or not isinstance(channels, int) or isinstance(channels, bool) or not 1 <= channels <= 8
            or not isinstance(frames, int) or isinstance(frames, bool) or not 1 <= frames <= 200_000_000
            or not isinstance(duration_ms, int) or isinstance(duration_ms, bool) or not 0 <= duration_ms <= 7_200_000
        ):
            raise StoreContractError("derivative audio metadata outside bounds")
        values = (
            _identifier(derivative_id, "derivative id"),
            _identifier(take_id, "take id"),
            _identifier(purpose, "derivative purpose"),
            sha256,
            path,
            sample_rate,
            channels,
            _identifier(sample_format, "sample format"),
            frames,
            duration_ms,
            _json(parameters, "derivative parameters"),
        )
        with self.transaction() as connection:
            self._insert_derivative(connection, values)

    def _insert_qc(self, connection: sqlite3.Connection, values: tuple[Any, ...]) -> None:
        existing = connection.execute(
            "SELECT qc_id,take_id,status,flags_json,metrics_json,transcript FROM qc_results "
            "WHERE qc_id=?", (values[0],)
        ).fetchone()
        if existing is not None:
            if tuple(existing) != values:
                raise StoreContractError(f"qc id collision: {values[0]}")
            return
        connection.execute(
            "INSERT INTO qc_results(qc_id,take_id,status,flags_json,metrics_json,transcript,created_at) "
            "VALUES (?,?,?,?,?,?,?)", values + (_now(),)
        )

    def add_qc(
        self,
        *,
        qc_id: str,
        take_id: str,
        status: str,
        flags: list[str],
        metrics: dict[str, Any],
        transcript: str | None,
    ) -> None:
        if transcript is not None:
            transcript = _bounded_text(transcript, "transcript", 16_000, empty=True)
        values = (
            _identifier(qc_id, "qc id"),
            _identifier(take_id, "take id"),
            _identifier(status, "qc status"),
            _json(flags, "qc flags"),
            _json(metrics, "qc metrics"),
            transcript,
        )
        with self.transaction() as connection:
            self._insert_qc(connection, values)

    def commit_capture(
        self,
        *,
        take_id: str,
        prompt_id: str,
        corpus_version: str,
        raw_path: str,
        raw_sha256: str,
        source: str,
        metadata: dict[str, Any],
        derivatives: list[dict[str, Any]],
        qc: dict[str, Any],
        expected_generation: int,
        override_reason: str | None = None,
    ) -> dict[str, Any]:
        """Atomically persist a take, derivatives, QC, and optional acceptance.

        Content is verified before the transaction. A non-passing take remains
        reviewable but cannot replace the accepted pointer without a stored
        override reason. P10: derivative and flag collections have fixed caps.
        """
        take_id = _identifier(take_id, "take id")
        prompt_id = _identifier(prompt_id, "prompt id")
        corpus_version = _bounded_text(corpus_version, "corpus version", 64)
        source = _identifier(source, "capture source")
        raw_sha256 = _digest(raw_sha256, "raw sha256")
        raw_path, _ = self._relative(raw_path, "raw")
        self._verify_content(raw_path, raw_sha256, "raw")
        if not isinstance(metadata, dict):
            raise StoreContractError("capture metadata must be an object")
        metadata_json = _json(metadata, "capture metadata")
        if (
            not isinstance(expected_generation, int)
            or isinstance(expected_generation, bool)
            or expected_generation < 0
        ):
            raise StoreContractError("invalid expected generation")
        if override_reason is not None:
            override_reason = _bounded_text(override_reason, "override reason", 2_000)

        if not isinstance(derivatives, list) or not 1 <= len(derivatives) <= MAX_DERIVATIVES_PER_TAKE:
            raise StoreContractError("derivative count outside bounds")
        derivative_keys = {
            "derivative_id",
            "purpose",
            "path",
            "sha256",
            "sample_rate",
            "channels",
            "sample_format",
            "frames",
            "duration_ms",
            "parameters",
        }
        prepared_derivatives: list[tuple[Any, ...]] = []
        seen_purposes: set[str] = set()
        for derivative in derivatives:
            if not isinstance(derivative, dict) or set(derivative) != derivative_keys:
                raise StoreContractError("derivative keys mismatch")
            purpose = _identifier(derivative.get("purpose"), "derivative purpose")
            if purpose in seen_purposes:
                raise StoreContractError("duplicate derivative purpose")
            seen_purposes.add(purpose)
            path, _ = self._relative(derivative.get("path"), "derivative")
            sha256 = _digest(derivative.get("sha256"), "derivative sha256")
            self._verify_content(path, sha256, "derivative")
            sample_rate = derivative.get("sample_rate")
            channels = derivative.get("channels")
            frames = derivative.get("frames")
            duration_ms = derivative.get("duration_ms")
            if (
                not isinstance(sample_rate, int)
                or isinstance(sample_rate, bool)
                or not 8_000 <= sample_rate <= 192_000
                or not isinstance(channels, int)
                or isinstance(channels, bool)
                or not 1 <= channels <= 8
                or not isinstance(frames, int)
                or isinstance(frames, bool)
                or not 1 <= frames <= 200_000_000
                or not isinstance(duration_ms, int)
                or isinstance(duration_ms, bool)
                or not 0 <= duration_ms <= 7_200_000
            ):
                raise StoreContractError("derivative audio metadata outside bounds")
            prepared_derivatives.append(
                (
                    _identifier(derivative.get("derivative_id"), "derivative id"),
                    take_id,
                    purpose,
                    sha256,
                    path,
                    sample_rate,
                    channels,
                    _identifier(derivative.get("sample_format"), "sample format"),
                    frames,
                    duration_ms,
                    _json(derivative.get("parameters"), "derivative parameters"),
                )
            )

        if not isinstance(qc, dict) or set(qc) != {
            "qc_id",
            "status",
            "flags",
            "metrics",
            "transcript",
        }:
            raise StoreContractError("qc keys mismatch")
        status = _identifier(qc.get("status"), "qc status")
        if status not in {"pass", "review_required", "unavailable"}:
            raise StoreContractError("unsupported qc status")
        flags = qc.get("flags")
        if (
            not isinstance(flags, list)
            or len(flags) > MAX_QC_FLAGS
            or any(
                not isinstance(flag, str) or not 1 <= len(flag) <= 128
                for flag in flags
            )
            or len(set(flags)) != len(flags)
            or (status == "pass") != (len(flags) == 0)
        ):
            raise StoreContractError("qc flags/status mismatch")
        if not isinstance(qc.get("metrics"), dict):
            raise StoreContractError("qc metrics must be an object")
        if status == "pass" and override_reason is not None:
            raise StoreContractError("passing qc must not carry an override reason")
        transcript = qc.get("transcript")
        if transcript is not None:
            transcript = _bounded_text(transcript, "transcript", 16_000, empty=True)
        qc_values = (
            _identifier(qc.get("qc_id"), "qc id"),
            take_id,
            status,
            _json(flags, "qc flags"),
            _json(qc.get("metrics"), "qc metrics"),
            transcript,
        )
        should_accept = status == "pass" or override_reason is not None

        with self.transaction() as connection:
            current = connection.execute(
                "SELECT take_id,generation FROM acceptances WHERE prompt_id=? AND corpus_version=?",
                (prompt_id, corpus_version),
            ).fetchone()
            generation = int(current["generation"]) if current is not None else 0
            if generation != expected_generation:
                raise StoreConflict(
                    f"acceptance generation is {generation}, expected {expected_generation}"
                )
            self._insert_take(
                connection,
                take_id=take_id,
                prompt_id=prompt_id,
                corpus_version=corpus_version,
                raw_path=raw_path,
                raw_sha256=raw_sha256,
                source=source,
                metadata_json=metadata_json,
            )
            for values in prepared_derivatives:
                self._insert_derivative(connection, values)
            self._insert_qc(connection, qc_values)
            if not should_accept:
                return {"accepted": False, "generation": generation, "take_id": take_id}
            if current is not None and current["take_id"] == take_id:
                return {"accepted": True, "generation": generation, "take_id": take_id}
            generation = self._accept_tx(
                connection,
                prompt_id=prompt_id,
                corpus_version=corpus_version,
                take_id=take_id,
                expected_generation=expected_generation,
                reason=override_reason,
            )
            return {"accepted": True, "generation": generation, "take_id": take_id}

    def _accept_tx(
        self,
        connection: sqlite3.Connection,
        *,
        prompt_id: str,
        corpus_version: str,
        take_id: str,
        expected_generation: int,
        reason: str | None,
    ) -> int:
        take = connection.execute(
            "SELECT prompt_id,corpus_version FROM takes WHERE take_id=?", (take_id,)
        ).fetchone()
        if take is None or tuple(take) != (prompt_id, corpus_version):
            raise StoreContractError("take does not belong to prompt snapshot")
        if connection.execute(
            "SELECT 1 FROM take_tombstones WHERE take_id=?", (take_id,)
        ).fetchone():
            raise StoreContractError("cannot accept a tombstoned take")
        current = connection.execute(
            "SELECT take_id,generation FROM acceptances WHERE prompt_id=? AND corpus_version=?",
            (prompt_id, corpus_version),
        ).fetchone()
        generation = int(current["generation"]) if current is not None else 0
        if expected_generation != generation:
            raise StoreConflict(f"acceptance generation is {generation}, expected {expected_generation}")
        new_generation = generation + 1
        action = "accept" if current is None or current["take_id"] is None else "repoint"
        timestamp = _now()
        connection.execute(
            "INSERT INTO acceptances(prompt_id,corpus_version,take_id,generation,accepted_at,override_reason) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(prompt_id,corpus_version) DO UPDATE SET "
            "take_id=excluded.take_id,generation=excluded.generation,accepted_at=excluded.accepted_at,"
            "override_reason=excluded.override_reason",
            (prompt_id, corpus_version, take_id, new_generation, timestamp, reason),
        )
        event_id = hashlib.sha256(
            f"{prompt_id}\0{corpus_version}\0{new_generation}\0{action}".encode()
        ).hexdigest()
        connection.execute(
            "INSERT INTO acceptance_history(event_id,prompt_id,corpus_version,take_id,action,"
            "reason,generation,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (event_id, prompt_id, corpus_version, take_id, action, reason, new_generation, timestamp),
        )
        return new_generation

    def accept(
        self,
        *,
        prompt_id: str,
        corpus_version: str,
        take_id: str,
        expected_generation: int,
        reason: str | None = None,
    ) -> int:
        prompt_id = _identifier(prompt_id, "prompt id")
        corpus_version = _bounded_text(corpus_version, "corpus version", 64)
        take_id = _identifier(take_id, "take id")
        if not isinstance(expected_generation, int) or isinstance(expected_generation, bool) or expected_generation < 0:
            raise StoreContractError("invalid expected generation")
        if reason is not None:
            reason = _bounded_text(reason, "acceptance reason", 2_000)
        with self.transaction() as connection:
            return self._accept_tx(
                connection,
                prompt_id=prompt_id,
                corpus_version=corpus_version,
                take_id=take_id,
                expected_generation=expected_generation,
                reason=reason,
            )

    def accepted(self, prompt_id: str, corpus_version: str) -> dict[str, Any] | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                "SELECT prompt_id,corpus_version,take_id,generation,accepted_at,override_reason "
                "FROM acceptances WHERE prompt_id=? AND corpus_version=? AND take_id IS NOT NULL",
                (prompt_id, corpus_version),
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            connection.close()

    def acceptance_state(self, prompt_id: str, corpus_version: str) -> dict[str, Any] | None:
        connection = self._connect(read_only=True)
        try:
            row = connection.execute(
                "SELECT prompt_id,corpus_version,take_id,generation,accepted_at,override_reason "
                "FROM acceptances WHERE prompt_id=? AND corpus_version=?",
                (prompt_id, corpus_version),
            ).fetchone()
            return dict(row) if row is not None else None
        finally:
            connection.close()

    def accepted_for_prompt(self, prompt_id: str) -> dict[str, Any] | None:
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                "SELECT prompt_id,corpus_version,take_id,generation,accepted_at,override_reason "
                "FROM acceptances WHERE prompt_id=? AND take_id IS NOT NULL ORDER BY corpus_version",
                (prompt_id,),
            ).fetchall()
            if len(rows) > 1:
                raise StoreContractError("prompt has multiple accepted corpus versions")
            return dict(rows[0]) if rows else None
        finally:
            connection.close()

    def acceptance_history(self, prompt_id: str, corpus_version: str) -> list[dict[str, Any]]:
        connection = self._connect(read_only=True)
        try:
            return [dict(row) for row in connection.execute(
                "SELECT event_id,prompt_id,corpus_version,take_id,action,reason,generation,created_at "
                "FROM acceptance_history WHERE prompt_id=? AND corpus_version=? ORDER BY generation",
                (prompt_id, corpus_version),
            )]
        finally:
            connection.close()

    def take_for_review(self, take_id: str) -> dict[str, Any] | None:
        """Return the bounded review projection for one immutable take."""

        take_id = _identifier(take_id, "take id")
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                "SELECT t.take_id,t.prompt_id,t.corpus_version,d.path,d.sha256,q.status,"
                "q.flags_json,CASE WHEN x.take_id IS NULL THEN 0 ELSE 1 END AS tombstoned "
                "FROM takes t "
                "JOIN derivatives d ON d.take_id=t.take_id AND d.purpose='serve-24k' "
                "JOIN qc_results q ON q.take_id=t.take_id "
                "LEFT JOIN take_tombstones x ON x.take_id=t.take_id WHERE t.take_id=?",
                (take_id,),
            ).fetchall()
        finally:
            connection.close()
        if len(rows) > 1:
            raise StoreContractError("take has multiple review derivatives")
        if not rows:
            return None
        row = dict(rows[0])
        try:
            flags = json.loads(row.pop("flags_json"))
        except (TypeError, json.JSONDecodeError) as exc:
            raise StoreContractError("stored review flags are invalid") from exc
        if not isinstance(flags, list) or len(flags) > MAX_QC_FLAGS:
            raise StoreContractError("stored review flags are outside bounds")
        row["flags"] = flags
        row["tombstoned"] = bool(row["tombstoned"])
        return row

    @staticmethod
    def _capture_claims(claims: dict[str, Any]) -> tuple[str, str, str, int, int, int]:
        if not isinstance(claims, dict) or set(claims) != {
            "nonce",
            "prompt_id",
            "corpus_version",
            "generation",
            "issued_at",
            "expires_at",
        }:
            raise StoreContractError("capture claims keys mismatch")
        nonce = _digest(claims.get("nonce"), "capture nonce")
        prompt_id = _identifier(claims.get("prompt_id"), "prompt id")
        corpus_version = _bounded_text(claims.get("corpus_version"), "corpus version", 64)
        integer_values = []
        for field in ("generation", "issued_at", "expires_at"):
            value = claims.get(field)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or value > 2**63 - 1
            ):
                raise StoreContractError(f"invalid capture {field}")
            integer_values.append(value)
        generation, issued_at, expires_at = integer_values
        if generation > 2**31 - 1 or not 1 <= expires_at - issued_at <= 600:
            raise StoreContractError("capture claims outside bounds")
        return nonce, prompt_id, corpus_version, generation, issued_at, expires_at

    def register_capture_session(
        self, claims: dict[str, Any], token_sha256: str
    ) -> dict[str, Any]:
        """Persist an immutable issuance receipt at the current CAS generation."""

        nonce, prompt_id, corpus_version, generation, issued_at, expires_at = (
            self._capture_claims(claims)
        )
        token_sha256 = _digest(token_sha256, "capture token sha256")
        with self.transaction() as connection:
            if connection.execute("SELECT COUNT(*) FROM capture_sessions").fetchone()[0] >= MAX_CAPTURE_SESSIONS:
                raise StoreContractError("capture session count outside bound")
            prompt = connection.execute(
                "SELECT 1 FROM prompts WHERE prompt_id=? AND corpus_version=?",
                (prompt_id, corpus_version),
            ).fetchone()
            if prompt is None:
                raise StoreContractError("unknown capture prompt snapshot")
            state = connection.execute(
                "SELECT generation FROM acceptances WHERE prompt_id=? AND corpus_version=?",
                (prompt_id, corpus_version),
            ).fetchone()
            current_generation = int(state["generation"]) if state is not None else 0
            if current_generation != generation:
                raise StoreConflict(
                    f"stale generation {generation}; current generation is {current_generation}"
                )
            timestamp = _now()
            try:
                connection.execute(
                    "INSERT INTO capture_sessions(nonce,token_sha256,prompt_id,corpus_version,"
                    "generation,issued_at,expires_at,state,consumed_at,created_at) "
                    "VALUES (?,?,?,?,?,?,?,'issued',NULL,?)",
                    (
                        nonce,
                        token_sha256,
                        prompt_id,
                        corpus_version,
                        generation,
                        issued_at,
                        expires_at,
                        timestamp,
                    ),
                )
                connection.execute(
                    "INSERT INTO capture_session_events(nonce,action,event_at) VALUES (?,'issued',?)",
                    (nonce, timestamp),
                )
            except sqlite3.IntegrityError as exc:
                raise StoreConflict("capture session already exists") from exc
            return dict(
                connection.execute(
                    "SELECT nonce,prompt_id,corpus_version,generation,issued_at,expires_at,state "
                    "FROM capture_sessions WHERE nonce=?",
                    (nonce,),
                ).fetchone()
            )

    def consume_capture_session(
        self,
        claims: dict[str, Any],
        token_sha256: str,
        *,
        now_epoch: int,
    ) -> dict[str, Any]:
        """Atomically consume one receipt if its prompt generation is still current."""

        nonce, prompt_id, corpus_version, generation, issued_at, expires_at = (
            self._capture_claims(claims)
        )
        token_sha256 = _digest(token_sha256, "capture token sha256")
        if (
            not isinstance(now_epoch, int)
            or isinstance(now_epoch, bool)
            or not 0 <= now_epoch <= 2**63 - 1
        ):
            raise StoreContractError("invalid capture consumption time")
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT nonce,token_sha256,prompt_id,corpus_version,generation,issued_at,"
                "expires_at,state FROM capture_sessions WHERE nonce=?",
                (nonce,),
            ).fetchone()
            if row is None:
                raise StoreConflict("unknown capture session")
            expected = (
                nonce,
                token_sha256,
                prompt_id,
                corpus_version,
                generation,
                issued_at,
                expires_at,
            )
            actual = tuple(
                row[key]
                for key in (
                    "nonce",
                    "token_sha256",
                    "prompt_id",
                    "corpus_version",
                    "generation",
                    "issued_at",
                    "expires_at",
                )
            )
            if actual != expected:
                raise StoreConflict("capture session binding mismatch")
            if row["state"] != "issued":
                raise StoreConflict("capture session already consumed")
            if now_epoch > expires_at:
                raise StoreConflict("capture session expired")
            state = connection.execute(
                "SELECT generation FROM acceptances WHERE prompt_id=? AND corpus_version=?",
                (prompt_id, corpus_version),
            ).fetchone()
            current_generation = int(state["generation"]) if state is not None else 0
            if current_generation != generation:
                raise StoreConflict(
                    f"stale generation {generation}; current generation is {current_generation}"
                )
            timestamp = _now()
            cursor = connection.execute(
                "UPDATE capture_sessions SET state='consumed',consumed_at=? "
                "WHERE nonce=? AND state='issued'",
                (now_epoch, nonce),
            )
            if cursor.rowcount != 1:
                raise StoreConflict("capture session already consumed")
            connection.execute(
                "INSERT INTO capture_session_events(nonce,action,event_at) VALUES (?,'consumed',?)",
                (nonce, timestamp),
            )
            return dict(
                connection.execute(
                    "SELECT nonce,prompt_id,corpus_version,generation,issued_at,expires_at,state,"
                    "consumed_at FROM capture_sessions WHERE nonce=?",
                    (nonce,),
                ).fetchone()
            )

    def tombstone(self, take_id: str, *, reason: str, expected_generation: int | None = None) -> int:
        take_id = _identifier(take_id, "take id")
        reason = _bounded_text(reason, "tombstone reason", 2_000)
        with self.transaction() as connection:
            take = connection.execute(
                "SELECT prompt_id,corpus_version FROM takes WHERE take_id=?", (take_id,)
            ).fetchone()
            if take is None:
                raise StoreContractError("unknown take")
            state = connection.execute(
                "SELECT take_id,generation FROM acceptances WHERE prompt_id=? AND corpus_version=?",
                tuple(take),
            ).fetchone()
            generation = int(state["generation"]) if state is not None else 0
            if expected_generation is not None and expected_generation != generation:
                raise StoreConflict(f"acceptance generation is {generation}, expected {expected_generation}")
            existing = connection.execute(
                "SELECT reason FROM take_tombstones WHERE take_id=?", (take_id,)
            ).fetchone()
            if existing is not None:
                if existing["reason"] != reason:
                    raise StoreContractError("take already tombstoned with a different reason")
                return generation
            timestamp = _now()
            connection.execute(
                "INSERT INTO take_tombstones(take_id,reason,created_at) VALUES (?,?,?)",
                (take_id, reason, timestamp),
            )
            if state is not None and state["take_id"] == take_id:
                generation += 1
                connection.execute(
                    "UPDATE acceptances SET take_id=NULL,generation=?,accepted_at=NULL,override_reason=NULL "
                    "WHERE prompt_id=? AND corpus_version=?",
                    (generation, take["prompt_id"], take["corpus_version"]),
                )
                event_id = hashlib.sha256(
                    f"{take['prompt_id']}\0{take['corpus_version']}\0{generation}\0tombstone".encode()
                ).hexdigest()
                connection.execute(
                    "INSERT INTO acceptance_history(event_id,prompt_id,corpus_version,take_id,action,"
                    "reason,generation,created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        event_id,
                        take["prompt_id"],
                        take["corpus_version"],
                        take_id,
                        "tombstone",
                        reason,
                        generation,
                        timestamp,
                    ),
                )
            return generation

    def load_accepted_records(self) -> list[dict[str, Any]]:
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                "SELECT p.prompt_id,p.corpus_version,p.prompt_text,p.section,p.kind,t.take_id,"
                "t.source,t.raw_path,t.metadata_json,t.created_at,a.generation,"
                "d.path AS processed_path,q.status AS qc_status,q.flags_json,q.metrics_json,"
                "q.transcript FROM acceptances a "
                "JOIN prompts p ON p.prompt_id=a.prompt_id AND p.corpus_version=a.corpus_version "
                "JOIN takes t ON t.take_id=a.take_id "
                "LEFT JOIN derivatives d ON d.derivative_id=(SELECT derivative_id FROM derivatives "
                "WHERE take_id=t.take_id ORDER BY CASE purpose WHEN 'serve-24k' THEN 0 ELSE 1 END,"
                "created_at,derivative_id LIMIT 1) "
                "LEFT JOIN qc_results q ON q.take_id=t.take_id "
                "WHERE a.take_id IS NOT NULL ORDER BY p.prompt_id,p.corpus_version"
            ).fetchall()
        finally:
            connection.close()
        if len(rows) > MAX_EXPORT_ROWS:
            raise StoreContractError("accepted record count outside bounds")
        records: list[dict[str, Any]] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else {}
            flags = json.loads(row["flags_json"]) if row["flags_json"] else []
            qc = {**metrics, "flags": flags, "status": row["qc_status"] or "unavailable"}
            records.append(
                {
                    **metadata,
                    "id": row["prompt_id"],
                    "corpus_version": row["corpus_version"],
                    "take_id": row["take_id"],
                    "generation": row["generation"],
                    "section": row["section"],
                    "kind": row["kind"],
                    "prompt_text": row["prompt_text"],
                    "raw_path": row["raw_path"],
                    "processed_path": row["processed_path"],
                    "transcript": row["transcript"] or metadata.get("transcript", ""),
                    "qc": qc,
                    "source": row["source"],
                    "ts": metadata.get("ts", row["created_at"]),
                }
            )
        return records

    def validate_accepted_checksums(self) -> dict[str, int]:
        connection = self._connect(read_only=True)
        try:
            rows = connection.execute(
                "SELECT t.take_id,t.raw_path,t.raw_sha256 FROM acceptances a "
                "JOIN takes t ON t.take_id=a.take_id WHERE a.take_id IS NOT NULL ORDER BY t.take_id"
            ).fetchall()
            derivatives = connection.execute(
                "SELECT d.take_id,d.path,d.sha256 FROM derivatives d "
                "JOIN acceptances a ON a.take_id=d.take_id WHERE a.take_id IS NOT NULL "
                "ORDER BY d.take_id,d.derivative_id"
            ).fetchall()
        finally:
            connection.close()
        if len(rows) > MAX_EXPORT_ROWS or len(derivatives) > MAX_EXPORT_ROWS * 8:
            raise StoreContractError("accepted checksum row count outside bounds")
        files = 0
        for row in rows:
            self._verify_content(row["raw_path"], row["raw_sha256"], "raw")
            files += 1
        for row in derivatives:
            self._verify_content(row["path"], row["sha256"], "derivative")
            files += 1
        return {"takes": len(rows), "files": files}

    def export_jsonl(self, destination: Path) -> None:
        lines = [
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
            for record in self.load_accepted_records()
        ]
        data = (("\n".join(lines) + "\n") if lines else "").encode("utf-8")
        _write_atomic(Path(destination), data, MAX_EXPORT_BYTES)

    def backup(self, destination: Path) -> None:
        if self.database_path.stat().st_size > MAX_DATABASE_BYTES:
            raise StoreContractError("database exceeds backup bound")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        os.close(descriptor)
        temporary = Path(name)
        temporary.unlink()
        source = self._connect(read_only=True)
        target: sqlite3.Connection | None = None
        try:
            pages = int(source.execute("PRAGMA page_count").fetchone()[0])
            page_size = int(source.execute("PRAGMA page_size").fetchone()[0])
            if pages * page_size > MAX_DATABASE_BYTES:
                raise StoreContractError("database page count exceeds backup bound")
            target = sqlite3.connect(temporary)
            source.backup(target, pages=256)
            target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            target.close()
            target = None
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary.replace(destination)
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            self.verify_backup(destination, content_root=self.root)
        finally:
            if target is not None:
                target.close()
            source.close()
            temporary.unlink(missing_ok=True)

    @classmethod
    def verify_backup(cls, backup: Path, *, content_root: Path) -> dict[str, int | str]:
        backup = Path(backup)
        if backup.is_symlink() or not backup.is_file() or backup.stat().st_size > MAX_DATABASE_BYTES:
            raise StoreContractError("backup file is invalid")
        store = cls(content_root, database_path=backup)
        connection = store._connect(read_only=True)
        try:
            integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
            foreign = connection.execute("PRAGMA foreign_key_check").fetchall()
        finally:
            connection.close()
        if integrity != "ok" or foreign:
            raise StoreContractError("backup database integrity failed")
        checks = store.validate_accepted_checksums()
        return {"integrity": "ok", **checks}

    def _legacy_report_path(self) -> Path:
        return self.root / "data" / "migration" / "legacy-manifest-report.json"

    def _save_legacy_report(self, report: dict[str, Any]) -> None:
        self._ensure_directory(self._legacy_report_path().parent)
        data = (
            json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8")
        _write_atomic(self._legacy_report_path(), data, MAX_JSON_BYTES)

    def import_legacy_manifest(self, manifest: Path) -> dict[str, Any]:
        manifest = Path(manifest)
        # Validate the deterministic report destination before any database write.
        self._ensure_directory(self._legacy_report_path().parent)
        data = _read_bounded(manifest, MAX_LEGACY_MANIFEST_BYTES)
        manifest_sha = hashlib.sha256(data).hexdigest()
        connection = self._connect(read_only=True)
        try:
            receipt = connection.execute(
                "SELECT report_json FROM legacy_imports WHERE manifest_sha256=?", (manifest_sha,)
            ).fetchone()
            take_count = int(connection.execute("SELECT COUNT(*) FROM takes").fetchone()[0])
        finally:
            connection.close()
        if receipt is not None:
            report = json.loads(receipt["report_json"])
            self._save_legacy_report(report)
            return report
        if take_count:
            raise StoreContractError("legacy import is allowed only for a new database")

        lines = data.splitlines()
        if not 1 <= len(lines) <= MAX_LEGACY_LINES:
            raise StoreContractError("legacy manifest line count outside bounds")
        prepared: list[dict[str, Any]] = []
        quarantine: list[dict[str, Any]] = []
        seen_prompts: set[tuple[str, str]] = set()
        prompt_connection = self._connect(read_only=True)
        try:
            for number, raw_line in enumerate(lines, start=1):
                reason: str | None = None
                if not 1 <= len(raw_line) <= MAX_LEGACY_LINE_BYTES:
                    reason = "line size outside bounds"
                record: dict[str, Any] | None = None
                if reason is None:
                    try:
                        candidate = json.loads(raw_line)
                        if not isinstance(candidate, dict):
                            raise ValueError("record is not an object")
                        record = candidate
                    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                        reason = "malformed JSON record"
                prompt: sqlite3.Row | None = None
                metadata_json: str | None = None
                raw_path = processed_path = raw_sha = processed_sha = None
                if reason is None and record is not None:
                    prompt_id = record.get("id")
                    try:
                        prompt_id = _identifier(prompt_id, "legacy prompt id")
                    except StoreContractError as exc:
                        reason = str(exc)
                    if reason is None:
                        matches = prompt_connection.execute(
                            "SELECT prompt_id,corpus_version,prompt_text,section,kind FROM prompts "
                            "WHERE prompt_id=? ORDER BY corpus_version", (prompt_id,)
                        ).fetchall()
                        if len(matches) != 1:
                            reason = "prompt id has no unique registered snapshot"
                        else:
                            prompt = matches[0]
                            if (
                                record.get("prompt_text") != prompt["prompt_text"]
                                or record.get("section") != prompt["section"]
                                or record.get("kind") != prompt["kind"]
                            ):
                                reason = "legacy prompt text or metadata mismatch"
                normalized: dict[str, Any] | None = None
                if reason is None and record is not None and prompt is not None:
                    qc = record.get("qc")
                    duration = qc.get("duration") if isinstance(qc, dict) else None
                    flags = qc.get("flags") if isinstance(qc, dict) else None
                    transcript = record.get("transcript", "")
                    try:
                        source = _identifier(record.get("source", "legacy"), "legacy source")
                    except StoreContractError:
                        source = ""
                    if (
                        not isinstance(qc, dict)
                        or not isinstance(duration, (int, float))
                        or isinstance(duration, bool)
                        or not math.isfinite(float(duration))
                        or not 0 <= float(duration) <= 7_200
                        or not isinstance(flags, list)
                        or len(flags) > 64
                        or any(not isinstance(flag, str) or not 1 <= len(flag) <= 128 for flag in flags)
                        or not isinstance(transcript, str)
                        or len(transcript) > 16_000
                        or not source
                    ):
                        reason = "invalid legacy qc/source/transcript metadata"
                    else:
                        normalized = {
                            "duration": float(duration),
                            "flags": flags,
                            "transcript": transcript,
                            "source": source,
                        }
                if reason is None and record is not None:
                    try:
                        metadata_json = _json(record, "legacy metadata")
                    except StoreContractError as exc:
                        reason = str(exc)
                if (
                    reason is None
                    and prompt is not None
                    and (prompt["prompt_id"], prompt["corpus_version"]) in seen_prompts
                ):
                    reason = "duplicate legacy prompt record"
                if reason is None and record is not None and prompt is not None:
                    try:
                        raw_path, raw_sha, _ = self.hash_content(
                            record.get("raw_path"), "legacy raw"
                        )
                        processed_path, processed_sha, _ = self.hash_content(
                            record.get("processed_path"), "legacy processed"
                        )
                    except StoreContractError as exc:
                        reason = str(exc)
                if reason is not None:
                    quarantine.append(
                        {"line": number, "line_sha256": hashlib.sha256(raw_line).hexdigest(), "reason": reason}
                    )
                    continue
                if (
                    record is None
                    or prompt is None
                    or normalized is None
                    or metadata_json is None
                    or not raw_path
                    or not processed_path
                    or not raw_sha
                    or not processed_sha
                ):
                    raise StoreContractError("legacy validation reached an invalid internal state")
                seen_prompts.add((prompt["prompt_id"], prompt["corpus_version"]))
                take_id = "legacy-" + hashlib.sha256(
                    f"{manifest_sha}\0{number}\0{raw_sha}\0{processed_sha}".encode()
                ).hexdigest()[:48]
                prepared.append(
                    {
                        "line": number,
                        "record": record,
                        "prompt": dict(prompt),
                        "take_id": take_id,
                        "raw_path": raw_path,
                        "raw_sha": raw_sha,
                        "processed_path": processed_path,
                        "processed_sha": processed_sha,
                        "normalized": normalized,
                        "metadata_json": metadata_json,
                    }
                )
        finally:
            prompt_connection.close()

        report = {
            "schema": "avaas/legacy-import-report@v1",
            "manifest_sha256": manifest_sha,
            "imported": len(prepared),
            "quarantined": len(quarantine),
            "quarantine": quarantine,
        }
        report_json = _json(report, "legacy import report")
        with self.transaction() as connection:
            for item in prepared:
                record = item["record"]
                normalized = item["normalized"]
                prompt = item["prompt"]
                take_id = item["take_id"]
                self._insert_take(
                    connection,
                    take_id=take_id,
                    prompt_id=prompt["prompt_id"],
                    corpus_version=prompt["corpus_version"],
                    raw_path=item["raw_path"],
                    raw_sha256=item["raw_sha"],
                    source=normalized["source"],
                    metadata_json=item["metadata_json"],
                )
                duration = normalized["duration"]
                frames = max(1, min(200_000_000, int(float(duration) * 24_000)))
                derivative_id = "legacy-derivative-" + hashlib.sha256(
                    f"{take_id}\0{item['processed_sha']}".encode()
                ).hexdigest()[:40]
                self._insert_derivative(
                    connection,
                    (
                        derivative_id,
                        take_id,
                        "serve-24k",
                        item["processed_sha"],
                        item["processed_path"],
                        24_000,
                        1,
                        "pcm_s16le",
                        frames,
                        max(0, int(float(duration) * 1_000)),
                        _json({"migration": "legacy-jsonl"}, "legacy derivative parameters"),
                    ),
                )
                flags = normalized["flags"]
                qc_id = "legacy-qc-" + hashlib.sha256(take_id.encode()).hexdigest()[:48]
                self._insert_qc(
                    connection,
                    (
                        qc_id,
                        take_id,
                        "pass" if not flags else "review_required",
                        _json(flags, "legacy qc flags"),
                        _json(record.get("qc", {}), "legacy qc metrics"),
                        normalized["transcript"],
                    ),
                )
                self._accept_tx(
                    connection,
                    prompt_id=prompt["prompt_id"],
                    corpus_version=prompt["corpus_version"],
                    take_id=take_id,
                    expected_generation=0,
                    reason="legacy JSONL migration",
                )
            connection.execute(
                "INSERT INTO legacy_imports(manifest_sha256,manifest_path,report_json,imported_count,"
                "quarantined_count,applied_at) VALUES (?,?,?,?,?,?)",
                (manifest_sha, str(manifest), report_json, len(prepared), len(quarantine), _now()),
            )
            for item in quarantine:
                connection.execute(
                    "INSERT INTO legacy_quarantine(manifest_sha256,line_number,line_sha256,reason) "
                    "VALUES (?,?,?,?)",
                    (manifest_sha, item["line"], item["line_sha256"], item["reason"]),
                )
        self._save_legacy_report(report)
        return report

"""Short-lived, one-use capture authorizations.

Tokens bind a browser recording to one immutable prompt snapshot and to the
current acceptance generation.  The SQLite receipt is the authority for
one-use semantics; the HMAC prevents a browser from relabelling a recording.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MAX_TOKEN_CHARS = 2_048
DEFAULT_TTL_SECONDS = 300
MAX_TTL_SECONDS = 600
CLOCK_SKEW_SECONDS = 5
KEY_BYTES = 32

_TOKEN_PART_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_NONCE_RE = re.compile(r"^[0-9a-f]{64}$")
_PROMPT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_PAYLOAD_KEYS = frozenset({"schema", "nonce", "prompt_id", "corpus_version", "generation", "issued_at", "expires_at"})


class CaptureTokenError(ValueError):
    """A token or signing-key contract failed closed."""


@dataclass(frozen=True, slots=True)
class CaptureClaims:
    nonce: str
    prompt_id: str
    corpus_version: str
    generation: int
    issued_at: int
    expires_at: int

    def as_dict(self) -> dict[str, str | int]:
        return asdict(self)


def _integer(value: Any, field: str, *, maximum: int = 2**63 - 1) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > maximum
    ):
        raise CaptureTokenError(f"invalid {field}")
    return value


def _claims(payload: dict[str, Any]) -> CaptureClaims:
    if set(payload) != _PAYLOAD_KEYS or payload.get("schema") != "avaas.capture.v1":
        raise CaptureTokenError("invalid token claims")
    nonce = payload.get("nonce")
    prompt_id = payload.get("prompt_id")
    corpus_version = payload.get("corpus_version")
    if not isinstance(nonce, str) or _NONCE_RE.fullmatch(nonce) is None:
        raise CaptureTokenError("invalid token nonce")
    if not isinstance(prompt_id, str) or _PROMPT_RE.fullmatch(prompt_id) is None:
        raise CaptureTokenError("invalid token prompt")
    if (
        not isinstance(corpus_version, str)
        or not 1 <= len(corpus_version) <= 64
        or any(ord(character) < 0x20 for character in corpus_version)
    ):
        raise CaptureTokenError("invalid token corpus version")
    generation = _integer(payload.get("generation"), "token generation", maximum=2**31 - 1)
    issued_at = _integer(payload.get("issued_at"), "token issuance")
    expires_at = _integer(payload.get("expires_at"), "token expiry")
    if not 1 <= expires_at - issued_at <= MAX_TTL_SECONDS:
        raise CaptureTokenError("invalid token lifetime")
    return CaptureClaims(
        nonce=nonce,
        prompt_id=prompt_id,
        corpus_version=corpus_version,
        generation=generation,
        issued_at=issued_at,
        expires_at=expires_at,
    )


def _encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _decode(value: str, field: str) -> bytes:
    if not value or _TOKEN_PART_RE.fullmatch(value) is None:
        raise CaptureTokenError(f"invalid token {field}")
    padding = "=" * (-len(value) % 4)
    try:
        return base64.b64decode(value + padding, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise CaptureTokenError(f"invalid token {field}") from exc


class CaptureTokenCodec:
    """Issue and verify bounded HMAC-SHA256 capture tokens."""

    def __init__(self, key: bytes) -> None:
        if not isinstance(key, bytes) or len(key) != KEY_BYTES:
            raise CaptureTokenError("capture token key must be exactly 32 bytes")
        self._key = key

    def issue(
        self,
        *,
        prompt_id: str,
        corpus_version: str,
        generation: int,
        now: int | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> str:
        issued_at = int(time.time()) if now is None else _integer(now, "token issuance")
        ttl = _integer(ttl_seconds, "token lifetime", maximum=MAX_TTL_SECONDS)
        if ttl < 1:
            raise CaptureTokenError("invalid token lifetime")
        claims = _claims(
            {
                "schema": "avaas.capture.v1",
                "nonce": secrets.token_hex(KEY_BYTES),
                "prompt_id": prompt_id,
                "corpus_version": corpus_version,
                "generation": generation,
                "issued_at": issued_at,
                "expires_at": issued_at + ttl,
            }
        )
        payload = json.dumps(
            {"schema": "avaas.capture.v1", **claims.as_dict()},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        signature = hmac.digest(self._key, payload, "sha256")
        token = f"{_encode(payload)}.{_encode(signature)}"
        if len(token) > MAX_TOKEN_CHARS:
            raise CaptureTokenError("token length exceeds bound")
        return token

    def verify(self, token: str, *, now: int | None = None) -> CaptureClaims:
        if not isinstance(token, str) or not 1 <= len(token) <= MAX_TOKEN_CHARS:
            raise CaptureTokenError("token length outside bound")
        if token.count(".") != 1:
            raise CaptureTokenError("invalid token shape")
        payload_part, signature_part = token.split(".")
        payload = _decode(payload_part, "payload")
        signature = _decode(signature_part, "signature")
        if len(signature) != hashlib.sha256().digest_size or not hmac.compare_digest(
            signature, hmac.digest(self._key, payload, "sha256")
        ):
            raise CaptureTokenError("invalid token signature")
        try:
            decoded = json.loads(payload.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CaptureTokenError("invalid token payload") from exc
        if not isinstance(decoded, dict):
            raise CaptureTokenError("invalid token claims")
        claims = _claims(decoded)
        current = int(time.time()) if now is None else _integer(now, "verification time")
        if claims.issued_at > current + CLOCK_SKEW_SECONDS:
            raise CaptureTokenError("token issued in the future")
        if current > claims.expires_at:
            raise CaptureTokenError("token expired")
        return claims


def token_sha256(token: str) -> str:
    if not isinstance(token, str) or not 1 <= len(token) <= MAX_TOKEN_CHARS:
        raise CaptureTokenError("token length outside bound")
    return hashlib.sha256(token.encode("ascii", errors="strict")).hexdigest()


def _read_key(path: Path) -> bytes:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise CaptureTokenError("cannot inspect capture token key") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise CaptureTokenError("capture token key must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise CaptureTokenError("capture token key must be a regular file")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise CaptureTokenError("capture token key permissions must be 0600")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise CaptureTokenError("capture token key has an unexpected owner")
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size != KEY_BYTES:
            raise CaptureTokenError("capture token key must be exactly 32 bytes")
        data = os.read(descriptor, KEY_BYTES + 1)
        if len(data) != KEY_BYTES:
            raise CaptureTokenError("capture token key must be exactly 32 bytes")
        return data
    except CaptureTokenError:
        raise
    except OSError as exc:
        raise CaptureTokenError("cannot read capture token key") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def load_or_create_key(root: Path) -> bytes:
    """Load the private key, or durably create it without following symlinks."""

    base = Path(root).resolve()
    data_directory = base / "data"
    try:
        data_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory_metadata = os.lstat(data_directory)
    except OSError as exc:
        raise CaptureTokenError("cannot create capture token key directory") from exc
    if stat.S_ISLNK(directory_metadata.st_mode) or not stat.S_ISDIR(directory_metadata.st_mode):
        raise CaptureTokenError("capture token key directory must not be a symlink")
    path = data_directory / "capture-token-v1.key"
    try:
        return _read_key(path)
    except CaptureTokenError:
        if path.exists() or path.is_symlink():
            raise

    key = secrets.token_bytes(KEY_BYTES)
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".capture-token-v1.",
            dir=data_directory,
        )
        temporary = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        written = os.write(descriptor, key)
        if written != KEY_BYTES:
            raise CaptureTokenError("short write while creating capture token key")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            return _read_key(path)
        temporary.unlink()
        temporary = None
        directory = os.open(data_directory, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return _read_key(path)
    except CaptureTokenError:
        raise
    except OSError as exc:
        raise CaptureTokenError("cannot create capture token key") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)

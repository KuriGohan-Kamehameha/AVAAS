from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from webui import capture_tokens
from webui.store import Store, StoreConflict


PROMPT = {
    "id": "sec13_001__satraj",
    "corpus_version": "2026.07.16",
    "speaker_id": "satraj",
    "voice_model_id": "satraj-piranesi",
    "identity": "satraj",
    "text": "This is Satraj.",
    "section": "sec13",
    "kind": "identity",
    "source": "avaas_local",
}


def test_signed_token_is_bounded_prompt_version_generation_and_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec = capture_tokens.CaptureTokenCodec(b"k" * 32)
    monkeypatch.setattr(capture_tokens.secrets, "token_hex", lambda _count: "a" * 64)
    token = codec.issue(
        prompt_id=PROMPT["id"],
        corpus_version=PROMPT["corpus_version"],
        generation=7,
        now=1_000,
        ttl_seconds=300,
    )
    assert 1 <= len(token) <= capture_tokens.MAX_TOKEN_CHARS
    claims = codec.verify(token, now=1_001)
    assert claims == capture_tokens.CaptureClaims(
        nonce="a" * 64,
        prompt_id=PROMPT["id"],
        corpus_version=PROMPT["corpus_version"],
        generation=7,
        issued_at=1_000,
        expires_at=1_300,
    )
    with pytest.raises(capture_tokens.CaptureTokenError, match="expired"):
        codec.verify(token, now=1_301)


def test_tamper_future_issuance_and_oversize_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    codec = capture_tokens.CaptureTokenCodec(b"s" * 32)
    monkeypatch.setattr(capture_tokens.secrets, "token_hex", lambda _count: "b" * 64)
    token = codec.issue(
        prompt_id=PROMPT["id"],
        corpus_version=PROMPT["corpus_version"],
        generation=0,
        now=2_000,
    )
    payload, signature = token.split(".")
    replacement = "A" if payload[0] != "A" else "B"
    with pytest.raises(capture_tokens.CaptureTokenError, match="signature"):
        codec.verify(replacement + payload[1:] + "." + signature, now=2_001)
    with pytest.raises(capture_tokens.CaptureTokenError, match="future"):
        codec.verify(token, now=1_000)
    with pytest.raises(capture_tokens.CaptureTokenError, match="length"):
        codec.verify("x" * (capture_tokens.MAX_TOKEN_CHARS + 1), now=2_001)


def test_key_is_private_persistent_and_symlink_safe(tmp_path: Path) -> None:
    first = capture_tokens.load_or_create_key(tmp_path)
    second = capture_tokens.load_or_create_key(tmp_path)
    assert first == second
    assert len(first) == 32
    key_path = tmp_path / "data" / "capture-token-v1.key"
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600

    other = tmp_path / "other-key"
    other.write_bytes(b"z" * 32)
    key_path.unlink()
    key_path.symlink_to(other)
    with pytest.raises(capture_tokens.CaptureTokenError, match="symlink"):
        capture_tokens.load_or_create_key(tmp_path)


def test_store_receipt_is_one_use_and_generation_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path)
    store.migrate()
    store.register_prompts([PROMPT])
    codec = capture_tokens.CaptureTokenCodec(b"r" * 32)
    nonces = iter(("c" * 64, "d" * 64))
    monkeypatch.setattr(capture_tokens.secrets, "token_hex", lambda _count: next(nonces))

    token = codec.issue(
        prompt_id=PROMPT["id"],
        corpus_version=PROMPT["corpus_version"],
        generation=0,
        now=3_000,
    )
    claims = codec.verify(token, now=3_001)
    store.register_capture_session(claims.as_dict(), capture_tokens.token_sha256(token))
    consumed = store.consume_capture_session(
        claims.as_dict(), capture_tokens.token_sha256(token), now_epoch=3_002
    )
    assert consumed["state"] == "consumed"
    with pytest.raises(StoreConflict, match="already consumed"):
        store.consume_capture_session(
            claims.as_dict(), capture_tokens.token_sha256(token), now_epoch=3_003
        )

    stale_token = codec.issue(
        prompt_id=PROMPT["id"],
        corpus_version=PROMPT["corpus_version"],
        generation=0,
        now=3_010,
    )
    stale = codec.verify(stale_token, now=3_011)
    store.register_capture_session(stale.as_dict(), capture_tokens.token_sha256(stale_token))

    raw = tmp_path / "data/raw/accepted.wav"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"accepted")
    digest = hashlib.sha256(b"accepted").hexdigest()
    store.create_take(
        take_id="take-accepted",
        prompt_id=PROMPT["id"],
        corpus_version=PROMPT["corpus_version"],
        raw_path="data/raw/accepted.wav",
        raw_sha256=digest,
        source="fixture",
        metadata={"fixture": True},
    )
    store.accept(
        prompt_id=PROMPT["id"],
        corpus_version=PROMPT["corpus_version"],
        take_id="take-accepted",
        expected_generation=0,
    )
    with pytest.raises(StoreConflict, match="stale generation"):
        store.consume_capture_session(
            stale.as_dict(), capture_tokens.token_sha256(stale_token), now_epoch=3_012
        )

    with sqlite3.connect(store.database_path) as connection:
        events = connection.execute(
            "SELECT action FROM capture_session_events WHERE nonce=? ORDER BY event_id",
            (claims.nonce,),
        ).fetchall()
    assert {row[0] for row in events} == {"issued", "consumed"}


def test_key_rejects_permissive_mode(tmp_path: Path) -> None:
    key_path = tmp_path / "data" / "capture-token-v1.key"
    key_path.parent.mkdir(parents=True)
    key_path.write_bytes(os.urandom(32))
    key_path.chmod(0o644)
    with pytest.raises(capture_tokens.CaptureTokenError, match="permissions"):
        capture_tokens.load_or_create_key(tmp_path)


def test_concurrent_first_key_loaders_converge_on_one_atomic_key(tmp_path: Path) -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        keys = list(pool.map(lambda _index: capture_tokens.load_or_create_key(tmp_path), range(32)))
    assert len(set(keys)) == 1
    assert not list((tmp_path / "data").glob(".capture-token-v1.*"))

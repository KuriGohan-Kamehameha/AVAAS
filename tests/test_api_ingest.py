from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from tests.audio_fixtures import speech_like
from webui import audio_contracts, corpus, prompts, server


REPO = Path(__file__).resolve().parents[1]


def _wav_bytes(audio: np.ndarray, sample_rate: int = 48_000) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _isolated_server(tmp_path: Path, monkeypatch) -> tuple[TestClient, dict]:
    prompt = next(
        record
        for record in prompts.compile_prompts(REPO)
        if record["id"] == "sec13_001__satraj"
    )
    sections = [
        {
            "section": "sec13",
            "num": 13,
            "title": "Identity",
            "target_min": 1.0,
            "prompts": [prompt],
        }
    ]
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "SETTINGS", tmp_path / "data" / "studio_settings.json")
    monkeypatch.setattr(server, "_SECTIONS", sections)
    monkeypatch.setattr(server, "_PROMPTS", {prompt["id"]: prompt})
    monkeypatch.setattr(server.trainer, "maybe_launch", lambda *_args, **_kwargs: {"state": "idle"})
    corpus.initialize(tmp_path, sections)
    return TestClient(server.app), prompt


def test_clip_api_publishes_exact_content_but_does_not_accept_unavailable_asr(
    tmp_path: Path, monkeypatch
) -> None:
    client, prompt = _isolated_server(tmp_path, monkeypatch)
    monkeypatch.setattr(
        server.qc_transcribe,
        "transcribe_result",
        lambda *_args, **_kwargs: {
            "status": "unavailable",
            "text": None,
            "model": "base.en",
            "reason": "model-unavailable",
        },
    )
    response = client.post(
        "/api/clip",
        data={"prompt_id": prompt["id"], "denoise": "false", "source": "browser-mic"},
        files={"file": ("clip.wav", _wav_bytes(speech_like()), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    record = response.json()["record"]
    assert record["persistence"]["accepted"] is False
    assert record["qc"]["status"] == "unavailable"
    assert "asr-unavailable" in record["qc"]["flags"]
    assert record["recording_accent"] == "natural"
    assert record["synthesis_presentation"] == "satraj.en-ca.neutral"
    assert corpus.load_manifest(tmp_path) == []

    master = tmp_path / record["raw_path"]
    audio_contracts.inspect_pcm16_wav(master, audio_contracts.AUDIO_SPECS["master-48k"])
    for derivative in record["derivatives"]:
        spec = audio_contracts.AUDIO_SPECS[derivative["purpose"]]
        observed = audio_contracts.inspect_pcm16_wav(tmp_path / derivative["path"], spec)
        assert observed["sha256"] == derivative["sha256"]
    with sqlite3.connect(tmp_path / "data" / "corpus.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM takes").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM acceptances").fetchone()[0] == 0
    assert list((tmp_path / "data" / "staging").iterdir()) == []


def test_multipart_gate_rejects_unframed_oversize_and_wrong_media_type(
    tmp_path: Path, monkeypatch
) -> None:
    client, _ = _isolated_server(tmp_path, monkeypatch)
    unframed = client.build_request(
        "POST",
        "/api/clip",
        headers={"content-type": "multipart/form-data; boundary=x"},
        content=iter((b"--x--\r\n",)),
    )
    assert client.send(unframed).status_code == 411

    oversize = client.post(
        "/api/clip",
        content=b"x",
        headers={
            "content-type": "multipart/form-data; boundary=x",
            "content-length": str(server.MAX_MULTIPART_BYTES + 1),
        },
    )
    assert oversize.status_code == 413
    assert client.post("/api/clip", content=b"x").status_code == 415


def test_clean_transcript_atomically_becomes_the_accepted_take(
    tmp_path: Path, monkeypatch
) -> None:
    client, prompt = _isolated_server(tmp_path, monkeypatch)
    monkeypatch.setattr(
        server.qc_transcribe,
        "transcribe_result",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "text": prompt["text"],
            "model": "base.en",
            "reason": None,
        },
    )
    audio = np.concatenate(
        (
            np.zeros(24_000, dtype=np.float32),
            speech_like(1.0),
            np.zeros(24_000, dtype=np.float32),
        )
    )
    response = client.post(
        "/api/clip",
        data={"prompt_id": prompt["id"], "denoise": "false", "source": "browser-mic"},
        files={"file": ("clip.wav", _wav_bytes(audio), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    record = response.json()["record"]
    assert record["qc"]["flags"] == []
    assert record["persistence"]["accepted"] is True
    accepted = corpus.load_manifest(tmp_path)
    assert len(accepted) == 1
    assert accepted[0]["id"] == prompt["id"]


def test_room_tone_is_bounded_content_addressed_and_referenced_atomically(
    tmp_path: Path, monkeypatch
) -> None:
    client, _ = _isolated_server(tmp_path, monkeypatch)
    quiet = np.zeros(2 * 48_000, dtype=np.float32)
    response = client.post(
        "/api/roomtone",
        files={"file": ("room.wav", _wav_bytes(quiet), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    assert response.json()["room_tone_sec"] == 2.0
    settings = server._read_settings()
    assert settings["room_tone_path"].startswith("data/audio/sha256/")
    assert server._room_tone() is not None
    assert server._room_tone().size == quiet.size
    assert list((tmp_path / "data" / "staging").iterdir()) == []

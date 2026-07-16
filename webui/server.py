#!/usr/bin/env python3
"""Voice corpus recording studio — web server.

Serves a single-page studio UI and the API behind it:

  GET  /                      the studio UI
  GET  /api/script            sections + prompts + per-prompt recorded/flag state
  GET  /api/progress          coverage, clean minutes, readiness, training state
  POST /api/clip              multipart {file, prompt_id, denoise?} -> standardize,
                              label, save, manifest-append, maybe auto-train
  GET  /api/clip/{id}/audio   serve the processed 24 kHz clip for review playback
  DELETE /api/clip/{id}       discard a clip (re-record)
  POST /api/roomtone          multipart {file} -> store the room-tone noise profile
  POST /api/train             manual training launch
  GET  /api/settings          read auto-train toggle
  POST /api/settings          set auto-train toggle

Run:  cd ~/voice && .venv/bin/python -m uvicorn webui.server:app --port 8731
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)

from . import corpus, processing, qc_transcribe, script_parser, trainer

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
SETTINGS = ROOT / "data" / "studio_settings.json"
MAX_UPLOAD_BYTES = 200 * 1024 * 1024     # 200 MB hard cap; P10 bounded input

app = FastAPI(title="AVAAS")

# Parsed script is cached; cheap to rebuild when the manifest changes.
_SECTIONS: list[dict] = []
_PROMPTS: dict[str, dict] = {}

# Server-Sent Events: push live state to every open studio tab. Background
# threads (the sync clip handler runs in FastAPI's threadpool) broadcast via
# _emit_threadsafe; the captured loop bridges thread -> asyncio.
_SSE_QUEUES: list[asyncio.Queue] = []
_EVENT_LOOP: asyncio.AbstractEventLoop | None = None
SSE_PING_SEC = 15.0


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


async def _emit(event: dict) -> None:
    payload = json.dumps(event)
    for q in list(_SSE_QUEUES):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass               # slow consumer drops a frame; next one resyncs


def _emit_threadsafe(event: dict) -> None:
    if _EVENT_LOOP is None:
        return
    asyncio.run_coroutine_threadsafe(_emit(event), _EVENT_LOOP)


def _state_event() -> dict:
    """The canonical 'something changed' payload — full progress + training."""
    prog = corpus.progress(ROOT, _SECTIONS)
    st = trainer.maybe_launch(ROOT, prog, _read_settings().get("auto_train", True), _now())
    return {"type": "state", "progress": prog, "training": st}


def _load_sections() -> None:
    global _SECTIONS, _PROMPTS
    _SECTIONS = script_parser.parse_script(ROOT)
    _PROMPTS = {p["id"]: p for s in _SECTIONS for p in s["prompts"]}


def _read_settings() -> dict:
    import json
    if SETTINGS.exists():
        try:
            return json.loads(SETTINGS.read_text())
        except json.JSONDecodeError:
            pass
    return {"auto_train": True, "denoise": True}


def _write_settings(data: dict) -> None:
    import json
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps(data, indent=2))


@app.on_event("startup")
def _startup() -> None:
    global _EVENT_LOOP
    _load_sections()
    if not _SECTIONS:
        raise RuntimeError("no structured prompt sections compiled")
    corpus.initialize(ROOT, _SECTIONS)
    _EVENT_LOOP = asyncio.get_event_loop()
    qc_transcribe.configure(_read_settings().get("whisper_model", "base.en"))


@app.get("/api/events")
async def api_events(request: Request) -> StreamingResponse:
    """SSE stream of state changes. Replaces client polling."""
    q: asyncio.Queue = asyncio.Queue(maxsize=64)
    _SSE_QUEUES.append(q)

    async def gen() -> AsyncIterator[str]:
        try:
            yield f"data: {json.dumps(_state_event())}\n\n"      # initial sync
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(q.get(), timeout=SSE_PING_SEC)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            if q in _SSE_QUEUES:
                _SSE_QUEUES.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (STATIC / "index.html").read_text()
    return HTMLResponse(html)


# --------------------------------------------------------------------------- #
# script + progress
# --------------------------------------------------------------------------- #
@app.get("/api/script")
def api_script() -> JSONResponse:
    records = {r["id"]: r for r in corpus.load_manifest(ROOT)}
    out = []
    for s in _SECTIONS:
        prompts = []
        for p in s["prompts"]:
            rec = records.get(p["id"])
            prompts.append({**p,
                            "recorded": rec is not None,
                            "flags": (rec or {}).get("qc", {}).get("flags", []),
                            "duration": (rec or {}).get("qc", {}).get("duration")})
        out.append({"section": s["section"], "title": s["title"],
                    "num": s["num"], "target_min": s["target_min"],
                    "prompts": prompts})
    return JSONResponse({"sections": out})


@app.get("/api/progress")
def api_progress() -> JSONResponse:
    prog = corpus.progress(ROOT, _SECTIONS)
    st = trainer.maybe_launch(ROOT, prog, _read_settings().get("auto_train", True), _now())
    return JSONResponse({"progress": prog, "training": st,
                         "qc_available": qc_transcribe.available()})


# --------------------------------------------------------------------------- #
# clip ingest  (sync def -> FastAPI threadpool; heavy DSP off the event loop)
# --------------------------------------------------------------------------- #
def _save_upload(upload: UploadFile) -> Path:
    data = upload.file.read()
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "upload too large")
    suffix = Path(upload.filename or "clip").suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return Path(tmp.name)


def _room_tone() -> np.ndarray | None:
    p = ROOT / "data" / "room_tone_48k.wav"
    if not p.exists():
        return None
    audio, _ = sf.read(str(p), dtype="float32")
    return audio if audio.ndim == 1 else audio.mean(axis=1)


@app.post("/api/clip")
def api_clip(prompt_id: str = Form(...),
             denoise: bool = Form(True),
             source: str = Form("browser-mic"),
             file: UploadFile = File(...)) -> JSONResponse:
    prompt = _PROMPTS.get(prompt_id)
    if prompt is None:
        raise HTTPException(404, f"unknown prompt_id {prompt_id}")

    tmp = _save_upload(file)
    capture_id = uuid.uuid4().hex
    raw_out = ROOT / "data" / "raw" / f"{prompt_id}_{capture_id}_{prompt['slug']}.wav"
    proc_out = ROOT / "data" / "processed" / f"{prompt_id}_{capture_id}_{prompt['slug']}.wav"
    try:
        qc = processing.standardize(tmp, raw_out, proc_out,
                                    room_tone=self_or_none(denoise, _room_tone()),
                                    denoise=denoise)
    except Exception as e:
        raise HTTPException(422, f"processing failed: {e}")
    finally:
        tmp.unlink(missing_ok=True)

    # Ground truth for scripted lines is the script itself; Whisper only QCs.
    hyp = qc_transcribe.transcribe(str(proc_out))
    wer_val = None
    if hyp is not None and prompt["kind"] not in ("spontaneous",):
        wer_val = qc_transcribe.wer(prompt["text"], hyp)
    transcript = prompt["text"] if prompt["kind"] != "spontaneous" else (hyp or "")
    qc["wer"] = wer_val
    qc["flags"] = corpus.qc_flags(qc, wer_val)

    record = {
        "id": prompt_id, "section": prompt["section"], "idx": prompt["idx"],
        "kind": prompt["kind"], "prompt_text": prompt["text"],
        "corpus_version": prompt["corpus_version"],
        "speaker_id": prompt["speaker_id"],
        "voice_model_id": prompt["voice_model_id"],
        "identity": prompt["identity"],
        "prompt_source": prompt["source"],
        "raw_path": str(raw_out.relative_to(ROOT)),
        "processed_path": str(proc_out.relative_to(ROOT)),
        "transcript": transcript, "asr_hypothesis": hyp,
        "qc": qc, "source": source, "ts": _now(),
    }
    corpus.append_record(ROOT, record)

    prog = corpus.progress(ROOT, _SECTIONS)
    st = trainer.maybe_launch(ROOT, prog, _read_settings().get("auto_train", True), _now())
    _emit_threadsafe({"type": "state", "progress": prog, "training": st,
                      "saved": record["id"], "flags": record["qc"]["flags"]})
    return JSONResponse({"record": record, "progress": prog, "training": st})


def self_or_none(denoise: bool, rt: np.ndarray | None) -> np.ndarray | None:
    return rt if denoise else None


@app.get("/api/clip/{prompt_id}/audio")
def api_clip_audio(prompt_id: str) -> FileResponse:
    rec = {r["id"]: r for r in corpus.load_manifest(ROOT)}.get(prompt_id)
    if rec is None:
        raise HTTPException(404, "no such clip")
    path = ROOT / rec["processed_path"]
    if not path.exists():
        raise HTTPException(404, "clip file missing")
    return FileResponse(str(path), media_type="audio/wav")


@app.delete("/api/clip/{prompt_id}")
def api_clip_delete(prompt_id: str) -> JSONResponse:
    records = corpus.load_manifest(ROOT)
    rec = {r["id"]: r for r in records}.get(prompt_id)
    if rec is None:
        raise HTTPException(404, "no such clip")
    corpus.tombstone_record(ROOT, prompt_id, reason="studio retake requested")
    _emit_threadsafe({"type": "state", "deleted": prompt_id,
                      "progress": corpus.progress(ROOT, _SECTIONS)})
    return JSONResponse({"deleted": prompt_id,
                         "progress": corpus.progress(ROOT, _SECTIONS)})

# --------------------------------------------------------------------------- #
# room tone + training + settings
# --------------------------------------------------------------------------- #
@app.post("/api/roomtone")
def api_roomtone(file: UploadFile = File(...)) -> JSONResponse:
    tmp = _save_upload(file)
    try:
        audio = processing.decode_to_48k_mono(tmp)
    except Exception as e:
        raise HTTPException(422, f"room-tone decode failed: {e}")
    finally:
        tmp.unlink(missing_ok=True)
    out = ROOT / "data" / "room_tone_48k.wav"
    sf.write(str(out), audio, processing.CAPTURE_SR, subtype="PCM_16")
    return JSONResponse({"room_tone_sec": round(audio.size / processing.CAPTURE_SR, 2),
                         "noise_floor_dbfs": processing.peak_dbfs(audio)})


@app.post("/api/train")
def api_train(force: bool = Form(False)) -> JSONResponse:
    prog = corpus.progress(ROOT, _SECTIONS)
    if not prog["ready"] and not force:
        raise HTTPException(409, {"msg": "corpus not at readiness gate", "progress": prog})
    st = trainer.launch(ROOT, _now(), force=bool(force))
    _emit_threadsafe({"type": "state", "progress": prog, "training": st})
    return JSONResponse({"training": st})


@app.get("/api/settings")
def api_get_settings() -> JSONResponse:
    return JSONResponse(_read_settings())


@app.post("/api/settings")
def api_set_settings(auto_train: bool = Form(...), denoise: bool = Form(True),
                     whisper_model: str = Form("base.en")) -> JSONResponse:
    if whisper_model not in qc_transcribe.ALLOWED_MODELS:
        raise HTTPException(400, f"whisper_model must be one of {qc_transcribe.ALLOWED_MODELS}")
    data = {"auto_train": bool(auto_train), "denoise": bool(denoise),
            "whisper_model": whisper_model}
    _write_settings(data)
    qc_transcribe.configure(whisper_model)
    return JSONResponse(data)

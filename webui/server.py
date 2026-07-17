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
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import AsyncIterator

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles

from . import (
    audio_contracts,
    capture_tokens,
    corpus,
    ingest,
    processing,
    qc_transcribe,
    readiness,
    script_parser,
    trainer,
    training_jobs,
)
from .store import Store, StoreConflict, StoreContractError

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
SETTINGS = ROOT / "data" / "studio_settings.json"
MAX_MULTIPART_BYTES = ingest.MAX_UPLOAD_BYTES + 1024 * 1024
MAX_ROOM_TONE_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_SETTINGS_BYTES = 64 * 1024
MAX_JSON_BODY_BYTES = 4 * 1024

@asynccontextmanager
async def _lifespan(_application: FastAPI):
    global _EVENT_LOOP
    _startup()
    try:
        yield
    finally:
        _EVENT_LOOP = None


app = FastAPI(title="AVAAS", lifespan=_lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.middleware("http")
async def require_bounded_write(request: Request, call_next):
    """Reject unframed or oversized writes before Starlette parses a body."""
    if request.method == "POST" and request.url.path in {"/api/clip", "/api/roomtone"}:
        content_type = request.headers.get("content-type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            return JSONResponse({"detail": "multipart/form-data required"}, status_code=415)
        raw_length = request.headers.get("content-length")
        if raw_length is None:
            return JSONResponse({"detail": "bounded Content-Length required"}, status_code=411)
        try:
            length = int(raw_length)
        except ValueError:
            return JSONResponse({"detail": "invalid Content-Length"}, status_code=400)
        maximum = (
            MAX_ROOM_TONE_UPLOAD_BYTES + 1024 * 1024
            if request.url.path == "/api/roomtone"
            else MAX_MULTIPART_BYTES
        )
        if not 1 <= length <= maximum:
            return JSONResponse({"detail": "request body too large"}, status_code=413)
    elif request.method == "POST" and (
        request.url.path == "/api/captures"
        or (
            request.url.path.startswith("/api/reviews/")
            and request.url.path.rsplit("/", 1)[-1] in {"accept", "reject"}
        )
    ):
        if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/json":
            return JSONResponse({"detail": "application/json required"}, status_code=415)
        raw_length = request.headers.get("content-length")
        if raw_length is None:
            return JSONResponse({"detail": "bounded Content-Length required"}, status_code=411)
        try:
            length = int(raw_length)
        except ValueError:
            return JSONResponse({"detail": "invalid Content-Length"}, status_code=400)
        if not 1 <= length <= MAX_JSON_BODY_BYTES:
            return JSONResponse({"detail": "request body too large"}, status_code=413)
    elif request.method == "POST" and request.url.path in {"/api/settings", "/api/train"}:
        content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
        if content_type not in {
            "application/x-www-form-urlencoded",
            "multipart/form-data",
        }:
            return JSONResponse({"detail": "form body required"}, status_code=415)
        raw_length = request.headers.get("content-length")
        if raw_length is None:
            return JSONResponse({"detail": "bounded Content-Length required"}, status_code=411)
        try:
            length = int(raw_length)
        except ValueError:
            return JSONResponse({"detail": "invalid Content-Length"}, status_code=400)
        if not 1 <= length <= MAX_SETTINGS_BYTES:
            return JSONResponse({"detail": "request body too large"}, status_code=413)
    return await call_next(request)

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


def _capture_codec() -> capture_tokens.CaptureTokenCodec:
    return capture_tokens.CaptureTokenCodec(capture_tokens.load_or_create_key(ROOT))


def _trainer_health() -> dict | None:
    return readiness.load_trainer_health(ROOT)


async def _json_object(request: Request) -> dict:
    raw = await request.body()
    if not 1 <= len(raw) <= MAX_JSON_BODY_BYTES:
        raise HTTPException(413, "request body too large")

    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(400, "invalid JSON object") from exc
    if not isinstance(value, dict):
        raise HTTPException(400, "JSON body must be an object")
    return value


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
    prog = readiness.evaluate(ROOT, _SECTIONS, trainer_health=_trainer_health())
    st = trainer.status(ROOT, prog)
    return {"type": "state", "progress": prog, "training": st}


def _load_sections() -> None:
    global _SECTIONS, _PROMPTS
    _SECTIONS = script_parser.parse_script(ROOT)
    _PROMPTS = {p["id"]: p for s in _SECTIONS for p in s["prompts"]}


def _read_settings() -> dict:
    defaults = {
        "auto_train": False,
        "denoise": True,
        "whisper_model": "base.en",
    }
    try:
        if SETTINGS.is_symlink() or not SETTINGS.is_file() or SETTINGS.stat().st_size > MAX_SETTINGS_BYTES:
            return defaults
        value = json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return defaults
    if not isinstance(value, dict) or set(value) - {
        "auto_train",
        "denoise",
        "whisper_model",
        "room_tone_path",
    }:
        return defaults
    if not isinstance(value.get("auto_train", False), bool) or not isinstance(
        value.get("denoise", True), bool
    ):
        return defaults
    if value.get("whisper_model", "base.en") not in qc_transcribe.ALLOWED_MODELS:
        return defaults
    room_tone_path = value.get("room_tone_path")
    if room_tone_path is not None and (
        not isinstance(room_tone_path, str) or not 1 <= len(room_tone_path) <= 1_024
    ):
        return defaults
    return {**defaults, **value}


def _write_settings(data: dict) -> None:
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    if SETTINGS.parent.is_symlink() or SETTINGS.is_symlink():
        raise StoreContractError("settings path must not be a symlink")
    encoded = (
        json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_SETTINGS_BYTES:
        raise StoreContractError("settings exceed bound")
    descriptor, name = tempfile.mkstemp(prefix=".studio-settings.", dir=SETTINGS.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(SETTINGS)
        directory = os.open(SETTINGS.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _startup() -> None:
    global _EVENT_LOOP
    _load_sections()
    if not _SECTIONS:
        raise RuntimeError("no structured prompt sections compiled")
    corpus.initialize(ROOT, _SECTIONS)
    _EVENT_LOOP = asyncio.get_running_loop()
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
    return HTMLResponse(
        html,
        headers={
            "Cache-Control": "no-store",
            "Content-Security-Policy": (
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
                "media-src 'self' blob:; img-src 'self' data:; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'"
            ),
            "Permissions-Policy": "microphone=(self)",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
        },
    )


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
    prog = readiness.evaluate(ROOT, _SECTIONS, trainer_health=_trainer_health())
    st = trainer.status(ROOT, prog)
    return JSONResponse({"progress": prog, "training": st,
                         "qc_available": qc_transcribe.available()})


@app.post("/api/captures", status_code=201)
async def api_capture_session(request: Request) -> JSONResponse:
    """Issue one short-lived capture capability for an immutable prompt."""

    body = await _json_object(request)
    if set(body) != {"prompt_id"} or not isinstance(body.get("prompt_id"), str):
        raise HTTPException(400, "capture request keys mismatch")
    prompt_id = body["prompt_id"]
    prompt = _PROMPTS.get(prompt_id)
    if prompt is None:
        raise HTTPException(404, "unknown prompt")
    store = Store(ROOT)
    state = store.acceptance_state(prompt_id, prompt["corpus_version"])
    generation = int(state["generation"]) if state is not None else 0
    codec = _capture_codec()
    token = codec.issue(
        prompt_id=prompt_id,
        corpus_version=prompt["corpus_version"],
        generation=generation,
    )
    claims = codec.verify(token)
    try:
        store.register_capture_session(claims.as_dict(), capture_tokens.token_sha256(token))
    except (StoreContractError, capture_tokens.CaptureTokenError) as exc:
        raise HTTPException(409, "capture session could not be issued") from exc
    return JSONResponse(
        {
            "capture_token": token,
            "prompt_id": claims.prompt_id,
            "corpus_version": claims.corpus_version,
            "generation": claims.generation,
            "issued_at": claims.issued_at,
            "expires_at": claims.expires_at,
        },
        status_code=201,
        headers={"Cache-Control": "no-store"},
    )


# --------------------------------------------------------------------------- #
# clip ingest  (sync def -> FastAPI threadpool; heavy DSP off the event loop)
# --------------------------------------------------------------------------- #
def _stage_upload(upload: UploadFile, *, maximum: int) -> ingest.StagedUpload:
    try:
        return ingest.stage_file(ROOT, upload.file, max_bytes=maximum)
    except ingest.IngestError as exc:
        message = str(exc)
        status = 408 if "deadline" in message else 413 if "exceed" in message else 400
        raise HTTPException(status, message) from exc


def _room_tone() -> np.ndarray | None:
    relative = _read_settings().get("room_tone_path")
    if relative is None:
        return None
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    path = ROOT.joinpath(*candidate.parts)
    try:
        resolved = path.resolve(strict=True)
        if ROOT.resolve() not in resolved.parents:
            return None
        audio_contracts.inspect_pcm16_wav(
            path,
            audio_contracts.AUDIO_SPECS["master-48k"],
            min_seconds=0.1,
            max_seconds=30.0,
        )
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if sample_rate != processing.CAPTURE_SR:
            return None
        return audio_contracts.validate_samples(
            audio,
            sample_rate,
            min_seconds=0.1,
            max_seconds=30.0,
            require_signal=False,
        )
    except (OSError, RuntimeError, audio_contracts.AudioContractError):
        return None


@app.post("/api/clip")
def api_clip(prompt_id: str = Form(...),
             capture_token: str = Form(...),
             denoise: bool = Form(True),
             source: str = Form("browser-mic"),
             file: UploadFile = File(...)) -> JSONResponse:
    current_epoch = int(time.time())
    try:
        claims = _capture_codec().verify(capture_token, now=current_epoch)
    except capture_tokens.CaptureTokenError as exc:
        raise HTTPException(401, "invalid or expired capture token") from exc
    if claims.prompt_id != prompt_id:
        raise HTTPException(409, "capture token prompt mismatch")
    prompt = _PROMPTS.get(prompt_id)
    if prompt is None:
        raise HTTPException(404, f"unknown prompt_id {prompt_id}")
    if claims.corpus_version != prompt["corpus_version"]:
        raise HTTPException(409, "capture token corpus version mismatch")
    if source not in {"browser-mic", "upload"}:
        raise HTTPException(400, "unsupported capture source")

    try:
        Store(ROOT).consume_capture_session(
            claims.as_dict(),
            capture_tokens.token_sha256(capture_token),
            now_epoch=current_epoch,
        )
    except StoreConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except (StoreContractError, capture_tokens.CaptureTokenError) as exc:
        raise HTTPException(400, "invalid capture receipt") from exc

    staged_input = _stage_upload(file, maximum=ingest.MAX_UPLOAD_BYTES)
    capture_id = claims.nonce
    staging = ROOT / "data" / "staging"
    output_paths = {
        purpose: staging / f"dsp-{capture_id}-{purpose}.wav"
        for purpose in audio_contracts.AUDIO_SPECS
    }
    try:
        processed = processing.standardize(
            staged_input.path,
            output_paths,
            room_tone=_room_tone() if denoise else None,
            denoise=denoise,
        )
        publications: dict[str, ingest.PublishedContent] = {}
        for purpose in audio_contracts.AUDIO_SPECS:
            receipt = ingest.receipt_for_staged_path(ROOT, output_paths[purpose])
            publications[purpose] = ingest.publish_content(
                ROOT,
                receipt,
                label=purpose,
                extension=".wav",
            )
    except processing.ProcessingError as exc:
        raise HTTPException(422, f"processing failed: {exc}") from exc
    except (ingest.IngestError, audio_contracts.AudioContractError) as exc:
        raise HTTPException(422, f"audio publication failed: {exc}") from exc
    finally:
        staged_input.cleanup()
        for path in output_paths.values():
            path.unlink(missing_ok=True)

    # Ground truth for scripted lines is the script itself; Whisper only QCs.
    serve_path = ROOT / publications["serve-24k"].relative_path
    asr = qc_transcribe.transcribe_result(str(serve_path), language="en")
    hyp = asr["text"]
    wer_val = None
    if asr["status"] == "ok" and hyp is not None and prompt["kind"] != "spontaneous":
        wer_val = qc_transcribe.wer(prompt["text"], hyp)
    transcript = prompt["text"] if prompt["kind"] != "spontaneous" else (hyp or "")
    qc = processed["qc"]
    qc.update(
        {
            "wer": wer_val,
            "asr_status": asr["status"],
            "asr_model": asr["model"],
            "asr_reason": asr["reason"],
        }
    )
    qc["flags"] = corpus.qc_flags(qc, wer_val, asr_status=asr["status"])
    qc["status"] = (
        "pass"
        if not qc["flags"]
        else "unavailable"
        if asr["status"] == "unavailable"
        else "review_required"
    )

    artifacts = {item["purpose"]: item for item in processed["artifacts"]}
    derivatives = []
    for purpose in ("serve-24k", "piper-22050", "wake-16k"):
        artifact = artifacts[purpose]
        publication = publications[purpose]
        derivatives.append(
            {
                "purpose": purpose,
                "path": publication.relative_path,
                "sha256": publication.sha256,
                "sample_rate": artifact["sample_rate"],
                "channels": artifact["channels"],
                "sample_format": artifact["sample_format"],
                "frames": artifact["frames"],
                "duration_ms": artifact["duration_ms"],
                "parameters": {
                    "schema": "avaas/audio-derivative@v1",
                    "master_sha256": publications["master-48k"].sha256,
                    "dsp": processed["dsp"],
                },
            }
        )

    record = {
        "id": prompt_id, "section": prompt["section"], "idx": prompt["idx"],
        "kind": prompt["kind"], "prompt_text": prompt["text"],
        "corpus_version": prompt["corpus_version"],
        "speaker_id": prompt["speaker_id"],
        "voice_model_id": prompt["voice_model_id"],
        "identity": prompt["identity"],
        "prompt_source": prompt["source"],
        "recording_profile_id": prompt["recording_profile_id"],
        "recording_language_tag": prompt["recording_language_tag"],
        "recording_accent": prompt["recording_accent"],
        "recording_delivery": prompt["recording_delivery"],
        "recording_safety_cue": prompt["recording_safety_cue"],
        "calibration_optional": prompt["calibration_optional"],
        "synthesis_presentation": prompt["synthesis_presentation"],
        "eligible_presentations": prompt["eligible_presentations"],
        "raw_path": publications["master-48k"].relative_path,
        "processed_path": publications["serve-24k"].relative_path,
        "derivatives": derivatives,
        "transcript": transcript,
        "asr_hypothesis": hyp,
        "asr": asr,
        "dsp": processed["dsp"],
        "qc": qc,
        "source": source,
        "ts": _now(),
    }
    try:
        persistence = corpus.append_record(
            ROOT,
            record,
            expected_generation=claims.generation,
        )
    except StoreContractError as exc:
        raise HTTPException(409, "capture could not be committed") from exc

    prog = readiness.evaluate(ROOT, _SECTIONS, trainer_health=_trainer_health())
    st = trainer.maybe_enqueue(
        ROOT,
        prog,
        auto=_read_settings().get("auto_train", False),
        now_iso=_now(),
    )
    event = {"type": "state", "progress": prog, "training": st,
             "flags": record["qc"]["flags"]}
    event["saved" if persistence["accepted"] else "review_required"] = record["id"]
    _emit_threadsafe(event)
    return JSONResponse({"record": {**record, "persistence": persistence,
                                     "review_audio_url": f"/api/take/{persistence['take_id']}/audio"},
                         "progress": prog, "training": st})


@app.get("/api/clip/{prompt_id}/audio")
def api_clip_audio(prompt_id: str) -> FileResponse:
    rec = {r["id"]: r for r in corpus.load_manifest(ROOT)}.get(prompt_id)
    if rec is None:
        raise HTTPException(404, "no such clip")
    path = ROOT / rec["processed_path"]
    if not path.exists():
        raise HTTPException(404, "clip file missing")
    return FileResponse(str(path), media_type="audio/wav")


@app.get("/api/take/{take_id}/audio")
def api_take_audio(take_id: str) -> FileResponse:
    store = Store(ROOT)
    try:
        take = store.take_for_review(take_id)
    except StoreContractError as exc:
        raise HTTPException(400, "invalid take id") from exc
    if take is None or take["tombstoned"]:
        raise HTTPException(404, "no reviewable take")
    try:
        path, digest, _size = store.hash_content(take["path"], "review audio")
    except StoreContractError as exc:
        raise HTTPException(409, "review audio failed integrity verification") from exc
    if digest != take["sha256"]:
        raise HTTPException(409, "review audio failed integrity verification")
    return FileResponse(str(ROOT / path), media_type="audio/wav")


@app.post("/api/reviews/{take_id}/accept")
async def api_accept_review(take_id: str, request: Request) -> JSONResponse:
    body = await _json_object(request)
    if set(body) != {"reason", "expected_generation"}:
        raise HTTPException(400, "review request keys mismatch")
    reason = body.get("reason")
    expected_generation = body.get("expected_generation")
    if not isinstance(reason, str) or not 12 <= len(reason.strip()) <= 1_000:
        raise HTTPException(400, "review reason must contain 12 to 1000 characters")
    if (
        not isinstance(expected_generation, int)
        or isinstance(expected_generation, bool)
        or expected_generation < 0
    ):
        raise HTTPException(400, "invalid expected generation")
    store = Store(ROOT)
    try:
        take = store.take_for_review(take_id)
    except StoreContractError as exc:
        raise HTTPException(400, "invalid take id") from exc
    if take is None or take["tombstoned"]:
        raise HTTPException(404, "no reviewable take")
    if take["status"] not in {"review_required", "unavailable"}:
        raise HTTPException(409, "take does not require human review")
    try:
        generation = store.accept(
            prompt_id=take["prompt_id"],
            corpus_version=take["corpus_version"],
            take_id=take_id,
            expected_generation=expected_generation,
            reason=reason.strip(),
        )
    except StoreConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except StoreContractError as exc:
        raise HTTPException(400, "take could not be accepted") from exc
    progress = readiness.evaluate(ROOT, _SECTIONS, trainer_health=_trainer_health())
    training = trainer.maybe_enqueue(
        ROOT,
        progress,
        auto=_read_settings().get("auto_train", False),
        now_iso=_now(),
    )
    _emit_threadsafe(
        {
            "type": "state",
            "saved": take["prompt_id"],
            "flags": take["flags"],
            "progress": progress,
            "training": training,
        }
    )
    return JSONResponse(
        {
            "take_id": take_id,
            "prompt_id": take["prompt_id"],
            "generation": generation,
            "progress": progress,
            "training": training,
        }
    )


@app.post("/api/reviews/{take_id}/reject")
async def api_reject_review(take_id: str, request: Request) -> JSONResponse:
    body = await _json_object(request)
    if set(body) != {"reason", "expected_generation"}:
        raise HTTPException(400, "review request keys mismatch")
    reason = body.get("reason")
    expected_generation = body.get("expected_generation")
    if not isinstance(reason, str) or not 12 <= len(reason.strip()) <= 1_000:
        raise HTTPException(400, "review reason must contain 12 to 1000 characters")
    if (
        not isinstance(expected_generation, int)
        or isinstance(expected_generation, bool)
        or expected_generation < 0
    ):
        raise HTTPException(400, "invalid expected generation")
    store = Store(ROOT)
    try:
        take = store.take_for_review(take_id)
    except StoreContractError as exc:
        raise HTTPException(400, "invalid take id") from exc
    if take is None or take["tombstoned"]:
        raise HTTPException(404, "no reviewable take")
    if take["status"] not in {"review_required", "unavailable"}:
        raise HTTPException(409, "take does not require human review")
    current = store.accepted(take["prompt_id"], take["corpus_version"])
    was_accepted = current is not None and current["take_id"] == take_id
    try:
        generation = store.tombstone(
            take_id,
            reason=reason.strip(),
            expected_generation=expected_generation,
        )
    except StoreConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except StoreContractError as exc:
        raise HTTPException(400, "take could not be rejected") from exc
    progress = readiness.evaluate(ROOT, _SECTIONS, trainer_health=_trainer_health())
    event = {"type": "state", "progress": progress}
    if was_accepted:
        event["deleted"] = take["prompt_id"]
    _emit_threadsafe(event)
    return JSONResponse(
        {
            "take_id": take_id,
            "prompt_id": take["prompt_id"],
            "generation": generation,
            "progress": progress,
        }
    )


@app.delete("/api/clip/{prompt_id}")
def api_clip_delete(prompt_id: str) -> JSONResponse:
    records = corpus.load_manifest(ROOT)
    rec = {r["id"]: r for r in records}.get(prompt_id)
    if rec is None:
        raise HTTPException(404, "no such clip")
    corpus.tombstone_record(ROOT, prompt_id, reason="studio retake requested")
    _emit_threadsafe({"type": "state", "deleted": prompt_id,
                      "progress": readiness.evaluate(ROOT, _SECTIONS, trainer_health=_trainer_health())})
    return JSONResponse({"deleted": prompt_id,
                         "progress": readiness.evaluate(ROOT, _SECTIONS, trainer_health=_trainer_health())})

# --------------------------------------------------------------------------- #
# room tone + training + settings
# --------------------------------------------------------------------------- #
@app.post("/api/roomtone")
def api_roomtone(file: UploadFile = File(...)) -> JSONResponse:
    staged = _stage_upload(file, maximum=MAX_ROOM_TONE_UPLOAD_BYTES)
    output = ROOT / "data" / "staging" / f"room-tone-{uuid.uuid4().hex}.wav"
    try:
        audio = processing.decode_to_48k_mono(
            staged.path,
            min_seconds=1.0,
            max_seconds=30.0,
            require_signal=False,
        )
        audio_contracts.write_pcm16_wav(
            output,
            audio,
            audio_contracts.AUDIO_SPECS["master-48k"],
            min_seconds=1.0,
            max_seconds=30.0,
            require_signal=False,
        )
        receipt = ingest.receipt_for_staged_path(ROOT, output)
        publication = ingest.publish_content(
            ROOT,
            receipt,
            label="room-tone-48k",
            extension=".wav",
        )
        settings = _read_settings()
        settings["room_tone_path"] = publication.relative_path
        _write_settings(settings)
    except (processing.ProcessingError, audio_contracts.AudioContractError, ingest.IngestError) as exc:
        raise HTTPException(422, f"room-tone processing failed: {exc}") from exc
    finally:
        staged.cleanup()
        output.unlink(missing_ok=True)
    return JSONResponse({"room_tone_sec": round(audio.size / processing.CAPTURE_SR, 2),
                         "noise_floor_dbfs": processing.peak_dbfs(audio),
                         "sha256": publication.sha256})


@app.post("/api/train")
def api_train(force: bool = Form(False),
              engine: str = Form("cosyvoice3"),
              profile: str = Form("expressive-zero-shot")) -> JSONResponse:
    prog = readiness.evaluate(ROOT, _SECTIONS, trainer_health=_trainer_health())
    if not prog["ready"] and not force:
        raise HTTPException(409, {"msg": "corpus not at readiness gate", "progress": prog})
    try:
        st = trainer.enqueue(
            ROOT,
            prog,
            engine=engine,
            profile=profile,
            now_iso=_now(),
            fixture=bool(force),
            promotable=not bool(force),
        )
    except (training_jobs.JobContractError, training_jobs.JobConflict) as exc:
        raise HTTPException(409, "training job could not be enqueued") from exc
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
    data = {**_read_settings(), "auto_train": bool(auto_train), "denoise": bool(denoise),
            "whisper_model": whisper_model}
    _write_settings(data)
    qc_transcribe.configure(whisper_model)
    return JSONResponse(data)

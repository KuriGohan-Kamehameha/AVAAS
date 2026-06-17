#!/usr/bin/env python3
"""Optional speech-QC via faster-whisper.

For scripted lines we already know the ground-truth text, so Whisper is used
only to (a) catch misreads via word-error-rate and (b) transcribe the
unscripted "spontaneous" section. Degrades gracefully: if faster-whisper is not
installed or the model can't load, transcription is skipped and callers fall
back to the script text with wer=None.

Lazy module-level singleton — the model loads once on first use.
"""
from __future__ import annotations

import re

# base.en: fast on Apple Silicon, ample for WER misread-detection (default).
# large-v3: slower, higher-fidelity transcripts — better for the unscripted
# spontaneous section where there is no ground-truth text to fall back on.
ALLOWED_MODELS = ("base.en", "small.en", "medium.en", "large-v3")

_MODEL = None
_TRIED = False
_MODEL_NAME = "base.en"


def configure(model_name: str) -> None:
    """Select the Whisper model. Resets the singleton if it changed."""
    global _MODEL, _TRIED, _MODEL_NAME
    if model_name not in ALLOWED_MODELS:
        return
    if model_name != _MODEL_NAME:
        _MODEL_NAME, _MODEL, _TRIED = model_name, None, False


def _get_model():
    global _MODEL, _TRIED
    if _MODEL is not None or _TRIED:
        return _MODEL
    _TRIED = True
    try:
        from faster_whisper import WhisperModel
        _MODEL = WhisperModel(_MODEL_NAME, device="cpu", compute_type="int8")
    except Exception:
        _MODEL = None          # graceful: QC disabled, not fatal
    return _MODEL


def available() -> bool:
    return _get_model() is not None


def _norm(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()


def wer(reference: str, hypothesis: str) -> float:
    """Levenshtein word-error-rate in [0, 1+]. Bounded DP, no recursion."""
    r, h = _norm(reference), _norm(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        cur = [i] + [0] * len(h)
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return round(prev[len(h)] / len(r), 3)


def transcribe(wav_path: str) -> str | None:
    """Return the transcript, or None if QC is unavailable."""
    model = _get_model()
    if model is None:
        return None
    try:
        segments, _ = model.transcribe(wav_path, language="en", beam_size=5)
        return " ".join(seg.text.strip() for seg in segments).strip()
    except Exception:
        return None


def model_name() -> str:
    return _MODEL_NAME

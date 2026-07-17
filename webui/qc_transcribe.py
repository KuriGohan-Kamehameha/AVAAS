#!/usr/bin/env python3
"""Bounded optional speech QC with explicit unavailable states."""
from __future__ import annotations

import re
from typing import Any


ALLOWED_MODELS = ("base.en", "small.en", "medium.en", "large-v3")
MAX_TRANSCRIPT_CHARS = 16_000
MAX_TRANSCRIPT_WORDS = 2_048
MAX_SEGMENTS = 512
MAX_SEGMENT_CHARS = 1_024
_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}$")

_MODEL = None
_TRIED = False
_MODEL_NAME = "base.en"


class QCContractError(ValueError):
    """Transcript input or inference output exceeded a fixed QC contract."""


def configure(model_name: str) -> None:
    global _MODEL, _TRIED, _MODEL_NAME
    if model_name not in ALLOWED_MODELS:
        raise QCContractError("unsupported ASR model")
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
        _MODEL = None
    return _MODEL


def available() -> bool:
    return _get_model() is not None


def _bounded_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) > MAX_TRANSCRIPT_CHARS:
        raise QCContractError(f"{field} length outside bounds")
    return value


def _norm(text: str) -> list[str]:
    words = re.sub(r"[^a-z0-9 ]+", " ", _bounded_text(text, "transcript").lower()).split()
    if len(words) > MAX_TRANSCRIPT_WORDS:
        raise QCContractError("transcript word count outside bounds")
    return words


def wer(reference: str, hypothesis: str) -> float:
    """Bounded Levenshtein word-error rate with O(hypothesis) memory."""
    reference_words, hypothesis_words = _norm(reference), _norm(hypothesis)
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    previous = list(range(len(hypothesis_words) + 1))
    for index, reference_word in enumerate(reference_words, start=1):
        current = [index] + [0] * len(hypothesis_words)
        for offset, hypothesis_word in enumerate(hypothesis_words, start=1):
            cost = 0 if reference_word == hypothesis_word else 1
            current[offset] = min(
                previous[offset] + 1,
                current[offset - 1] + 1,
                previous[offset - 1] + cost,
            )
        previous = current
    return round(previous[len(hypothesis_words)] / len(reference_words), 3)


def transcribe_result(wav_path: str, *, language: str = "en") -> dict[str, str | None]:
    """Return typed QC evidence; internal model errors never leak to clients."""
    if not isinstance(wav_path, str) or not 1 <= len(wav_path) <= 1_024:
        raise QCContractError("WAV path length outside bounds")
    if not isinstance(language, str) or not _LANGUAGE_RE.fullmatch(language):
        raise QCContractError("ASR language is invalid")
    model = _get_model()
    if model is None:
        return {
            "status": "unavailable",
            "text": None,
            "model": _MODEL_NAME,
            "reason": "model-unavailable",
        }
    try:
        segments, _ = model.transcribe(
            wav_path,
            language=language,
            beam_size=5,
            vad_filter=True,
        )
        pieces: list[str] = []
        characters = 0
        for index, segment in enumerate(segments, start=1):
            if index > MAX_SEGMENTS:
                raise QCContractError("ASR segment count outside bounds")
            text = getattr(segment, "text", None)
            if not isinstance(text, str) or len(text) > MAX_SEGMENT_CHARS:
                raise QCContractError("ASR segment length outside bounds")
            text = text.strip()
            characters += len(text) + (1 if pieces else 0)
            if characters > MAX_TRANSCRIPT_CHARS:
                raise QCContractError("ASR transcript length outside bounds")
            if text:
                pieces.append(text)
        transcript = " ".join(pieces)
        _norm(transcript)
        return {
            "status": "ok",
            "text": transcript,
            "model": _MODEL_NAME,
            "reason": None,
        }
    except QCContractError:
        return {
            "status": "unavailable",
            "text": None,
            "model": _MODEL_NAME,
            "reason": "output-outside-bounds",
        }
    except Exception:
        return {
            "status": "unavailable",
            "text": None,
            "model": _MODEL_NAME,
            "reason": "inference-failed",
        }


def transcribe(wav_path: str) -> str | None:
    """Compatibility façade for older callers."""
    return transcribe_result(wav_path)["text"]


def model_name() -> str:
    return _MODEL_NAME

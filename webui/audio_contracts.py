#!/usr/bin/env python3
"""Exact bounded audio contracts shared by capture, training, and wakeword export."""
from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


MIN_CLIP_SECONDS = 0.4
MAX_CLIP_SECONDS = 180.0
MAX_WAV_BYTES = 24 * 1024 * 1024
HASH_CHUNK_BYTES = 64 * 1024
MAX_HASH_CHUNKS = MAX_WAV_BYTES // HASH_CHUNK_BYTES + 1
SILENCE_RMS = 1e-5


class AudioContractError(ValueError):
    """Decoded samples or an on-disk WAV violated a fixed audio contract."""


@dataclass(frozen=True)
class AudioSpec:
    purpose: str
    sample_rate: int
    channels: int = 1
    subtype: str = "PCM_16"
    sample_format: str = "pcm_s16le"


AUDIO_SPECS = {
    "master-48k": AudioSpec("master-48k", 48_000),
    "serve-24k": AudioSpec("serve-24k", 24_000),
    "piper-22050": AudioSpec("piper-22050", 22_050),
    "wake-16k": AudioSpec("wake-16k", 16_000),
}


def validate_samples(
    audio: np.ndarray,
    sample_rate: int,
    *,
    min_seconds: float = MIN_CLIP_SECONDS,
    max_seconds: float = MAX_CLIP_SECONDS,
    require_signal: bool = True,
) -> np.ndarray:
    """Return a contiguous float32 mono array after bounded validation."""
    if (
        not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or not 8_000 <= sample_rate <= 192_000
    ):
        raise AudioContractError("sample rate outside bounds")
    if (
        not isinstance(audio, np.ndarray)
        or audio.ndim != 1
        or not 0.0 <= min_seconds <= max_seconds <= MAX_CLIP_SECONDS
    ):
        raise AudioContractError("audio shape or duration contract is invalid")
    minimum = int(min_seconds * sample_rate)
    maximum = int(max_seconds * sample_rate)
    if not minimum <= audio.size <= maximum:
        raise AudioContractError("decoded duration outside bounds")
    if not np.issubdtype(audio.dtype, np.number) or not np.isfinite(audio).all():
        raise AudioContractError("audio samples must be finite")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.001:
        raise AudioContractError("audio amplitude outside normalized bounds")
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
    if require_signal and rms <= SILENCE_RMS:
        raise AudioContractError("decoded audio is silent")
    return np.ascontiguousarray(audio, dtype=np.float32)


def sha256_file(path: Path, *, maximum: int = MAX_WAV_BYTES) -> tuple[str, int]:
    path = Path(path)
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= maximum:
            raise AudioContractError("audio file size outside bounds")
        digest = hashlib.sha256()
        read_bytes = 0
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            for _ in range(MAX_HASH_CHUNKS):
                chunk = handle.read(HASH_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                read_bytes += len(chunk)
            if handle.read(1):
                raise AudioContractError("audio hash loop bound exceeded")
        if read_bytes != metadata.st_size:
            raise AudioContractError("audio file changed while hashing")
        return digest.hexdigest(), read_bytes
    except AudioContractError:
        raise
    except OSError as exc:
        raise AudioContractError("cannot read audio file") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def inspect_pcm16_wav(
    path: Path,
    spec: AudioSpec,
    *,
    min_seconds: float = MIN_CLIP_SECONDS,
    max_seconds: float = MAX_CLIP_SECONDS,
) -> dict[str, int | str]:
    path = Path(path)
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise AudioContractError("audio file is missing") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise AudioContractError("audio path must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= MAX_WAV_BYTES:
        raise AudioContractError("audio file size outside bounds")
    try:
        info = sf.info(path)
    except (RuntimeError, TypeError) as exc:
        raise AudioContractError("audio is not a readable WAV") from exc
    if info.format != "WAV":
        raise AudioContractError("audio container must be WAV")
    if info.subtype != spec.subtype:
        raise AudioContractError("audio subtype must be PCM_16")
    if info.channels != spec.channels:
        raise AudioContractError("audio channel count mismatch")
    if info.samplerate != spec.sample_rate:
        raise AudioContractError("audio sample rate mismatch")
    minimum = int(min_seconds * spec.sample_rate)
    maximum = int(max_seconds * spec.sample_rate)
    if not minimum <= info.frames <= maximum:
        raise AudioContractError("audio duration outside bounds")
    digest, size = sha256_file(path)
    return {
        "purpose": spec.purpose,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "sample_format": spec.sample_format,
        "frames": info.frames,
        "duration_ms": round(info.frames * 1_000 / info.samplerate),
        "sha256": digest,
        "size": size,
    }


def write_pcm16_wav(
    path: Path,
    audio: np.ndarray,
    spec: AudioSpec,
    *,
    min_seconds: float = MIN_CLIP_SECONDS,
    max_seconds: float = MAX_CLIP_SECONDS,
    require_signal: bool = True,
) -> dict[str, int | str]:
    """Write, fsync, re-open, and validate a PCM16 WAV before atomic publication."""
    samples = validate_samples(
        audio,
        spec.sample_rate,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        require_signal=require_signal,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise AudioContractError("output path must not be a symlink")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".wav", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        sf.write(temporary, samples, spec.sample_rate, subtype=spec.subtype, format="WAV")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        observed = inspect_pcm16_wav(
            temporary,
            spec,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
        )
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return observed
    except AudioContractError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise AudioContractError("cannot write contracted PCM16 WAV") from exc
    finally:
        temporary.unlink(missing_ok=True)

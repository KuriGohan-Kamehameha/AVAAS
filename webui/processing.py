#!/usr/bin/env python3
"""Bounded, deterministic AVAAS audio decode and derivative pipeline."""
from __future__ import annotations

import math
import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyloudnorm as pyln
import scipy
import soundfile as sf
from scipy.signal import istft, resample_poly, stft

from .audio_contracts import (
    AUDIO_SPECS,
    MAX_CLIP_SECONDS,
    AudioContractError,
    validate_samples,
    write_pcm16_wav,
)


CAPTURE_SR = 48_000
TARGET_SR = 24_000
TARGET_LUFS = -23.0
PAD_SEC = 0.15
TRIM_DB = 35.0
DEAD_AIR_MIN_SEC = 0.30
SUB_ALPHA = 1.6
SUB_FLOOR = 0.06
MIN_DENOISE_IMPROVEMENT_DB = 0.75
DENOISE_CHUNK_SECONDS = 4.0
DENOISE_OVERLAP_SECONDS = 0.2
NOISE_PROFILE_MAX_SECONDS = 2.0
MAX_DENOISE_CHUNKS = 64
FFMPEG_TIMEOUT_SECONDS = 30.0
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_DECODED_WAV_BYTES = 36 * 1024 * 1024
DSP_REVISION = "avaas-dsp/1"

_FFMPEG = shutil.which("ffmpeg")


class ProcessingError(ValueError):
    """A codec, sample, DSP, or output contract failed closed."""


def _regular_bounded(path: Path, maximum: int) -> int:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise ProcessingError("decode input is missing") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise ProcessingError("decode input must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= maximum:
        raise ProcessingError("decode input size outside bounds")
    return metadata.st_size


def decode_to_48k_mono(
    src: Path,
    *,
    min_seconds: float = 0.4,
    max_seconds: float = MAX_CLIP_SECONDS,
    require_signal: bool = True,
) -> np.ndarray:
    """Decode one bounded audio stream to finite mono float32 at 48 kHz.

    ffmpeg writes to a bounded temporary file and has no captured stdout/stderr,
    preventing an adversarial codec from growing an in-memory pipe.
    """
    if _FFMPEG is None:
        raise ProcessingError("ffmpeg decode is unavailable")
    if not 0.0 <= min_seconds <= max_seconds <= MAX_CLIP_SECONDS:
        raise ProcessingError("decode duration limits are invalid")
    src = Path(src)
    _regular_bounded(src, MAX_INPUT_BYTES)
    descriptor, name = tempfile.mkstemp(prefix="avaas-decode-", suffix=".wav")
    os.close(descriptor)
    decoded = Path(name)
    try:
        command = [
            _FFMPEG,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-y",
            "-i",
            str(src),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            str(CAPTURE_SR),
            # Decode one second past the accepted ceiling so oversize input is
            # distinguishable from an exact maximum-length clip.
            "-t",
            f"{max_seconds + 1.0:.1f}",
            "-fs",
            str(MAX_DECODED_WAV_BYTES),
            "-f",
            "wav",
            "-acodec",
            "pcm_f32le",
            str(decoded),
        ]
        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=FFMPEG_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ProcessingError("ffmpeg decode deadline exceeded") from exc
        if result.returncode != 0:
            raise ProcessingError("ffmpeg decode failed")
        _regular_bounded(decoded, MAX_DECODED_WAV_BYTES)
        try:
            info = sf.info(decoded)
        except RuntimeError as exc:
            raise ProcessingError("ffmpeg decode produced an invalid WAV") from exc
        if (
            info.format != "WAV"
            or info.subtype != "FLOAT"
            or info.channels != 1
            or info.samplerate != CAPTURE_SR
            or info.frames > int((max_seconds + 0.5) * CAPTURE_SR)
        ):
            raise ProcessingError("ffmpeg decode output contract mismatch")
        audio, sample_rate = sf.read(decoded, dtype="float32", always_2d=False)
        if sample_rate != CAPTURE_SR:
            raise ProcessingError("ffmpeg decode sample rate mismatch")
        return validate_samples(
            audio,
            CAPTURE_SR,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            require_signal=require_signal,
        )
    except AudioContractError as exc:
        raise ProcessingError(str(exc)) from exc
    finally:
        decoded.unlink(missing_ok=True)


def peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return float(20.0 * np.log10(peak)) if peak > 1e-9 else -120.0


def _frame_rms(audio: np.ndarray, sample_rate: int, win_ms: float = 20.0) -> tuple[np.ndarray, int]:
    window = max(1, int(sample_rate * win_ms / 1_000.0))
    count = audio.size // window
    if count < 1:
        value = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if audio.size else 0.0
        return np.array([value], dtype=np.float64), window
    framed = audio[: count * window].reshape(count, window)
    return np.sqrt(np.mean(np.square(framed, dtype=np.float64), axis=1) + 1e-12), window


def find_dead_air(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    rms, window = _frame_rms(audio, sample_rate)
    needed = max(1, int(DEAD_AIR_MIN_SEC * sample_rate / window))
    if rms.size < needed:
        return audio[: int(DEAD_AIR_MIN_SEC * sample_rate)].copy()
    cumulative = np.cumsum(np.insert(rms, 0, 0.0))
    means = (cumulative[needed:] - cumulative[:-needed]) / needed
    start = int(np.argmin(means)) * window
    return audio[start : min(audio.size, start + needed * window)].copy()


def estimate_snr_db(audio: np.ndarray, sample_rate: int) -> float:
    """Return a bounded robust active-to-floor estimate for selection evidence."""
    rms, _ = _frame_rms(audio, sample_rate)
    floor = max(float(np.percentile(rms, 10.0)), 1e-9)
    active = max(float(np.percentile(rms, 90.0)), floor)
    return float(20.0 * np.log10(active / floor))


def _noise_magnitude(noise: np.ndarray, sample_rate: int, segment: int) -> np.ndarray:
    profile = noise[: min(noise.size, int(NOISE_PROFILE_MAX_SECONDS * sample_rate))]
    _, _, spectrum = stft(
        profile,
        fs=sample_rate,
        nperseg=segment,
        noverlap=segment * 3 // 4,
    )
    return np.median(np.abs(spectrum), axis=1, keepdims=True)


def _subtract_chunk(
    chunk: np.ndarray,
    sample_rate: int,
    segment: int,
    noise_magnitude: np.ndarray,
) -> np.ndarray:
    _, _, spectrum = stft(
        chunk,
        fs=sample_rate,
        nperseg=segment,
        noverlap=segment * 3 // 4,
    )
    magnitude = np.abs(spectrum)
    clean = np.maximum(magnitude - SUB_ALPHA * noise_magnitude, SUB_FLOOR * magnitude)
    _, reconstructed = istft(
        clean * np.exp(1j * np.angle(spectrum)),
        fs=sample_rate,
        nperseg=segment,
        noverlap=segment * 3 // 4,
    )
    reconstructed = np.asarray(reconstructed[: chunk.size], dtype=np.float32)
    if reconstructed.size < chunk.size:
        reconstructed = np.pad(reconstructed, (0, chunk.size - reconstructed.size))
    return reconstructed


def reduce_noise(audio: np.ndarray, sample_rate: int, noise: np.ndarray) -> np.ndarray:
    """Spectral subtraction in fixed chunks with bounded cross-fade state."""
    if audio.size < sample_rate // 10 or noise.size < 256:
        return audio.copy()
    segment = 1_024
    noise_magnitude = _noise_magnitude(noise, sample_rate, segment)
    chunk_frames = int(DENOISE_CHUNK_SECONDS * sample_rate)
    overlap = int(DENOISE_OVERLAP_SECONDS * sample_rate)
    step = chunk_frames - overlap
    output = np.empty_like(audio)
    previous_end = 0
    for index in range(MAX_DENOISE_CHUNKS):
        start = index * step
        if start >= audio.size:
            return output
        end = min(audio.size, start + chunk_frames)
        reconstructed = _subtract_chunk(
            audio[start:end], sample_rate, segment, noise_magnitude
        )
        blend = min(overlap, max(0, previous_end - start), reconstructed.size)
        if blend:
            fade = np.linspace(0.0, 1.0, blend, endpoint=False, dtype=np.float32)
            output[start : start + blend] = (
                output[start : start + blend] * (1.0 - fade)
                + reconstructed[:blend] * fade
            )
        output[start + blend : end] = reconstructed[blend:]
        previous_end = end
        if end >= audio.size:
            return output
    raise ProcessingError("denoise chunk count exceeded bound")


def _select_denoised(
    audio: np.ndarray,
    sample_rate: int,
    noise: np.ndarray,
    requested: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    baseline = estimate_snr_db(audio, sample_rate)
    decision: dict[str, Any] = {
        "requested": bool(requested),
        "applied": False,
        "reason": "disabled",
        "baseline_snr_db": round(baseline, 2),
        "candidate_snr_db": None,
        "improvement_db": 0.0,
    }
    if not requested:
        return audio, decision
    candidate = reduce_noise(audio, sample_rate, noise)
    try:
        candidate = validate_samples(candidate, sample_rate)
    except AudioContractError:
        decision["reason"] = "candidate-invalid"
        return audio, decision
    candidate_score = estimate_snr_db(candidate, sample_rate)
    improvement = candidate_score - baseline
    decision.update(
        {
            "candidate_snr_db": round(candidate_score, 2),
            "improvement_db": round(improvement, 2),
        }
    )
    original_rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    candidate_rms = float(np.sqrt(np.mean(np.square(candidate, dtype=np.float64))))
    retained = candidate_rms / max(original_rms, 1e-12)
    if improvement < MIN_DENOISE_IMPROVEMENT_DB:
        decision["reason"] = "insufficient-improvement"
        return audio, decision
    if not 0.35 <= retained <= 1.25:
        decision["reason"] = "level-distortion"
        return audio, decision
    decision["applied"] = True
    decision["reason"] = "measured-improvement"
    return candidate, decision


def normalize_lufs(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    try:
        loudness = float(pyln.Meter(sample_rate).integrated_loudness(audio))
    except (ValueError, RuntimeError) as exc:
        raise ProcessingError("loudness measurement failed") from exc
    if not math.isfinite(loudness):
        raise ProcessingError("loudness measurement is non-finite")
    gain = 10.0 ** ((TARGET_LUFS - loudness) / 20.0)
    normalized = np.asarray(audio * gain, dtype=np.float32)
    if not math.isfinite(gain) or not np.isfinite(normalized).all():
        raise ProcessingError("loudness normalization is non-finite")
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 0.999:
        normalized = normalized * (0.999 / peak)
    return np.asarray(normalized, dtype=np.float32), loudness


def trim_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    rms, window = _frame_rms(audio, sample_rate)
    threshold = float(np.max(rms)) * (10.0 ** (-TRIM_DB / 20.0))
    voiced = np.flatnonzero(rms > threshold)
    if voiced.size == 0:
        raise ProcessingError("trim found no voiced audio")
    pad = int(PAD_SEC * sample_rate)
    start = max(0, int(voiced[0]) * window - pad)
    end = min(audio.size, (int(voiced[-1]) + 1) * window + pad)
    return np.ascontiguousarray(audio[start:end], dtype=np.float32)


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.copy()
    divisor = math.gcd(source_rate, target_rate)
    resampled = np.asarray(
        resample_poly(audio, target_rate // divisor, source_rate // divisor),
        dtype=np.float32,
    )
    peak = float(np.max(np.abs(resampled))) if resampled.size else 0.0
    if peak > 0.999:
        resampled = resampled * (0.999 / peak)
    return np.asarray(resampled, dtype=np.float32)


def resample_24k(audio: np.ndarray) -> np.ndarray:
    return resample_audio(audio, CAPTURE_SR, TARGET_SR)


def standardize(
    src: Path,
    outputs: dict[str, Path],
    *,
    room_tone: np.ndarray | None = None,
    denoise: bool = True,
) -> dict[str, Any]:
    """Write exact 48/24/22.05/16 kHz PCM16 artifacts and return evidence."""
    if not isinstance(outputs, dict) or set(outputs) != set(AUDIO_SPECS):
        raise ProcessingError("output purpose set mismatch")
    paths = [Path(outputs[purpose]) for purpose in AUDIO_SPECS]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ProcessingError("output paths must be unique")
    try:
        audio = decode_to_48k_mono(src)
        input_peak = peak_dbfs(audio)
        clipping = input_peak > -0.1
        if room_tone is None:
            noise = find_dead_air(audio, CAPTURE_SR)
            noise_source = "clip-dead-air"
        else:
            try:
                noise = validate_samples(
                    room_tone,
                    CAPTURE_SR,
                    min_seconds=0.1,
                    max_seconds=30.0,
                    require_signal=False,
                )
            except AudioContractError as exc:
                raise ProcessingError(f"room tone contract failed: {exc}") from exc
            noise_source = "captured-room-tone"

        selected, noise_decision = _select_denoised(audio, CAPTURE_SR, noise, denoise)
        normalized, measured_loudness = normalize_lufs(selected, CAPTURE_SR)
        trimmed = trim_silence(normalized, CAPTURE_SR)
        try:
            trimmed = validate_samples(trimmed, CAPTURE_SR)
        except AudioContractError as exc:
            raise ProcessingError(f"processed audio contract failed: {exc}") from exc

        rendered = {
            "master-48k": audio,
            "serve-24k": resample_audio(trimmed, CAPTURE_SR, 24_000),
            "piper-22050": resample_audio(trimmed, CAPTURE_SR, 22_050),
            "wake-16k": resample_audio(trimmed, CAPTURE_SR, 16_000),
        }
        artifacts: list[dict[str, Any]] = []
        for purpose, spec in AUDIO_SPECS.items():
            observed = write_pcm16_wav(outputs[purpose], rendered[purpose], spec)
            artifacts.append({**observed, "path": str(Path(outputs[purpose]))})

        serve = next(item for item in artifacts if item["purpose"] == "serve-24k")
        qc = {
            "duration": round(int(serve["frames"]) / 24_000, 3),
            "peak_dbfs": round(input_peak, 2),
            "clipping": bool(clipping),
            "loudness_lufs": round(measured_loudness, 2),
            "snr_db": noise_decision["baseline_snr_db"],
            "noise_reduced": noise_decision["applied"],
        }
        dsp = {
            "revision": DSP_REVISION,
            "target_lufs": TARGET_LUFS,
            "noise_source": noise_source,
            "noise_reduction": noise_decision,
            "tools": {
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "soundfile": sf.__version__,
                "ffmpeg": str(_FFMPEG),
            },
        }
        return {"qc": qc, "dsp": dsp, "artifacts": artifacts}
    except (AudioContractError, ProcessingError):
        for path in paths:
            path.unlink(missing_ok=True)
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        for path in paths:
            path.unlink(missing_ok=True)
        raise ProcessingError("audio processing failed") from exc

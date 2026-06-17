#!/usr/bin/env python3
"""Audio standardization workflow for the voice corpus.

Takes any browser-recorded blob or uploaded file in any ffmpeg-decodable format
and produces a training-ready clip:

  1. ffmpeg decode  -> 48 kHz mono float32              (any input format)
  2. measure peak / clipping                            (QC)
  3. auto-sample room tone from dead air -> denoise     (spectral subtraction)
  4. LUFS-normalize to -23 (broadcast standard)
  5. trim leading/trailing silence (keep 0.15 s pad)
  6. resample 48 -> 24 kHz
  7. (optional) faster-whisper transcript + WER vs the script line

Deps: numpy, scipy, soundfile, pyloudnorm, system ffmpeg. faster-whisper is
optional — QC degrades gracefully (no transcript / no WER) when absent.

All functions are pure and side-effect-free except `standardize`, which writes
the two WAV artifacts. P10: small functions, checked returns, bounded work.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import resample_poly, stft, istft

CAPTURE_SR = 48_000
TARGET_SR = 24_000
TARGET_LUFS = -23.0
PAD_SEC = 0.15
TRIM_DB = 35.0           # below (peak - TRIM_DB) counts as silence for trimming
DEAD_AIR_MIN_SEC = 0.30  # shortest usable room-tone sample
SUB_ALPHA = 1.6          # spectral over-subtraction factor
SUB_FLOOR = 0.06         # spectral floor (fraction of original) — limits musical noise
FFMPEG_TIMEOUT = 120     # P10: bounded external call

_FFMPEG = shutil.which("ffmpeg")


# --------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------- #
def decode_to_48k_mono(src: Path) -> np.ndarray:
    """ffmpeg-decode any format to float32 mono @ 48 kHz in [-1, 1]."""
    assert _FFMPEG, "ffmpeg not found on PATH"
    src = Path(src)
    assert src.exists() and src.stat().st_size > 0, f"empty/missing input: {src}"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = Path(tmp.name)
    cmd = [_FFMPEG, "-nostdin", "-y", "-i", str(src),
           "-ac", "1", "-ar", str(CAPTURE_SR), "-f", "wav",
           "-acodec", "pcm_f32le", str(out)]
    proc = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT)
    if proc.returncode != 0:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode()[-400:]}")
    audio, sr = sf.read(str(out), dtype="float32")
    out.unlink(missing_ok=True)
    assert sr == CAPTURE_SR, f"unexpected decode sr {sr}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


# --------------------------------------------------------------------------- #
# metering / QC primitives
# --------------------------------------------------------------------------- #
def peak_dbfs(audio: np.ndarray) -> float:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return float(20.0 * np.log10(peak)) if peak > 1e-9 else -120.0


def _frame_rms(audio: np.ndarray, sr: int, win_ms: float = 20.0) -> tuple[np.ndarray, int]:
    win = max(1, int(sr * win_ms / 1000.0))
    n = audio.size // win
    if n < 1:
        return np.array([float(np.sqrt(np.mean(audio ** 2)) if audio.size else 0.0)]), win
    framed = audio[: n * win].reshape(n, win)
    return np.sqrt(np.mean(framed ** 2, axis=1) + 1e-12), win


def find_dead_air(audio: np.ndarray, sr: int) -> np.ndarray:
    """Return the quietest contiguous >=DEAD_AIR_MIN_SEC slice as a noise sample.

    Falls back to the leading DEAD_AIR_MIN_SEC if no clear dead air is found.
    """
    rms, win = _frame_rms(audio, sr)
    need = max(1, int(DEAD_AIR_MIN_SEC * sr / win))
    if rms.size < need:
        return audio[: int(DEAD_AIR_MIN_SEC * sr)].copy()
    # sliding-window mean energy; pick the calmest run (bounded, vectorized)
    csum = np.cumsum(np.insert(rms, 0, 0.0))
    window_means = (csum[need:] - csum[:-need]) / need
    start_frame = int(np.argmin(window_means))
    a = start_frame * win
    b = min(audio.size, a + need * win)
    return audio[a:b].copy()


def snr_db(audio: np.ndarray, noise: np.ndarray) -> float:
    sig_p = float(np.mean(audio ** 2) + 1e-12)
    noise_p = float(np.mean(noise ** 2) + 1e-12)
    return float(10.0 * np.log10(sig_p / noise_p))


# --------------------------------------------------------------------------- #
# denoise — spectral subtraction against a room-tone profile
# --------------------------------------------------------------------------- #
def reduce_noise(audio: np.ndarray, sr: int, noise: np.ndarray) -> np.ndarray:
    """Spectral-subtract the room-tone magnitude spectrum from the signal.

    Classic over-subtraction with a spectral floor to suppress musical noise.
    Phase is preserved from the original signal.
    """
    if audio.size < sr // 10 or noise.size < 256:
        return audio
    nperseg = 1024
    f, t, Z = stft(audio, fs=sr, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    _, _, N = stft(noise, fs=sr, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    noise_mag = np.median(np.abs(N), axis=1, keepdims=True)
    mag = np.abs(Z)
    phase = np.angle(Z)
    clean = np.maximum(mag - SUB_ALPHA * noise_mag, SUB_FLOOR * mag)
    _, rec = istft(clean * np.exp(1j * phase), fs=sr,
                   nperseg=nperseg, noverlap=nperseg * 3 // 4)
    rec = rec[: audio.size].astype(np.float32)
    if rec.size < audio.size:                       # pad if istft returned short
        rec = np.concatenate([rec, np.zeros(audio.size - rec.size, np.float32)])
    return rec


# --------------------------------------------------------------------------- #
# normalize / trim / resample
# --------------------------------------------------------------------------- #
def normalize_lufs(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    meter = pyln.Meter(sr)
    loud = meter.integrated_loudness(audio)
    if not np.isfinite(loud):
        return audio, float("nan")
    out = pyln.normalize.loudness(audio, loud, TARGET_LUFS)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0.999:                                # guard re-clip after gain
        out = out * (0.999 / peak)
    return out.astype(np.float32), float(loud)


def trim_silence(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.size < sr // 10:
        return audio
    rms, win = _frame_rms(audio, sr)
    thresh = np.max(rms) * (10.0 ** (-TRIM_DB / 20.0))
    voiced = np.where(rms > thresh)[0]
    if voiced.size == 0:
        return audio
    pad = int(PAD_SEC * sr)
    a = max(0, voiced[0] * win - pad)
    b = min(audio.size, (voiced[-1] + 1) * win + pad)
    return audio[a:b]


def resample_24k(audio: np.ndarray) -> np.ndarray:
    if CAPTURE_SR == TARGET_SR:
        return audio
    g = np.gcd(CAPTURE_SR, TARGET_SR)
    return resample_poly(audio, TARGET_SR // g, CAPTURE_SR // g).astype(np.float32)


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def standardize(src: Path, raw_out: Path, proc_out: Path,
                room_tone: np.ndarray | None = None,
                denoise: bool = True) -> dict:
    """Full pipeline. Writes raw (48k, archival) + processed (24k) WAVs.

    Returns a QC dict: duration, peak_dbfs, clipping, loudness_lufs, snr_db,
    noise_reduced. Raises on decode failure (caller flags the clip).
    """
    audio = decode_to_48k_mono(src)
    assert audio.size > 0, "decoded audio is empty"
    in_peak = peak_dbfs(audio)
    clipping = in_peak > -0.1

    have_profile = room_tone is not None and room_tone.size >= 256
    noise = room_tone if have_profile else find_dead_air(audio, CAPTURE_SR)
    # Only trust an SNR figure when the noise reference is a genuine floor: a
    # captured room-tone profile, or a dead-air sample clearly quieter than the
    # clip overall. Otherwise (no lead-in silence) report unknown, not "noisy".
    clip_rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
    noise_rms = float(np.sqrt(np.mean(noise ** 2) + 1e-12))
    # Real room tone sits >=12 dB under the clip; a mere speech gap does not.
    # Only then is an SNR figure trustworthy enough to drive the "noisy" flag.
    confident = have_profile or noise_rms < clip_rms * 0.25
    snr = snr_db(audio, noise) if confident else None

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(raw_out), audio, CAPTURE_SR, subtype="PCM_16")   # archival original

    work = reduce_noise(audio, CAPTURE_SR, noise) if denoise else audio
    work, loud = normalize_lufs(work, CAPTURE_SR)
    work = trim_silence(work, CAPTURE_SR)
    work = resample_24k(work)

    proc_out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(proc_out), work, TARGET_SR, subtype="PCM_16")

    return {
        "duration": round(work.size / TARGET_SR, 3),
        "peak_dbfs": round(in_peak, 2),
        "clipping": bool(clipping),
        "loudness_lufs": None if not np.isfinite(loud) else round(loud, 2),
        "snr_db": None if snr is None else round(snr, 2),
        "noise_reduced": bool(denoise),
    }

#!/usr/bin/env python3
"""Preprocess raw recordings into training-ready segments + manifest.

Per file:
  1. Load 48kHz wav from data/raw/
  2. Loudness-normalize to -23 LUFS (broadcast standard)
  3. Trim leading/trailing silence (keep ~0.15s pad each side)
  4. Resample 48 → 24 kHz
  5. Transcribe with faster-whisper
  6. Write 24kHz wav to data/processed/
  7. Append to data/processed/manifest.jsonl

Run:
  python scripts/preprocess.py
  python scripts/preprocess.py --model large-v3 --device cuda
  python scripts/preprocess.py --limit 10              # smoke test
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm

ROOT        = Path(__file__).resolve().parent.parent
RAW_DIR     = ROOT / "data" / "raw"
OUT_DIR     = ROOT / "data" / "processed"
MANIFEST    = OUT_DIR / "manifest.jsonl"
TARGET_LUFS = -23.0
TARGET_SR   = 24_000
PAD_SEC     = 0.15


def _load_noise_profile(path: Path) -> tuple[np.ndarray, int] | None:
    """Load a noise-profile WAV (48 kHz) for spectral subtraction."""
    try:
        audio, sr = librosa.load(str(path), sr=None, mono=True)
        assert len(audio) > 0, "empty noise profile"
        return audio, int(sr)
    except Exception as e:
        print(f"  ! noise profile load failed: {e}")
        return None


def _apply_noise_reduction(
    audio: np.ndarray,
    sr: int,
    noise: np.ndarray,
    noise_sr: int,
) -> np.ndarray:
    """Spectral-gate denoising via noisereduce. Fails gracefully."""
    try:
        import noisereduce as nr
        noise_r = (
            librosa.resample(noise, orig_sr=noise_sr, target_sr=sr)
            if noise_sr != sr else noise
        )
        return nr.reduce_noise(y=audio, sr=sr, y_noise=noise_r,
                               prop_decrease=0.75, stationary=True)
    except Exception as e:
        print(f"  ! noise reduction skipped: {e}")
        return audio


def _load_existing_sources() -> set[str]:
    """Return basenames already recorded in the manifest."""
    seen: set[str] = set()
    if not MANIFEST.exists():
        return seen
    try:
        for row in MANIFEST.read_text().splitlines():
            if row.strip():
                src = json.loads(row).get("source", "")
                seen.add(Path(src).name)
    except Exception:
        pass
    return seen


def process_one(
    path: Path,
    meter: pyln.Meter,
    asr,
    noise: np.ndarray | None = None,
    noise_sr: int = 48_000,
) -> dict:
    audio, sr = librosa.load(str(path), sr=None, mono=True)
    if noise is not None:
        audio = _apply_noise_reduction(audio, int(sr), noise, noise_sr)
    loudness = meter.integrated_loudness(audio)
    if np.isfinite(loudness):
        audio = pyln.normalize.loudness(audio, loudness, TARGET_LUFS)
    audio, _ = librosa.effects.trim(audio, top_db=35)
    pad   = int(PAD_SEC * sr)
    audio = np.concatenate([np.zeros(pad), audio, np.zeros(pad)])
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=int(sr), target_sr=TARGET_SR)
    out_path = OUT_DIR / path.name
    sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")
    segments, info = asr.transcribe(str(out_path), language="en", beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return {
        "audio_filepath": str(out_path.relative_to(ROOT)),
        "duration": len(audio) / TARGET_SR,
        "text": text,
        "language": info.language,
        "src_loudness_lufs": float(loudness) if np.isfinite(loudness) else None,
        "source": str(path.relative_to(ROOT)),
    }


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",         default="large-v3")
    ap.add_argument("--device",        default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--limit",         type=int, default=None)
    ap.add_argument("--new-only",      action="store_true",
                    help="Skip files already in manifest (append mode)")
    ap.add_argument("--noise-profile", default=None,
                    help="Path to a WAV noise profile (from room-tone capture)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from faster_whisper import WhisperModel

    device       = pick_device(args.device)
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"faster-whisper {args.model} · {device} ({compute_type})")
    asr   = WhisperModel(args.model, device=device, compute_type=compute_type)
    meter = pyln.Meter(48_000)

    noise, noise_sr = None, 48_000
    if args.noise_profile:
        result = _load_noise_profile(Path(args.noise_profile))
        if result is not None:
            noise, noise_sr = result
            print(f"Noise profile loaded: {args.noise_profile} ({noise_sr} Hz)")

    already = _load_existing_sources() if args.new_only else set()
    files   = [f for f in sorted(RAW_DIR.glob("*.wav")) if f.name not in already]
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("No new wav files to process")
        return 0

    records: list[dict] = []
    for f in tqdm(files, desc="preprocess"):
        try:
            records.append(process_one(f, meter, asr, noise, noise_sr))
        except Exception as e:
            print(f"  ! {f.name}: {e}")

    # Append mode when --new-only, overwrite otherwise
    mode = "a" if args.new_only else "w"
    with open(MANIFEST, mode) as fp:
        for r in records:
            fp.write(json.dumps(r) + "\n")

    total = sum(r["duration"] for r in records)
    print(f"\n{len(records)} segments · {total / 60:.1f} min")
    print(f"Manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

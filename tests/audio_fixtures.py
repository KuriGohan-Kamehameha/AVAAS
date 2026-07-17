from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def speech_like(seconds: float = 1.25, sample_rate: int = 48_000) -> np.ndarray:
    """Deterministic, non-silent fixture with quiet lead/trail and two tones."""
    frames = int(seconds * sample_rate)
    time = np.arange(frames, dtype=np.float64) / sample_rate
    envelope = np.ones(frames, dtype=np.float64)
    edge = min(frames // 4, int(0.18 * sample_rate))
    envelope[:edge] = np.linspace(0.0, 1.0, edge, endpoint=False)
    envelope[-edge:] = np.linspace(1.0, 0.0, edge, endpoint=False)
    signal = 0.18 * np.sin(2.0 * np.pi * 173.0 * time)
    signal += 0.06 * np.sin(2.0 * np.pi * 311.0 * time)
    return np.asarray(signal * envelope, dtype=np.float32)


def write_input(
    path: Path,
    audio: np.ndarray | None = None,
    *,
    sample_rate: int = 48_000,
    subtype: str = "PCM_16",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, speech_like(sample_rate=sample_rate) if audio is None else audio, sample_rate, subtype=subtype)
    return path

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from tests.audio_fixtures import speech_like, write_input
from webui import audio_contracts, processing


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_standardize_emits_exact_immutable_master_and_derivatives(tmp_path: Path) -> None:
    source = write_input(tmp_path / "source.wav")
    source_before = _digest(source)
    outputs = {
        "master-48k": tmp_path / "stage" / "master.wav",
        "serve-24k": tmp_path / "stage" / "serve.wav",
        "piper-22050": tmp_path / "stage" / "piper.wav",
        "wake-16k": tmp_path / "stage" / "wake.wav",
    }
    result = processing.standardize(source, outputs, denoise=False)

    assert _digest(source) == source_before
    assert set(result) == {"qc", "dsp", "artifacts"}
    assert result["dsp"]["revision"] == processing.DSP_REVISION
    assert result["dsp"]["noise_reduction"]["requested"] is False
    assert result["dsp"]["noise_reduction"]["applied"] is False
    assert {item["purpose"] for item in result["artifacts"]} == set(outputs)

    for purpose, spec in audio_contracts.AUDIO_SPECS.items():
        observed = audio_contracts.inspect_pcm16_wav(outputs[purpose], spec)
        assert observed["sample_rate"] == spec.sample_rate
        assert observed["channels"] == 1
        assert observed["sample_format"] == "pcm_s16le"
        assert np.isfinite(sf.read(outputs[purpose], dtype="float32")[0]).all()
        artifact = next(item for item in result["artifacts"] if item["purpose"] == purpose)
        assert artifact["sha256"] == _digest(outputs[purpose])
        assert artifact["frames"] == observed["frames"]

    master_duration = sf.info(outputs["master-48k"]).duration
    assert master_duration == pytest.approx(1.25, abs=1 / 48_000)
    derivative_durations = [sf.info(outputs[name]).duration for name in outputs if name != "master-48k"]
    assert max(derivative_durations) - min(derivative_durations) < 1 / 16_000


def test_decode_and_sample_contracts_reject_malformed_nonfinite_silence_and_extremes(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.bin"
    malformed.write_bytes(b"not audio")
    with pytest.raises(processing.ProcessingError, match="decode"):
        processing.decode_to_48k_mono(malformed)

    with pytest.raises(audio_contracts.AudioContractError, match="finite"):
        audio_contracts.validate_samples(
            np.array([0.0, np.nan], dtype=np.float32), 48_000, min_seconds=0.0
        )
    with pytest.raises(audio_contracts.AudioContractError, match="silent"):
        audio_contracts.validate_samples(np.zeros(48_000, dtype=np.float32), 48_000)
    with pytest.raises(audio_contracts.AudioContractError, match="duration"):
        audio_contracts.validate_samples(np.ones(100, dtype=np.float32) * 0.1, 48_000)


def test_decode_accepts_wavex_float_container_emitted_by_debian_ffmpeg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_input(tmp_path / "source.wav")
    real_info = processing.sf.info

    def debian_info(path: Path) -> SimpleNamespace:
        observed = real_info(path)
        return SimpleNamespace(
            format="WAVEX",
            subtype=observed.subtype,
            channels=observed.channels,
            samplerate=observed.samplerate,
            frames=observed.frames,
        )

    monkeypatch.setattr(processing.sf, "info", debian_info)
    decoded = processing.decode_to_48k_mono(source)
    assert decoded.shape == (60_000,)
    assert decoded.dtype == np.float32


def test_clipping_is_preserved_as_review_required_evidence(tmp_path: Path) -> None:
    clipped = np.ones(48_000, dtype=np.float32)
    source = write_input(tmp_path / "clipped.wav", clipped)
    outputs = {name: tmp_path / f"{name}.wav" for name in audio_contracts.AUDIO_SPECS}
    result = processing.standardize(source, outputs, denoise=False)
    assert result["qc"]["clipping"] is True
    assert result["qc"]["peak_dbfs"] >= -0.1


def test_denoise_is_selected_only_after_measured_improvement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_input(tmp_path / "source.wav")
    outputs = {name: tmp_path / "a" / f"{name}.wav" for name in audio_contracts.AUDIO_SPECS}
    monkeypatch.setattr(processing, "reduce_noise", lambda audio, _sr, _noise: audio * 0.9)
    scores = iter((10.0, 10.2))
    monkeypatch.setattr(processing, "estimate_snr_db", lambda _audio, _sr: next(scores))
    rejected = processing.standardize(source, outputs, denoise=True)
    assert rejected["dsp"]["noise_reduction"]["applied"] is False
    assert rejected["dsp"]["noise_reduction"]["reason"] == "insufficient-improvement"

    outputs = {name: tmp_path / "b" / f"{name}.wav" for name in audio_contracts.AUDIO_SPECS}
    scores = iter((10.0, 12.0))
    monkeypatch.setattr(processing, "estimate_snr_db", lambda _audio, _sr: next(scores))
    accepted = processing.standardize(source, outputs, denoise=True)
    assert accepted["dsp"]["noise_reduction"]["applied"] is True
    assert accepted["dsp"]["noise_reduction"]["improvement_db"] == 2.0


def test_denoise_spectral_allocations_are_chunk_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_stft = processing.stft
    observed_sizes: list[int] = []

    def bounded_stft(audio, *args, **kwargs):
        observed_sizes.append(audio.size)
        return original_stft(audio, *args, **kwargs)

    monkeypatch.setattr(processing, "stft", bounded_stft)
    audio = np.tile(speech_like(1.0), 9)
    noise = np.tile(speech_like(1.0) * 0.02, 3)
    result = processing.reduce_noise(audio, 48_000, noise)
    assert result.shape == audio.shape
    assert np.isfinite(result).all()
    assert max(observed_sizes) <= int(processing.DENOISE_CHUNK_SECONDS * 48_000)
    assert observed_sizes[0] <= int(processing.NOISE_PROFILE_MAX_SECONDS * 48_000)


def test_pcm_validator_rejects_wrong_rate_channel_subtype_and_symlink(tmp_path: Path) -> None:
    spec = audio_contracts.AUDIO_SPECS["serve-24k"]
    wrong_rate = write_input(tmp_path / "wrong-rate.wav", sample_rate=48_000)
    with pytest.raises(audio_contracts.AudioContractError, match="sample rate"):
        audio_contracts.inspect_pcm16_wav(wrong_rate, spec)

    stereo = np.column_stack((speech_like(sample_rate=24_000), speech_like(sample_rate=24_000)))
    stereo_path = write_input(tmp_path / "stereo.wav", stereo, sample_rate=24_000)
    with pytest.raises(audio_contracts.AudioContractError, match="channel"):
        audio_contracts.inspect_pcm16_wav(stereo_path, spec)

    float_path = write_input(tmp_path / "float.wav", sample_rate=24_000, subtype="FLOAT")
    with pytest.raises(audio_contracts.AudioContractError, match="subtype"):
        audio_contracts.inspect_pcm16_wav(float_path, spec)

    link = tmp_path / "link.wav"
    link.symlink_to(wrong_rate)
    with pytest.raises(audio_contracts.AudioContractError, match="symlink"):
        audio_contracts.inspect_pcm16_wav(link, spec)

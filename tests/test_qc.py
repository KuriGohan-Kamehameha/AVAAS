from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from webui import corpus, qc_transcribe


def test_wer_is_bounded_and_deterministic() -> None:
    assert qc_transcribe.wer("This is Satraj", "This is Satraj") == 0.0
    assert qc_transcribe.wer("This is Satraj", "This is Piranesi") == pytest.approx(1 / 3, abs=0.001)
    with pytest.raises(qc_transcribe.QCContractError, match="length"):
        qc_transcribe.wer("x" * (qc_transcribe.MAX_TRANSCRIPT_CHARS + 1), "x")


def test_transcription_reports_load_and_inference_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"fixture")
    monkeypatch.setattr(qc_transcribe, "_get_model", lambda: None)
    assert qc_transcribe.transcribe_result(str(wav)) == {
        "status": "unavailable",
        "text": None,
        "model": qc_transcribe.model_name(),
        "reason": "model-unavailable",
    }

    class Broken:
        def transcribe(self, *_args, **_kwargs):
            raise RuntimeError("secret internal detail")

    monkeypatch.setattr(qc_transcribe, "_get_model", lambda: Broken())
    failed = qc_transcribe.transcribe_result(str(wav))
    assert failed["status"] == "unavailable"
    assert failed["reason"] == "inference-failed"
    assert "secret" not in str(failed)


def test_transcription_success_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    class Fake:
        def transcribe(self, *_args, **_kwargs):
            return iter((SimpleNamespace(text=" This is"), SimpleNamespace(text=" Satraj. "))), None

    monkeypatch.setattr(qc_transcribe, "_get_model", lambda: Fake())
    assert qc_transcribe.transcribe_result("clip.wav", language="en") == {
        "status": "ok",
        "text": "This is Satraj.",
        "model": qc_transcribe.model_name(),
        "reason": None,
    }


def test_asr_unavailable_cannot_look_like_clean_qc() -> None:
    qc = {"duration": 1.0, "peak_dbfs": -6.0, "clipping": False, "snr_db": 30.0}
    assert corpus.qc_flags(qc, None, asr_status="unavailable") == ["asr-unavailable"]
    assert corpus.qc_flags(qc, 0.0, asr_status="ok") == []

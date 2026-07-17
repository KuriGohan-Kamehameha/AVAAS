from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_capture_state_browser_contracts() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    result = subprocess.run(
        [node, "--test", "tests/browser/capture-state.test.mjs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_studio_uses_text_nodes_and_safe_training_default() -> None:
    html = (ROOT / "webui/static/index.html").read_text(encoding="utf-8")
    assert "innerHTML" not in html
    auto_train = html.split('id="auto-train"', 1)[0].rsplit("<input", 1)[-1]
    assert "checked" not in auto_train
    assert "recording_accent" in html
    assert "synthesis_presentation" in html

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_list_mode_uses_canonical_identity_variant_ids_without_audio_deps() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/capture.py", "--list", "--section", "sec13"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "sec13_001__satraj" in result.stdout
    assert "sec13_001__piranesi" in result.stdout
    assert "[satraj]" not in result.stdout
    assert "[piranesi]" not in result.stdout
    assert "80/80" not in result.stdout
    assert result.stdout.rstrip().endswith("0/80 recorded")


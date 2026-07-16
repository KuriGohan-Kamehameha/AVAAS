from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_container_copies_and_materializes_prompt_assets_before_startup() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "COPY prompts /app/prompts" in dockerfile
    assert "COPY scripts/materialize_prompt_corpora.py" in dockerfile
    assert "materialize_prompt_corpora.py --fetch" in dockerfile
    assert "python -m webui.prompts --write" in dockerfile
    assert "COPY RECORDING-SCRIPT.md" not in dockerfile
    assert dockerfile.index("materialize_prompt_corpora.py --fetch") < dockerfile.index("ENTRYPOINT")


def test_harvard_text_is_a_build_asset_not_public_git_content() -> None:
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "prompts/corpora/harvard_100.txt"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    ignored = (ROOT / ".gitignore").read_text()
    assert "prompts/corpora/harvard_100.txt" in ignored
    docker_ignored = (ROOT / ".dockerignore").read_text()
    assert "prompts/corpora/harvard_100.txt" in docker_ignored


def test_complete_generated_script_is_not_redistributed_by_git() -> None:
    import subprocess

    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "RECORDING-SCRIPT.md"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0
    assert "RECORDING-SCRIPT.md" in (ROOT / ".gitignore").read_text()
    assert "RECORDING-SCRIPT.md" in (ROOT / ".dockerignore").read_text()

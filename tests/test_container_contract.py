from __future__ import annotations

from pathlib import Path

import yaml


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


def test_release_image_is_reproducible_nonroot_and_health_checked() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert (
        "FROM python:3.12.13-slim-bookworm@sha256:"
        "d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b"
    ) in dockerfile
    assert "ffmpeg=7:5.1.9-0+deb12u1" in dockerfile
    assert "libsndfile1=1.2.0-1+deb12u1" in dockerfile
    assert "pip install --no-cache-dir --require-hashes -r /app/requirements.lock" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert "HEALTHCHECK" in dockerfile and "/readyz" in dockerfile
    assert "--limit-concurrency" in dockerfile
    assert "--timeout-keep-alive" in dockerfile

    runtime_lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    development_lock = (ROOT / "requirements-dev.lock").read_text(encoding="utf-8")
    assert "--hash=sha256:" in runtime_lock
    assert "--hash=sha256:" in development_lock


def test_compose_confines_the_canary_to_explicit_writable_storage() -> None:
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    service = compose["services"]["studio"]
    assert service["user"] == "10001:10001"
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]
    assert service["pids_limit"] <= 256
    assert service["volumes"] == [
        {
            "type": "bind",
            "source": "${AVAAS_DATA_DIR:-./data}",
            "target": "/app/data",
            "bind": {"create_host_path": False},
        }
    ]
    assert service["ports"] == ["127.0.0.1:${AVAAS_PORT:-8732}:8731"]
    assert any(str(item).startswith("/tmp:") for item in service["tmpfs"])

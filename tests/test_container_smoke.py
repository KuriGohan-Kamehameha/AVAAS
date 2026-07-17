from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


@pytest.mark.container
def test_built_container_starts_with_compiled_identity_corpus() -> None:
    """Opt-in image smoke: AVAAS_RUN_CONTAINER_TESTS=1 pytest -m container."""
    if os.environ.get("AVAAS_RUN_CONTAINER_TESTS") != "1":
        pytest.skip("set AVAAS_RUN_CONTAINER_TESTS=1 to build and start the image")

    suffix = str(os.getpid())
    image = f"avaas-corpus-smoke:{suffix}"
    container = f"avaas-corpus-smoke-{suffix}"
    volume = f"avaas-corpus-smoke-data-{suffix}"
    try:
        _run("docker", "build", "--quiet", "--tag", image, ".", timeout=900)
        _run("docker", "volume", "create", volume)
        _run(
            "docker",
            "run",
            "--detach",
            "--name",
            container,
            "--user",
            "10001:10001",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            "256",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=256m,mode=1777",
            "--mount",
            f"source={volume},target=/app/data",
            "--publish",
            "127.0.0.1::8731",
            image,
        )
        port_output = _run("docker", "port", container, "8731/tcp").stdout.strip()
        port = int(port_output.rsplit(":", 1)[1])
        deadline = time.monotonic() + 60
        response: dict | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/api/script", timeout=2
                ) as handle:
                    if handle.status == 200:
                        response = json.load(handle)
                        break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.25)
        assert response is not None, _run("docker", "logs", container).stdout[-8_192:]
        sections = response["sections"]
        assert len(sections) == 13
        identity_prompts = next(item for item in sections if item["section"] == "sec13")[
            "prompts"
        ]
        assert len(identity_prompts) == 80
        assert {item["identity"] for item in identity_prompts} == {"satraj", "piranesi"}
        assert all("Sat" not in item["text"].split() for item in identity_prompts)
        readiness = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/readyz", timeout=2).read()
        )
        assert readiness["ready"] is True
        assert readiness["prompt_count"] == 932
        assert _run("docker", "inspect", "--format", "{{.Config.User}}", container).stdout.strip() == "10001:10001"
        _run(
            "docker",
            "exec",
            container,
            "sh",
            "-c",
            'test ! -w /app && test -w /app/data && test "$(id -u)" = 10001',
        )
        _run(
            "docker",
            "exec",
            container,
            "python",
            "/app/scripts/backup.py",
            "/app",
            "/app/data/backups",
            "--backup-id",
            "container-smoke",
        )
        _run(
            "docker",
            "exec",
            container,
            "python",
            "/app/scripts/restore_verify.py",
            "/app/data/backups/container-smoke",
            "/app/data/restored-smoke",
        )
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        subprocess.run(
            ["docker", "volume", "rm", "--force", volume],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        subprocess.run(
            ["docker", "image", "rm", "--force", image],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )

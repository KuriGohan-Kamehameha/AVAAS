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
    try:
        _run("docker", "build", "--quiet", "--tag", image, ".", timeout=900)
        _run(
            "docker",
            "run",
            "--detach",
            "--name",
            container,
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
    finally:
        subprocess.run(
            ["docker", "rm", "--force", container],
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

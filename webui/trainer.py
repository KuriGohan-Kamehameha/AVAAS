#!/usr/bin/env python3
"""Training auto-trigger.

When the corpus reaches the readiness gate (corpus.TARGET_MIN clean minutes with
every section >= 80% covered) and auto-train is enabled, this launches the
training orchestration exactly once. State is persisted to
data/training_state.json so a server restart never double-launches.

The heavy lift (Chatterbox LoRA, ~12-22 h) runs on the training host; this module only
syncs data and kicks off training/launch_training.sh, then tracks the job.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

STATE_IDLE = "idle"
STATE_READY = "ready"
STATE_LAUNCHED = "launched"
STATE_ERROR = "error"
LAUNCH_TIMEOUT = 60          # seconds for the launcher to detach; P10-bounded


def _state_path(root: Path) -> Path:
    return root / "data" / "training_state.json"


def read_state(root: Path) -> dict:
    p = _state_path(root)
    if not p.exists():
        return {"state": STATE_IDLE, "launched_at": None, "log": None, "detail": ""}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {"state": STATE_IDLE, "launched_at": None, "log": None, "detail": ""}


def _write_state(root: Path, state: dict) -> None:
    p = _state_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(p)


def already_launched(root: Path) -> bool:
    return read_state(root)["state"] in (STATE_LAUNCHED,)


def launch(root: Path, now_iso: str, force: bool = False) -> dict:
    """Run the launcher script once. Idempotent unless force=True."""
    if already_launched(root) and not force:
        return read_state(root)

    script = root / "training" / "launch_training.sh"
    log = root / "data" / "training_launch.log"
    if not script.exists():
        st = {"state": STATE_ERROR, "launched_at": now_iso, "log": str(log),
              "detail": f"launcher missing: {script}"}
        _write_state(root, st)
        return st

    try:
        with open(log, "ab") as lf:
            # Detached: the launcher rsyncs + ssh-starts the remote run, then exits.
            subprocess.run(["bash", str(script)], stdout=lf, stderr=lf,
                           timeout=LAUNCH_TIMEOUT, check=True)
        st = {"state": STATE_LAUNCHED, "launched_at": now_iso, "log": str(log),
              "detail": "training launched on the training host"}
    except subprocess.TimeoutExpired:
        st = {"state": STATE_LAUNCHED, "launched_at": now_iso, "log": str(log),
              "detail": "launcher still running at timeout (likely rsync) — detached"}
    except subprocess.CalledProcessError as e:
        st = {"state": STATE_ERROR, "launched_at": now_iso, "log": str(log),
              "detail": f"launcher exit {e.returncode} — see log"}
    _write_state(root, st)
    return st


def maybe_launch(root: Path, prog: dict, auto: bool, now_iso: str) -> dict:
    """Decide + act based on a progress dict. Returns the training state."""
    cur = read_state(root)
    if cur["state"] == STATE_LAUNCHED:
        return cur
    if not prog.get("ready"):
        if cur["state"] != STATE_IDLE:
            _write_state(root, {**cur, "state": STATE_IDLE,
                                "detail": "corpus no longer at readiness gate"})
        return read_state(root)
    if not auto:
        st = {**cur, "state": STATE_READY,
              "detail": "ready — auto-train off; POST /api/train to start"}
        _write_state(root, st)
        return st
    return launch(root, now_iso)

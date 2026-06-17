#!/usr/bin/env bash
# Launch the AVAAS web UI.  Open http://localhost:8731 in a browser
# (localhost is a secure context, so mic access works without https).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${VOICE_STUDIO_PORT:-8731}"
cd "$ROOT"
[[ -d .venv ]] || { echo "no .venv — run: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 3; }
exec .venv/bin/python -m uvicorn webui.server:app --host 127.0.0.1 --port "$PORT" "$@"

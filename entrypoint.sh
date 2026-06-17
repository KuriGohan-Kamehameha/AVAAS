#!/bin/sh
set -e
mkdir -p /app/data/raw /app/data/processed /app/data/transcripts
[ -f /app/data/studio_settings.json ] || printf '%s\n' '{"auto_train": false, "denoise": true, "whisper_model": "base.en"}' > /app/data/studio_settings.json
exec uvicorn webui.server:app --host 0.0.0.0 --port 8731

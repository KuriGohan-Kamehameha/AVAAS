#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

DATA_ROOT=/app/data
if [ -L "$DATA_ROOT" ] || [ ! -d "$DATA_ROOT" ] || [ ! -w "$DATA_ROOT" ]; then
    echo "AVAAS data root must be a writable, non-symlink directory" >&2
    exit 1
fi

mkdir -p \
    "$DATA_ROOT/audio" \
    "$DATA_ROOT/cache/huggingface" \
    "$DATA_ROOT/captures" \
    "$DATA_ROOT/derivatives" \
    "$DATA_ROOT/migration" \
    "$DATA_ROOT/raw" \
    "$DATA_ROOT/staging" \
    "$DATA_ROOT/training-jobs"

exec "$@"

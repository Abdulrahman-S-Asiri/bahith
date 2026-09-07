#!/bin/sh
set -eu

if [ -z "${BAHITH_ALLOWED_HOSTS:-}" ]; then
    echo 'Set BAHITH_ALLOWED_HOSTS to the exact public hostname before starting.' >&2
    exit 1
fi

exec python -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-7860}" \
    --workers 1 --no-access-log --no-proxy-headers

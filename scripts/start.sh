#!/usr/bin/env bash
# Starts the backend (FastAPI on 127.0.0.1:8765) and the Vite dev UI (http://127.0.0.1:5173).
# Use `./scripts/start.sh --prod` to build the UI once and serve everything from the backend.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
if ! "$PY" -c "import fastapi, uvicorn, psutil, numpy" 2>/dev/null; then
  echo "Backend dependencies missing — run ./scripts/setup.sh first"; exit 1
fi
[ -d node_modules ] || npm install

if [ "${1:-}" = "--prod" ]; then
  npm run build
  echo "==> Studio: http://127.0.0.1:${AI_STUDIO_PORT:-8765}"
  exec "$PY" -m server
fi

"$PY" -m server &
BACKEND=$!
trap 'kill $BACKEND 2>/dev/null' EXIT INT TERM
echo "==> Studio UI: http://127.0.0.1:5173  (backend PID $BACKEND)"
npx vite

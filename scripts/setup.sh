#!/usr/bin/env bash
# One-time setup: Python venv for the backend + Unsloth training stack, and the web UI deps.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  echo "==> Creating Python virtual environment (.venv)"
  "$PY" -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip wheel

echo "==> Installing backend requirements"
pip install -r server/requirements.txt

if [ "${AI_STUDIO_AUTO_INSTALL:-1}" != "0" ]; then
  echo "==> Installing the Unsloth training stack (torch, transformers, trl, peft, bitsandbytes, ...)"
  python -c "from server.environment import install_command; import subprocess, sys; sys.exit(subprocess.call(install_command()))" \
    || echo "!! Unsloth install failed — fix it later from the Studio's 'Environment & Unsloth' tab or the terminal."
fi

echo "==> Installing web UI dependencies"
npm install

echo
echo "Done. Start the studio with:  ./scripts/start.sh"

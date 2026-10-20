#!/usr/bin/env bash
# Creates an isolated Python environment with Unsloth + the Hugging Face fine-tuning stack.
# Launched by AI Studio ("Install Fine-Tuning Toolkit") but safe to run by hand too:
#   bash server/scripts/install_finetune_env.sh
set -euo pipefail

export STUDIO_HOME="${STUDIO_HOME:-$HOME/.ai_studio}"
VENV="$STUDIO_HOME/venv"
TOTAL=6
step() { echo ""; echo "==> [$1/$TOTAL] $2"; }

mkdir -p "$STUDIO_HOME"
echo "AI Studio fine-tuning toolkit installer"
echo "Install location: $VENV"

# ---------------------------------------------------------------------------
step 1 "Checking system: GPU driver, CUDA and Python"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "WARNING: nvidia-smi not found. Unsloth needs an NVIDIA GPU with a working driver for training."
  echo "         Install the NVIDIA driver first (e.g. 'sudo ubuntu-drivers autoinstall'), then re-run."
fi

PY=""
for cand in python3.12 python3.11 python3.13 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then PY="$(command -v "$cand")"; break; fi
done
if [ -z "$PY" ]; then
  echo "ERROR: Python 3.10-3.13 not found. Install it with: sudo apt install python3 python3-venv python3-pip"
  exit 1
fi
echo "Using Python: $PY ($("$PY" --version 2>&1))"

USE_UV=0
if command -v uv >/dev/null 2>&1; then USE_UV=1; echo "Found uv: $(uv --version) (fast installer will be used)"; fi

# ---------------------------------------------------------------------------
step 2 "Creating virtual environment"
if [ -x "$VENV/bin/python" ]; then
  echo "Virtual environment already exists, reusing it."
elif [ "$USE_UV" = 1 ]; then
  uv venv --python "$PY" "$VENV"
else
  if ! "$PY" -m venv "$VENV"; then
    echo "ERROR: could not create venv. On Debian/Ubuntu run: sudo apt install python3-venv"
    exit 1
  fi
fi

pipi() {
  if [ "$USE_UV" = 1 ]; then
    uv pip install --python "$VENV/bin/python" "$@"
  else
    "$VENV/bin/python" -m pip install --progress-bar off "$@"
  fi
}

# ---------------------------------------------------------------------------
step 3 "Upgrading pip / setuptools / wheel"
if [ "$USE_UV" = 1 ]; then
  pipi --upgrade pip setuptools wheel
else
  "$VENV/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
  pipi --upgrade pip setuptools wheel
fi

# ---------------------------------------------------------------------------
step 4 "Installing Unsloth (pulls PyTorch + CUDA wheels, this is the big download ~3-5 GB)"
pipi --upgrade unsloth

# ---------------------------------------------------------------------------
step 5 "Installing training stack: trl, peft, accelerate, bitsandbytes, datasets, transformers"
pipi --upgrade trl peft accelerate bitsandbytes datasets transformers sentencepiece protobuf hf_transfer

# ---------------------------------------------------------------------------
step 6 "Verifying installation"
"$VENV/bin/python" - <<'PY'
import importlib, json, os
report = {}
for mod in ["torch", "transformers", "trl", "peft", "accelerate", "bitsandbytes", "datasets"]:
    try:
        m = importlib.import_module(mod)
        report[mod] = getattr(m, "__version__", "ok")
        print(f"  [✓] {mod:<13} {report[mod]}")
    except Exception as e:
        report[mod] = None
        print(f"  [✗] {mod:<13} {e}")

import torch
cuda = torch.cuda.is_available()
report["cuda"] = cuda
print(f"  [{'✓' if cuda else '!'}] CUDA available: {cuda}" + (f" ({torch.cuda.get_device_name(0)})" if cuda else ""))

try:
    import unsloth  # noqa: F401  (import fails without a GPU, which is expected on CPU-only boxes)
    report["unsloth"] = getattr(unsloth, "__version__", "ok")
    print(f"  [✓] unsloth       {report['unsloth']}")
except Exception as e:
    report["unsloth"] = None
    print(f"  [!] unsloth import: {e}")

path = os.path.join(os.environ.get("STUDIO_HOME", os.path.expanduser("~/.ai_studio")), "env_ready.json")
with open(path, "w") as f:
    json.dump(report, f, indent=2)
print(f"Report written to {path}")
PY

echo ""
echo "Successfully installed the AI Studio fine-tuning toolkit."
echo "Python for training: $VENV/bin/python"

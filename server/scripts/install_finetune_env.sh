#!/usr/bin/env bash
# Creates an isolated Python environment with Unsloth + the Hugging Face fine-tuning stack,
# matched to THIS machine's NVIDIA driver. Launched by AI Studio ("Install Everything
# Automatically") but safe to run by hand too:
#   bash server/scripts/install_finetune_env.sh            # install / repair
#   FRESH=1 bash server/scripts/install_finetune_env.sh    # wipe and reinstall from scratch
#
# Why the matching matters: the default PyTorch wheel on PyPI targets the newest CUDA
# (e.g. cu130 needs driver >= 580). On an older driver torch imports fine but
# torch.cuda.is_available() is False ("NVIDIA driver on your system is too old").
set -euo pipefail

export STUDIO_HOME="${STUDIO_HOME:-$HOME/.ai_studio}"
VENV="$STUDIO_HOME/venv"
PY_VERSION="3.12"   # every CUDA build of torch 2.4+ and xformers ships wheels for 3.12
TOTAL=7
step() { echo ""; echo "==> [$1/$TOTAL] $2"; }
fail() { echo ""; echo "ERROR: $*"; exit 1; }

mkdir -p "$STUDIO_HOME"
echo "AI Studio fine-tuning toolkit installer"
echo "Install location: $VENV"

# ---------------------------------------------------------------------------
step 1 "Checking GPU, NVIDIA driver and disk space"
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found. Install the NVIDIA proprietary driver first:
  Ubuntu/Kubuntu/Mint : sudo ubuntu-drivers install
  Fedora              : sudo dnf install akmod-nvidia   (needs RPM Fusion)
  Arch / Manjaro      : sudo pacman -S nvidia nvidia-utils
then reboot and click Install again."

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || fail "nvidia-smi is installed but cannot talk to the GPU. Reboot (or check 'lsmod | grep nvidia'), then retry."
DRIVER_CUDA="$(nvidia-smi | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' || true)"
[ -n "$DRIVER_CUDA" ] || fail "Could not read the CUDA version supported by the driver from nvidia-smi."
CUDA_NUM=$(( ${DRIVER_CUDA%%.*} * 10 + ${DRIVER_CUDA##*.} ))
echo "Driver supports CUDA up to: $DRIVER_CUDA"

# PyTorch CUDA builds this driver can run, best first.
if   [ "$CUDA_NUM" -ge 130 ]; then CANDIDATES="cu130 cu129 cu128 cu126"
elif [ "$CUDA_NUM" -ge 129 ]; then CANDIDATES="cu129 cu128 cu126"
elif [ "$CUDA_NUM" -ge 128 ]; then CANDIDATES="cu128 cu126 cu124"
elif [ "$CUDA_NUM" -ge 126 ]; then CANDIDATES="cu126 cu124"
elif [ "$CUDA_NUM" -ge 124 ]; then CANDIDATES="cu124 cu121 cu118"
elif [ "$CUDA_NUM" -ge 121 ]; then CANDIDATES="cu121 cu118"
elif [ "$CUDA_NUM" -ge 118 ]; then CANDIDATES="cu118"
else fail "Driver CUDA $DRIVER_CUDA is too old for PyTorch 2.4+. Update the NVIDIA driver (535 or newer) and retry."
fi
echo "Compatible PyTorch builds: $CANDIDATES"
if [ "$CUDA_NUM" -lt 128 ]; then
  echo "TIP: this works fine, but updating to NVIDIA driver 570+ unlocks newer PyTorch / Unsloth features."
fi

FREE_GB=$(( $(df -Pk "$STUDIO_HOME" | awk 'NR==2 {print $4}') / 1024 / 1024 ))
echo "Free disk space in $STUDIO_HOME: ${FREE_GB} GB"
[ "$FREE_GB" -ge 12 ] || fail "At least 12 GB free disk space is needed (PyTorch + CUDA libraries + model downloads)."

# ---------------------------------------------------------------------------
step 2 "Getting the uv package manager (fast, and provides Python $PY_VERSION)"
export PATH="$STUDIO_HOME/bin:$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$STUDIO_HOME/bin" INSTALLER_NO_MODIFY_PATH=1 sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$STUDIO_HOME/bin" INSTALLER_NO_MODIFY_PATH=1 sh
  else
    fail "Neither curl nor wget is installed. Install one (e.g. sudo apt install curl) and retry."
  fi
fi
command -v uv >/dev/null 2>&1 || fail "uv installation failed."
echo "uv: $(uv --version)"

# ---------------------------------------------------------------------------
step 3 "Preparing Python $PY_VERSION virtual environment"
HEALTHY=0
if [ -x "$VENV/bin/python" ] && [ "${FRESH:-0}" != 1 ]; then
  if "$VENV/bin/python" -c "
import sys, torch
assert sys.version_info[:2] == tuple(map(int, '$PY_VERSION'.split('.')))
assert torch.cuda.is_available()
" >/dev/null 2>&1; then
    HEALTHY=1
    echo "Existing environment is healthy (Python $PY_VERSION, CUDA works) — keeping PyTorch, upgrading the rest."
  else
    echo "Existing environment is broken or mismatched (wrong Python or PyTorch built for the wrong CUDA) — rebuilding it."
  fi
fi
if [ "$HEALTHY" = 0 ]; then
  rm -rf "$VENV"
  uv venv --python "$PY_VERSION" "$VENV"   # downloads a standalone Python if the system lacks 3.12
fi
PY="$VENV/bin/python"
echo "Python: $("$PY" --version)"

# ---------------------------------------------------------------------------
step 4 "Installing PyTorch built for your driver (CUDA $DRIVER_CUDA) — ~2.5 GB download"
if [ "$HEALTHY" = 1 ]; then
  echo "Keeping $("$PY" -c 'import torch; print(torch.__version__)')"
else
  INSTALLED=""
  for cu in $CANDIDATES; do
    echo "Trying PyTorch index: https://download.pytorch.org/whl/$cu"
    if uv pip install --python "$PY" "torch>=2.4,<2.15" torchvision --index-url "https://download.pytorch.org/whl/$cu"; then
      INSTALLED="$cu"; break
    fi
    echo "No usable build on $cu, trying the next one..."
  done
  [ -n "$INSTALLED" ] || fail "Could not install a CUDA build of PyTorch for driver CUDA $DRIVER_CUDA."
fi

TORCH_VER="$("$PY" -c 'import torch; print(torch.__version__.split("+")[0])')"
VISION_VER="$("$PY" -c 'import torchvision; print(torchvision.__version__.split("+")[0])')"
TORCH_MINOR="$(echo "$TORCH_VER" | cut -d. -f2)"
"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('PyTorch', torch.__version__, '| CUDA OK on', torch.cuda.get_device_name(0))" \
  || fail "PyTorch $TORCH_VER cannot use the GPU. Run 'nvidia-smi' to confirm the driver works, then click Install again."

# Pin torch so nothing later swaps it for the PyPI default build, and pick the torchao
# release made for this torch (unsloth-zoo needs torchao>=0.13; newer ones spam warnings on old torch).
CONSTRAINTS="$STUDIO_HOME/constraints.txt"
{
  echo "torch==$TORCH_VER"
  echo "torchvision==$VISION_VER"
  if   [ "$TORCH_MINOR" -le 8 ]; then echo "torchao==0.13.0"
  elif [ "$TORCH_MINOR" -eq 9 ]; then echo "torchao==0.14.1"
  fi
} > "$CONSTRAINTS"
echo "Pinned versions:"; sed 's/^/  /' "$CONSTRAINTS"

# ---------------------------------------------------------------------------
step 5 "Installing Unsloth + training stack (trl, peft, accelerate, bitsandbytes, datasets, transformers)"
UPGRADES=()
for p in unsloth unsloth_zoo trl peft accelerate bitsandbytes datasets transformers xformers torchao; do UPGRADES+=(--upgrade-package "$p"); done
uv pip install --python "$PY" -c "$CONSTRAINTS" "${UPGRADES[@]}" \
  unsloth unsloth_zoo trl peft accelerate bitsandbytes datasets transformers xformers torchao \
  sentencepiece protobuf hf_transfer

# ---------------------------------------------------------------------------
step 6 "Checking nothing replaced the CUDA build of PyTorch"
NOW_TORCH="$("$PY" -c 'import torch; print(torch.__version__)')"
echo "torch: $NOW_TORCH"
[ "${NOW_TORCH%%+*}" = "$TORCH_VER" ] || fail "torch changed to $NOW_TORCH during install. Re-run with FRESH=1."

# ---------------------------------------------------------------------------
step 7 "Verifying on the GPU (CUDA kernel, 4-bit quantization, Unsloth import)"
"$PY" - <<'PY'
import importlib, json, logging, os, sys, warnings

warnings.filterwarnings("ignore")          # third-party deprecation noise is not an install problem
logging.disable(logging.WARNING)
report, problems = {}, []

for mod in ["torch", "torchvision", "transformers", "trl", "peft", "accelerate", "bitsandbytes", "datasets", "torchao", "xformers"]:
    try:
        report[mod] = getattr(importlib.import_module(mod), "__version__", "ok")
        print(f"  [✓] {mod:<13} {report[mod]}")
    except Exception as e:
        report[mod] = None
        problems.append(f"{mod}: {e}")
        print(f"  [✗] {mod:<13} {e}")

import torch
try:
    x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize(); (x @ x).sum().item()
    report["cuda"] = True
    report["gpu"] = torch.cuda.get_device_name(0)
    report["torch_cuda"] = torch.version.cuda
    print(f"  [✓] CUDA matmul   {report['gpu']} (torch CUDA {torch.version.cuda})")
except Exception as e:
    report["cuda"] = False
    problems.append(f"CUDA: {e}")
    print(f"  [✗] CUDA          {e}")

if report["cuda"]:
    try:
        import bitsandbytes.functional as F
        q, state = F.quantize_4bit(torch.randn(256, 256, device="cuda", dtype=torch.float16), quant_type="nf4")
        F.dequantize_4bit(q, state)
        report["bnb_4bit"] = True
        print("  [✓] 4-bit NF4     bitsandbytes quantize/dequantize on GPU")
    except Exception as e:
        report["bnb_4bit"] = False
        problems.append(f"bitsandbytes 4-bit: {e}")
        print(f"  [✗] 4-bit NF4     {e}")

try:
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel  # noqa: F401
    report["unsloth"] = getattr(unsloth, "__version__", "ok")
    print(f"  [✓] unsloth       {report['unsloth']}")
except Exception as e:
    report["unsloth"] = None
    problems.append(f"unsloth: {e}")
    print(f"  [✗] unsloth       {e}")

report["ok"] = not problems
path = os.path.join(os.environ["STUDIO_HOME"], "env_ready.json")
with open(path, "w") as f:
    json.dump(report, f, indent=2)
print(f"Report written to {path}")

if problems:
    print("\nERROR: the toolkit is installed but these checks failed:")
    for p in problems:
        print("  - " + p)
    sys.exit(1)
PY

echo ""
echo "Successfully installed the AI Studio fine-tuning toolkit — all GPU checks passed."
echo "Python for training: $PY"

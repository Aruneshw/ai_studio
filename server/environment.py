"""Python package status for the training stack + automatic Unsloth installation."""
import importlib.metadata as md
import json
import shutil
import sys
import urllib.request

from . import jobs
from .config import OLLAMA_URL, ROOT_DIR
from .system_info import cuda_version, gpus

TRAINING_PACKAGES = ["unsloth", "unsloth_zoo", "torch", "transformers", "trl", "peft",
                     "accelerate", "bitsandbytes", "datasets", "xformers", "triton",
                     "sentencepiece", "protobuf", "hf_transfer"]
OPTIONAL_PACKAGES = ["llama_cpp_python", "huggingface_hub", "tokenizers"]

# What "install all Unsloth packages" means; pip resolves compatible versions from unsloth's pins.
INSTALL_SPEC = ["unsloth", "unsloth_zoo", "trl", "peft", "accelerate", "bitsandbytes",
                "datasets", "transformers", "sentencepiece", "protobuf", "hf_transfer"]

def version(pkg):
    for name in (pkg, pkg.replace("_", "-")):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            continue
    return None


def ollama_status():
    info = {"installed": shutil.which("ollama") is not None, "running": False, "version": None,
            "url": OLLAMA_URL}
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/version", timeout=1.5) as r:
            info["running"] = True
            info["version"] = json.loads(r.read()).get("version")
    except Exception:  # noqa: BLE001
        pass
    return info


def install_job():
    for j in jobs.all_jobs():
        if j.kind == "install":
            return j
    return None


def status():
    install = install_job()
    return {
        "python": sys.executable,
        "pythonVersion": sys.version.split()[0],
        "virtualenv": sys.prefix != sys.base_prefix,
        "packages": {p: version(p) for p in TRAINING_PACKAGES},
        "optional": {p: version(p) for p in OPTIONAL_PACKAGES},
        "unslothReady": version("unsloth") is not None,
        "gpu": bool(gpus()),
        "cuda": cuda_version(),
        "nvcc": shutil.which("nvcc") is not None,
        "ollama": ollama_status(),
        "install": install.summary() if install else None,
        "installCommand": install_command_str(),
    }


def install_command():
    return [sys.executable, "-m", "pip", "install", "--upgrade", *INSTALL_SPEC]


def install_command_str():
    return " ".join(install_command())


def start_install():
    current = install_job()
    if current and current.status == "running":
        return current
    return jobs.start("install", "Install Unsloth training stack", install_command(), cwd=str(ROOT_DIR))


def auto_install():
    """Called on startup: install the Unsloth stack if it's missing."""
    if version("unsloth") is None:
        return start_install()
    return None

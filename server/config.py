"""Paths and settings shared by the backend."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("AI_STUDIO_DATA", ROOT_DIR / "data")).expanduser()
DATASETS_DIR = DATA_DIR / "datasets"
RUNS_DIR = DATA_DIR / "runs"
EXPORTS_DIR = DATA_DIR / "exports"
LIBRARY_FILE = DATA_DIR / "library.json"

HOST = os.environ.get("AI_STUDIO_HOST", "127.0.0.1")
PORT = int(os.environ.get("AI_STUDIO_PORT", "8765"))
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if not OLLAMA_URL.startswith("http"):
    OLLAMA_URL = "http://" + OLLAMA_URL

# Install Unsloth + training stack automatically on first start (set to 0 to disable).
AUTO_INSTALL = os.environ.get("AI_STUDIO_AUTO_INSTALL", "1") != "0"

for d in (DATA_DIR, DATASETS_DIR, RUNS_DIR, EXPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

"""Download a model repo from the Hugging Face Hub into the standard HF cache."""
import argparse
import os

p = argparse.ArgumentParser()
p.add_argument("repo")
p.add_argument("--include", default="", help="glob, e.g. '*Q4_K_M.gguf' for GGUF repos")
a = p.parse_args()

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
try:
    import hf_transfer  # noqa: F401
except ImportError:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import snapshot_download  # noqa: E402

patterns = [x.strip() for x in a.include.split(",") if x.strip()] or None
if patterns:
    patterns += ["*.json", "tokenizer*", "*.model", "*.jinja"]
print(f"Downloading {a.repo} {'(' + ', '.join(patterns) + ')' if patterns else ''}", flush=True)
path = snapshot_download(a.repo, allow_patterns=patterns)
print(f"Saved to {path}", flush=True)

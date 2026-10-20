"""Export worker: LoRA adapter copy, merged 16-bit/4-bit model, GGUF, and optional Ollama import.

    python -m server.workers.export --source <adapter-or-model-dir> --format gguf --quant q4_k_m --out <dir>
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--format", required=True, choices=["lora", "merged_16bit", "merged_4bit", "gguf"])
p.add_argument("--quant", default="q4_k_m")
p.add_argument("--out", required=True)
p.add_argument("--ollama-name", default="")
a = p.parse_args()

src, out = Path(a.source), Path(a.out)
out.mkdir(parents=True, exist_ok=True)

if a.format == "lora":
    if not (src / "adapter_config.json").exists():
        sys.exit("Source is not a LoRA adapter directory")
    shutil.copytree(src, out, dirs_exist_ok=True)
    print(f"Adapter copied to {out}", flush=True)
    sys.exit(0)

print("Importing Unsloth...", flush=True)
from unsloth import FastLanguageModel  # noqa: E402

print(f"Loading {src}", flush=True)
model, tok = FastLanguageModel.from_pretrained(str(src), max_seq_length=4096, dtype=None,
                                               load_in_4bit=(a.format == "merged_4bit"))
if a.format in ("merged_16bit", "merged_4bit"):
    model.save_pretrained_merged(str(out), tok, save_method=a.format)
    print(f"Merged model written to {out}", flush=True)
else:
    model.save_pretrained_gguf(str(out), tok, quantization_method=a.quant)
    print(f"GGUF written to {out}", flush=True)
    if a.ollama_name:
        ggufs = sorted(Path(out).rglob("*.gguf"), key=lambda f: f.stat().st_mtime, reverse=True)
        ggufs = [g for g in ggufs if "mmproj" not in g.name] or ggufs
        if not ggufs:
            sys.exit("No .gguf produced; cannot create Ollama model")
        modelfile = out / "Modelfile"
        if not modelfile.exists():
            modelfile.write_text(f"FROM {ggufs[0].resolve()}\n")
        print(f"$ ollama create {a.ollama_name} -f {modelfile}", flush=True)
        r = subprocess.run(["ollama", "create", a.ollama_name, "-f", str(modelfile)], cwd=str(out))
        sys.exit(r.returncode)

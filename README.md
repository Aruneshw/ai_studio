# AI Model Studio

A local workstation for **inspecting, visualizing, fine-tuning (Unsloth) and chatting with the LLMs installed on your machine**.
Everything shown in the UI comes from your real files, hardware and processes. There is no demo or mock data.

```
React + Three.js UI (Vite)  ──HTTP / WebSocket──▶  FastAPI backend (python -m server)
                                                     ├─ model discovery + safetensors/GGUF readers (numpy)
                                                     ├─ PTY terminal (/ws/terminal)
                                                     ├─ job runner: pip installs, downloads, training, export
                                                     └─ inference: Ollama / Unsloth+Transformers / llama.cpp
```

## Quick start (Linux, NVIDIA GPU recommended)

```bash
./scripts/setup.sh     # creates .venv, installs the backend + the Unsloth training stack, npm install
./scripts/start.sh     # backend on 127.0.0.1:8765 + UI on http://127.0.0.1:5173
# or: ./scripts/start.sh --prod   (builds the UI and serves everything from http://127.0.0.1:8765)
```

**Unsloth is installed by default.** `setup.sh` installs it, and every time the backend starts it checks for it and
installs it in the background if it's missing. The command it runs is
`pip install --upgrade unsloth unsloth_zoo trl peft accelerate bitsandbytes datasets transformers sentencepiece protobuf hf_transfer`.
You can watch the log under **Terminal → Environment & Unsloth**. Set `AI_STUDIO_AUTO_INSTALL=0` to turn this off.

## Where models come from

Only models that actually exist on this machine are listed:

| Source | Location scanned |
| --- | --- |
| Ollama | `$OLLAMA_MODELS`, `~/.ollama/models`, `/usr/share/ollama/.ollama/models` (GGUF blobs read via manifests) |
| Hugging Face | `$HF_HUB_CACHE` / `~/.cache/huggingface/hub` (safetensors models, adapters, GGUF files) |
| LM Studio | `~/.lmstudio/models`, `~/.cache/lm-studio/models` |
| Local paths | anything you import in **Model Library** (a file, a model folder, or a folder to watch) |
| Studio fine-tunes | adapters/models produced by your training runs in `data/runs/` |

You can also download from the Hugging Face Hub or run `ollama pull` from the Model Library.

## Screens

1. **Dashboard**: live CPU/RAM/disk/GPU, environment status, and a mini 3D network of the active model.
2. **Model Library**: discovered models, import by path with a folder browser, HF / Ollama downloads.
3. **Model Overview**: real config, parameter breakdown per layer, module and dtype, plus the full tensor table.
4. **Neural Network (3D)**: a live WebGL network with bloom. Each glass slab is a real layer. Each neuron's glow is the
   mean |weight| of a block of that layer's tensor, read from the safetensors/GGUF file. Pulses travel along the connections and light up neurons as they arrive.
5. **Layer view**: the tensors in a block, with heatmaps and stats.
6. **Tensor inspector**: 3D surface, heatmap, histogram and raw values of any tensor. GGUF is dequantized for
   F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K.
7. **Tokenizer**: the model's real tokenizer (`tokenizer.json`, or rebuilt from GGUF vocab), plus a 3D PCA of real embedding rows.
8. **Datasets**: upload jsonl/json/csv/txt/parquet. The format is auto-detected (chat, ShareGPT, Alpaca, prompt/completion, Q&A, text, DPO).
9. **Fine-tuning**: QLoRA / LoRA / full / DPO via Unsloth, with target modules detected from the model and a VRAM estimate.
   It generates the exact command, which you approve or edit before it runs.
10. **Training live**: loss / eval / LR / grad-norm charts, log tail and GPU memory from the running process. Stop
    gracefully (saves the adapter) or kill.
11. **Before vs after**: the same prompt through the base model (adapter disabled) and the fine-tuned model, plus ‖ΔW‖ per layer×module
    and a 3D ΔW = B·A·α/r surface.
12. **Terminal**: a real interactive shell (xterm.js + PTY). `python`/`pip` point at the Studio's venv.
    Also has process history and an environment doctor.
13. **GPU & System**: live nvidia-smi / psutil charts.
14. **Experiments**: every run with its config, metrics and outputs, and loss-curve comparison.
15. **Chat**: Ollama models via the Ollama API, HF models/adapters via Unsloth or Transformers, `.gguf` via llama-cpp-python.
16. **Export**: LoRA adapter, merged 16/4-bit, GGUF (q4_k_m, q8_0, …), and optional `ollama create`.

## Security notes

The backend can run shell commands, so it is locked to your machine:
- it binds to `127.0.0.1` only, and requests whose `Host` isn't localhost are rejected (blocks DNS rebinding);
- state-changing API calls require an `X-AI-Studio: 1` header and a localhost `Origin` (blocks cross-site requests);
- the terminal WebSocket rejects non-localhost `Origin`s (blocks cross-site WebSocket hijacking).

Don't expose port 8765 to a network.

## Layout

```
server/            FastAPI backend
  formats/         GGUF + safetensors readers (header-only, row sampling, dequantization)
  workers/         train.py (Unsloth), export.py, hf_download.py — run as subprocesses
src/               React UI (views/, components/3d/ Three.js scenes)
scripts/           setup.sh, start.sh
data/              runtime data (git-ignored): datasets/, runs/, exports/, library.json
```

## Requirements

- Linux (the terminal uses POSIX PTYs), Python 3.10+, Node 18+
- Training/export: an NVIDIA GPU with a recent driver (Unsloth requires CUDA). Inspection, tokenizer, datasets and
  Ollama chat work without a GPU.

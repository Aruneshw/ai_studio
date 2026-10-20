# Local AI Studio

3D model visualization + **real** LoRA/QLoRA fine-tuning workstation for local models (Ollama + Unsloth).

## Quick start

```bash
npm install
npm run dev          # UI + local backend on http://localhost:5173
ollama serve         # in another terminal, if Ollama isn't already running as a service
```

The backend (`server/`) runs inside the Vite dev server, so `npm run dev` (or `npm run preview`) is all you need.
It only answers same-origin requests to `localhost`, because it executes commands on your machine.

## Fine-tuning workflow

1. **9. Fine-Tuning Setup → Install Everything Automatically**
   Runs `server/scripts/install_finetune_env.sh`, which creates `~/.ai_studio/venv` and installs
   `unsloth`, `torch` (CUDA), `trl`, `peft`, `accelerate`, `bitsandbytes`, `datasets`, `transformers`.
   Output streams live into the terminal panel. The first install downloads ~3–5 GB.
   You can also run the script by hand: `bash server/scripts/install_finetune_env.sh`.
2. **Pick a model.** Every model from `ollama list` appears under *Ollama — local*. Pull new ones from the
   *Pull from Ollama* box (also streamed live).
3. Ollama stores quantized **GGUF** files, which cannot be trained. The studio maps each Ollama tag to the
   matching Hugging Face checkpoint (e.g. `llama3.2:3b` → `unsloth/Llama-3.2-3B-Instruct`, see
   `server/ollamaMap.js`) and downloads those weights on the first run. Edit the repo field for custom models.
4. **Dataset:** a `.jsonl`/`.json` path or a Hugging Face dataset id. Alpaca (`instruction/input/output`),
   chat (`messages`) and raw `text` rows are supported. A 16-row sample ships in `server/sample_data/`.
5. **Start Fine-Tuning** → review/edit the exact command → **Approve & Execute**. Training logs, loss, LR and
   VRAM stream live; the LoRA adapter is saved to `~/.ai_studio/runs/<run-name>/lora_adapter`.
   Fill *Export to Ollama as* to also build a GGUF and register it with `ollama create`.

### Built-in safety net

- **Pre-launch checks** run on every *Start*: model id format and existence on Hugging Face, gated-model token,
  dataset file/rows/format, GPU memory fit, disk space, run/export names and llama.cpp build tools.
  Errors block the launch and each one shows a concrete fix.
- **Best fit for my GPU** picks the largest recommended model that fits your VRAM; models that won't fit are flagged.
- Ollama tags missing from the built-in table are matched to Unsloth checkpoints via Hugging Face search.
- Loaded Ollama models are unloaded from VRAM before training starts.
- The trainer retries CUDA out-of-memory automatically (smaller micro-batch, then shorter context), skips unusable
  dataset rows, falls back to all-linear LoRA targets for unusual architectures, and ends failures with a plain
  "what went wrong / how to fix" card.
- The installer matches PyTorch to your NVIDIA driver's CUDA version and verifies on the GPU before reporting ready.

**12. Linux Terminal** runs any non-interactive command for real, with live output, job history, Stop
(kills the whole process group) and an environment doctor.

## Requirements

- Linux with an NVIDIA GPU + driver (`nvidia-smi` must work). A 6 GB card handles 1–3B models in 4-bit comfortably.
- `curl` or `wget`. The installer fetches [`uv`](https://docs.astral.sh/uv/) and a Python 3.12 runtime itself; no system Python setup needed.
- Node 20+, Ollama.

Environment variables: `STUDIO_HOME` (default `~/.ai_studio`), `OLLAMA_HOST` (default `127.0.0.1:11434`).

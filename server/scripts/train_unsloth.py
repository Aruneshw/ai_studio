"""LoRA / QLoRA supervised fine-tuning with Unsloth, launched from AI Studio.

Dataset formats auto-detected per row (JSONL / JSON file or a Hugging Face dataset id):
  * Alpaca:   {"instruction": ..., "input": ..., "output": ...}
  * Chat:     {"messages": [{"role": "user", "content": ...}, ...]}  (also ShareGPT "conversations")
  * Raw text: {"text": ...}

Machine-readable lines for the UI (everything else is plain log output):
  [[STAGE]]  {"stage": ..., "label": ...}       current phase
  [[METRIC]] {"step": ..., "loss": ...}          training progress
  [[ERROR]]  {"title": ..., "detail": ..., "fix": ...}   friendly failure explanation

Robustness: CUDA out-of-memory during training is retried automatically with a smaller
micro-batch (keeping the effective batch via gradient accumulation), then a shorter context.
Rows that can't be converted are skipped instead of crashing the run.
"""
import argparse
import gc
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import traceback


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AI Studio Unsloth trainer")
    p.add_argument("--model_name", required=True, help="Hugging Face repo id or local path")
    p.add_argument("--dataset", required=True, help="Path to .jsonl/.json or HF dataset id")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--max_steps", type=int, default=-1, help="Overrides epochs when > 0")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    p.add_argument("--load_in_4bit", default="True")
    p.add_argument("--gradient_checkpointing", default="True")
    p.add_argument("--save_steps", type=int, default=0, help="0 disables intermediate checkpoints")
    p.add_argument("--export_ollama", default="", help="If set, export GGUF and register it in Ollama under this name")
    p.add_argument("--gguf_quant", default="q4_k_m")
    return p.parse_args(argv)


def truthy(v):
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def emit(tag, **kw):
    print(f"[[{tag}]] " + json.dumps(kw, default=str), flush=True)


def stage(name, label):
    emit("STAGE", stage=name, label=label)
    print(f"--- {label}", flush=True)


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- errors
def explain_error(exc, args):
    """Map common failures to a plain-language explanation and a concrete fix."""
    name = type(exc).__name__
    msg = str(exc)
    low = msg.lower()
    if "outofmemory" in name.lower() or "out of memory" in low:
        return ("GPU ran out of memory",
                "Even after automatic retries with a smaller batch and context, the model did not fit in VRAM.",
                "Pick a smaller model (click 'Best fit for my GPU'), keep 4-bit on, set Max Sequence Length to 1024, "
                "and close other GPU apps.")
    if "gated" in low or "401" in msg or "access to model" in low or "restricted" in low:
        return ("Model needs Hugging Face access",
                f"{args.model_name} is gated or private.",
                "Accept its license on huggingface.co and log in: in 12. Linux Terminal run "
                "\"$STUDIO_HOME/venv/bin/hf\" auth login --token hf_xxx — or use an unsloth/ mirror (no token needed).")
    if ("repo id must" in low or "repositorynotfound" in name.lower() or "not a valid model identifier" in low
            or "can't find 'adapter_config.json'" in low or "404" in msg):
        return ("Model not found",
                f"'{args.model_name}' is not a model on Hugging Face or a local model folder.",
                "Use an id like unsloth/Qwen3-4B (owner/name), or click 'Best fit for my GPU'.")
    if "no space left" in low:
        return ("Disk full", msg[:300], "Free some disk space (model downloads go to ~/.cache/huggingface) and retry.")
    if any(s in low for s in ("connection", "timed out", "max retries", "name resolution", "offline")):
        return ("Network problem", msg[:300],
                "Check your internet connection. After the first download the weights are cached and work offline.")
    if "dataset" in low or "json" in name.lower() or isinstance(exc, (KeyError, ValueError)) and "row" in low:
        return ("Dataset problem", msg[:400],
                "Each row needs instruction/output (Alpaca), messages (chat) or text. Check the file in 8. Dataset Management.")
    if "cuda" in low and ("driver" in low or "not available" in low or "no kernel image" in low):
        return ("CUDA / driver problem", msg[:400],
                "Re-run 'Install Everything Automatically' — it re-matches PyTorch to your NVIDIA driver.")
    if "unsupported" in low and "architecture" in low or "does not recognize this architecture" in low:
        return ("Model architecture not supported", msg[:400],
                "Update the toolkit (Reinstall / Upgrade) or choose another model from the recommended list.")
    return (f"Training failed ({name})", msg[:500] or name,
            "See the traceback above. If it mentions a package, click 'Reinstall / Upgrade Toolkit'.")


def is_oom(exc):
    return "outofmemory" in type(exc).__name__.lower() or "out of memory" in str(exc).lower()


# --------------------------------------------------------------------------- data
def load_rows(path_or_id):
    from datasets import load_dataset

    expanded = os.path.expanduser(path_or_id)
    if os.path.exists(expanded):
        return load_dataset("json", data_files=os.path.abspath(expanded), split="train")
    return load_dataset(path_or_id, split="train")


def build_text_fn(tokenizer):
    has_template = bool(getattr(tokenizer, "chat_template", None))
    eos = getattr(tokenizer, "eos_token", None) or ""
    role_map = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant",
                "model": "assistant", "bot": "assistant", "system": "system"}

    def chat(msgs):
        if has_template:
            return tokenizer.apply_chat_template(msgs, tokenize=False)
        return "\n".join(f"{m['role']}: {m['content']}" for m in msgs) + eos

    def to_text(row):
        msgs = row.get("messages") or row.get("conversations")
        if msgs:
            norm = []
            for m in msgs:
                role = role_map.get(str(m.get("role") or m.get("from") or "user").lower(), "user")
                content = m.get("content") if m.get("content") is not None else m.get("value")
                if content is None:
                    continue
                norm.append({"role": role, "content": str(content)})
            if not norm:
                return None
            return chat(norm)
        if row.get("instruction") is not None:
            user = str(row["instruction"]) + (f"\n\n{row['input']}" if row.get("input") else "")
            out = str(row.get("output") or row.get("response") or "")
            if has_template:
                return chat([{"role": "user", "content": user}, {"role": "assistant", "content": out}])
            return f"### Instruction:\n{user}\n\n### Response:\n{out}{eos}"
        if row.get("text") is not None:
            return str(row["text"]) + eos
        if row.get("prompt") is not None and (row.get("completion") is not None or row.get("response") is not None):
            out = str(row.get("completion") if row.get("completion") is not None else row.get("response"))
            return chat([{"role": "user", "content": str(row["prompt"])}, {"role": "assistant", "content": out}])
        return None

    return to_text


def prepare_dataset(dataset, to_text):
    """Convert rows to text, skipping (and counting) rows that can't be converted."""
    def convert(row):
        try:
            text = to_text(row)
        except Exception:  # a malformed row must not kill the run
            text = None
        return {"text": text or ""}

    converted = dataset.map(convert, remove_columns=dataset.column_names)
    before = len(converted)
    converted = converted.filter(lambda r: bool(r["text"].strip()))
    skipped = before - len(converted)
    if len(converted) == 0:
        raise ValueError("Dataset problem: no usable rows. Rows need instruction/output, messages, prompt/completion or text fields.")
    return converted, skipped


# --------------------------------------------------------------------------- training
def oom_schedule(batch_size, grad_accum, seq_len, min_seq=256):
    """Yield (batch, grad_accum, seq) settings to try: shrink micro-batch first, then context."""
    b, g, s = batch_size, grad_accum, seq_len
    yield b, g, s
    while b > 1:
        b, g = max(1, b // 2), g * 2
        yield b, g, s
    while s > min_seq:
        s = max(min_seq, s // 2)
        yield b, g, s


def train_with_oom_retry(make_trainer, args, free_memory):
    last_exc = None
    for attempt, (b, g, s) in enumerate(oom_schedule(args.batch_size, args.gradient_accumulation_steps, args.max_seq_length)):
        if attempt:
            log(f"Retrying with batch_size={b}, gradient_accumulation={g}, max_seq_length={s} (effective batch kept at {b * g})")
            emit("STAGE", stage="retry", label=f"Out of memory - retrying (batch {b}, context {s})")
        trainer = make_trainer(b, g, s)
        try:
            return trainer, trainer.train()
        except Exception as e:  # noqa: BLE001 - only OOM is retried
            if not is_oom(e):
                raise
            last_exc = e
            log(f"CUDA out of memory with batch_size={b}, max_seq_length={s}.")
            del trainer
            free_memory()
    raise last_exc


def build_sft(model, tokenizer, dataset, args, batch, accum, seq_len, callbacks):
    import torch
    from trl import SFTConfig, SFTTrainer

    # TRL renamed arguments across releases; only pass what this version understands.
    cfg_params = inspect.signature(SFTConfig.__init__).parameters
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cfg = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=5,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        dataset_text_field="text",
        dataset_num_proc=1,
        packing=False,
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
    )
    cfg["max_length" if "max_length" in cfg_params else "max_seq_length"] = seq_len
    cfg = {k: v for k, v in cfg.items() if k in cfg_params}

    kw = dict(model=model, train_dataset=dataset, args=SFTConfig(**cfg), callbacks=callbacks)
    params = inspect.signature(SFTTrainer.__init__).parameters
    kw["processing_class" if "processing_class" in params else "tokenizer"] = tokenizer
    return SFTTrainer(**kw)


def check_export_tools():
    missing = [t for t in ("cmake", "git") if not shutil.which(t)]
    if not (shutil.which("gcc") or shutil.which("cc") or shutil.which("clang")):
        missing.append("gcc")
    return missing


def export_to_ollama(model, tokenizer, args):
    missing = check_export_tools()
    if missing:
        raise RuntimeError(
            f"GGUF export needs build tools that are missing: {', '.join(missing)}. Install them "
            "(Ubuntu: sudo apt install build-essential cmake git libcurl4-openssl-dev | Fedora: sudo dnf install gcc-c++ cmake git "
            "libcurl-devel | Arch: sudo pacman -S base-devel cmake git) and run the export again.")
    if not shutil.which("ollama"):
        raise RuntimeError("The 'ollama' command is not on PATH, so the GGUF cannot be registered.")
    gguf_dir = os.path.join(args.output_dir, "gguf")
    stage("export", f"Exporting merged GGUF ({args.gguf_quant}) - builds llama.cpp on first use")
    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=args.gguf_quant)
    ggufs = sorted((os.path.join(r, f) for r, _, fs in os.walk(args.output_dir) for f in fs if f.endswith(".gguf")),
                   key=os.path.getmtime)
    if not ggufs:
        raise RuntimeError("GGUF export produced no .gguf file.")
    modelfile = os.path.join(args.output_dir, "Modelfile")
    with open(modelfile, "w") as f:
        f.write(f"FROM {ggufs[-1]}\n")
    stage("ollama", f"Registering in Ollama as '{args.export_ollama}'")
    subprocess.run(["ollama", "create", args.export_ollama, "-f", modelfile], check=True)
    log(f"Successfully created Ollama model: ollama run {args.export_ollama}")


def main(args):
    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    stage("import", "Loading Unsloth and PyTorch")
    from unsloth import FastLanguageModel  # must be imported before transformers/trl
    import torch
    from transformers import TrainerCallback

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available: PyTorch cannot see the GPU (driver not loaded?).")
    props = torch.cuda.get_device_properties(0)
    log(f"GPU: {props.name} | VRAM {props.total_memory / 1024**3:.2f} GB")

    def free_memory():
        gc.collect()
        torch.cuda.empty_cache()

    stage("model", f"Loading base model {args.model_name} (first run downloads the weights)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=os.path.expanduser(args.model_name),
        max_seq_length=args.max_seq_length,
        load_in_4bit=truthy(args.load_in_4bit),
        dtype=None,
    )
    max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if max_pos and args.max_seq_length > max_pos:
        log(f"Max sequence length {args.max_seq_length} exceeds the model's limit; using {max_pos}.")
        args.max_seq_length = max_pos

    targets = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    stage("lora", f"Attaching LoRA adapters (r={args.lora_r}, alpha={args.lora_alpha})")
    try:
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0, bias="none",
            target_modules=targets,
            use_gradient_checkpointing="unsloth" if truthy(args.gradient_checkpointing) else False,
            random_state=3407,
        )
    except ValueError as e:
        # Some architectures name their projections differently; let PEFT pick all linear layers.
        if "target" not in str(e).lower():
            raise
        log(f"Target modules {targets} not found in this architecture; using all linear layers instead.")
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0, bias="none",
            target_modules="all-linear",
            use_gradient_checkpointing="unsloth" if truthy(args.gradient_checkpointing) else False,
            random_state=3407,
        )

    stage("data", f"Preparing dataset {args.dataset}")
    dataset, skipped = prepare_dataset(load_rows(args.dataset), build_text_fn(tokenizer))
    log(f"Dataset rows: {len(dataset)}" + (f" ({skipped} unusable rows skipped)" if skipped else ""))
    log("Sample:\n" + dataset[0]["text"][:400])

    class StudioCallback(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            if not logs:
                return
            mem = torch.cuda.max_memory_reserved() / 1024**3
            emit("METRIC", step=state.global_step, max_steps=state.max_steps, epoch=round(state.epoch or 0, 3),
                 vram_gb=round(mem, 2), **{k: v for k, v in logs.items() if isinstance(v, (int, float))})

    stage("train", "Training")
    trainer, stats = train_with_oom_retry(
        lambda b, g, s: build_sft(model, tokenizer, dataset, args, b, g, s, [StudioCallback()]),
        args, free_memory)
    log(f"Training finished in {stats.metrics.get('train_runtime', 0):.0f}s | final loss {stats.metrics.get('train_loss')}")

    stage("save", "Saving LoRA adapter")
    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    log(f"Successfully saved LoRA adapter to {adapter_dir}")

    if args.export_ollama:
        export_to_ollama(model, tokenizer, args)

    stage("done", f"All done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    cli_args = parse_args()
    try:
        main(cli_args)
    except KeyboardInterrupt:
        log("Interrupted.")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001 - top-level reporter
        traceback.print_exc()
        title, detail, fix = explain_error(exc, cli_args)
        emit("ERROR", title=title, detail=detail, fix=fix)
        sys.exit(1)

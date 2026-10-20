"""LoRA / QLoRA supervised fine-tuning with Unsloth, launched from AI Studio.

Dataset formats auto-detected per row (JSONL / JSON file or a Hugging Face dataset id):
  * Alpaca:   {"instruction": ..., "input": ..., "output": ...}
  * Chat:     {"messages": [{"role": "user", "content": ...}, ...]}  (also "conversations")
  * Raw text: {"text": ...}

Progress lines prefixed with [[METRIC]] are JSON so the UI can chart them.
"""
import argparse
import inspect
import json
import os
import subprocess
import sys
import time


def parse_args():
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
    return p.parse_args()


def truthy(v):
    return str(v).lower() in ("1", "true", "yes", "y", "on")


def metric(**kw):
    print("[[METRIC]] " + json.dumps(kw), flush=True)


def load_rows(path_or_id):
    from datasets import load_dataset

    if os.path.exists(path_or_id):
        return load_dataset("json", data_files=os.path.abspath(path_or_id), split="train")
    return load_dataset(path_or_id, split="train")


def build_text_fn(tokenizer):
    has_template = bool(getattr(tokenizer, "chat_template", None))
    eos = tokenizer.eos_token or ""

    def to_text(row):
        msgs = row.get("messages") or row.get("conversations")
        if msgs:
            # ShareGPT style {"from": "human", "value": ...} -> {"role", "content"}
            norm = []
            for m in msgs:
                role = m.get("role") or {"human": "user", "gpt": "assistant", "system": "system"}.get(m.get("from"), "user")
                norm.append({"role": role, "content": m.get("content") or m.get("value") or ""})
            if has_template:
                return tokenizer.apply_chat_template(norm, tokenize=False)
            return "\n".join(f"{m['role']}: {m['content']}" for m in norm) + eos
        if row.get("instruction") is not None:
            user = row["instruction"] + (f"\n\n{row['input']}" if row.get("input") else "")
            if has_template:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": user}, {"role": "assistant", "content": row.get("output", "")}],
                    tokenize=False,
                )
            return f"### Instruction:\n{user}\n\n### Response:\n{row.get('output', '')}{eos}"
        if row.get("text") is not None:
            return row["text"] + eos
        raise ValueError(f"Unrecognised dataset row keys: {list(row.keys())}")

    return to_text


def main():
    args = parse_args()
    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    print(f"Loading base model: {args.model_name}", flush=True)
    from unsloth import FastLanguageModel  # must be imported before transformers/trl
    import torch
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | VRAM {props.total_memory / 1024**3:.2f} GB", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=truthy(args.load_in_4bit),
        dtype=None,
    )

    targets = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    print(f"Attaching LoRA adapters: r={args.lora_r} alpha={args.lora_alpha} targets={targets}", flush=True)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        target_modules=targets,
        use_gradient_checkpointing="unsloth" if truthy(args.gradient_checkpointing) else False,
        random_state=3407,
    )

    print(f"Loading dataset: {args.dataset}", flush=True)
    to_text = build_text_fn(tokenizer)
    dataset = load_rows(args.dataset)
    dataset = dataset.map(lambda r: {"text": to_text(r)}, remove_columns=dataset.column_names)
    print(f"Dataset rows: {len(dataset)}", flush=True)
    print("Sample:\n" + dataset[0]["text"][:400], flush=True)

    # TRL renamed a few arguments across releases; only pass what this version understands.
    cfg_params = inspect.signature(SFTConfig.__init__).parameters
    cfg = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )
    cfg["max_length" if "max_length" in cfg_params else "max_seq_length"] = args.max_seq_length
    cfg = {k: v for k, v in cfg.items() if k in cfg_params}

    class StudioCallback(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            if not logs:
                return
            mem = torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
            metric(step=state.global_step, max_steps=state.max_steps, epoch=round(state.epoch or 0, 3),
                   vram_gb=round(mem, 2), **{k: v for k, v in logs.items() if isinstance(v, (int, float))})

    trainer_kw = dict(model=model, train_dataset=dataset, args=SFTConfig(**cfg), callbacks=[StudioCallback()])
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_kw["processing_class" if "processing_class" in trainer_params else "tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kw)

    print("Starting training...", flush=True)
    stats = trainer.train()
    print(f"Training finished in {stats.metrics.get('train_runtime', 0):.0f}s | final loss {stats.metrics.get('train_loss')}", flush=True)

    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Successfully saved LoRA adapter to {adapter_dir}", flush=True)

    if args.export_ollama:
        gguf_dir = os.path.join(args.output_dir, "gguf")
        print(f"Exporting merged GGUF ({args.gguf_quant}) to {gguf_dir} (builds llama.cpp on first use)...", flush=True)
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method=args.gguf_quant)
        ggufs = sorted(
            (os.path.join(r, f) for r, _, fs in os.walk(args.output_dir) for f in fs if f.endswith(".gguf")),
            key=os.path.getmtime,
        )
        if not ggufs:
            print("ERROR: GGUF export produced no .gguf file", flush=True)
            sys.exit(1)
        modelfile = os.path.join(args.output_dir, "Modelfile")
        with open(modelfile, "w") as f:
            f.write(f"FROM {ggufs[-1]}\n")
        print(f"Registering in Ollama as '{args.export_ollama}'", flush=True)
        subprocess.run(["ollama", "create", args.export_ollama, "-f", modelfile], check=True)
        print(f"Successfully created Ollama model: ollama run {args.export_ollama}", flush=True)

    print(f"All done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

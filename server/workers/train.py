"""Unsloth fine-tuning worker. Launched by the Studio as a subprocess:

    python -m server.workers.train --config data/runs/<run>/config.json

Writes metrics as JSON lines to <run>/metrics.jsonl and saves the adapter to <run>/adapter
(or the full model to <run>/model). Touch <run>/STOP to stop gracefully after the current step.
"""
import argparse
import dataclasses
import inspect
import json
import os
import sys
import time
import traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()
cfg = json.loads(Path(args.config).read_text())
run_dir = Path(cfg["run_dir"])
metrics_f = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")


def emit(event, **kw):
    rec = {"event": event, "time": time.time(), **kw}
    metrics_f.write(json.dumps(rec) + "\n")
    metrics_f.flush()
    print(f"[studio] {event}: " + json.dumps(kw), flush=True)


def only_fields(cls, kw):
    names = {f.name for f in dataclasses.fields(cls)}
    renamed = dict(kw)
    # TRL renamed a few arguments over time
    if "max_seq_length" in renamed and "max_seq_length" not in names and "max_length" in names:
        renamed["max_length"] = renamed.pop("max_seq_length")
    if "eval_strategy" in renamed and "eval_strategy" not in names and "evaluation_strategy" in names:
        renamed["evaluation_strategy"] = renamed.pop("eval_strategy")
    return {k: v for k, v in renamed.items() if k in names}


def trainer_kwargs(cls, tokenizer, **kw):
    params = inspect.signature(cls.__init__).parameters
    kw["processing_class" if "processing_class" in params else "tokenizer"] = tokenizer
    return {k: v for k, v in kw.items() if k in params}


ALPACA = ("Below is an instruction that describes a task{ctx}. Write a response that appropriately "
          "completes the request.\n\n### Instruction:\n{instruction}\n\n{input}### Response:\n{output}")


def load_records(path):
    from datasets import load_dataset
    ext = Path(path).suffix.lower()
    kind = {".jsonl": "json", ".json": "json", ".csv": "csv", ".txt": "text", ".parquet": "parquet"}[ext]
    return load_dataset(kind, data_files=str(path), split="train")


def to_text(ds, tokenizer, fmt):
    eos = tokenizer.eos_token or ""
    has_template = bool(getattr(tokenizer, "chat_template", None))

    def conv(msgs):
        norm = [{"role": {"human": "user", "gpt": "assistant"}.get(m.get("role") or m.get("from"),
                          m.get("role") or m.get("from")),
                 "content": m.get("content") or m.get("value") or ""} for m in msgs]
        if has_template:
            return tokenizer.apply_chat_template(norm, tokenize=False)
        return "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in norm) + eos

    def fn(r):
        if fmt == "conversation":
            return {"text": conv(r.get("messages") or r.get("conversations"))}
        if fmt == "alpaca":
            inp = r.get("input") or ""
            return {"text": ALPACA.format(ctx=", paired with an input that provides further context" if inp else "",
                                          instruction=r["instruction"],
                                          input=f"### Input:\n{inp}\n\n" if inp else "",
                                          output=r["output"]) + eos}
        if fmt == "completion":
            return {"text": f"{r['prompt']}{r['completion']}{eos}"}
        if fmt == "qa":
            if has_template:
                return {"text": conv([{"role": "user", "content": r["question"]},
                                      {"role": "assistant", "content": r["answer"]}])}
            return {"text": f"Question: {r['question']}\nAnswer: {r['answer']}{eos}"}
        return {"text": r["text"] + ("" if r["text"].endswith(eos) else eos)}

    return ds.map(fn, remove_columns=[c for c in ds.column_names if c != "text"])


def main():
    emit("loading", message="Importing Unsloth")
    from unsloth import FastLanguageModel  # must be imported before transformers/trl
    import torch
    from transformers import TrainerCallback

    method = cfg["method"]  # lora | qlora | full | dpo
    load_4bit = method == "qlora" or (method == "dpo" and cfg.get("load_in_4bit", True))
    emit("loading", message=f"Loading {cfg['model_path']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_path"], max_seq_length=cfg["max_seq_length"], dtype=None,
        load_in_4bit=load_4bit, full_finetuning=(method == "full"))
    if method != "full":
        model = FastLanguageModel.get_peft_model(
            model, r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg.get("lora_dropout", 0),
            target_modules=cfg["target_modules"], bias="none",
            use_gradient_checkpointing="unsloth" if cfg.get("gradient_checkpointing", True) else False,
            random_state=cfg.get("seed", 3407))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    emit("model", trainable=trainable, total=total)

    ds = load_records(cfg["dataset_path"])
    if method != "dpo":
        ds = to_text(ds, tokenizer, cfg["dataset_format"])
    eval_ds = None
    if cfg.get("eval_ratio", 0) > 0 and len(ds) > 10:
        split = ds.train_test_split(test_size=cfg["eval_ratio"], seed=cfg.get("seed", 3407))
        ds, eval_ds = split["train"], split["test"]
    emit("dataset", records=len(ds), eval_records=len(eval_ds) if eval_ds else 0)

    stop_file = run_dir / "STOP"

    class StudioCallback(TrainerCallback):
        def on_train_begin(self, a, state, control, **kw):
            emit("start", max_steps=state.max_steps, epochs=a.num_train_epochs)

        def on_log(self, a, state, control, logs=None, **kw):
            rec = {k: v for k, v in (logs or {}).items() if isinstance(v, (int, float))}
            if torch.cuda.is_available():
                rec["gpu_mem_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
            emit("log", **{"step": state.global_step, "max_steps": state.max_steps, "epoch": state.epoch, **rec})

        def on_step_end(self, a, state, control, **kw):
            if stop_file.exists():
                emit("stopping", step=state.global_step)
                control.should_training_stop = True
            return control

    optim = cfg.get("optim", "adamw_8bit")
    if optim.endswith("8bit"):
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            print("[studio] bitsandbytes not available, falling back to adamw_torch", flush=True)
            optim = "adamw_torch"

    common = dict(
        output_dir=str(run_dir / "checkpoints"), per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"], num_train_epochs=cfg["epochs"],
        max_steps=cfg.get("max_steps") or -1, learning_rate=cfg["learning_rate"],
        warmup_steps=cfg.get("warmup_steps", 5), logging_steps=1, optim=optim,
        weight_decay=cfg.get("weight_decay", 0.01), lr_scheduler_type=cfg.get("scheduler", "linear"),
        seed=cfg.get("seed", 3407), report_to="none",
        save_strategy="steps" if cfg.get("save_checkpoints") else "no",
        save_steps=cfg.get("save_steps", 200), save_total_limit=2,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.get("eval_steps", 50), max_seq_length=cfg["max_seq_length"],
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )
    if method == "dpo":
        from trl import DPOConfig, DPOTrainer
        try:
            from unsloth import PatchDPOTrainer
            PatchDPOTrainer()
        except ImportError:
            pass
        targs = DPOConfig(**only_fields(DPOConfig, {**common, "beta": cfg.get("dpo_beta", 0.1),
                                                     "max_prompt_length": cfg["max_seq_length"] // 2}))
        trainer = DPOTrainer(**trainer_kwargs(DPOTrainer, tokenizer, model=model, ref_model=None, args=targs,
                                              train_dataset=ds, eval_dataset=eval_ds))
    else:
        from trl import SFTConfig, SFTTrainer
        targs = SFTConfig(**only_fields(SFTConfig, {**common, "dataset_text_field": "text", "packing": False}))
        trainer = SFTTrainer(**trainer_kwargs(SFTTrainer, tokenizer, model=model, args=targs,
                                              train_dataset=ds, eval_dataset=eval_ds))
    trainer.add_callback(StudioCallback())
    result = trainer.train()

    out = run_dir / ("model" if method == "full" else "adapter")
    emit("saving", path=str(out))
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    emit("done", path=str(out), train_loss=result.training_loss,
         runtime=result.metrics.get("train_runtime"), stopped=stop_file.exists())


if __name__ == "__main__":
    os.chdir(run_dir)
    try:
        main()
    except Exception as e:  # noqa: BLE001
        emit("error", message=f"{type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

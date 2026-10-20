"""Fine-tuning runs: each lives in data/runs/<id>/ with config, metrics, log and outputs."""
import json
import shlex
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from . import jobs
from . import model_registry as reg
from . import datasets_service as dsvc
from .config import RUNS_DIR, ROOT_DIR

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _slug(s):
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in s).strip("-")[:40]


def build_config(body):
    m, _ = reg.get(body["modelId"])
    if m["kind"] not in ("hf", "adapter"):
        raise ValueError("Unsloth fine-tunes Hugging Face (safetensors) models. GGUF/Ollama models are "
                         "inference-only — download the HF version of this model to train it.")
    ds_path = dsvc.path_of(body["dataset"])
    if not ds_path.exists():
        raise ValueError("Dataset not found")
    analysis = dsvc.analyze(ds_path)
    method = body.get("method", "qlora")
    if method == "dpo" and analysis["format"] != "preference":
        raise ValueError("DPO needs a dataset with prompt / chosen / rejected fields")
    if method != "dpo" and analysis["format"] in ("unknown", "empty", "preference"):
        raise ValueError(f"Dataset format '{analysis['format']}' can't be used for SFT")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}-{_slug(m['name'].split('/')[-1])}-{method}"
    run_dir = RUNS_DIR / run_id
    cfg = {
        "run_id": run_id, "run_dir": str(run_dir), "created": time.time(),
        "model_id": m["id"], "model_name": m["name"], "model_path": m["path"],
        "dataset": body["dataset"], "dataset_path": str(ds_path), "dataset_format": analysis["format"],
        "dataset_records": analysis["records"],
        "method": method,
        "learning_rate": float(body.get("learningRate", 2e-4)),
        "batch_size": int(body.get("batchSize", 2)), "grad_accum": int(body.get("gradAccum", 4)),
        "epochs": float(body.get("epochs", 1)), "max_steps": int(body.get("maxSteps") or 0),
        "max_seq_length": int(body.get("maxSeqLength", 2048)),
        "lora_r": int(body.get("loraRank", 16)), "lora_alpha": int(body.get("loraAlpha", 16)),
        "lora_dropout": float(body.get("loraDropout", 0)),
        "target_modules": body.get("targetModules") or DEFAULT_TARGETS,
        "gradient_checkpointing": bool(body.get("gradientCheckpointing", True)),
        "save_checkpoints": bool(body.get("saveCheckpoints", False)),
        "eval_ratio": float(body.get("evalRatio", 0)),
        "warmup_steps": int(body.get("warmupSteps", 5)),
        "scheduler": body.get("scheduler", "linear"),
        "dpo_beta": float(body.get("dpoBeta", 0.1)),
        "seed": int(body.get("seed", 3407)),
    }
    return cfg


def command_for(cfg):
    return f"{shlex.quote(sys.executable)} -m server.workers.train --config {shlex.quote(str(Path(cfg['run_dir']) / 'config.json'))}"


def prepare(body):
    cfg = build_config(body)
    return {"config": cfg, "command": command_for(cfg)}


def launch(cfg, command=None):
    if jobs.running("train"):
        raise ValueError("A training run is already in progress")
    run_dir = Path(cfg["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    def on_exit(job):
        _write_status(run_dir, {"state": job.status, "returncode": job.returncode, "ended": job.ended})

    job = jobs.start("train", f"Fine-tune {cfg['model_name']} ({cfg['method']})",
                     command or command_for(cfg), cwd=str(ROOT_DIR), log_file=str(run_dir / "train.log"),
                     meta={"runId": cfg["run_id"]}, on_exit=on_exit)
    _write_status(run_dir, {"state": "running", "jobId": job.id, "started": job.started})
    return job


def _write_status(run_dir, st):
    p = Path(run_dir) / "status.json"
    try:
        cur = json.loads(p.read_text())
    except (OSError, ValueError):
        cur = {}
    cur.update(st)
    p.write_text(json.dumps(cur))


def read_metrics(run_dir):
    out = []
    try:
        with open(Path(run_dir) / "metrics.jsonl", encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def summarize(run_dir):
    run_dir = Path(run_dir)
    try:
        cfg = json.loads((run_dir / "config.json").read_text())
    except (OSError, ValueError):
        return None
    try:
        st = json.loads((run_dir / "status.json").read_text())
    except (OSError, ValueError):
        st = {}
    state = st.get("state", "unknown")
    if state == "running":
        try:
            job = jobs.get(st.get("jobId"))
            state = job.status
        except KeyError:
            state = "interrupted"  # server restarted while it ran
    metrics = read_metrics(run_dir)
    logs = [x for x in metrics if x["event"] == "log" and "loss" in x]
    last = next((x for x in reversed(metrics) if x["event"] == "log"), {})
    err = next((x["message"] for x in reversed(metrics) if x["event"] == "error"), None)
    done = next((x for x in reversed(metrics) if x["event"] == "done"), None)
    peak = max((x.get("gpu_mem_gb", 0) for x in metrics if x["event"] == "log"), default=None)
    return {
        "id": cfg["run_id"], "config": cfg, "state": state, "jobId": st.get("jobId"),
        "started": st.get("started", cfg.get("created")), "ended": st.get("ended"),
        "step": last.get("step", 0), "maxSteps": last.get("max_steps"), "epoch": last.get("epoch"),
        "lossStart": logs[0]["loss"] if logs else None, "lossLast": logs[-1]["loss"] if logs else None,
        "evalLoss": next((x["eval_loss"] for x in reversed(metrics) if "eval_loss" in x), None),
        "gpuPeakGb": peak, "error": err, "output": done["path"] if done else None,
    }


def list_runs():
    out = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if d.is_dir():
            s = summarize(d)
            if s:
                out.append(s)
    return out


def run_dir_of(run_id):
    d = (RUNS_DIR / run_id).resolve()
    if d.parent != RUNS_DIR.resolve() or not d.is_dir():
        raise KeyError(run_id)
    return d


def detail(run_id):
    d = run_dir_of(run_id)
    return {**summarize(d), "metrics": read_metrics(d)}


def request_stop(run_id):
    d = run_dir_of(run_id)
    (d / "STOP").write_text("stop")


def delete(run_id):
    d = run_dir_of(run_id)
    s = summarize(d)
    if s and s["state"] == "running":
        raise ValueError("Stop the run before deleting it")
    shutil.rmtree(d)

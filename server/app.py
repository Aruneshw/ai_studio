"""AI Model Studio backend (FastAPI). Run with:  python -m server"""
import shlex
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import datasets_service as dsvc
from . import environment, inference, jobs, runs, system_info, tensors, tokenizer_service
from . import model_registry as reg
from .config import EXPORTS_DIR, ROOT_DIR, PORT, AUTO_INSTALL
from .terminal import run_terminal

app = FastAPI(title="AI Model Studio")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


# ----------------------------------------------------------------------------- security
# The API can run shell commands, so only the local UI may talk to it:
#  * Host must be localhost (blocks DNS-rebinding),
#  * state-changing requests need the X-AI-Studio header (forces a CORS preflight that we never allow),
#  * WebSocket Origin must be a localhost page.

def _host_ok(host_header):
    host = (host_header or "").rsplit(":", 1)[0] if not (host_header or "").startswith("[") else host_header.split("]")[0] + "]"
    return host in LOCAL_HOSTS


def _origin_ok(origin):
    if not origin:
        return True  # non-browser client
    return (urlparse(origin).hostname or "") in LOCAL_HOSTS


@app.middleware("http")
async def guard(request: Request, call_next):
    if not _host_ok(request.headers.get("host")):
        return JSONResponse({"detail": "forbidden host"}, status_code=403)
    if request.url.path.startswith("/api/") and request.method not in ("GET", "HEAD", "OPTIONS"):
        if request.headers.get("x-ai-studio") != "1" or not _origin_ok(request.headers.get("origin")):
            return JSONResponse({"detail": "missing studio header"}, status_code=403)
    return await call_next(request)


def bad(e, code=400):
    raise HTTPException(status_code=code, detail=str(e))


@app.on_event("startup")
def _startup():
    if AUTO_INSTALL:
        environment.auto_install()


@app.on_event("shutdown")
def _shutdown():
    jobs.stop_all()


# ----------------------------------------------------------------------------- system & env

@app.get("/api/system")
def system():
    return system_info.snapshot()


@app.get("/api/env")
def env():
    return environment.status()


@app.post("/api/env/install")
def env_install():
    return environment.start_install().summary()


# ----------------------------------------------------------------------------- models

@app.get("/api/models")
def models():
    return {"models": reg.scan_all(), "library": reg.load_library()}


@app.post("/api/models/import")
async def models_import(req: Request):
    body = await req.json()
    try:
        key = reg.add_path(body["path"])
    except (ValueError, KeyError) as e:
        bad(e)
    return {"added": key, "models": reg.scan_all()}


@app.post("/api/models/remove")
async def models_remove(req: Request):
    body = await req.json()
    reg.remove_path(body["path"])
    return {"models": reg.scan_all()}


@app.post("/api/models/download")
async def models_download(req: Request):
    body = await req.json()
    src, name = body.get("source"), (body.get("name") or "").strip()
    if not name or any(c in name for c in " ;&|`$<>\n"):
        bad("invalid model name")
    if src == "ollama":
        if not shutil.which("ollama"):
            bad("Ollama is not installed (https://ollama.com/download)")
        job = jobs.start("download", f"ollama pull {name}", ["ollama", "pull", name])
    elif src == "hf":
        cmd = [sys.executable, "-m", "server.workers.hf_download", name]
        if body.get("include"):
            cmd += ["--include", body["include"]]
        job = jobs.start("download", f"HF download {name}", cmd, cwd=str(ROOT_DIR))
    else:
        bad("source must be 'ollama' or 'hf'")
    return job.summary()


@app.get("/api/browse")
def browse(path: str = "~"):
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        bad("not a directory")
    entries = []
    try:
        for c in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if c.name.startswith("."):
                continue
            try:
                is_dir = c.is_dir()
            except OSError:
                continue
            if is_dir or c.suffix in (".gguf", ".safetensors"):
                entries.append({"name": c.name, "path": str(c), "dir": is_dir,
                                "model": reg._looks_like_model(c) if is_dir else True})
    except PermissionError:
        bad("permission denied")
    return {"path": str(p), "parent": str(p.parent), "entries": entries[:500]}


def _model(model_id):
    try:
        return reg.get(model_id)
    except KeyError:
        bad("model not found", 404)


@app.get("/api/models/{model_id}/tensors")
def model_tensors(model_id: str):
    _model(model_id)
    try:
        return {"tensors": tensors.list_tensors(model_id)}
    except ValueError as e:
        bad(e)


@app.get("/api/models/{model_id}/tensor")
def model_tensor(model_id: str, name: str, grid: int = 48):
    _model(model_id)
    try:
        return tensors.tensor_detail(model_id, name, max(8, min(grid, 128)))
    except KeyError:
        bad("tensor not found", 404)


@app.get("/api/models/{model_id}/network")
def model_network(model_id: str, planes: int = 8, grid: int = 16):
    _model(model_id)
    try:
        return tensors.network_map(model_id, max(2, min(planes, 16)), max(4, min(grid, 32)))
    except ValueError as e:
        bad(e)


@app.get("/api/models/{model_id}/embeddings")
def model_embeddings(model_id: str, ids: str = ""):
    _model(model_id)
    try:
        return tensors.embeddings_3d(model_id, [int(x) for x in ids.split(",") if x.strip().lstrip("-").isdigit()])
    except ValueError as e:
        bad(e)


@app.get("/api/models/{model_id}/lora")
def model_lora(model_id: str):
    _model(model_id)
    try:
        return tensors.lora_modules(model_id)
    except ValueError as e:
        bad(e)


@app.get("/api/models/{model_id}/lora/delta")
def model_lora_delta(model_id: str, module: str, grid: int = 48):
    _model(model_id)
    try:
        return tensors.lora_delta(model_id, module, max(8, min(grid, 96)))
    except (ValueError, StopIteration) as e:
        bad(e or "module not found")


# ----------------------------------------------------------------------------- tokenizer

@app.post("/api/models/{model_id}/tokenize")
async def tokenize(model_id: str, req: Request):
    _model(model_id)
    body = await req.json()
    try:
        return tokenizer_service.tokenize(model_id, body.get("text", ""), body.get("addSpecial", False))
    except (ValueError, ImportError) as e:
        bad(e)


# ----------------------------------------------------------------------------- datasets

@app.get("/api/datasets")
def datasets():
    return {"datasets": dsvc.list_datasets()}


@app.post("/api/datasets/upload")
async def dataset_upload(file: UploadFile = File(...)):
    try:
        dest = dsvc.path_of(file.filename)
    except ValueError as e:
        bad(e)
    with open(dest, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    try:
        return dsvc.analyze(dest)
    except Exception as e:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        bad(f"Could not parse dataset: {e}")


@app.get("/api/datasets/{name}/preview")
def dataset_preview(name: str, offset: int = 0, limit: int = 20):
    try:
        return {"rows": dsvc.preview(name, offset, min(limit, 200)), "analysis": dsvc.analyze(dsvc.path_of(name))}
    except (ValueError, OSError) as e:
        bad(e)


@app.delete("/api/datasets/{name}")
def dataset_delete(name: str):
    p = dsvc.path_of(name)
    p.unlink(missing_ok=True)
    return {"ok": True}


# ----------------------------------------------------------------------------- training runs

@app.post("/api/train/prepare")
async def train_prepare(req: Request):
    try:
        return runs.prepare(await req.json())
    except (ValueError, KeyError) as e:
        bad(e)


@app.post("/api/train/launch")
async def train_launch(req: Request):
    body = await req.json()
    try:
        cfg = runs.build_config(body["params"])
        job = runs.launch(cfg, body.get("command") or None)
    except (ValueError, KeyError) as e:
        bad(e)
    return {"runId": cfg["run_id"], "job": job.summary()}


@app.get("/api/runs")
def runs_list():
    return {"runs": runs.list_runs()}


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str):
    try:
        return runs.detail(run_id)
    except KeyError:
        bad("run not found", 404)


@app.get("/api/runs/{run_id}/log")
def run_log(run_id: str, lines: int = 400):
    try:
        f = runs.run_dir_of(run_id) / "train.log"
    except KeyError:
        bad("run not found", 404)
    if not f.exists():
        return {"lines": []}
    with open(f, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 512 * 1024))
        tail = fh.read().decode("utf-8", errors="replace").splitlines()
    return {"lines": tail[-max(1, min(lines, 5000)):]}


@app.post("/api/runs/{run_id}/stop")
def run_stop(run_id: str):
    try:
        runs.request_stop(run_id)
    except KeyError:
        bad("run not found", 404)
    return {"ok": True}


@app.delete("/api/runs/{run_id}")
def run_delete(run_id: str):
    try:
        runs.delete(run_id)
    except KeyError:
        bad("run not found", 404)
    except ValueError as e:
        bad(e)
    return {"ok": True}


# ----------------------------------------------------------------------------- jobs

@app.get("/api/jobs")
def jobs_list():
    return {"jobs": [j.summary() for j in jobs.all_jobs()]}


@app.get("/api/jobs/{job_id}")
def job_log(job_id: str, since: int = 0):
    try:
        return jobs.get(job_id).log(since)
    except KeyError:
        bad("job not found", 404)


@app.post("/api/jobs/{job_id}/stop")
def job_stop(job_id: str):
    try:
        jobs.get(job_id).stop()
    except KeyError:
        bad("job not found", 404)
    return {"ok": True}


@app.post("/api/jobs/run")
async def job_run(req: Request):
    """Runs an approved shell command (from the command-approval dialog)."""
    body = await req.json()
    cmd = (body.get("command") or "").strip()
    if not cmd:
        bad("empty command")
    return jobs.start("command", body.get("title") or cmd[:80], cmd, cwd=str(ROOT_DIR)).summary()


# ----------------------------------------------------------------------------- inference

@app.post("/api/chat")
def chat(body: dict):
    _model(body.get("modelId", ""))
    try:
        text, stats = inference.chat(body["modelId"], body["messages"], float(body.get("temperature", 0.7)),
                                     int(body.get("maxTokens", 512)), bool(body.get("disableAdapter")))
    except (RuntimeError, ValueError, ImportError, OSError) as e:
        bad(e)
    return {"text": text, "stats": stats}


@app.post("/api/chat/unload")
def chat_unload():
    inference.unload()
    return {"ok": True}


# ----------------------------------------------------------------------------- export

@app.post("/api/export")
async def export(req: Request):
    body = await req.json()
    m, _ = _model(body.get("modelId", ""))
    fmt = body.get("format")
    if fmt not in ("lora", "merged_16bit", "merged_4bit", "gguf"):
        bad("unknown export format")
    if m["kind"] not in ("hf", "adapter"):
        bad("Only Hugging Face models / LoRA adapters can be exported")
    name = "".join(c if c.isalnum() or c in "-_." else "-" for c in (body.get("name") or m["name"]))[:60]
    out = EXPORTS_DIR / f"{name}-{fmt}"
    cmd = [sys.executable, "-m", "server.workers.export", "--source", m["path"], "--format", fmt,
           "--quant", body.get("quant", "q4_k_m"), "--out", str(out)]
    ollama_name = (body.get("ollamaName") or "").strip()
    if ollama_name:
        if any(c in ollama_name for c in " ;&|`$<>"):
            bad("invalid Ollama model name")
        cmd += ["--ollama-name", ollama_name]
    job = jobs.start("export", f"Export {m['name']} → {fmt}", cmd, cwd=str(ROOT_DIR),
                     meta={"out": str(out)})
    return {**job.summary(), "commandPreview": " ".join(shlex.quote(c) for c in cmd)}


# ----------------------------------------------------------------------------- terminal

@app.websocket("/ws/terminal")
async def terminal(ws: WebSocket):
    if not _host_ok(ws.headers.get("host")) or not _origin_ok(ws.headers.get("origin")):
        await ws.close(code=1008)
        return
    await ws.accept()
    await run_terminal(ws, int(ws.query_params.get("cols", 120)), int(ws.query_params.get("rows", 32)))


# ----------------------------------------------------------------------------- built frontend

DIST = ROOT_DIR / "dist"
if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = (DIST / path).resolve()
        if path and f.is_file() and DIST.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(DIST / "index.html")


def main():
    import uvicorn
    from .config import HOST
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")

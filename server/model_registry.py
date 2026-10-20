"""Discovers models that actually exist on this machine.

Sources: Ollama blob store, Hugging Face hub cache, LM Studio folders, user-added
paths/folders (library.json) and adapters produced by Studio fine-tuning runs.
"""
import hashlib
import json
import os
import re
import threading
from pathlib import Path

from .config import LIBRARY_FILE, RUNS_DIR
from .formats.gguf import GGUFFile, FILE_TYPES
from .formats.safetensors_reader import SafetensorsModel

LAYER_RE = re.compile(r"(?:^|\.)(?:layers|h|blk|blocks|block|layer)\.(\d+)\.")

_lock = threading.Lock()
_cache = {}  # path -> (mtime, model dict)
_handles = {}  # model id -> (model dict, reader)


# --------------------------------------------------------------------------- library

def load_library():
    try:
        return json.loads(LIBRARY_FILE.read_text())
    except (OSError, ValueError):
        return {"paths": [], "folders": []}


def save_library(lib):
    LIBRARY_FILE.write_text(json.dumps(lib, indent=2))


def add_path(path):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise ValueError(f"Path does not exist: {p}")
    lib = load_library()
    if _looks_like_model(p):
        key = "paths"
    elif p.is_dir():
        key = "folders"
    else:
        raise ValueError("Not a model: expected a .gguf/.safetensors file, a directory with "
                         "config.json/adapter_config.json, or a folder containing models")
    if str(p) not in lib[key]:
        lib[key].append(str(p))
    save_library(lib)
    return key


def remove_path(path):
    lib = load_library()
    for key in ("paths", "folders"):
        lib[key] = [x for x in lib[key] if x != path]
    save_library(lib)


def _looks_like_model(p):
    try:
        if p.is_file():
            return p.suffix in (".gguf", ".safetensors")
        if (p / "config.json").exists() or (p / "adapter_config.json").exists():
            return True
        return any(p.glob("*.gguf")) and not any(c.is_dir() for c in p.iterdir())
    except OSError:
        return False


# --------------------------------------------------------------------------- helpers

def _mid(path):
    return hashlib.sha1(str(path).encode()).hexdigest()[:12]


def _dir_size(p):
    p = Path(p)
    if p.is_file():
        return p.stat().st_size
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size  # follows HF cache symlinks to blobs
        except OSError:
            pass
    return total


def params_label(n):
    if not n:
        return "?"
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n / div:.2f}".rstrip("0").rstrip(".") + suf
    return str(n)


AVATARS = [("llama", "🦙"), ("mistral", "🌪️"), ("mixtral", "🌪️"), ("qwen", "🐉"),
           ("deepseek", "🐋"), ("phi", "⚡"), ("gemma", "💎"), ("gpt", "🤖"),
           ("falcon", "🦅"), ("smol", "🐣"), ("granite", "🪨")]


def _avatar(*names):
    s = " ".join(n or "" for n in names).lower()
    for key, a in AVATARS:
        if key in s:
            return a
    return "🧠"


def _base(mid, name, kind, source, path):
    return {
        "id": mid, "name": name, "kind": kind, "source": source, "path": str(path),
        "format": None, "quantization": None, "architecture": None, "modelType": None,
        "params": None, "paramsLabel": "?", "diskBytes": None, "layers": None,
        "hiddenSize": None, "intermediateSize": None, "heads": None, "kvHeads": None,
        "headDim": None, "vocabSize": None, "contextLength": None, "activation": None,
        "tensorCount": None, "config": {}, "baseModel": None, "ollamaName": None,
        "capabilities": {"tensors": False, "tokenizer": False, "finetune": False, "chat": False},
        "error": None,
    }


# --------------------------------------------------------------------------- describers

def describe_hf_dir(path, source, name=None):
    path = Path(path)
    m = _base(_mid(path), name or path.name, "hf", source, path)
    cfg = {}
    try:
        cfg = json.loads((path / "config.json").read_text())
    except (OSError, ValueError):
        pass
    text_cfg = cfg.get("text_config", cfg)  # multimodal configs nest the LM
    m["config"] = cfg
    m["architecture"] = (cfg.get("architectures") or [None])[0]
    m["modelType"] = cfg.get("model_type")
    m["layers"] = text_cfg.get("num_hidden_layers") or text_cfg.get("n_layer")
    m["hiddenSize"] = text_cfg.get("hidden_size") or text_cfg.get("n_embd")
    m["intermediateSize"] = text_cfg.get("intermediate_size")
    m["heads"] = text_cfg.get("num_attention_heads") or text_cfg.get("n_head")
    m["kvHeads"] = text_cfg.get("num_key_value_heads") or m["heads"]
    m["headDim"] = text_cfg.get("head_dim") or (
        m["hiddenSize"] // m["heads"] if m["hiddenSize"] and m["heads"] else None)
    m["vocabSize"] = text_cfg.get("vocab_size")
    m["contextLength"] = text_cfg.get("max_position_embeddings") or text_cfg.get("n_positions")
    m["activation"] = text_cfg.get("hidden_act") or text_cfg.get("activation_function")
    q = cfg.get("quantization_config")
    if q:
        bits = "4-bit" if q.get("load_in_4bit") else "8-bit" if q.get("load_in_8bit") else ""
        m["quantization"] = f"{q.get('quant_method', 'quantized')} {bits}".strip()
    else:
        m["quantization"] = cfg.get("torch_dtype") or text_cfg.get("torch_dtype") or "unknown"
    has_st = any(path.glob("*.safetensors"))
    m["format"] = "Safetensors" if has_st else "PyTorch .bin" if any(path.glob("*.bin")) else "HF"
    reader = None
    if has_st:
        try:
            reader = SafetensorsModel(path)
            m["tensorCount"] = len(reader.tensors)
            m["params"] = sum(reader.tensor_info(n)["numel"] for n in reader.tensors)
            m["capabilities"]["tensors"] = True
        except Exception as e:  # noqa: BLE001
            m["error"] = f"safetensors header: {e}"
    m["paramsLabel"] = params_label(m["params"])
    m["diskBytes"] = _dir_size(path)
    m["capabilities"]["tokenizer"] = (path / "tokenizer.json").exists() or \
        (path / "tokenizer.model").exists()
    m["capabilities"]["finetune"] = True
    m["capabilities"]["chat"] = True
    m["avatar"] = _avatar(m["name"], m["modelType"])
    return m, reader


def describe_adapter(path, source, name=None):
    path = Path(path)
    m = _base(_mid(path), name or path.name, "adapter", source, path)
    try:
        cfg = json.loads((path / "adapter_config.json").read_text())
    except (OSError, ValueError):
        cfg = {}
    m["config"] = cfg
    m["format"] = "LoRA Adapter"
    m["baseModel"] = cfg.get("base_model_name_or_path")
    m["quantization"] = f"r={cfg.get('r')} α={cfg.get('lora_alpha')}"
    m["architecture"] = cfg.get("peft_type", "LORA")
    reader = None
    st = path / "adapter_model.safetensors"
    if st.exists():
        reader = SafetensorsModel(st)
        m["tensorCount"] = len(reader.tensors)
        m["params"] = sum(reader.tensor_info(n)["numel"] for n in reader.tensors)
        layers = {int(x.group(1)) for n in reader.tensors if (x := LAYER_RE.search(n))}
        m["layers"] = len(layers) or None
        m["capabilities"]["tensors"] = True
    m["paramsLabel"] = params_label(m["params"])
    m["diskBytes"] = _dir_size(path)
    m["capabilities"]["chat"] = True
    m["capabilities"]["tokenizer"] = (path / "tokenizer.json").exists()
    m["avatar"] = "🧩"
    return m, reader


def describe_gguf(path, source, name=None, ollama_name=None):
    path = Path(path)
    m = _base(_mid(path), name or path.stem, "ollama" if ollama_name else "gguf", source, path)
    m["format"] = "GGUF"
    m["ollamaName"] = ollama_name
    try:
        g = GGUFFile(path)
    except Exception as e:  # noqa: BLE001
        m["error"] = f"GGUF header: {e}"
        m["diskBytes"] = path.stat().st_size
        m["avatar"] = _avatar(m["name"])
        return m, None
    md = g.metadata
    arch = md.get("general.architecture", "")
    a = lambda k: md.get(f"{arch}.{k}")  # noqa: E731
    m["config"] = {k: v for k, v in md.items() if not isinstance(v, list)}
    m["architecture"] = arch
    m["modelType"] = arch
    if not ollama_name and md.get("general.name"):
        m["name"] = md["general.name"] if not name else name
    m["layers"] = a("block_count")
    m["hiddenSize"] = a("embedding_length")
    m["intermediateSize"] = a("feed_forward_length")
    m["heads"] = a("attention.head_count")
    m["kvHeads"] = a("attention.head_count_kv") or m["heads"]
    m["headDim"] = a("attention.key_length") or (
        m["hiddenSize"] // m["heads"] if m["hiddenSize"] and isinstance(m["heads"], int) and m["heads"] else None)
    m["contextLength"] = a("context_length")
    m["vocabSize"] = g.array_lengths.get("tokenizer.ggml.tokens") or a("vocab_size")
    ft = md.get("general.file_type")
    m["quantization"] = FILE_TYPES.get(ft, f"type {ft}") if ft is not None else None
    m["tensorCount"] = len(g.tensors)
    m["params"] = sum(g.tensor_info(n)["numel"] for n in g.tensors)
    m["paramsLabel"] = md.get("general.size_label") or params_label(m["params"])
    m["diskBytes"] = path.stat().st_size
    m["capabilities"]["tensors"] = True
    m["capabilities"]["tokenizer"] = md.get("tokenizer.ggml.model") in ("gpt2", "llama")
    m["capabilities"]["chat"] = True
    m["avatar"] = _avatar(m["name"], arch)
    return m, g


def _describe_cached(path, fn, *args, **kw):
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        return None
    key = (str(path), args, tuple(sorted(kw.items())))
    hit = _cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        result = fn(path, *args, **kw)
    except Exception as e:  # noqa: BLE001
        m = _base(_mid(path), Path(path).name, "unknown", args[0] if args else "?", path)
        m["error"] = str(e)
        result = (m, None)
    _cache[key] = (mtime, result)
    return result


def describe_path(path, source):
    p = Path(path)
    if p.is_file() and p.suffix == ".gguf":
        return [_describe_cached(p, describe_gguf, source)]
    if p.is_file() and p.suffix == ".safetensors":
        return [_describe_cached(p.parent, describe_hf_dir, source)]
    if (p / "adapter_config.json").exists():
        return [_describe_cached(p, describe_adapter, source)]
    if (p / "config.json").exists():
        return [_describe_cached(p, describe_hf_dir, source)]
    return [_describe_cached(f, describe_gguf, source) for f in sorted(p.glob("*.gguf"))
            if not _is_aux_gguf(f)]


def _is_aux_gguf(f):
    n = f.name.lower()
    return "mmproj" in n or re.search(r"-0000[2-9]-of-", n) is not None


# --------------------------------------------------------------------------- sources

def _ollama_roots():
    roots = []
    if os.environ.get("OLLAMA_MODELS"):
        roots.append(Path(os.environ["OLLAMA_MODELS"]))
    roots += [Path.home() / ".ollama/models", Path("/usr/share/ollama/.ollama/models"),
              Path("/var/lib/ollama/.ollama/models"), Path("/var/lib/ollama/models")]
    seen, out = set(), []
    for r in roots:
        try:
            if (r / "manifests").is_dir() and r.resolve() not in seen:
                seen.add(r.resolve())
                out.append(r)
        except OSError:
            pass
    return out


def scan_ollama():
    found = []
    for root in _ollama_roots():
        for mf in (root / "manifests").rglob("*"):
            if not mf.is_file():
                continue
            try:
                manifest = json.loads(mf.read_text())
            except (OSError, ValueError):
                continue
            rel = mf.relative_to(root / "manifests").parts  # host/ns/model/tag
            if len(rel) < 4:
                continue
            host, ns, model, tag = rel[0], rel[1], rel[2], rel[3]
            name = f"{model}:{tag}" if ns == "library" and host == "registry.ollama.ai" \
                else f"{ns}/{model}:{tag}" if host == "registry.ollama.ai" else "/".join(rel[:3]) + f":{tag}"
            for layer in manifest.get("layers", []):
                if layer.get("mediaType") == "application/vnd.ollama.image.model":
                    blob = root / "blobs" / layer["digest"].replace(":", "-")
                    if blob.exists():
                        r = _describe_cached(blob, describe_gguf, "Ollama", name=name, ollama_name=name)
                        if r:
                            found.append(r)
                    break
    return found


def _hf_cache_dir():
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    return home / "hub"


def scan_hf_cache():
    found = []
    root = _hf_cache_dir()
    if not root.is_dir():
        return found
    for repo in root.glob("models--*"):
        repo_id = repo.name[len("models--"):].replace("--", "/")
        snaps = sorted((repo / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not snaps:
            continue
        snap = snaps[0]
        if (snap / "adapter_config.json").exists():
            r = _describe_cached(snap, describe_adapter, "Hugging Face Cache", name=repo_id)
            found.append(r)
        elif (snap / "config.json").exists() and (any(snap.glob("*.safetensors")) or any(snap.glob("*.bin"))):
            found.append(_describe_cached(snap, describe_hf_dir, "Hugging Face Cache", name=repo_id))
        for g in sorted(snap.rglob("*.gguf")):
            if not _is_aux_gguf(g):
                found.append(_describe_cached(g, describe_gguf, "Hugging Face Cache",
                                              name=f"{repo_id}/{g.name}"))
    return found


def _scan_folder(folder, source, depth=4):
    found = []
    folder = Path(folder)
    if not folder.is_dir():
        return found

    def walk(d, level):
        try:
            if (d / "config.json").exists() or (d / "adapter_config.json").exists():
                found.extend(r for r in describe_path(d, source) if r)
                return
            for f in sorted(d.iterdir()):
                if f.is_file() and f.suffix == ".gguf" and not _is_aux_gguf(f):
                    found.append(_describe_cached(f, describe_gguf, source))
                elif f.is_dir() and level < depth and not f.name.startswith("."):
                    walk(f, level + 1)
        except OSError:
            pass

    walk(folder, 0)
    return found


def scan_lmstudio():
    found = []
    for d in (Path.home() / ".lmstudio/models", Path.home() / ".cache/lm-studio/models"):
        found += _scan_folder(d, "LM Studio")
    return found


def scan_runs():
    found = []
    for run in sorted(RUNS_DIR.glob("*")):
        for sub, fn in (("adapter", describe_adapter), ("model", describe_hf_dir)):
            p = run / sub
            if p.is_dir() and any(p.iterdir()):
                r = _describe_cached(p, fn, "Studio Fine-tune", name=f"{run.name} ({sub})")
                if r:
                    found.append(r)
    return found


def scan_all():
    with _lock:
        lib = load_library()
        results = []
        results += scan_ollama()
        results += scan_hf_cache()
        results += scan_lmstudio()
        for p in lib.get("paths", []):
            if Path(p).exists():
                results += [r for r in describe_path(p, "Local Path") if r]
        for f in lib.get("folders", []):
            results += _scan_folder(f, "Local Folder")
        results += scan_runs()
        seen, models = set(), []
        _handles.clear()
        for r in results:
            if not r:
                continue
            m, reader = r
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            models.append(m)
            _handles[m["id"]] = (m, reader)
        return models


def get(model_id):
    if model_id not in _handles:
        scan_all()
    if model_id not in _handles:
        raise KeyError(model_id)
    return _handles[model_id]


def layer_of(name):
    x = LAYER_RE.search(name)
    return int(x.group(1)) if x else None

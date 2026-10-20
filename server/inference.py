"""Text generation with whatever runtime fits the model: Ollama, Transformers/Unsloth, llama.cpp."""
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import model_registry as reg
from .config import OLLAMA_URL

_lock = threading.RLock()  # re-entrant: chat() -> _load() -> unload()
_loaded = {"id": None, "model": None, "tokenizer": None, "kind": None}


def _ollama_chat(name, messages, temperature, max_tokens):
    body = json.dumps({"model": name, "messages": messages, "stream": False,
                       "options": {"temperature": temperature, "num_predict": max_tokens}}).encode()
    req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama is not reachable at {OLLAMA_URL} — start it with `ollama serve` ({e})")
    stats = {"tokens": data.get("eval_count"),
             "tokensPerSec": (data["eval_count"] / (data["eval_duration"] / 1e9))
             if data.get("eval_duration") else None}
    return data["message"]["content"], stats


def unload():
    with _lock:
        _loaded.update(id=None, model=None, tokenizer=None, kind=None)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def _load_hf(m, load_in_4bit):
    path = m["path"]
    try:
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(path, max_seq_length=4096,
                                                       load_in_4bit=load_in_4bit, dtype=None)
        FastLanguageModel.for_inference(model)
        return model, tok
    except ImportError:
        pass
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if m["kind"] == "adapter":
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(path, torch_dtype="auto", device_map="auto")
        tok_path = path if (Path(path) / "tokenizer_config.json").exists() else m["baseModel"]
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", device_map="auto")
        tok_path = path
    model.eval()
    return model, AutoTokenizer.from_pretrained(tok_path)


def _load(m, load_in_4bit=True):
    if _loaded["id"] == m["id"]:
        return
    unload()
    if m["kind"] in ("hf", "adapter"):
        model, tok = _load_hf(m, load_in_4bit)
        _loaded.update(id=m["id"], model=model, tokenizer=tok, kind="hf")
    elif m["kind"] == "gguf":
        try:
            from llama_cpp import Llama
        except ImportError:
            raise RuntimeError("Chatting with a standalone .gguf needs llama-cpp-python "
                               "(`pip install llama-cpp-python`) — or import it into Ollama.")
        llm = Llama(model_path=m["path"], n_ctx=4096, n_gpu_layers=-1, verbose=False)
        _loaded.update(id=m["id"], model=llm, tokenizer=None, kind="llama_cpp")


def _hf_generate(messages, temperature, max_tokens, disable_adapter=False):
    import torch
    model, tok = _loaded["model"], _loaded["tokenizer"]
    if getattr(tok, "chat_template", None):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    kw = dict(max_new_tokens=max_tokens, do_sample=temperature > 0,
              pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if temperature > 0:
        kw["temperature"] = temperature
    t0 = time.time()
    with torch.no_grad():
        if disable_adapter and hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                out = model.generate(**inputs, **kw)
        else:
            out = model.generate(**inputs, **kw)
    new = out[0][inputs["input_ids"].shape[1]:]
    dt = time.time() - t0
    return tok.decode(new, skip_special_tokens=True), {"tokens": len(new), "tokensPerSec": len(new) / dt if dt else None}


def chat(model_id, messages, temperature=0.7, max_tokens=512, disable_adapter=False):
    m, _ = reg.get(model_id)
    if m["kind"] == "ollama":
        return _ollama_chat(m["ollamaName"], messages, temperature, max_tokens)
    with _lock:
        _load(m)
        if _loaded["kind"] == "llama_cpp":
            t0 = time.time()
            r = _loaded["model"].create_chat_completion(messages=messages, temperature=temperature,
                                                        max_tokens=max_tokens)
            n = r["usage"]["completion_tokens"]
            return r["choices"][0]["message"]["content"], {"tokens": n, "tokensPerSec": n / (time.time() - t0)}
        return _hf_generate(messages, temperature, max_tokens, disable_adapter)


def loaded():
    return {"id": _loaded["id"], "kind": _loaded["kind"]}

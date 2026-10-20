"""Tokenizes text with the model's own tokenizer (tokenizer.json or GGUF vocab)."""
import json
from pathlib import Path

from . import model_registry as reg
from .formats.gguf import GGUFFile

_tok_cache = {}


def _gguf_tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    g = GGUFFile(path, load_arrays={"tokenizer.ggml.tokens", "tokenizer.ggml.merges",
                                    "tokenizer.ggml.scores", "tokenizer.ggml.token_type"})
    md = g.metadata
    tokens = md.get("tokenizer.ggml.tokens") or []
    kind = md.get("tokenizer.ggml.model")
    types = md.get("tokenizer.ggml.token_type") or []
    if kind == "gpt2":
        vocab = {t: i for i, t in enumerate(tokens)}
        merges = [tuple(m.split(" ", 1)) for m in md.get("tokenizer.ggml.merges") or []]
        tok = Tokenizer(models.BPE(vocab, merges, ignore_merges=True))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tok.decoder = decoders.ByteLevel()
    elif kind == "llama":
        scores = md.get("tokenizer.ggml.scores") or [0.0] * len(tokens)
        tok = Tokenizer(models.Unigram([(t, float(s)) for t, s in zip(tokens, scores)],
                                       unk_id=md.get("tokenizer.ggml.unknown_token_id", 0),
                                       byte_fallback=True))
        tok.pre_tokenizer = pre_tokenizers.Metaspace()
        tok.decoder = decoders.Metaspace()
    else:
        raise ValueError(f"GGUF tokenizer type '{kind}' is not supported")
    special = {i for i, t in enumerate(types) if t in (3, 4)}  # control / user-defined
    if special:
        from tokenizers import AddedToken
        tok.add_special_tokens([AddedToken(tokens[i], special=True, normalized=False) for i in sorted(special)])
    info = {"type": f"GGUF {kind} vocab (rebuilt locally; may differ slightly from llama.cpp)",
            "bos": md.get("tokenizer.ggml.bos_token_id"), "eos": md.get("tokenizer.ggml.eos_token_id"),
            "chatTemplate": md.get("tokenizer.chat_template"), "vocabSize": len(tokens),
            "special": [tokens[i] for i in sorted(special)][:200]}
    return tok, info


def _hf_tokenizer(path):
    from tokenizers import Tokenizer
    path = Path(path)
    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    cfg = {}
    try:
        cfg = json.loads((path / "tokenizer_config.json").read_text())
    except (OSError, ValueError):
        pass
    template = cfg.get("chat_template")
    if not template and (path / "chat_template.jinja").exists():
        template = (path / "chat_template.jinja").read_text()
    added = cfg.get("added_tokens_decoder", {})
    info = {"type": "Hugging Face tokenizer.json", "bos": cfg.get("bos_token"), "eos": cfg.get("eos_token"),
            "chatTemplate": template, "vocabSize": tok.get_vocab_size(),
            "special": [v.get("content") for v in added.values() if v.get("special")][:200]}
    return tok, info


def get_tokenizer(model_id):
    if model_id in _tok_cache:
        return _tok_cache[model_id]
    m, _ = reg.get(model_id)
    if m["format"] == "GGUF":
        res = _gguf_tokenizer(m["path"])
    elif (Path(m["path"]) / "tokenizer.json").exists():
        res = _hf_tokenizer(m["path"])
    else:
        raise ValueError("No tokenizer.json found for this model")
    _tok_cache[model_id] = res
    return res


def tokenize(model_id, text, add_special=False):
    tok, info = get_tokenizer(model_id)
    enc = tok.encode(text, add_special_tokens=add_special)
    special = set(info.get("special") or [])
    toks = []
    for tid, piece, (s, e) in zip(enc.ids, enc.tokens, enc.offsets):
        kind = "special" if piece in special else "byte" if piece.startswith("<0x") else \
            "punct" if not any(ch.isalnum() for ch in text[s:e]) and text[s:e].strip() else \
            "space" if not text[s:e].strip() else "word"
        toks.append({"id": tid, "piece": piece, "text": text[s:e], "start": s, "end": e, "kind": kind})
    return {"tokens": toks, "count": len(toks), "chars": len(text), "info": info}


def info(model_id):
    _, i = get_tokenizer(model_id)
    return i

"""Reads real weights from model files and turns them into stats / grids for the UI."""
import re
import numpy as np

from . import model_registry as reg

_net_cache = {}


def list_tensors(model_id):
    m, reader = reg.get(model_id)
    if reader is None:
        raise ValueError("This model's weights can't be read (no safetensors/GGUF file)")
    out = []
    for name in reader.tensors:
        info = reader.tensor_info(name)
        out.append({"name": name, "layer": reg.layer_of(name), **info})
    return out


def _sample_rows(reader, name, max_rows):
    n = reader.n_rows(name)
    if n <= max_rows:
        idx = np.arange(n)
    else:
        idx = np.unique(np.linspace(0, n - 1, max_rows).astype(np.int64))
    return reader.read_rows(name, idx), idx, n


def _downsample(a, gr, gc):
    """Block-mean a 2D array to at most gr x gc cells."""
    r, c = a.shape
    gr, gc = min(gr, r), min(gc, c)
    rb = np.array_split(np.arange(r), gr)
    cb = np.array_split(np.arange(c), gc)
    out = np.empty((gr, gc), dtype=np.float32)
    for i, rr in enumerate(rb):
        block = a[rr]
        for j, cc in enumerate(cb):
            out[i, j] = block[:, cc].mean()
    return out


def _as_2d(rows):
    if rows.shape[0] == 1:  # 1D tensor -> square-ish grid
        v = rows[0]
        side = int(np.ceil(np.sqrt(v.size)))
        pad = np.full(side * side, np.nan, dtype=np.float32)
        pad[: v.size] = v
        return pad.reshape(side, side)
    return rows


def tensor_detail(model_id, name, grid=48):
    m, reader = reg.get(model_id)
    if reader is None or name not in reader.tensors:
        raise KeyError(name)
    info = reader.tensor_info(name)
    res = {"name": name, "layer": reg.layer_of(name), **info, "sampled": False}
    if not info["supported"]:
        res["error"] = f"{info['dtype']} weights can't be decoded for preview"
        return res
    rows, idx, n_rows = _sample_rows(reader, name, max_rows=256)
    res["sampled"] = len(idx) < n_rows
    res["sampledRows"] = int(len(idx))
    flat = rows.reshape(-1)
    flat = flat[np.isfinite(flat)]
    res["stats"] = {
        "mean": float(flat.mean()), "std": float(flat.std()),
        "min": float(flat.min()), "max": float(flat.max()),
        "absMean": float(np.abs(flat).mean()),
        "nearZero": float((np.abs(flat) < 1e-3).mean()),
        "l2PerRow": float(np.linalg.norm(rows, axis=1).mean()),
    }
    lo, hi = np.percentile(flat, [0.1, 99.9])
    counts, edges = np.histogram(np.clip(flat, lo, hi), bins=60, range=(float(lo), float(hi)))
    res["histogram"] = {"counts": counts.tolist(), "edges": edges.tolist()}
    a2 = _as_2d(rows)
    g = _downsample(np.nan_to_num(a2), grid, grid)
    res["grid"] = g.tolist()
    # raw top-left values from the *sampled* rows
    res["raw"] = {"rowIndices": idx[:12].tolist(), "values": rows[:12, :12].tolist()}
    return res


# ------------------------------------------------------------------ network map for 3D view

def _pick(names, patterns):
    for p in patterns:
        for n in names:
            if re.search(p, n):
                return n
    return None


def network_map(model_id, planes=8, grid=16):
    key = (model_id, planes, grid)
    if key in _net_cache:
        return _net_cache[key]
    m, reader = reg.get(model_id)
    if reader is None:
        raise ValueError("weights not readable")
    names = list(reader.tensors)
    by_layer = {}
    for n in names:
        lyr = reg.layer_of(n)
        if lyr is not None:
            by_layer.setdefault(lyr, []).append(n)
    layer_ids = sorted(by_layer)
    total = len(layer_ids)
    k = min(planes, total)
    chosen = sorted({layer_ids[round(i * (total - 1) / max(k - 1, 1))] for i in range(k)}) if total else []

    def plane(label, kind, tname, layer=None):
        p = {"label": label, "kind": kind, "layer": layer, "tensor": tname, "grid": None}
        if not tname:
            return p
        info = reader.tensor_info(tname)
        p["shape"] = info["shape"]
        p["dtype"] = info["dtype"]
        if not info["supported"]:
            p["error"] = f"{info['dtype']} not decodable"
            return p
        try:
            rows, _, _ = _sample_rows(reader, tname, max_rows=grid * 6)
            a = np.abs(np.nan_to_num(_as_2d(rows)))
            g = _downsample(a, grid, grid)
            lo, hi = np.percentile(g, [2, 98])
            p["grid"] = np.clip((g - lo) / max(hi - lo, 1e-12), 0, 1).round(4).tolist()
            p["absMean"] = float(a.mean())
        except Exception as e:  # noqa: BLE001
            p["error"] = str(e)
        return p

    embed = _pick(names, [r"embed_tokens\.weight$", r"token_embd\.weight$", r"wte\.weight$", r"embed.*weight$"])
    head = _pick(names, [r"lm_head\.weight$", r"^output\.weight$", r"embed_out\.weight$"])
    out = [plane("Token Embedding", "embed", embed)]
    for lyr in chosen:
        ln = by_layer[lyr]
        t = _pick(ln, [r"mlp\.down_proj", r"ffn_down\.", r"mlp\.c_proj", r"o_proj", r"attn_output",
                       r"lora_B", r"\.weight$"])
        out.append(plane(f"Block {lyr}", "block", t, lyr))
    out.append(plane("LM Head" if head else "LM Head (tied)", "head", head or embed))
    result = {"modelId": model_id, "totalLayers": total, "planes": out}
    _net_cache[key] = result
    return result


# ------------------------------------------------------------------ embeddings (PCA 3D)

def embeddings_3d(model_id, token_ids, background=400):
    m, reader = reg.get(model_id)
    names = list(reader.tensors) if reader else []
    emb = _pick(names, [r"embed_tokens\.weight$", r"token_embd\.weight$", r"wte\.weight$"])
    if not emb:
        raise ValueError("No token embedding matrix found in this model")
    info = reader.tensor_info(emb)
    if not info["supported"]:
        raise ValueError(f"Embedding stored as {info['dtype']} — not decodable")
    vocab = reader.n_rows(emb)
    rng = np.random.default_rng(0)
    bg = rng.choice(vocab, size=min(background, vocab), replace=False)
    ids = [int(t) for t in token_ids if 0 <= int(t) < vocab]
    all_ids = np.array(ids + [int(b) for b in bg if int(b) not in ids])
    order = np.argsort(all_ids)
    vecs = np.empty((len(all_ids), reader.read_rows(emb, [0]).shape[1]), dtype=np.float32)
    vecs[order] = reader.read_rows(emb, all_ids[order])
    x = vecs - vecs.mean(axis=0)
    _, s, vt = np.linalg.svd(x, full_matrices=False)
    pts = x @ vt[:3].T
    pts /= np.abs(pts).max() + 1e-9
    var = (s[:3] ** 2) / (s ** 2).sum()
    return {
        "tensor": emb,
        "explainedVariance": var.tolist(),
        "points": [{"id": int(i), "xyz": p.round(4).tolist(), "query": idx < len(ids)}
                   for idx, (i, p) in enumerate(zip(all_ids, pts))],
    }


# ------------------------------------------------------------------ LoRA delta analysis

def lora_modules(model_id):
    m, reader = reg.get(model_id)
    if m["kind"] != "adapter":
        raise ValueError("Not a LoRA adapter")
    cfg = m["config"]
    r = cfg.get("r") or 1
    alpha = cfg.get("lora_alpha") or r
    scale = alpha / r
    mods = {}
    for n in reader.tensors:
        if ".lora_A." in n:
            base = n.split(".lora_A.")[0]
            b = n.replace(".lora_A.", ".lora_B.")
            if b in reader.tensors:
                mods[base] = (n, b)
    out = []
    for base, (a, b) in sorted(mods.items()):
        A = reader.read_full(a)  # (r, in)
        B = reader.read_full(b)  # (out, r)
        # ||B A||_F computed in rank space
        fro = float(np.sqrt(max(np.sum((B.T @ B) * (A @ A.T)), 0.0))) * scale
        out.append({"module": base, "layer": reg.layer_of(base),
                    "type": base.rsplit(".", 1)[-1], "shape": [B.shape[0], A.shape[1]],
                    "deltaNorm": fro})
    return {"scale": scale, "rank": r, "alpha": alpha, "modules": out}


def lora_delta(model_id, module, grid=48):
    m, reader = reg.get(model_id)
    cfg = m["config"]
    scale = (cfg.get("lora_alpha") or 1) / (cfg.get("r") or 1)
    a, b = f"{module}.lora_A.weight", f"{module}.lora_B.weight"
    if a not in reader.tensors:
        a = next(n for n in reader.tensors if n.startswith(module + ".lora_A"))
        b = a.replace(".lora_A.", ".lora_B.")
    A = reader.read_full(a)
    B = reader.read_full(b)
    ri = np.unique(np.linspace(0, B.shape[0] - 1, min(256, B.shape[0])).astype(int))
    ci = np.unique(np.linspace(0, A.shape[1] - 1, min(256, A.shape[1])).astype(int))
    d = (B[ri] @ A[:, ci]) * scale
    g = _downsample(d, grid, grid)
    return {"module": module, "shape": [B.shape[0], A.shape[1]], "grid": g.tolist(),
            "stats": {"mean": float(d.mean()), "std": float(d.std()),
                      "min": float(d.min()), "max": float(d.max())}}

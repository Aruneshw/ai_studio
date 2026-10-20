"""Datasets the user uploaded (or registered by path) for fine-tuning."""
import csv
import json
import re
from pathlib import Path

from .config import DATASETS_DIR

ALLOWED = {".json", ".jsonl", ".csv", ".txt", ".parquet"}

FORMATS = [  # (name, required fields, description)
    ("conversation", {"messages"}, "chat messages [{role, content}]"),
    ("conversation", {"conversations"}, "ShareGPT conversations [{from, value}]"),
    ("preference", {"prompt", "chosen", "rejected"}, "DPO preference pairs"),
    ("alpaca", {"instruction", "output"}, "instruction / input / output"),
    ("completion", {"prompt", "completion"}, "prompt / completion"),
    ("qa", {"question", "answer"}, "question / answer"),
    ("text", {"text"}, "raw text"),
]


def safe_name(name):
    name = Path(name).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if Path(name).suffix.lower() not in ALLOWED:
        raise ValueError(f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED))}")
    return name


def path_of(name):
    p = (DATASETS_DIR / safe_name(name)).resolve()
    if p.parent != DATASETS_DIR.resolve():
        raise ValueError("bad dataset name")
    return p


def read_records(path, limit=None):
    path = Path(path)
    ext = path.suffix.lower()
    rows = []
    if ext == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                    if limit and len(rows) >= limit:
                        break
    elif ext == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = next((v for v in data.values() if isinstance(v, list)), [data])
        rows = data[:limit] if limit else data
    elif ext == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
                if limit and len(rows) >= limit:
                    break
    elif ext == ".txt":
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append({"text": line.rstrip("\n")})
                    if limit and len(rows) >= limit:
                        break
    elif ext == ".parquet":
        import pyarrow.parquet as pq  # optional, comes with `datasets`
        t = pq.read_table(path)
        rows = t.slice(0, limit).to_pylist() if limit else t.to_pylist()
    return rows


def detect_format(rows):
    if not rows:
        return "empty", "no records"
    keys = set(rows[0].keys()) if isinstance(rows[0], dict) else set()
    for name, req, desc in FORMATS:
        if req <= keys:
            return name, desc
    return "unknown", f"fields: {', '.join(sorted(keys)) or 'n/a'}"


def _record_text(r):
    return " ".join(json.dumps(v) if not isinstance(v, str) else v for v in r.values())


def analyze(path):
    path = Path(path)
    rows = read_records(path)
    fmt, desc = detect_format(rows)
    lengths = [len(_record_text(r)) for r in rows if isinstance(r, dict)]
    missing = {}
    if rows and isinstance(rows[0], dict):
        for k in rows[0]:
            missing[k] = sum(1 for r in rows if not r.get(k))
    lengths_sorted = sorted(lengths)
    hist = []
    if lengths:
        top = lengths_sorted[int(len(lengths_sorted) * 0.99) - 1] if len(lengths_sorted) > 1 else lengths_sorted[0]
        bins = 20
        width = max(1, top // bins + 1)
        hist = [0] * bins
        for n in lengths:
            hist[min(bins - 1, n // width)] += 1
        hist = {"width": width, "counts": hist}
    return {
        "name": path.name, "path": str(path), "size": path.stat().st_size,
        "records": len(rows), "format": fmt, "formatDescription": desc,
        "fields": list(rows[0].keys()) if rows and isinstance(rows[0], dict) else [],
        "avgChars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "maxChars": max(lengths) if lengths else 0,
        "approxTokens": round(sum(lengths) / len(lengths) / 4) if lengths else 0,
        "emptyFields": missing, "lengthHistogram": hist,
        "valid": fmt not in ("empty", "unknown"),
    }


def list_datasets():
    out = []
    for p in sorted(DATASETS_DIR.iterdir()):
        if p.suffix.lower() in ALLOWED and p.is_file():
            try:
                out.append(analyze(p))
            except Exception as e:  # noqa: BLE001
                out.append({"name": p.name, "path": str(p), "size": p.stat().st_size,
                            "valid": False, "error": str(e), "records": 0, "format": "error"})
    return out


def preview(name, offset=0, limit=20):
    rows = read_records(path_of(name), limit=offset + limit)
    return rows[offset: offset + limit]

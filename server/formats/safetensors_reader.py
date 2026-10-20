"""Header-only safetensors reader with row sampling (no torch needed)."""
import json
import struct
from pathlib import Path
import numpy as np

_NP = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16, "I64": np.int64,
    "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_,
}
_SIZES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2,
          "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
FLOAT_TYPES = {"F64", "F32", "F16", "BF16"}


def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


class SafetensorsModel:
    """Merges every *.safetensors shard in a directory (or a single file)."""

    def __init__(self, path):
        path = Path(path)
        files = [path] if path.is_file() else sorted(path.glob("*.safetensors"))
        self.tensors = {}
        for fp in files:
            header, data_start = read_header(fp)
            for name, info in header.items():
                self.tensors[name] = {
                    "file": str(fp), "dtype": info["dtype"], "shape": info["shape"],
                    "start": data_start + info["data_offsets"][0],
                    "end": data_start + info["data_offsets"][1],
                }

    def tensor_info(self, name):
        t = self.tensors[name]
        n = int(np.prod(t["shape"])) if t["shape"] else 1
        return {"shape": t["shape"], "dtype": t["dtype"], "numel": n,
                "nbytes": t["end"] - t["start"], "supported": t["dtype"] in FLOAT_TYPES}

    def n_rows(self, name):
        shape = self.tensors[name]["shape"]
        return int(shape[0]) if len(shape) > 1 else 1

    def read_rows(self, name, row_indices):
        t = self.tensors[name]
        if t["dtype"] not in FLOAT_TYPES:
            raise ValueError(f"{t['dtype']} tensors (packed/quantized) cannot be previewed")
        shape = t["shape"]
        row_len = int(np.prod(shape[1:])) if len(shape) > 1 else (int(shape[0]) if shape else 1)
        item = _SIZES[t["dtype"]]
        out = np.empty((len(row_indices), row_len), dtype=np.float32)
        with open(t["file"], "rb") as f:
            for i, r in enumerate(row_indices):
                f.seek(t["start"] + int(r) * row_len * item)
                raw = f.read(row_len * item)
                if t["dtype"] == "BF16":
                    out[i] = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
                else:
                    out[i] = np.frombuffer(raw, _NP[t["dtype"]]).astype(np.float32)
        return out

    def read_full(self, name):
        return self.read_rows(name, range(self.n_rows(name))).reshape(self.tensors[name]["shape"])

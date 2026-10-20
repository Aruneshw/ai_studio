"""Minimal GGUF reader: metadata, tensor table and row dequantization (numpy only)."""
import struct
import numpy as np

GGUF_MAGIC = b"GGUF"

# value types
_SCALARS = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
    6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
_STRING, _ARRAY = 8, 9

# ggml tensor types: id -> (name, block_size, type_size)
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
}

FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
    11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
    17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
    23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
}


class GGUFFile:
    """Parses header lazily. Large arrays are skipped unless listed in `load_arrays`."""

    def __init__(self, path, load_arrays=()):
        self.path = str(path)
        self.metadata = {}
        self.array_lengths = {}
        self.tensors = {}
        with open(self.path, "rb") as f:
            self._f = f
            if f.read(4) != GGUF_MAGIC:
                raise ValueError("not a GGUF file")
            (self.version,) = struct.unpack("<I", f.read(4))
            if self.version == 1:
                n_tensors, n_kv = struct.unpack("<II", f.read(8))
            else:
                n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
            for _ in range(n_kv):
                key = self._str()
                (vtype,) = struct.unpack("<I", f.read(4))
                if vtype == _ARRAY:
                    (atype,) = struct.unpack("<I", f.read(4))
                    (alen,) = struct.unpack("<Q", f.read(8))
                    self.array_lengths[key] = alen
                    if key in load_arrays:
                        self.metadata[key] = [self._value(atype) for _ in range(alen)]
                    else:
                        self._skip_array(atype, alen)
                else:
                    self.metadata[key] = self._value(vtype)
            for _ in range(n_tensors):
                name = self._str()
                (n_dims,) = struct.unpack("<I", f.read(4))
                dims = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))
                ttype, offset = struct.unpack("<IQ", f.read(12))
                self.tensors[name] = {"dims": dims, "type": ttype, "offset": offset}
            align = int(self.metadata.get("general.alignment", 32))
            pos = f.tell()
            self.data_offset = pos + (align - pos % align) % align
        self._f = None

    # --- header helpers -------------------------------------------------
    def _str(self):
        (n,) = struct.unpack("<Q", self._f.read(8))
        return self._f.read(n).decode("utf-8", errors="replace")

    def _value(self, vtype):
        if vtype == _STRING:
            return self._str()
        if vtype == _ARRAY:
            (atype,) = struct.unpack("<I", self._f.read(4))
            (alen,) = struct.unpack("<Q", self._f.read(8))
            return [self._value(atype) for _ in range(alen)]
        fmt, size = _SCALARS[vtype]
        return struct.unpack(fmt, self._f.read(size))[0]

    def _skip_array(self, atype, alen):
        if atype == _STRING:
            for _ in range(alen):
                (n,) = struct.unpack("<Q", self._f.read(8))
                self._f.seek(n, 1)
        elif atype == _ARRAY:
            for _ in range(alen):
                (sub,) = struct.unpack("<I", self._f.read(4))
                (sublen,) = struct.unpack("<Q", self._f.read(8))
                self._skip_array(sub, sublen)
        else:
            self._f.seek(_SCALARS[atype][1] * alen, 1)

    # --- tensor helpers -------------------------------------------------
    def tensor_info(self, name):
        t = self.tensors[name]
        tname, bs, ts = GGML_TYPES.get(t["type"], (f"type{t['type']}", 1, 0))
        # ggml dims are innermost-first; expose PyTorch-style shape (outer first)
        shape = list(reversed(t["dims"]))
        n = int(np.prod(t["dims"])) if t["dims"] else 1
        return {"shape": shape, "dtype": tname, "numel": n,
                "nbytes": n // bs * ts if ts else None,
                "supported": tname in DEQUANT}

    def read_rows(self, name, row_indices):
        """Dequantize selected rows (outer index) of a tensor into float32 [len(rows), ne0]."""
        t = self.tensors[name]
        tname, bs, ts = GGML_TYPES[t["type"]]
        if tname not in DEQUANT:
            raise ValueError(f"dequantization of {tname} is not supported")
        ne0 = t["dims"][0]
        row_bytes = ne0 // bs * ts
        base = self.data_offset + t["offset"]
        out = np.empty((len(row_indices), ne0), dtype=np.float32)
        with open(self.path, "rb") as f:
            for i, r in enumerate(row_indices):
                f.seek(base + int(r) * row_bytes)
                raw = np.frombuffer(f.read(row_bytes), dtype=np.uint8)
                out[i] = DEQUANT[tname](raw, ne0)
        return out

    def n_rows(self, name):
        dims = self.tensors[name]["dims"]
        return int(np.prod(dims[1:])) if len(dims) > 1 else 1


# --- dequantization kernels (operate on one row of raw bytes) --------------

def _blocks(raw, size):
    return raw.reshape(-1, size)


def _dq_f32(raw, n):
    return raw.view(np.float32)[:n]


def _dq_f16(raw, n):
    return raw.view(np.float16).astype(np.float32)[:n]


def _dq_bf16(raw, n):
    return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)[:n]


def _dq_q8_0(raw, n):
    b = _blocks(raw, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    q = b[:, 2:].view(np.int8).astype(np.float32)
    return (q * d).reshape(-1)[:n]


def _dq_q4_0(raw, n):
    b = _blocks(raw, 18)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    qs = b[:, 2:]
    q = np.concatenate([qs & 0xF, qs >> 4], axis=1).astype(np.float32) - 8
    return (q * d).reshape(-1)[:n]


def _dq_q4_1(raw, n):
    b = _blocks(raw, 20)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    m = b[:, 2:4].copy().view(np.float16).astype(np.float32)
    qs = b[:, 4:]
    q = np.concatenate([qs & 0xF, qs >> 4], axis=1).astype(np.float32)
    return (q * d + m).reshape(-1)[:n]


def _q5_bits(qh_bytes, qs):
    qh = qh_bytes.copy().view(np.uint32).reshape(-1, 1)
    j = np.arange(16, dtype=np.uint32)
    h0 = ((qh >> j) << 4) & 0x10
    h1 = (qh >> (j + 12)) & 0x10
    lo = (qs & 0xF).astype(np.uint32) | h0
    hi = (qs >> 4).astype(np.uint32) | h1
    return np.concatenate([lo, hi], axis=1).astype(np.float32)


def _dq_q5_0(raw, n):
    b = _blocks(raw, 22)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    q = _q5_bits(b[:, 2:6], b[:, 6:]) - 16
    return (q * d).reshape(-1)[:n]


def _dq_q5_1(raw, n):
    b = _blocks(raw, 24)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    m = b[:, 2:4].copy().view(np.float16).astype(np.float32)
    q = _q5_bits(b[:, 4:8], b[:, 8:])
    return (q * d + m).reshape(-1)[:n]


def _k_scales(s):
    """Unpack the 12-byte 6-bit scales/mins of Q4_K/Q5_K into (nb, 8) arrays."""
    s = s.astype(np.uint8)
    sc = np.empty((s.shape[0], 8), dtype=np.float32)
    mn = np.empty((s.shape[0], 8), dtype=np.float32)
    for j in range(4):
        sc[:, j] = s[:, j] & 63
        mn[:, j] = s[:, j + 4] & 63
    for j in range(4, 8):
        sc[:, j] = (s[:, j + 4] & 0xF) | ((s[:, j - 4] >> 6) << 4)
        mn[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    return sc, mn


def _dq_q4_k(raw, n):
    b = _blocks(raw, 144)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32)
    sc, mn = _k_scales(b[:, 4:16])
    qs = b[:, 16:].reshape(-1, 4, 32)
    q = np.stack([qs & 0xF, qs >> 4], axis=2).astype(np.float32)  # (nb,4,2,32)
    scale = (d * sc).reshape(-1, 4, 2, 1)
    mins = (dmin * mn).reshape(-1, 4, 2, 1)
    return (q * scale - mins).reshape(-1)[:n]


def _dq_q5_k(raw, n):
    b = _blocks(raw, 176)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32)
    sc, mn = _k_scales(b[:, 4:16])
    qh = b[:, 16:48]  # (nb,32)
    qs = b[:, 48:].reshape(-1, 4, 32)
    out = np.empty((b.shape[0], 4, 2, 32), dtype=np.float32)
    for c in range(4):
        out[:, c, 0] = (qs[:, c] & 0xF) + (((qh >> (2 * c)) & 1) << 4)
        out[:, c, 1] = (qs[:, c] >> 4) + (((qh >> (2 * c + 1)) & 1) << 4)
    scale = (d * sc).reshape(-1, 4, 2, 1)
    mins = (dmin * mn).reshape(-1, 4, 2, 1)
    return (out * scale - mins).reshape(-1)[:n]


_Q6_IDX = np.array([[2 * k + l // 16 for l in range(32)] for k in range(4)])


def _dq_q6_k(raw, n):
    b = _blocks(raw, 210)
    ql = b[:, :128].reshape(-1, 2, 64)
    qh = b[:, 128:192].reshape(-1, 2, 32)
    sc = b[:, 192:208].copy().view(np.int8).astype(np.float32).reshape(-1, 2, 8)
    d = b[:, 208:210].copy().view(np.float16).astype(np.float32).reshape(-1, 1, 1, 1)
    la, lb = ql[:, :, :32].astype(np.int16), ql[:, :, 32:].astype(np.int16)
    h = qh.astype(np.int16)
    q = np.stack([
        (la & 0xF) | ((h & 3) << 4),
        (lb & 0xF) | (((h >> 2) & 3) << 4),
        (la >> 4) | (((h >> 4) & 3) << 4),
        (lb >> 4) | (((h >> 6) & 3) << 4),
    ], axis=2).astype(np.float32) - 32  # (nb,2,4,32)
    scales = sc[:, :, _Q6_IDX]  # (nb,2,4,32)
    return (q * scales * d).reshape(-1)[:n]


DEQUANT = {
    "F32": _dq_f32, "F16": _dq_f16, "BF16": _dq_bf16, "Q8_0": _dq_q8_0,
    "Q4_0": _dq_q4_0, "Q4_1": _dq_q4_1, "Q5_0": _dq_q5_0, "Q5_1": _dq_q5_1,
    "Q4_K": _dq_q4_k, "Q5_K": _dq_q5_k, "Q6_K": _dq_q6_k,
}

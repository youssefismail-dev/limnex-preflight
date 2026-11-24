"""GGUF: header, key/value metadata (names, chat templates, tokenizer data) and tensor names.

Only the header region is parsed, so a file whose first few MB are available is enough.
"""

from __future__ import annotations

import io
import struct
from typing import BinaryIO, List

from ..context import MB, ScanContext
from ..detectors import MODE_SECRETS
from .binary import scan_stream_deep

_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_VOCAB_KEYS = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type", "tokenizer.ggml.scores")
MAX_STRING = 64 * MB
MAX_ARRAY = 5_000_000


class _Truncated(Exception):
    pass


class _Reader:
    def __init__(self, f: BinaryIO, v1: bool, size: int):
        self.f, self.v1, self.size = f, v1, size

    def take(self, n: int) -> bytes:
        # check against the file size BEFORE reading: f.read(n) allocates n bytes up front, and n can come from a damaged
        # count (8 * ndims of a few billion asked for 20 GB; found by evaluation/fuzz_formats.py)
        if n < 0 or self.f.tell() + n > self.size:
            raise _Truncated()
        b = self.f.read(n)
        if len(b) < n:
            raise _Truncated()
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def length(self) -> int:
        return self.u32() if self.v1 else self.u64()

    def string(self) -> str:
        n = self.length()
        if n > MAX_STRING or self.f.tell() + n > self.size:
            raise _Truncated()
        return self.take(n).decode("utf-8", "replace")


def scan_gguf(f: BinaryIO, ctx: ScanContext, name: str) -> None:
    f.seek(0, io.SEEK_END)
    size = f.tell()
    f.seek(0)
    if f.read(4) != b"GGUF":
        ctx.note("error", "gguf: bad magic")
        return
    version = struct.unpack("<I", f.read(4))[0]
    if version not in (1, 2, 3):
        ctx.note("error", f"gguf: unsupported version {version}")
        return
    r = _Reader(f, version == 1, size)
    try:
        tensor_count = r.length()
        kv_count = r.length()
        if kv_count > 1_000_000 or tensor_count > 10_000_000:
            ctx.note("error", "gguf: implausible header counts")
            return
        for _ in range(kv_count):
            key = r.string()
            ctx.scan_text(key, "gguf metadata key")
            vtype = r.u32()
            _value(r, ctx, key, vtype, 0)
        for _ in range(tensor_count):
            tname = r.string()
            ctx.scan_text(tname, "gguf tensor name")
            ndims = r.u32()
            r.take(8 * ndims if not r.v1 else 4 * ndims)
            r.u32()
            r.take(8 if not r.v1 else 4)
        data_start = f.tell()
    except _Truncated:
        ctx.note("degraded", "gguf: header truncated or file cut short; metadata read so far was scanned")
        return
    except struct.error:
        ctx.note("error", "gguf: malformed header")
        return
    if ctx.options.deep:
        scan_stream_deep(f, ctx, data_start, "gguf tensor data")


def _value(r: _Reader, ctx: ScanContext, key: str, vtype: int, depth: int) -> None:
    loc = f'gguf metadata "{key}"'
    if vtype == 8:
        ctx.scan_text(r.string(), loc, key=key)
    elif vtype in _FIXED:
        r.take(_FIXED[vtype])
    elif vtype == 9:
        if depth > 3:
            raise _Truncated()
        etype = r.u32()
        count = r.length()
        if count > MAX_ARRAY:
            raise _Truncated()
        if etype == 8:
            items: List[str] = [r.string() for _ in range(count)]
            if not items:
                return
            blob = "\n".join(items)
            mode = MODE_SECRETS if key in _VOCAB_KEYS else "full"
            ctx.scan_text(blob, f"{loc} (array of {count} strings)", mode=mode, key=key, dedupe=False)
        elif etype in _FIXED:
            r.f.seek(count * _FIXED[etype], io.SEEK_CUR)
            if r.f.tell() > r.size:
                raise _Truncated()
        elif etype == 9:
            for _ in range(count):
                _value(r, ctx, key, 9, depth + 1)
        else:
            raise _Truncated()
    else:
        raise _Truncated()

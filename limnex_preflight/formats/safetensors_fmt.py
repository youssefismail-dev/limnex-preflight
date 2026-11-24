"""safetensors: an 8-byte little-endian length, a JSON header (tensor table + __metadata__), then raw tensors."""

from __future__ import annotations

import json
import struct
from typing import BinaryIO

from ..context import MB, ScanContext
from ..detectors import MODE_BINARY
from .binary import _PRINTABLE_RUN, read_limited, scan_stream_deep
from .textual import walk_json

MAX_HEADER = 100 * MB  # same cap the reference implementation uses
TAIL_CAP = 64 * MB


def looks_like_safetensors(head: bytes, size: int) -> bool:
    if len(head) < 10 or size < 10:
        return False
    n = struct.unpack("<Q", head[:8])[0]
    return 0 < n <= min(MAX_HEADER, size - 8) and head[8:9] == b"{"


def scan_safetensors(f: BinaryIO, ctx: ScanContext, name: str) -> None:
    f.seek(0)
    raw = f.read(8)
    if len(raw) < 8:
        ctx.note("error", "safetensors: file shorter than 8 bytes")
        return
    n = struct.unpack("<Q", raw)[0]
    if n == 0 or n > MAX_HEADER:
        ctx.note("error", f"safetensors: implausible header length {n}")
        return
    header = f.read(n)
    if len(header) < n:
        ctx.note("error", "safetensors: file truncated inside the header")
        return
    ctx.charge(n)
    try:
        obj = json.loads(header.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        ctx.note("error", "safetensors: header is not valid JSON")
        return
    walk_json(ctx, obj, root="header")
    ends = [v["data_offsets"][1] for k, v in obj.items() if k != "__metadata__" and isinstance(v, dict)
            and isinstance(v.get("data_offsets"), list) and len(v["data_offsets"]) == 2
            and isinstance(v["data_offsets"][1], int)] if isinstance(obj, dict) else []
    here = f.tell()
    f.seek(0, 2)
    size = f.tell()
    f.seek(here)
    data_end = 8 + n + (max(ends) if ends else 0)
    if size < data_end:
        ctx.note("degraded", f"safetensors: tensor data cut short ({size - 8 - n} of {max(ends)} bytes); "
                             "file truncated")
    elif size > data_end:
        _scan_after_tensors(f, ctx, data_end, size - data_end)
    if ctx.options.deep:
        scan_stream_deep(f, ctx, 8 + n, "safetensors tensor data")


def _scan_after_tensors(f: BinaryIO, ctx: ScanContext, start: int, length: int) -> None:
    """Review 2026-11-11: bytes after the last tensor belong to no tensor, so no loader needs them; they were never read.
    They are now carved for text like other binary data (up to TAIL_CAP)."""
    f.seek(start)
    data = read_limited(f, min(length, TAIL_CAP))
    ctx.charge(len(data))
    for m in _PRINTABLE_RUN.finditer(data):
        ctx.scan_text(m.group(0).decode("ascii", "replace"), f"safetensors bytes after the last tensor, offset {start + m.start()}",
                      mode=MODE_BINARY)
    if length > TAIL_CAP:
        ctx.note("skipped", f"safetensors: {length} bytes after the last tensor; only the first {TAIL_CAP // MB} MB scanned")

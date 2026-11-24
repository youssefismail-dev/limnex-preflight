"""NumPy .npy. Object arrays are pickles (scanned statically); string arrays are decoded and scanned."""

from __future__ import annotations

import ast
import re
import struct
from typing import BinaryIO

from ..context import MB, ScanContext
from ..detectors import MODE_FULL
from .binary import scan_blob_secrets
from .pickle_scan import scan_pickle_stream

MAX_STRING_ARRAY = 64 * MB


_ITEM = re.compile(r"^[<>|=]?([a-zA-Z])(\d+)")


def _note_if_truncated(f: BinaryIO, ctx: ScanContext, data_start: int, descr, shape) -> None:
    """A file shorter than its header declares is damaged: say so (degraded), never let it pass as clean."""
    m = _ITEM.match(descr) if isinstance(descr, str) else None
    if not m or not isinstance(shape, tuple) or not all(isinstance(x, int) and x >= 0 for x in shape):
        return  # structured or unusual dtypes: no size claim is made
    itemsize = int(m.group(2)) * (4 if m.group(1) == "U" else 1)
    if m.group(1) == "O":
        return  # object arrays hold a pickle, which reports its own truncation
    count = 1
    for x in shape:
        count *= x
    here = f.tell()
    f.seek(0, 2)
    size = f.tell()
    f.seek(here)
    if size < data_start + count * itemsize:
        # strings are read and scanned, numbers are not: lost strings are lost content (pre-release test 5, T7)
        ctx.note("error" if m.group(1) in "US" else "degraded",
                 f"npy: file is shorter than its header declares ({size - data_start} of {count * itemsize} payload bytes); "
                 "truncated")


def scan_npy(f: BinaryIO, ctx: ScanContext, name: str) -> None:
    f.seek(0)
    if f.read(6) != b"\x93NUMPY":
        ctx.note("error", "npy: bad magic")
        return
    major, _minor = f.read(2)
    if major == 1:
        (hlen,) = struct.unpack("<H", f.read(2))
    elif major in (2, 3):
        (hlen,) = struct.unpack("<I", f.read(4))
    else:
        ctx.note("error", f"npy: unsupported format version {major}")
        return
    if hlen > 10 * MB:
        ctx.note("error", "npy: implausible header size")
        return
    raw = f.read(hlen)
    if len(raw) < hlen:  # the padding is cut, so the dict may still parse; found by evaluation/fuzz_formats.py
        ctx.note("error", f"npy: header cut short ({len(raw)} of {hlen} bytes); truncated")   # test 4, S5
    header = raw.decode("utf-8" if major == 3 else "latin-1", "replace")
    ctx.scan_text(header, "npy header")
    data_start = f.tell()
    try:
        meta = ast.literal_eval(header.strip())  # literals only; never evaluates code
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        ctx.note("error", "npy: header is not a plain literal; payload not analysed")
        return
    descr = meta.get("descr") if isinstance(meta, dict) else None
    shape = meta.get("shape") if isinstance(meta, dict) else None
    _note_if_truncated(f, ctx, data_start, descr, shape)
    if isinstance(descr, str) and "O" in descr or (isinstance(descr, list) and "O" in repr(descr)):
        if len(raw) == hlen and not f.read(1):
            ctx.note("degraded", "npy: object array without its pickle payload; truncated")
            return
        f.seek(data_start)
        scan_pickle_stream(f, ctx, "npy object-array pickle")
        return
    if isinstance(descr, str) and len(descr) >= 3 and descr[1] in "US" and isinstance(shape, tuple):
        kind, width = descr[1], int(descr[2:] or 0)
        count = 1
        for x in shape:
            count *= int(x)
        nbytes = count * width * (4 if kind == "U" else 1)
        if 0 < nbytes <= MAX_STRING_ARRAY:
            blob = f.read(nbytes)
            if kind == "U":
                text = blob.decode("utf-32-be" if descr[0] == ">" else "utf-32-le", "replace")
                elements = [text[i: i + width] for i in range(0, len(text), width)] if width else []
            else:
                text = blob.decode("latin-1")
                elements = [text[i: i + width] for i in range(0, len(text), width)] if width else []
            # fixed-width arrays only pad to the longest element: split per element, never run neighbours together
            joined = "\n".join(e.rstrip("\x00") for e in elements)
            ctx.scan_text(joined, "npy string array")
        else:
            ctx.note("skipped", "npy string array too large; payload not scanned")
        return
    if ctx.options.deep:
        data = f.read(512 * MB)
        scan_blob_secrets(ctx, data, "npy payload")

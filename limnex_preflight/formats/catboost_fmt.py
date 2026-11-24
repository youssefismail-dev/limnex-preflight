"""CatBoost .cbm: FlatBuffers core plus embedded JSON-ish training options. Printable strings are carved.

Layout: b"CBM1", a little-endian uint32 with the size of the FlatBuffers model core, the core (it holds the training
options), then optional extra parts (CTR data) that are not parsed.
"""

from __future__ import annotations

import io
import struct
from typing import BinaryIO

from ..context import MB, ScanContext
from .binary import scan_binary_strings

CAP = 256 * MB


def scan_catboost(f: BinaryIO, ctx: ScanContext, name: str) -> None:
    # a file shorter than its declared core is cut short (found by evaluation/fuzz_formats.py: carving alone reported a
    # truncated model as clean). A cut after the core, inside the unparsed extra parts, cannot be seen this way.
    f.seek(0, io.SEEK_END)
    size = f.tell()
    f.seek(0)
    head = f.read(8)
    if head[:4] == b"CBM1" and len(head) < 8:
        ctx.note("degraded", "catboost: the header is cut short (no core size); truncated")
    elif len(head) == 8 and head[:4] == b"CBM1":
        core = struct.unpack("<I", head[4:])[0]
        if 8 + core > size:
            ctx.note("degraded", f"catboost: the model core is cut short ({size - 8} of {core} bytes); truncated")
    if not scan_binary_strings(f, ctx, "catboost", CAP):
        ctx.note("skipped", "catboost: only the first 256 MB scanned")

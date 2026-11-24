"""Helpers for opaque or loosely-structured binary data."""

from __future__ import annotations

import re
from typing import BinaryIO

from ..context import MB, ScanContext
from ..detectors import MODE_BINARY, MODE_SECRETS

_WINDOW = 16 * MB
_OVERLAP = 4096
_CARVE_WINDOW = 64 * MB
_PRINTABLE_RUN = re.compile(rb"[\x20-\x7e\t\r\n]{8,}")


_READ_CHUNK = 1 * MB


def read_limited(f: BinaryIO, cap: int) -> bytes:
    """At most `cap` bytes (exactly `cap` when the stream is at least that long).

    Read in chunks: a single ``f.read(cap + 1)`` makes CPython allocate the whole cap up front (256 MB for every text file,
    1 GB for an ONNX file inside an archive), however small the file is. Found by evaluation/fuzz_formats.py.
    """
    chunks, total = [], 0
    while total <= cap:
        chunk = f.read(min(_READ_CHUNK, cap + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)[:cap]


def scan_blob_secrets(ctx: ScanContext, data: bytes, location: str) -> None:
    """Provider-specific secrets and key blocks only, in 16 MB windows (used for raw payloads)."""
    n = len(data)
    if n < 8:
        return
    step = _WINDOW - _OVERLAP
    for off in range(0, n, step):
        if not ctx.charge(0):
            return
        text = data[off: off + _WINDOW].decode("latin-1")
        loc = f"{location} (+{off})" if off else location
        ctx.scan_text(text, loc, mode=MODE_SECRETS, dedupe=False)
        if off + _WINDOW >= n:
            break


def scan_binary_strings(f: BinaryIO, ctx: ScanContext, label: str, cap: int) -> bool:
    """Carve printable ASCII runs from a binary file and run the text detectors over each.

    Returns False if the file was larger than ``cap`` (only the first ``cap`` bytes were read).
    """
    f.seek(0)
    pos = 0
    carry = b""
    complete = True
    while True:
        chunk = read_limited(f, _CARVE_WINDOW)  # not f.read(_CARVE_WINDOW): that allocates 64 MB even for a 2 KB file
        if not chunk:
            break
        buf = carry + chunk
        base = pos - len(carry)
        last_end = 0
        for m in _PRINTABLE_RUN.finditer(buf):
            # keep a run that touches the end of the window for the next round
            if m.end() == len(buf) and len(chunk) == _CARVE_WINDOW:
                break
            last_end = m.end()
            ctx.scan_text(m.group(0).decode("ascii", "replace"), f"{label} offset {base + m.start()}", mode=MODE_BINARY)
        carry = buf[last_end:][-_OVERLAP * 8:] if len(chunk) == _CARVE_WINDOW else b""
        pos += len(chunk)
        if pos >= cap:
            complete = not f.read(1)
            break
        if not ctx.charge(len(chunk)):
            return False
    return complete


def scan_stream_deep(f: BinaryIO, ctx: ScanContext, start: int, label: str, cap: int = 4096 * MB) -> None:
    """--deep: carve provider-specific secrets out of raw payload bytes (tensor data) from ``start``."""
    f.seek(start)
    pos = start
    carry = b""
    while pos - start < cap:
        chunk = f.read(_WINDOW)
        if not chunk:
            break
        buf = carry + chunk
        scan_blob_secrets(ctx, buf, f"{label} offset {pos - len(carry)}")
        carry = buf[-_OVERLAP:]
        pos += len(chunk)
        if not ctx.charge(len(chunk)):
            return
    if pos - start >= cap and f.read(1):
        ctx.note("skipped", f"{label}: --deep reads at most {cap // MB} MB of one payload; the rest was not scanned")

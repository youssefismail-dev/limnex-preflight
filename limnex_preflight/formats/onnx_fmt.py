"""ONNX: a small protobuf wire-format reader (no ``onnx`` package needed, nothing executed).

Scanned: producer/domain/doc strings, ``metadata_props`` key/values, graph and node names, node doc strings,
tensor (initializer) names, value-info names, string attributes, string tensors and external-data locations.
Raw tensor payloads (``raw_data`` etc.) are skipped.
"""

from __future__ import annotations

import mmap
import re
from typing import BinaryIO, Dict, Optional, Tuple

from ..context import MB, ScanContext
from ..masking import clean
from .binary import read_limited

# field number -> ("s", name) string | ("m", schema, name) message | absent = ignored
_SCHEMAS: Dict[str, Dict[int, tuple]] = {
    "Model": {2: ("s", "producer_name"), 3: ("s", "producer_version"), 4: ("s", "domain"), 6: ("s", "doc_string"),
              7: ("m", "Graph", "graph"), 14: ("m", "StrStr", "metadata_props"), 25: ("m", "Function", "functions")},
    "Graph": {1: ("m", "Node", "node"), 2: ("s", "name"), 5: ("m", "Tensor", "initializer"), 10: ("s", "doc_string"),
              11: ("m", "ValueInfo", "input"), 12: ("m", "ValueInfo", "output"), 13: ("m", "ValueInfo", "value_info"),
              15: ("m", "SparseTensor", "sparse_initializer"), 16: ("m", "StrStr", "metadata_props")},
    "Node": {1: ("s", "input"), 2: ("s", "output"), 3: ("s", "name"), 4: ("s", "op_type"), 5: ("m", "Attr", "attribute"),
             6: ("s", "doc_string"), 7: ("s", "domain"), 9: ("m", "StrStr", "metadata_props")},
    "Attr": {1: ("s", "name"), 4: ("s", "s"), 5: ("m", "Tensor", "t"), 6: ("m", "Graph", "g"), 9: ("s", "strings"),
             10: ("m", "Tensor", "tensors"), 11: ("m", "Graph", "graphs"), 13: ("s", "doc_string"), 21: ("s", "ref_attr_name")},
    "Tensor": {8: ("s", "name"), 12: ("s", "doc_string"), 6: ("s", "string_data"), 13: ("m", "StrStr", "external_data"),
               16: ("m", "StrStr", "metadata_props")},
    "ValueInfo": {1: ("s", "name"), 3: ("s", "doc_string"), 4: ("m", "StrStr", "metadata_props")},
    "StrStr": {1: ("s", "key"), 2: ("s", "value")},
    "Function": {1: ("s", "name"), 4: ("s", "input"), 5: ("s", "output"), 6: ("s", "attribute"), 7: ("m", "Node", "node"),
                 8: ("s", "doc_string"), 10: ("s", "domain"), 14: ("m", "StrStr", "metadata_props")},
    "SparseTensor": {1: ("m", "Tensor", "values"), 2: ("m", "Tensor", "indices")},
}

MAX_DEPTH = 40
IN_MEMORY_CAP = 1024 * MB
_DRIVE = re.compile(r"^[A-Za-z]:")


def _outside_folder(location: str) -> bool:
    """Review 2026-11-11: external data must sit next to the model. An absolute path, a drive, a home directory, a URL
    or a '..' segment leaves the folder, a known class of loader weakness (path traversal)."""
    p = location.strip().replace("\\", "/")
    return (p.startswith(("/", "~")) or bool(_DRIVE.match(p)) or "://" in p
            or any(part == ".." for part in p.split("/")))


class _Bad(Exception):
    pass


def _varint(buf, i: int, end: int) -> Tuple[int, int]:
    shift = 0
    result = 0
    while True:
        if i >= end:
            raise _Bad("truncated varint")
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7
        if shift > 70:
            raise _Bad("varint too long")


def _fields(buf, i: int, end: int):
    """Yield (field_number, wire_type, value) where value is an int or a (start, end) slice."""
    while i < end:
        tag, i = _varint(buf, i, end)
        fno, wire = tag >> 3, tag & 7
        if wire == 0:
            val, i = _varint(buf, i, end)
            yield fno, wire, val
        elif wire == 2:
            ln, i = _varint(buf, i, end)
            if i + ln > end:
                raise _Bad("length-delimited field past end")
            yield fno, wire, (i, i + ln)
            i += ln
        elif wire == 1:
            i += 8
        elif wire == 5:
            i += 4
        else:
            raise _Bad(f"unsupported wire type {wire}")
        if i > end:
            raise _Bad("field past end")


def looks_like_onnx(head: bytes) -> bool:
    # ModelProto normally starts with ir_version (field 1, varint)
    return len(head) >= 2 and head[0] == 0x08


class _Walk:
    def __init__(self, ctx: ScanContext, buf):
        self.ctx, self.buf = ctx, buf

    def text(self, s: int, e: int) -> str:
        return bytes(self.buf[s:e]).decode("utf-8", "replace")

    def message(self, schema: str, s: int, e: int, loc: str, depth: int) -> None:
        if depth > MAX_DEPTH:
            raise _Bad("nesting too deep")
        spec = _SCHEMAS[schema]
        counters: Dict[str, int] = {}
        if schema == "StrStr":
            self.strstr(s, e, loc)
            return
        for fno, wire, val in _fields(self.buf, s, e):
            item: Optional[tuple] = spec.get(fno)
            if item is None or wire != 2:
                continue
            name = item[-1]
            idx = counters.get(name, 0)
            counters[name] = idx + 1
            sub_loc = f"{loc}.{name}" if loc else name
            if item[0] == "s":
                self.ctx.scan_text(self.text(*val), f"onnx {sub_loc}" + (f"[{idx}]" if name in ("input", "output", "strings") else ""))
            else:
                self.message(item[1], val[0], val[1], f"{sub_loc}[{idx}]" if name not in ("graph", "t", "g", "values", "indices") else sub_loc, depth + 1)

    def strstr(self, s: int, e: int, loc: str) -> None:
        key = value = None
        for fno, wire, val in _fields(self.buf, s, e):
            if wire != 2:
                continue
            if fno == 1:
                key = self.text(*val)
            elif fno == 2:
                value = self.text(*val)
        where = f'onnx {loc}["{(key or "")[:60]}"]'
        if key:
            self.ctx.scan_text(key, f"{where} (key)")
        if value:
            self.ctx.scan_text(value, where, key=key)
            if key == "location" and ".external_data" in loc and _outside_folder(value):
                self.ctx.add("onnx-external-data-path", value, clean(value, 100), where,
                             message="ONNX external data points outside the model's folder: a loader that follows it "
                                     "reads (or, when saving, writes) a file elsewhere on the machine")


def scan_onnx(f: BinaryIO, ctx: ScanContext, name: str) -> None:
    buf = None
    mm = None
    try:
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)  # real file: no copy
            buf = mm
        except (AttributeError, OSError, ValueError, io_unsupported()):
            f.seek(0)
            data = read_limited(f, IN_MEMORY_CAP)
            if len(data) == IN_MEMORY_CAP:
                ctx.note("skipped", "onnx: larger than 1 GB inside an archive; not scanned")
                return
            buf = data
        ctx.charge(min(len(buf), 64 * MB))
        if ctx.options.deep:
            from .binary import scan_stream_deep

            scan_stream_deep(f, ctx, 0, "onnx raw payload")
        try:
            _Walk(ctx, buf).message("Model", 0, len(buf), "", 0)
        except _Bad as exc:
            ctx.note("error", f"onnx: protobuf parse stopped early ({exc}); strings read so far were scanned")
        except RecursionError:
            ctx.note("error", "onnx: nesting too deep")
    finally:
        if mm is not None:
            mm.close()


def io_unsupported():
    import io

    return io.UnsupportedOperation

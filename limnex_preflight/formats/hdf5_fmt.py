"""HDF5 / Keras .h5. With h5py: attributes, object names and small string datasets. Without: raw strings.

The h5py part runs in a child process (hdf5_child.py; pre-release test 2026-11-19, finding F1: a damaged file made h5py
loop forever, so a scan or a CI job hung). The child reports what it reads at least every half second; when nothing
arrives for STALL_S seconds it is stopped, and the file is scanned as raw strings with the reduced coverage reported, as
when h5py fails. A healthy file is never cut short for being big: 200,000 objects (the limit) took 35 s, steadily
reporting. The file is never written to disk: a file on disk is passed by its path, an uncompressed archive member by
the archive's path and the member's place in it (read in place), anything else through a pipe, in chunks (retest of
release candidate 8: a .keras weights member was read whole and then piped, so a 763 MB model took 1.55 GB).
"""

from __future__ import annotations

import importlib.util
import inspect
import io
import json
import math
import os
import queue
import subprocess
import sys
import threading
from typing import BinaryIO, Optional

from ..context import MB, ScanContext
from . import hdf5_child
from .binary import scan_binary_strings
from .textual import walk_json

MAX_STRINGS = 10000
MAX_OBJECTS = 200000
PIPE_CHUNK = 1 * MB
RAW_CAP = 256 * MB
STALL_ENV = "LIMNEX_PREFLIGHT_HDF5_STALL_SECONDS"
STALL_S = 10.0                       # see _stall_s()
HAVE_H5PY = importlib.util.find_spec("h5py") is not None and importlib.util.find_spec("numpy") is not None
_CHILD_SOURCE: Optional[str] = None


def _stall_s() -> float:
    """Seconds without any report from the child before it counts as stuck. The child reports at least every 0.5 s
    while it works (hdf5_child.HEARTBEAT_S), so 10 s is 20 missed reports; a single h5py call that takes longer than
    that on a small model file is the hang this guards against. Override with LIMNEX_PREFLIGHT_HDF5_STALL_SECONDS."""
    try:
        value = float(os.environ.get(STALL_ENV, STALL_S))
    except ValueError:
        return STALL_S
    if not math.isfinite(value):   # pre-release test 2, N18: "inf" made every HDF5 file fail with OverflowError
        return STALL_S
    return min(max(1.0, value), 3600.0)


class _Skips(hdf5_child.Skips):
    """Parts h5py could not give us, or that were left out by a limit. Review 2026-11-11: each used to be skipped
    silently, so the file looked fully scanned. They are reported as reduced coverage ("degraded"), once per kind."""

    @classmethod
    def from_dict(cls, d: dict) -> "_Skips":
        s = cls()
        s.attributes, s.datasets = int(d.get("attributes", 0)), int(d.get("datasets", 0))
        s.large_string_datasets, s.object_limit = int(d.get("large_string_datasets", 0)), bool(d.get("object_limit"))
        return s

    def report(self, ctx: ScanContext) -> None:
        if self.attributes:
            ctx.note("degraded", f"hdf5: {self.attributes} attribute(s) could not be read; not scanned")
        if self.datasets:
            ctx.note("degraded", f"hdf5: {self.datasets} string dataset(s) could not be read; not scanned")
        if self.large_string_datasets:
            ctx.note("degraded", f"hdf5: {self.large_string_datasets} string dataset(s) with more than {MAX_STRINGS} "
                                 "entries; not scanned")
        if self.object_limit:
            ctx.note("degraded", f"hdf5: more than {MAX_OBJECTS} objects; the rest was not scanned")


def _scan_value(ctx: ScanContext, s: str, what: str, key: str) -> None:
    st = s.lstrip()
    if st[:1] in "{[" and len(st) > 1:
        try:
            walk_json(ctx, json.loads(s), root=f"{what}: $")
            return
        except ValueError:
            pass
    ctx.scan_text(s, what, key=key)


def _apply(ctx: ScanContext, item) -> None:
    kind, text, what, key = item
    if kind == "value":
        _scan_value(ctx, text, what, key)
    else:
        ctx.scan_text(text, what)


def _attrs(ctx: ScanContext, attrs, where: str, skips: _Skips) -> None:
    for item in hdf5_child.attrs_items(attrs, where, skips, MAX_STRINGS):
        _apply(ctx, item)


def _child_command(header: dict) -> list:
    global _CHILD_SOURCE
    if _CHILD_SOURCE is None:
        _CHILD_SOURCE = inspect.getsource(hdf5_child)
    return [sys.executable, "-E", "-c", _CHILD_SOURCE, json.dumps(header)]


def _run_child(f: BinaryIO, ctx: Optional[ScanContext] = None) -> dict:
    """Run the child; scan its items as they arrive (when `ctx` is given). Returns {"done": True, "skips": ...},
    {"error": ...} or {"stalled": seconds}; "items" holds the items when no `ctx` is given (tests, measurements)."""
    path = getattr(f, "name", None)
    on_disk = isinstance(f, io.BufferedReader) and isinstance(path, str) and os.path.isfile(path)
    window = None if on_disk else getattr(f, "preflight_window", None)     # set by archive.scan_zip
    cwd = os.getcwd()
    header = {"path": os.path.abspath(path) if on_disk else None, "max_strings": MAX_STRINGS, "max_objects": MAX_OBJECTS,
              "sys_path": [p for p in sys.path if p and os.path.abspath(p) != cwd]}
    if window is not None:
        header["path"], header["offset"], header["length"] = window
    piped = header["path"] is None
    if piped:
        f.seek(0)
    try:
        proc = subprocess.Popen(_child_command(header), stdin=subprocess.PIPE if piped else subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError as exc:
        return {"error": type(exc).__name__}
    lines: "queue.Queue[Optional[bytes]]" = queue.Queue()
    read_failed: list = []

    def read() -> None:
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def write() -> None:
        try:
            while True:
                chunk = f.read(PIPE_CHUNK)
                if not chunk:
                    break
                try:
                    proc.stdin.write(chunk)
                except OSError:          # the child stopped reading (done, failed or stopped)
                    return
        except Exception as exc:         # reading the member failed (e.g. its CRC at the end): raised below, as before
            read_failed.append(exc)
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass
    threading.Thread(target=read, daemon=True).start()
    writer = threading.Thread(target=write, daemon=True) if piped else None
    if writer is not None:
        writer.start()
    stall = _stall_s()
    collected: list = []
    result: dict = {"error": "ChildProcessError"}
    try:
        while True:
            try:
                line = lines.get(timeout=stall)
            except queue.Empty:
                result = {"stalled": stall}
                break
            if line is None:
                break
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            for item in msg.get("items", ()):
                if ctx is None:
                    collected.append(item)
                else:
                    _apply(ctx, item)
            if msg.get("done"):
                result = {"done": True, "skips": msg.get("skips", {})}
            elif "error" in msg:
                result = {"error": str(msg["error"])}
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        if writer is not None:
            writer.join()                # `f` is not shared with the raw-string fallback while it still reads
    if read_failed:
        raise read_failed[0]
    if ctx is None:
        result["items"] = collected
    return result


def scan_hdf5(f: BinaryIO, ctx: ScanContext, name: str) -> None:
    if HAVE_H5PY:
        res = _run_child(f, ctx)
        if res.get("done"):
            _Skips.from_dict(res.get("skips", {})).report(ctx)
            return
        if "stalled" in res:
            ctx.note("degraded", f"hdf5: h5py made no progress for {res['stalled']:g} s and was stopped; "
                                 "scanned as raw strings only")
        else:
            ctx.note("degraded", f"hdf5: h5py could not read the file ({res['error']}); scanned as raw strings only")
    else:
        ctx.note("degraded", "hdf5: h5py not installed; scanned as raw strings only (pip install 'limnex-preflight[hdf5]')")
    f.seek(0)
    complete = scan_binary_strings(f, ctx, "hdf5 raw", RAW_CAP)
    if not complete:
        ctx.note("skipped", f"hdf5: only the first {RAW_CAP // MB} MB scanned in raw-string mode")

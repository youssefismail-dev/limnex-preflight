"""The h5py side of the HDF5 / Keras .h5 reader, run in a child process that the parent stops when it makes no progress.

Pre-release test 2026-11-19 (finding F1): an 11.5 KB damaged file made h5py loop forever at 100% CPU (Linux builds), so a
scan, and a CI job, hung until killed. h5py's C library cannot be interrupted from Python, so the parse runs in its own
process. A healthy file can take long (35 s at the 200,000-object limit, measured 2026-11-19), so the parent does not use
a fixed time limit: the child reports what it has read at least every HEARTBEAT_S seconds, and the parent stops it only
when nothing arrives for a while (hdf5_fmt.py). This module only READS: it returns the texts to scan (names, attribute
values, small string datasets) and what it had to skip; the parent scans them, so a finding is made exactly as before.

Self-contained on purpose (standard library, h5py and numpy only): the parent runs this file's source with
`python -E -c`, so it works whether or not the package is installed. Its first statement removes the working folder from
the import path, so nothing there can shadow an import; it then uses the parent's own import path, so it finds h5py
exactly where the parent would (`-I` would also hide the user site-packages, where h5py or numpy may be installed).

Protocol: argv[1] is a JSON header {"path": str or null, "max_strings": int, "max_objects": int, "sys_path": [...]},
optionally with "offset" and "length": the file is then those bytes of "path" (an uncompressed archive member, read in
place). With "path" null the file's bytes come on stdin. Then JSON lines on stdout: {"items": [[kind, text, what, key], ...]} as
the reading goes (possibly empty: a heartbeat), and last {"done": true, "skips": {...}} or {"error": "<exception type>"}.
kind "text" is scanned as text, kind "value" may be JSON (see hdf5_fmt._scan_value).
"""

import sys

if __name__ == "__main__":  # the child: before any other import, the working folder leaves the import path
    sys.path[:] = [p for p in sys.path if p != ""]

import io  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

HEARTBEAT_S = 0.5
CHUNK = 2000


class Skips:
    """Parts h5py could not give us, or that were left out by a limit (reported as reduced coverage by the parent)."""

    def __init__(self):
        self.attributes = 0
        self.datasets = 0
        self.large_string_datasets = 0
        self.object_limit = False

    def as_dict(self):
        return {"attributes": self.attributes, "datasets": self.datasets,
                "large_string_datasets": self.large_string_datasets, "object_limit": self.object_limit}


class Emitter:
    """Collects items; with a stream, writes them out in chunks and at least every HEARTBEAT_S seconds."""

    def __init__(self, stream=None):
        self.stream, self.items, self.last = stream, [], time.monotonic()

    def add(self, item):
        self.items.append(item)
        self.tick()

    def tick(self, force=False):
        if self.stream is None:
            return
        now = time.monotonic()
        if force or len(self.items) >= CHUNK or now - self.last >= HEARTBEAT_S:
            self.stream.write(json.dumps({"items": self.items}) + "\n")
            self.stream.flush()
            self.items, self.last = [], now


class Window(io.RawIOBase):
    """Bytes [offset, offset + length) of an open file, read as a file of its own (an uncompressed archive member)."""

    def __init__(self, f, offset, length):
        super().__init__()
        self.f, self.offset, self.length, self.pos = f, offset, length, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, pos, whence=0):
        new = {0: 0, 1: self.pos, 2: self.length}[whence] + pos
        if new < 0:
            raise ValueError("negative seek position")
        self.pos = new
        return new

    def readinto(self, b):
        n = max(0, min(len(b), self.length - self.pos))
        if n == 0:
            return 0
        self.f.seek(self.offset + self.pos)
        got = self.f.readinto(memoryview(b)[:n]) or 0
        self.pos += got
        return got


def strings_of(value, max_strings):
    import numpy as np
    out = []
    if isinstance(value, bytes):
        out.append(value.decode("utf-8", "replace"))
    elif isinstance(value, str):
        out.append(value)
    elif isinstance(value, np.ndarray) and value.dtype.kind in "SUO":
        for item in value.ravel().tolist()[:max_strings]:
            out.extend(strings_of(item, max_strings))
    elif isinstance(value, (np.bytes_, np.str_)):
        out.append(str(value, "utf-8", "replace") if isinstance(value, np.bytes_) else str(value))
    elif isinstance(value, (list, tuple)):
        for item in list(value)[:max_strings]:
            out.extend(strings_of(item, max_strings))
    return out


def attrs_items(attrs, where, skips, max_strings, out=None):
    """The texts of one object's attributes. With `out` (an Emitter) they are added to it as they are read."""
    items = [] if out is None else None
    add = items.append if out is None else out.add
    for key in list(attrs.keys())[:max_strings]:
        add(["text", str(key), "hdf5 attribute name at " + where, None])
        try:
            value = attrs[key]
        except Exception:
            skips.attributes += 1
            continue
        for s in strings_of(value, max_strings):
            add(["value", s, 'hdf5 attribute "%s" of %s' % (key, where), str(key)])
    return items


def extract(fileobj, max_strings, max_objects, out=None):
    """Read one HDF5 file. Returns {"items": [...], "skips": {...}}; with `out`, items go to it and "items" is empty."""
    import h5py
    out = out or Emitter()
    skips = Skips()
    with h5py.File(fileobj, "r") as h:
        attrs_items(h.attrs, "/", skips, max_strings, out)
        count = [0]

        def visit(path, obj):
            count[0] += 1
            if count[0] > max_objects:
                skips.object_limit = True
                return
            where = "/" + path
            out.add(["text", where, "hdf5 object name", None])
            attrs_items(obj.attrs, where, skips, max_strings, out)
            if isinstance(obj, h5py.Dataset):
                kind = obj.dtype.kind
                is_str = kind in "SUO" and (kind != "O" or h5py.check_string_dtype(obj.dtype) is not None)
                if not is_str:
                    return
                if obj.size > max_strings:
                    skips.large_string_datasets += 1
                    return
                try:
                    values = strings_of(obj[()], max_strings)
                except Exception:
                    skips.datasets += 1
                    return
                for s in values:
                    out.add(["value", s, "hdf5 dataset " + where, path.rsplit("/", 1)[-1]])

        h.visititems(visit)
    return {"items": out.items if out.stream is None else [], "skips": skips.as_dict()}


def main():
    header = json.loads(sys.argv[1])
    if header.get("sys_path"):
        sys.path[:] = [p for p in header["sys_path"] if p]
    out = Emitter(sys.stdout)
    try:
        if header.get("path"):
            with open(header["path"], "rb") as f:
                if header.get("offset") is not None:
                    f = Window(f, int(header["offset"]), int(header["length"]))
                res = extract(f, header["max_strings"], header["max_objects"], out)
        else:
            res = extract(io.BytesIO(sys.stdin.buffer.read()), header["max_strings"], header["max_objects"], out)
        out.tick(force=True)
        sys.stdout.write(json.dumps({"done": True, "skips": res["skips"]}) + "\n")
    except Exception as exc:  # corrupt file, unsupported feature ...
        out.tick(force=True)
        sys.stdout.write(json.dumps({"error": type(exc).__name__}) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()

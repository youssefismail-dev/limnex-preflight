"""Archives: zip (also .pt/.pth/.npz/.keras/.whl ...), tar, and single-stream gzip/bz2/xz/zlib wrappers.

Nothing is extracted to disk; members are streamed straight into the matching scanner.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import os
import re
import tarfile
import zipfile
import zlib
from typing import BinaryIO, Callable, Optional

from ..context import MB, ScanContext
from ..masking import clean

Dispatch = Callable[[BinaryIO, ScanContext, str, int], None]

# Raw tensor storages inside PyTorch zip archives: "<archive>/data/0", "<archive>/data/12" ...
_TORCH_STORAGE = re.compile(r"(?:^|/)data/\d+$")
BOMB_RATIO = 1000
BOMB_MIN_SIZE = 100 * MB
# PyTorch's loader looks its pickle up by this member name. Zip member names are never compressed, so the name can be
# found in the raw bytes even when the archive cannot be opened.
_TORCH_PICKLE_NAME = b"data.pkl"
_NAME_SEARCH_HEAD = 64 * 1024
_NAME_SEARCH_TAIL = 4 * MB


def _looks_like_torch_archive(f: BinaryIO) -> bool:
    """True when the raw bytes hold PyTorch's pickle member name (local headers come first, the directory last)."""
    try:
        pos = f.tell()
        f.seek(0)
        head = f.read(_NAME_SEARCH_HEAD)
        f.seek(0, io.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - _NAME_SEARCH_TAIL))
        tail = f.read(_NAME_SEARCH_TAIL)
        f.seek(pos)
    except (OSError, ValueError):
        return False
    return _TORCH_PICKLE_NAME in head or _TORCH_PICKLE_NAME in tail


def _check_member_name(ctx: ScanContext, name: str, kind: str) -> None:
    """A member whose name leaves the extraction folder ("../../x", "/etc/x", "C:\\x") writes outside the target when
    the archive is unpacked (a .nemo or .keras file is unpacked by its loader). User review before launch: such a tar
    member passed unflagged. The name is reported masked like any value."""
    n = name.replace("\\", "/")
    if n.startswith("/") or re.match(r"^[A-Za-z]:", n) or ".." in n.split("/"):
        ctx.add("archive-path-traversal", name, clean(name, 100), f"{kind} member name")


def _unscannable(ctx: ScanContext, what: str, exc: BaseException) -> None:
    """Review 2026-11-11: PyTorch reads zips with its own, more tolerant reader. An archive that it may still load but that
    zipfile cannot read must not pass the default gate as a mere scan problem (a warning in the GitHub Action)."""
    value = f"{what}: {type(exc).__name__}"
    ctx.add("unscannable-model-archive", value, clean(value, 100), what,
            message=f"PyTorch-style archive that could not be read ({what}: {type(exc).__name__}); its pickle was not "
                    "checked, and PyTorch may still load it")


def scan_zip(f: BinaryIO, ctx: ScanContext, name: str, dispatch: Dispatch) -> None:
    if ctx.depth >= ctx.options.max_depth:
        ctx.note("skipped", f"archive nesting deeper than {ctx.options.max_depth}; not opened")
        return
    try:
        zf = zipfile.ZipFile(f)
    except (zipfile.BadZipFile, OSError, ValueError, NotImplementedError) as exc:
        ctx.note("error", f"zip: cannot open ({type(exc).__name__}: {exc})")
        if _looks_like_torch_archive(f):
            _unscannable(ctx, "zip directory", exc)
        return
    with zf:
        _check_zip_extent(f, zf, ctx)
        if zf.comment:
            ctx.scan_text(zf.comment.decode("utf-8", "replace"), "zip comment")
        infos = zf.infolist()
        if len(infos) > ctx.options.max_members:
            ctx.note("skipped", f"zip has {len(infos)} members; only the first {ctx.options.max_members} scanned")
            infos = infos[: ctx.options.max_members]
        for info in infos:
            mname = info.filename
            ctx.scan_text(mname, "zip member name")
            _check_member_name(ctx, mname, "zip")
            if info.is_dir():
                continue
            child = ctx.child(mname)
            if info.flag_bits & 0x1:
                child.note("skipped", "encrypted zip member; cannot be scanned")
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                child.note("skipped", "symbolic link inside archive ignored")
                continue
            if info.file_size > ctx.options.max_member_bytes:
                child.note("skipped", f"member larger than {ctx.options.max_member_bytes // MB} MB")
                continue
            if info.file_size > BOMB_MIN_SIZE and info.compress_size and info.file_size / info.compress_size > BOMB_RATIO:
                child.note("skipped", "suspicious compression ratio (possible zip bomb); not read")
                continue
            if _TORCH_STORAGE.search(mname) and not ctx.options.deep:
                continue  # raw tensor bytes; only read with --deep
            try:
                with zf.open(info) as mf:
                    window = stored_window(f, info)
                    if window is not None:
                        mf.preflight_window = window          # the HDF5 reader opens it in place (hdf5_fmt.py)
                    dispatch(mf, child, mname, info.file_size)
            except (zipfile.BadZipFile, RuntimeError, NotImplementedError, EOFError, OSError, zlib.error) as exc:
                child.note("error", f"cannot read zip member ({type(exc).__name__}: {exc})")
                if mname.endswith(".pkl"):
                    _unscannable(child, "pickle member", exc)
            if ctx.shared.over_budget():
                ctx.note("skipped", "scan budget exhausted; remaining zip members not scanned")
                return


def stored_window(f: BinaryIO, info: zipfile.ZipInfo) -> Optional[tuple]:
    """Where an uncompressed (stored) member's bytes are in an archive on disk: (path, offset, length), or None.

    Retest of release candidate 8: a .keras file's weights member went to the HDF5 child through a pipe, read whole first,
    so a 763 MB model took 1.55 GB. Keras stores that member uncompressed, so the child can open it in place instead.
    The offset comes from the member's own local header (the bytes zipfile itself reads them from)."""
    path = getattr(f, "name", None)
    if info.compress_type != zipfile.ZIP_STORED or not isinstance(f, io.BufferedReader) or not isinstance(path, str) \
            or not os.path.isfile(path):
        return None
    try:
        pos = f.tell()
        f.seek(info.header_offset)
        local = f.read(30)
        f.seek(pos)
        size = os.path.getsize(path)
    except (OSError, ValueError):
        return None
    if len(local) != 30 or local[:4] != b"PK\x03\x04":
        return None
    start = info.header_offset + 30 + int.from_bytes(local[26:28], "little") + int.from_bytes(local[28:30], "little")
    if start + info.file_size > size:
        return None
    return os.path.abspath(path), start, info.file_size


def _check_zip_extent(f: BinaryIO, zf: zipfile.ZipFile, ctx: ScanContext) -> None:
    """The archive zipfile found must span the whole stream; otherwise part of the file is not being scanned.

    zipfile looks for the LAST end-of-central-directory record. A truncated archive that contains a stored inner zip is
    then read as that inner zip, and two archives glued together are read as the second one. Either way members exist
    that are never listed. Found by evaluation/fuzz_formats.py (a cut-off zip came back clean).
    """
    try:
        pos = f.tell()
        f.seek(0, io.SEEK_END)
        size = f.tell()
        end_record = size - 22 - len(zf.comment)
        f.seek(max(0, end_record))
        ends_at_eof = end_record >= 0 and f.read(4) == b"PK\x05\x06"
        f.seek(pos)
    except (OSError, ValueError):
        return  # a stream that cannot seek: zipfile could not have opened it either
    first = min((i.header_offset for i in zf.infolist()), default=0)
    problems = []
    if first > 0:
        problems.append(f"{first} bytes before the first member")
    if not ends_at_eof:
        problems.append("data after the end of the archive")
    if problems:
        ctx.note("degraded", f"zip: {' and '.join(problems)} were not scanned; the file may be truncated or several "
                             "archives joined together, and only the members of the last one were read")


def scan_tar(f: BinaryIO, ctx: ScanContext, name: str, dispatch: Dispatch) -> None:
    if ctx.depth >= ctx.options.max_depth:
        ctx.note("skipped", f"archive nesting deeper than {ctx.options.max_depth}; not opened")
        return
    try:
        tf = tarfile.open(fileobj=f, mode="r:*")
    except (tarfile.TarError, OSError, EOFError) as exc:
        ctx.note("error", f"tar: cannot open ({type(exc).__name__}: {exc})")
        return
    with tf:
        _scan_tar_members(tf, ctx, dispatch, buffer_cap=None)


def _scan_tar_members(tf: tarfile.TarFile, ctx: ScanContext, dispatch: Dispatch, buffer_cap: Optional[int]) -> None:
    """Every member of an open tar. With `buffer_cap` (a tar read as a stream, which cannot seek) each member is read into
    memory first, up to that many bytes; a larger member is reported and left out, the others are still scanned."""
    count = 0
    for member in tf:
        count += 1
        if count > ctx.options.max_members:
            ctx.note("skipped", f"tar has more than {ctx.options.max_members} members; remainder not scanned")
            return
        ctx.scan_text(member.name, "tar member name")
        _check_member_name(ctx, member.name, "tar")
        child = ctx.child(member.name)
        if not member.isreg():
            continue
        if member.size > ctx.options.max_member_bytes:
            child.note("skipped", "member too large")
            continue
        if _TORCH_STORAGE.search(member.name) and not ctx.options.deep:
            continue
        if buffer_cap is not None and member.size > buffer_cap:
            child.note("skipped", f"member decompresses to more than {buffer_cap // MB} MB; not scanned")
            continue
        try:
            mf = tf.extractfile(member)
            if mf is not None:
                if buffer_cap is None:
                    dispatch(mf, child, member.name, member.size)
                else:
                    data = mf.read()
                    ctx.charge(len(data))
                    dispatch(io.BytesIO(data), child, member.name, len(data))
        except (tarfile.TarError, OSError, EOFError, zlib.error, lzma.LZMAError) as exc:
            child.note("error", f"cannot read tar member ({type(exc).__name__}: {exc})")
            if buffer_cap is not None:   # a stream cannot skip past a member it failed to read
                return
        if ctx.shared.over_budget():
            return


class _Prefixed(io.RawIOBase):
    """The bytes already read from a stream, then the rest of the stream (a decompressor cannot be rewound)."""

    def __init__(self, head: bytes, rest: BinaryIO):
        self._head, self._rest = head, rest

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        if self._head:
            n = min(len(b), len(self._head))
            b[:n] = self._head[:n]
            self._head = self._head[n:]
            return n
        chunk = self._rest.read(len(b))
        b[:len(chunk)] = chunk
        return len(chunk)


def _inner_name(name: str) -> str:
    inner = re.sub(r"\.(gz|gzip|bz2|xz|z|zlib)$", "", name, flags=re.I) or name
    inner = re.sub(r"\.tgz$", ".tar", inner, flags=re.I)
    return inner if inner != name else name + ".decompressed"


def _bounded(read: Callable[[int], bytes], cap: int, ctx: ScanContext, label: str) -> bytes | None:
    data = read(cap + 1)
    if len(data) > cap:
        ctx.note("skipped", f"{label}: decompresses to more than {cap // MB} MB; not scanned")
        return None
    return data


def scan_compressed(f: BinaryIO, ctx: ScanContext, name: str, kind: str, dispatch: Dispatch,
                    pickle_inside: bool = False) -> None:
    """gzip / bz2 / xz / zlib single-stream wrapper (also how joblib compresses its pickles). ``pickle_inside``: the
    content is read as a pickle whatever it starts with (scanner._pickle_inside)."""
    if ctx.depth >= ctx.options.max_depth:
        ctx.note("skipped", "nesting too deep; not decompressed")
        return
    cap = ctx.options.max_decompressed_bytes
    f.seek(0)
    try:
        if kind == "zlib":
            d = zlib.decompressobj()
            data = d.decompress(f.read(), cap + 1)
            if len(data) > cap:
                ctx.note("skipped", f"zlib stream decompresses to more than {cap // MB} MB; not scanned")
                return
            # zlib reports neither of these as an error (gzip/bz2/xz raise EOFError for a cut stream); found by fuzzing
            if not d.eof:   # content is lost, as with a cut gzip/bz2/xz stream: an error (pre-release test 4, S5)
                ctx.note("error", "zlib: the compressed stream is cut short; only the part that decompressed was scanned")
            elif d.unused_data:
                ctx.note("degraded", f"zlib: {len(d.unused_data)} bytes after the end of the compressed stream were not scanned")
        else:
            dec_cm = gzip.GzipFile(fileobj=f) if kind == "gzip" else (bz2.BZ2File(f) if kind == "bz2" else lzma.LZMAFile(f))
            with dec_cm as dec:
                head = dec.read(512)
                if len(head) >= 262 and head[257:262] == b"ustar":
                    # A compressed tar (model.tar.gz, .tgz ...): read as a stream, member by member, so the archive as a
                    # whole may be larger than the limit. Pre-release test 2026-11-19, finding F8: it was decompressed
                    # whole and dropped whole above 512 MB, so a typical SageMaker model.tar.gz was not scanned at all.
                    tctx = ctx.child(_inner_name(name))
                    if tctx.depth >= ctx.options.max_depth:
                        tctx.note("skipped", f"archive nesting deeper than {ctx.options.max_depth}; not opened")
                        return
                    with tarfile.open(fileobj=io.BufferedReader(_Prefixed(head, dec)), mode="r|") as tf:
                        _scan_tar_members(tf, tctx, dispatch, buffer_cap=cap)
                    return
                got = _bounded(dec.read, cap - len(head), ctx, kind)
            if got is None:
                return
            data = head + got
    except (OSError, EOFError, zlib.error, lzma.LZMAError, ValueError, tarfile.TarError) as exc:
        ctx.note("error", f"{kind}: cannot decompress ({type(exc).__name__}: {exc})")
        return
    inner = _inner_name(name)
    ctx.charge(len(data))
    # the member keeps its name in reports; only the name its format is judged by gets a pickle extension
    dispatch(io.BytesIO(data), ctx.child(inner), inner + ".pkl" if pickle_inside else inner, len(data))

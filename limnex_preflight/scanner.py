"""File-type detection, dispatch and the public ``scan_paths`` API."""

from __future__ import annotations

import codecs
import fnmatch
import io
import lzma
import os
import re
import stat
import struct
import zlib
from dataclasses import dataclass, field
from typing import BinaryIO, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .context import MB, ScanContext, ScanOptions
from .detectors import MODE_FULL
from .formats import archive
from .formats.binary import scan_stream_deep
from .formats.catboost_fmt import scan_catboost
from .formats.gguf import scan_gguf
from .formats.hdf5_fmt import scan_hdf5
from .formats.npy import scan_npy
from .formats.onnx_fmt import scan_onnx
from .formats.pickle_scan import is_whole_legacy_pickle, scan_pickle_stream
from .formats.safetensors_fmt import looks_like_safetensors, scan_safetensors
from .formats.textual import decode_text, scan_plain_text
from .ignore import IgnoreList
from .models import Finding, Note, Severity

PICKLE_EXTS = (".pkl", ".pickle", ".pck", ".pt", ".pth", ".ckpt", ".bin", ".joblib", ".sav", ".dill", ".jl", ".p", ".model", ".state")
# Extensions shared with many non-pickle formats: only treated as pickle when the stream starts like one.
AMBIGUOUS_EXTS = (".bin", ".model", ".state", ".p", ".jl")
# Model repositories also publish custom model code (.py) and run scripts (.sh). Data tables (.csv/.tsv) are left to
# the size-limited sniff on purpose: dogfooding on URL datasets (50-100 MB each) made a scan 8x slower and buried the
# model findings under thousands of third-party values from the data. Scanning datasets needs its own design.
TEXT_EXTS = (".json", ".jsonl", ".ndjson", ".yaml", ".yml", ".txt", ".cfg", ".ini", ".toml", ".md", ".env", ".conf",
             ".properties", ".log", ".ipynb", ".py", ".sh")
TEXT_NAMES = (".env",)
SKIP_DIRS = {".git", ".hg", ".svn"}  # VCS internals only: .ipynb_checkpoints and friends get published, so they are scanned
SNIFF_TEXT_MAX = 8 * MB
# Extensions that promise a container with a magic number. If the content does not match, the file is damaged or
# misnamed and must not pass as clean (found by evaluation/fuzz_formats.py: a .zip cut to its first byte was sniffed as
# text). .pt/.pth/.ckpt are absent on purpose: they are a zip or a legacy pickle, and both are recognised by content.
EXPECTED_KINDS = {".zip": ("zip",), ".npz": ("zip",), ".whl": ("zip",), ".keras": ("zip", "hdf5"), ".h5": ("hdf5",),
                  ".hdf5": ("hdf5",), ".gguf": ("gguf",), ".npy": ("npy",), ".cbm": ("catboost",), ".gz": ("gzip",),
                  ".tgz": ("gzip",), ".bz2": ("bz2",), ".xz": ("xz",),
                  ".tar": ("tar", "gzip", "bz2", "xz")}   # tar reads a compressed .tar too (user review before launch)


@dataclass
class ScannedFile:
    path: str
    kind: str
    size: int


@dataclass
class ScanResult:
    findings: List[Finding] = field(default_factory=list)
    ignored: List[Finding] = field(default_factory=list)
    notes: List[Note] = field(default_factory=list)
    files: List[ScannedFile] = field(default_factory=list)
    ignore: IgnoreList = field(default_factory=IgnoreList)

    def worst(self) -> Optional[Severity]:
        return max((f.severity for f in self.findings), default=None)

    def failing(self, threshold: Severity) -> List[Finding]:
        return [f for f in self.findings if f.severity >= threshold]

    @property
    def errors(self) -> List[Note]:
        return [n for n in self.notes if n.kind == "error"]


# --------------------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------------------
def _is_zlib_header(head: bytes) -> bool:
    return len(head) >= 2 and head[0] == 0x78 and ((head[0] << 8) | head[1]) % 31 == 0


def _is_bz2_stream(head: bytes) -> bool:
    """The bz2 signature: BZh, a block size digit, then a block or end-of-stream magic (ten bytes no text starts with)."""
    return head[:3] == b"BZh" and head[3:4].isdigit() and head[3:4] != b"0" and head[4:10] in (b"1AY&SY", b"\x17rE8P\x90")


XZ_MAGIC = b"\xfd7zXZ\x00"
LZ4_MAGIC = b"\x04\x22\x4d\x18"   # an LZ4 frame: joblib's compress="lz4"
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


def _is_lfs_pointer(f: BinaryIO) -> bool:
    """A Git LFS pointer: the spec line, then "oid sha256:<64 hex>" (and a size), in a few hundred bytes."""
    pos = f.tell()
    try:
        f.seek(0)
        data = f.read(1024)
    finally:
        f.seek(pos)
    return len(data) < 1024 and data.startswith(LFS_POINTER) and re.search(rb"\noid sha256:[0-9a-f]{64}\b", data) is not None
_TRIAL_INPUT = 64 * 1024


def _trial(f: BinaryIO, decompressor, consumed) -> bool:
    """Is the stream, from its first byte, real compressed data of this kind? Its first 64 KB are decompressed on trial:
    no error, and either the stream ends or the trial got well into it (1 KB read or 4 KB out). Whatever it decompresses
    to counts (pre-release test 4, S1: a protocol-0/1 pickle has no marker byte, so a probe that wanted the protocol 2+
    marker let a compressed joblib under a text name, or behind a link with a joblib name, be read as text). Text fails
    this within a few bytes: deflate and LZMA data that are not real break their own rules almost at once."""
    pos = f.tell()
    try:
        f.seek(0)
        chunk = f.read(_TRIAL_INPUT)
        out = decompressor.decompress(chunk, 4096)
    except (zlib.error, lzma.LZMAError, EOFError, ValueError):
        return False
    finally:
        f.seek(pos)
    return bool(decompressor.eof) or len(out) >= 4096 or consumed(decompressor, chunk) >= 1024


def _is_zlib_stream(f: BinaryIO) -> bool:
    """joblib's default compression: a bare zlib stream (no signature beyond a 2-byte header)."""
    return _trial(f, zlib.decompressobj(), lambda d, chunk: len(chunk) - len(d.unconsumed_tail) - len(d.unused_data))


def _is_lzma_alone(f: BinaryIO, head: bytes) -> bool:
    """joblib's compress="lzma" writes the legacy .lzma format, which has no signature (its first byte is the properties
    byte, 0x5D for every preset joblib uses). Recognised by a trial decompression, like zlib."""
    if not head or head[0] != 0x5D:
        return False
    return _trial(f, lzma.LZMADecompressor(lzma.FORMAT_ALONE), lambda d, chunk: len(chunk) if d.needs_input else 0)


def detect_kind(head: bytes, name: str, size: int, f: Optional[BinaryIO] = None) -> str:
    lower = os.path.basename(name).lower()
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        return "zip"
    if head[:8] == b"\x89HDF\r\n\x1a\n":
        return "hdf5"
    if head[:4] == b"GGUF":
        return "gguf"
    if head[:6] == b"\x93NUMPY":
        return "npy"
    if head[:4] == b"CBM1":
        return "catboost"
    if head[:2] == b"\x1f\x8b":
        return "gzip"
    if _is_bz2_stream(head) or (head[:3] == b"BZh" and lower.endswith(("bz2", ".pkl", ".pickle", ".joblib", ".json", ".txt"))):
        return "bz2"
    if head.startswith(LFS_POINTER[:16]) and f is not None and _is_lfs_pointer(f):
        return "lfs-pointer"                             # under any name
    if head[:4] == LZ4_MAGIC:
        return "lz4"                                     # under any name (pre-release test 5, T2)
    if head[:6] == XZ_MAGIC or (f is not None and _is_lzma_alone(f, head)):
        return "xz"                                      # both decompressed by lzma.LZMAFile (format auto-detected)
    if f is not None and _is_zlib_header(head) and _is_zlib_stream(f):
        return "zlib"                                    # under any name, before ONNX and the pickle names too
    if f is not None and size >= 512:
        pos = f.tell()
        f.seek(257)
        magic = f.read(5)
        f.seek(pos)
        if magic == b"ustar":
            return "tar"
    # a pickle is still a pickle under a format name that has no signature (pre-release test 5, T5: model.onnx and
    # model.safetensors holding a pickle got only a parse error). A safetensors header length can start with the same two
    # bytes, so there the real header check decides first; an ONNX file never starts with them.
    if (lower.endswith((".onnx", ".safetensors")) and f is not None and not looks_like_safetensors(head, size)
            and ((head[:1] == b"\x80" and len(head) > 1 and 2 <= head[1] <= 5) or is_whole_legacy_pickle(f, size, name))):
        return "pickle"
    if lower.endswith(".safetensors") or (not lower.endswith(TEXT_EXTS) and looks_like_safetensors(head, size)):
        return "safetensors"
    if lower.endswith(".onnx"):
        return "onnx"
    if head[:1] == b"\x80" and len(head) > 1 and 2 <= head[1] <= 5:
        return "pickle"
    if lower.endswith(PICKLE_EXTS):
        if lower.endswith(AMBIGUOUS_EXTS):
            # .bin/.model/... without the protocol 2+ marker: a whole protocol-0/1 pickle (pre-release test F3), or some
            # other binary format (OpenVINO, ggml ...)
            if f is not None and is_whole_legacy_pickle(f, size, name):
                return "pickle"
            return "unknown"
        if _is_zlib_header(head):
            return "zlib"                                # a damaged stream: decompressing it says what is wrong
        return "pickle"
    # A loader does not look at the name (pickle.load / torch.load read "model.data" or "weights" too), so a whole
    # protocol-0/1 pickle is analysed whatever it is called (pre-release test F3, follow-up). Text fails this probe within
    # its first bytes: the whole file must parse as pickle instructions up to STOP and end there.
    if f is not None and is_whole_legacy_pickle(f, size, name):
        return "pickle"
    if lower.endswith(TEXT_EXTS) or lower in TEXT_NAMES:
        return "text"
    return "unknown"


def _printable_ratio(b: bytes) -> float:
    """Share of printable characters. UTF-8 is judged by its characters: an Arabic letter or an em dash is one
    printable character, not a printable lead byte plus 'unprintable' continuation bytes (found by dogfooding:
    non-ASCII text files were reported unsupported while their ASCII twins were scanned)."""
    if not b:
        return 1.0
    try:
        text = codecs.getincrementaldecoder("utf-8")().decode(b, final=False)  # a cut last character stays pending
    except UnicodeDecodeError:
        text = ""
    if text:
        return sum(1 for ch in text if ch.isprintable() or ch in "\t\n\r") / len(text)
    ok = sum(1 for c in b if 32 <= c < 127 or c in (9, 10, 13) or c >= 0xC2)
    return ok / len(b)


def scan_stream(f: BinaryIO, ctx: ScanContext, name: str, size: int) -> str:
    """Detect the format of ``f`` and run the matching handler. Returns the detected kind."""
    f.seek(0)
    head = f.read(16)
    f.seek(0)
    kind = detect_kind(head, name, size, f)
    ext = os.path.splitext(os.path.basename(name).lower())[1]
    if ext in EXPECTED_KINDS and kind not in EXPECTED_KINDS[ext]:
        ctx.note("degraded", f"named *{ext} but the content is not a {EXPECTED_KINDS[ext][0]} file (damaged or misnamed); "
                             "scanned as what it looks like instead")
    handler = _HANDLERS.get(kind)
    if handler is not None:
        handler(f, ctx, name)
        return kind
    if kind in ("zip", "tar"):
        (archive.scan_zip if kind == "zip" else archive.scan_tar)(f, ctx, name, _dispatch)
        return kind
    if kind in ("gzip", "bz2", "xz", "zlib"):
        archive.scan_compressed(f, ctx, name, kind, _dispatch, pickle_inside=_pickle_inside(kind, head, name))
        return kind
    # unknown extension: small, mostly-printable files are scanned as text
    if size <= SNIFF_TEXT_MAX:
        data = f.read(SNIFF_TEXT_MAX)
        if _printable_ratio(data[:65536]) >= 0.95 and b"\x00" not in data[:4096]:
            ctx.charge(len(data))
            ctx.scan_text(decode_text(data), "", line_numbers=True)
            return "text(sniffed)"
    if ctx.options.deep:
        scan_stream_deep(f, ctx, 0, "unsupported file")
        return "binary(deep)"
    ctx.note("skipped", "unsupported file type (not scanned)")
    return "unsupported"


def _pickle_inside(kind: str, head: bytes, name: str) -> bool:
    """Is the decompressed content read as a pickle, whatever it starts with? zlib and the legacy .lzma format are joblib's
    own wrappers (joblib.load reads them under any name), and a compressed file with a pickle name holds one. Content that
    then does not parse as a pickle is an incomplete analysis (MEDIUM), never "not scanned" (pre-release test 4, S1: a
    protocol-0/1 joblib with a NumPy array inside was "not scanned" even as model.joblib, and passed)."""
    if kind == "zlib" or (kind == "xz" and head[:6] != XZ_MAGIC):
        return True
    lower = os.path.basename(name).lower()
    return lower.endswith(PICKLE_EXTS) and not lower.endswith(AMBIGUOUS_EXTS)


def _dispatch(f: BinaryIO, ctx: ScanContext, name: str, size: int) -> None:
    scan_stream(f, ctx, name, size)


def _scan_pickle(f, ctx, name):
    scan_pickle_stream(f, ctx, "pickle")


def _scan_lfs_pointer(f, ctx, name):
    f.seek(0)
    m = re.search(rb"oid sha256:([0-9a-f]{64})", f.read(1024))
    oid = m.group(1).decode() if m else "?"
    ctx.add("git-lfs-pointer", oid, f"oid sha256:{oid[:12]}...", "Git LFS pointer")


def _scan_lz4(f, ctx, name):
    """Python cannot decompress LZ4 without a third-party package, and this scanner has no dependencies. joblib.load reads
    such a file under any name, so it never passes as clean or as text: a MEDIUM finding says what was not analysed."""
    ctx.add("pickle-incomplete-analysis", "lz4-frame", "LZ4 frame", "LZ4 frame",
            message="LZ4-compressed file (joblib's compress=\"lz4\"): it cannot be decompressed without the lz4 package, so "
                    "the pickle joblib.load would read was not analysed")


_HANDLERS: Dict[str, Callable[[BinaryIO, ScanContext, str], None]] = {
    "pickle": _scan_pickle,
    "lz4": _scan_lz4,
    "lfs-pointer": _scan_lfs_pointer,
    "npy": scan_npy,
    "safetensors": scan_safetensors,
    "onnx": scan_onnx,
    "hdf5": scan_hdf5,
    "gguf": scan_gguf,
    "catboost": scan_catboost,
    "text": lambda f, ctx, name: scan_plain_text(f, ctx, name),
}


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def excluded_by(rel: str, patterns: Sequence[str]) -> Optional[str]:
    """The --exclude pattern that leaves out `rel` (a path relative to the scanned folder, with "/"), or None. A pattern
    without "/" matches a folder or file of that name anywhere (".venv", "*.log"); a pattern with "/" matches the path from
    the scanned folder ("data/*.csv", "logs/run1"). Case-sensitive, shell-style wildcards (fnmatch)."""
    name = rel.rsplit("/", 1)[-1]
    for pat in patterns:
        p = pat.strip().replace("\\", "/").strip("/")
        if p.startswith("./"):
            p = p[2:]
        if not p:
            continue
        if fnmatch.fnmatchcase(rel if "/" in p else name, p):
            return pat.strip()
    return None


def _inside(path: str, root: str) -> bool:
    try:
        p, r = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(root))
        return os.path.commonpath([p, r]) == r
    except ValueError:  # another drive (Windows)
        return False


def symlink_decision(target: str, exists: bool, *, explicit: bool, scan_root: Optional[str], follow_all: bool,
                     confine_to: Optional[str]) -> Tuple[bool, Optional[str]]:
    """(follow the link?, why not). Pre-release test 2026-11-19, finding F14: every link used to be refused, so a Hugging
    Face cache snapshot (a tree of links) scanned nothing. `target` is the link's resolved path.
      * never out of `confine_to` (the Action's workspace): a pull request could turn the scanned folder into a link to
        the runner's own files, and their findings would be printed in a public log;
      * never into version-control data (.git, .hg, .svn), which is never scanned (pre-release test 4, S6: a committed link
        into .git was followed, and actions/checkout keeps the job's token in .git/config by default);
      * a path named on the command line is followed;
      * a link to a place inside the scanned folder is not followed: that place is scanned anyway;
      * inside a Hugging Face cache, a link from a snapshot to its own repository's blobs folder is followed (that is
        how the cache stores every file; user review before launch: a snapshot scanned nothing and passed), and so is a
        link into the cache-wide blob store (`<cache>/blobs/<2 hex>/<hash>`) that newer huggingface_hub versions use for
        Xet-backed files, with the repository's own blobs/<etag> entry linking onward to it (retest of release
        candidate 8: most of the repository's files were skipped and the scan passed);
      * a link out of the scanned folder is followed only with --follow-symlinks."""
    if not exists:
        return False, "broken symbolic link; not followed"
    if in_vcs_data(target):
        return False, VCS_LINK
    if confine_to and not _inside(target, confine_to):
        return False, "symbolic link points outside the workspace; not followed"
    if explicit:
        return True, None
    if scan_root and _inside(target, scan_root):
        return False, "symbolic link to a place inside the scanned folder; scanned there, not followed"
    if scan_root and _same_cache_repo_blob(target, scan_root):
        return True, None
    if follow_all:
        return True, None
    return False, "symbolic link points outside the scanned folder; not followed (--follow-symlinks to follow it)"


VCS_LINK = "symbolic link into version-control data (.git, .hg, .svn), which is never scanned; not followed"


def in_vcs_data(path: str) -> bool:
    """Is `path` (a resolved path) inside a .git, .hg or .svn folder?"""
    return any(part in SKIP_DIRS for part in os.path.normcase(os.path.abspath(path)).replace("\\", "/").split("/"))


_CACHE_REPO = re.compile(r"^(?:models|datasets|spaces)--.+")


def _cache_repo(path: str) -> Optional[str]:
    """The Hugging Face cache repository folder ("models--org--name", holding blobs/ and snapshots/) that `path` is in."""
    p = os.path.abspath(path)
    while True:
        if _CACHE_REPO.match(os.path.basename(p)) and os.path.isdir(os.path.join(p, "blobs")) \
                and os.path.isdir(os.path.join(p, "snapshots")):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent


_SHARED_REFS = re.compile(r"^[0-9a-f]{64}\.refs$")
_TWO_HEX = re.compile(r"^[0-9a-f]{2}$")
HF_METADATA = ("Hugging Face cache metadata (huggingface_hub's own file lists, not repository content; the files they "
               "name are scanned through the snapshots); left out on purpose")


def hf_cache_metadata(path: str) -> bool:
    """huggingface_hub's own bookkeeping inside a cache: the shared blob store's `blobs/<2 hex>/<64 hex>.refs` lists and a
    cache repository's `trees/<commit>.json` file lists. Retest of release candidate 11: scanning a cache root reported
    their hash paths and run-folder names as LOW noise. Anything else (a `trees/` folder in a project) is scanned."""
    p = os.path.abspath(path)
    name, parent = os.path.basename(p), os.path.dirname(p)
    in_shared_store = _TWO_HEX.match(os.path.basename(parent)) and os.path.basename(os.path.dirname(parent)) == "blobs"
    if _SHARED_REFS.match(name) and in_shared_store:
        return True
    if name.endswith(".json") and os.path.basename(parent) == "trees":
        repo = os.path.dirname(parent)
        return _cache_repo(repo) == repo
    return False


def _same_cache_repo_blob(target: str, scan_root: str) -> bool:
    """Is `target` a blob of the cache repository `scan_root` is in: in the repository's own blobs/ folder, or in the
    cache-wide store `<cache>/blobs/` (the folder holding "models--org--name") that huggingface_hub's shared-blob layout
    links the repository's blobs/<etag> entries to?"""
    repo = _cache_repo(scan_root)
    if repo is None:
        return False
    if _inside(target, os.path.join(repo, "blobs")):
        return True
    shared = os.path.join(os.path.dirname(repo), "blobs")
    return os.path.isdir(shared) and _inside(target, shared)


def _through_a_link(path: str) -> bool:
    """Does `path`, or a folder above it, go through a symbolic link? (A path outside the workspace that involves no link
    is the workflow author's own choice and is scanned as before.)"""
    p = os.path.abspath(path)
    while True:
        if os.path.islink(p):
            return True
        parent = os.path.dirname(p)
        if parent == p:
            return False
        p = parent


def _link(path: str, **kw) -> Tuple[bool, Optional[str]]:
    target = os.path.realpath(path)
    return symlink_decision(target, os.path.exists(target), **kw)


def excluded_path(rel: str, patterns: Sequence[str]) -> Optional[str]:
    """Like excluded_by, for a whole path given at once (the Action's changed files, pre-release test 2, N4): a pattern
    without "/" may match any folder on the way ("models/.venv/env.txt" is left out by ".venv")."""
    parts = [x for x in rel.replace("\\", "/").split("/") if x not in ("", ".")]
    for i in range(1, len(parts) + 1):
        pat = excluded_by("/".join(parts[:i]), patterns)
        if pat:
            return pat
    return None


def iter_files(paths: Sequence[str], follow_symlinks: bool = False, exclude: Sequence[str] = (),
               confine_to: Optional[str] = None) -> Iterable[Tuple[str, Optional[str], Optional[str]]]:
    """Yield (path, note kind, reason): kind None for a file to scan, else "excluded", "info", "skipped" or "error".
    Directories are walked in sorted order. A folder that cannot be listed (no permission, a path too long for the system)
    is an "error": pre-release test 2026-11-19, finding F7, it used to be skipped silently. Inside a folder, what an
    --exclude pattern matches is left out and reported ("excluded": a choice, not a gap in coverage, pre-release test 2,
    N5); a path named here is always scanned. Symbolic links: see symlink_decision and _walk; a folder reached twice (a
    loop) is walked once."""
    for p in paths:
        if os.path.islink(p):
            follow, why = _link(p, explicit=True, scan_root=None, follow_all=follow_symlinks, confine_to=confine_to)
            if not follow:
                yield p, "skipped", why
                continue
        elif confine_to and _through_a_link(p) and not _inside(os.path.realpath(p), confine_to):
            yield p, "skipped", "symbolic link points outside the workspace; not followed"   # a folder above it is a link
            continue
        if not os.path.isdir(p):
            yield p, None, None
            continue
        yield from _walk(p, follow_symlinks, exclude, confine_to)


def _walk(p: str, follow_symlinks: bool, exclude: Sequence[str],
          confine_to: Optional[str]) -> Iterable[Tuple[str, Optional[str], Optional[str]]]:
    """One folder. Links to places INSIDE it are decided after the walk (pre-release test 2, N6 and N7; test 3, R1):
      * a link to a file: the file is scanned once. A file that is walked anyway and has an extension of its own is
        scanned under its own name, and the links to it are noted: a link's name must never decide how a file is read
        ("notes.txt -> model.joblib" made the scanner read the joblib as text). A file without an extension (a Hugging
        Face cache blob is named by its hash), or one in a place that is not walked, is scanned once, under the name of
        the link that says most about it (_link_name_rank; pre-release test 4, S7: "a_copy.txt" and "model.onnx" to one
        blob read the model as text); detection_name() still judges its format by its own name whenever it has one;
      * a link into a place that --exclude left out is left out too, and reported as such (it used to say "scanned
        there" although nothing was);
      * a link to a folder that is walked anyway is noted ("info"), one to a folder that is not walked is "skipped"."""
    scan_root = os.path.realpath(p)
    visited: set = set()
    failed: List[OSError] = []
    files: List[Tuple[str, str]] = []            # (path, real path) of regular files to scan
    file_links: List[Tuple[str, str]] = []       # (link, target) for links to files inside the folder
    out_links: List[Tuple[str, str]] = []        # (link, target) for followed links to files outside it
    dir_links: List[Tuple[str, str]] = []        # (link, target) for links to folders inside the folder
    left_out: List[Tuple[str, str]] = []         # (real path, pattern) of what --exclude left out

    def inside_link(path: str) -> Optional[str]:
        """The target of a link that points to an existing place inside the scanned folder (and the workspace)."""
        target = os.path.realpath(path)
        if os.path.exists(target) and _inside(target, scan_root) and not (confine_to and not _inside(target, confine_to)) \
                and not in_vcs_data(target):
            return target
        return None

    for root, dirs, names in os.walk(p, followlinks=True, onerror=failed.append):
        while failed:
            e = failed.pop(0)
            yield (e.filename or root), "error", f"folder could not be listed ({type(e).__name__}: {e.strerror or e})"
        real_root = os.path.realpath(root)
        if real_root in visited:
            yield root, "info", "symbolic link leads back into a folder already scanned; scanned there, not followed"
            dirs[:] = []
            continue
        visited.add(real_root)
        rel_root = os.path.relpath(root, p).replace(os.sep, "/")
        rel_root = "" if rel_root == "." else rel_root + "/"
        kept = []
        for d in sorted(d for d in dirs if d not in SKIP_DIRS):
            full_d = os.path.join(root, d)
            pat = excluded_by(rel_root + d, exclude) if exclude else None
            if pat:
                left_out.append((os.path.realpath(full_d), pat))
                yield full_d, "excluded", f"excluded by --exclude {pat}"
                continue
            if os.path.islink(full_d):
                target = inside_link(full_d)
                if target:
                    dir_links.append((full_d, target))
                    continue
                follow, why = _link(full_d, explicit=False, scan_root=scan_root, follow_all=follow_symlinks,
                                    confine_to=confine_to)
                if not follow:
                    yield full_d, "skipped", why
                    continue
            kept.append(d)
        dirs[:] = kept
        for fn in sorted(names):
            full = os.path.join(root, fn)
            pat = excluded_by(rel_root + fn, exclude) if exclude else None
            if pat:
                left_out.append((os.path.realpath(full), pat))
                yield full, "excluded", f"excluded by --exclude {pat}"
                continue
            if hf_cache_metadata(full):
                yield full, "excluded", HF_METADATA
                continue
            if os.path.islink(full):
                target = inside_link(full)
                if target:
                    file_links.append((full, target))
                    continue
                follow, why = _link(full, explicit=False, scan_root=scan_root, follow_all=follow_symlinks,
                                    confine_to=confine_to)
                if not follow:
                    yield full, "skipped", why
                    continue
                out_links.append((full, os.path.realpath(full)))
                continue
            files.append((full, os.path.realpath(full)))
    for e in failed:                     # an error on the last folder walked
        yield (e.filename or p), "error", f"folder could not be listed ({type(e).__name__}: {e.strerror or e})"

    def left_out_by(target: str) -> Optional[str]:
        for real, pat in left_out:
            if target == real or _inside(target, real):
                return pat
        return None

    walked = {real: path.replace("\\", "/") for path, real in files}
    chosen: dict = {}          # target -> the link it is scanned under (a file without a walked name of its own)
    for link, target in file_links:
        if left_out_by(target) or (target in walked and _has_extension(target)):
            continue
        if target not in chosen or _link_name_rank(link) < _link_name_rank(chosen[target]):
            chosen[target] = link
    for link, target in file_links:
        pat = left_out_by(target)
        if pat:
            yield link, "excluded", f"symbolic link into a place left out by --exclude {pat}; not scanned"
        elif target in walked and _has_extension(target):
            yield link, "info", f"symbolic link to {walked[target]}; scanned there, under that file's own name"
        elif chosen[target] != link:
            yield link, "info", (f"symbolic link to the same file as {chosen[target].replace(chr(92), '/')}; scanned once, "
                                 "under that name")
        else:
            yield link, None, None
    # Followed links to files OUTSIDE the folder: also one read per file (a follow-up to the retest fix, 2026-11-22:
    # in huggingface_hub's shared-blob layout a cache repository holds two links to each file, snapshots/<rev>/<name> and
    # blobs/<etag>, both leading out to <cache>/blobs/; scanning the repository folder read every file twice and reported
    # every finding twice, once under a hash name). The repository's own blobs/<etag> link is the last choice of name, as in
    # the older layout, where that blob was a plain file and the snapshot's name was used.
    out_chosen: dict = {}
    for link, target in out_links:
        if target in walked and _has_extension(target):
            continue
        if target not in out_chosen or _out_link_rank(link) < _out_link_rank(out_chosen[target]):
            out_chosen[target] = link
    for link, target in out_links:
        if target in walked and _has_extension(target):
            yield link, "info", f"symbolic link to {walked[target]}; scanned there, under that file's own name"
        elif out_chosen[target] != link:
            yield link, "info", (f"symbolic link to the same file as {out_chosen[target].replace(chr(92), '/')}; scanned "
                                 "once, under that name")
        else:
            yield link, None, None
    for path, real in files:
        once = chosen.get(real) or out_chosen.get(real)
        if once:
            yield path, "info", (f"the same file as the symbolic link {once.replace(chr(92), '/')}; scanned once, "
                                 "under that name")
        else:
            yield path, None, None
    for link, target in dir_links:
        pat = left_out_by(target)
        if pat:
            yield link, "excluded", f"symbolic link into a folder left out by --exclude {pat}; not scanned"
        elif target in visited:
            yield link, "info", "symbolic link to a folder inside the scanned folder; scanned there, not followed"
        else:
            yield link, "skipped", "symbolic link to a folder inside the scanned folder that is not scanned; not followed"


def _link_name_rank(path: str) -> int:
    """Which link name to read a file without an extension of its own by: a format name first (.onnx, .safetensors, a
    pickle name ...: formats with no signature are known only by their name), then a name without an extension, a text
    name last. Equal ranks keep the walk order."""
    lower = os.path.basename(path).lower()
    if lower.endswith(TEXT_EXTS) or lower in TEXT_NAMES:
        return 2
    return 0 if _has_extension(path) else 1


def _out_link_rank(path: str) -> Tuple[int, int]:
    """_link_name_rank, with a Hugging Face cache repository's own blobs/<etag> link always last (its name is a hash)."""
    parent = os.path.dirname(os.path.abspath(path))
    own_blob = os.path.basename(parent) == "blobs" and _cache_repo(parent) is not None
    return (1 if own_blob else 0, _link_name_rank(path))


def _has_extension(path: str) -> bool:
    return bool(os.path.splitext(os.path.basename(path))[1])


def detection_name(path: str) -> str:
    """The name a file's format is judged by: the file's own name. A symbolic link's name is used only when the file it
    leads to has no extension (a Hugging Face cache blob is named by its hash, so "model.onnx" is the only hint).
    Pre-release test 3, R1: the link's name used to decide, so "notes.txt -> model.joblib" was read as text."""
    if os.path.islink(path):
        real = os.path.realpath(path)
        if _has_extension(real):
            return os.path.basename(real)
    return os.path.basename(path)


def scan_file(path: str, options: ScanOptions, result: ScanResult) -> None:
    ctx = ScanContext(path.replace("\\", "/"), options)
    try:
        if not stat.S_ISREG(os.stat(path).st_mode):
            # pre-release test 3, R4: opening a named pipe waits for a writer, so the scan hung; never open one
            ctx.note("skipped", "not a regular file (a named pipe, socket or device); not scanned")
            result.files.append(ScannedFile(ctx.display, "unsupported", 0))
            result.notes.extend(ctx.notes)
            return
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            kind = scan_stream(f, ctx, detection_name(path), size)
        result.files.append(ScannedFile(ctx.display, kind, size))
    except Exception as exc:  # one broken file must not stop the run
        ctx.note("error", f"scan failed ({type(exc).__name__}: {exc})")
        result.files.append(ScannedFile(ctx.display, "error", 0))
    result.findings.extend(ctx.findings)
    result.notes.extend(ctx.notes)


def scan_paths(paths: Sequence[str], options: Optional[ScanOptions] = None, ignore: Optional[IgnoreList] = None,
               entries: Optional[Iterable[Tuple[str, Optional[str], Optional[str]]]] = None) -> ScanResult:
    """Scan ``paths``. ``entries``: (path, note kind, reason) already decided by iter_files, used instead of walking
    ``paths`` (the Action's changed-only keeps the part of a full scan's walk that concerns the changed files)."""
    options = options or ScanOptions()
    result = ScanResult(ignore=ignore or IgnoreList())
    if entries is None:
        entries = iter_files(list(paths), options.follow_symlinks, options.exclude, options.confine_to)
    for path, level, reason in entries:
        if level:
            result.notes.append(Note(level, str(path).replace("\\", "/"), reason))
            continue
        if not os.path.exists(path):
            result.notes.append(Note("error", path, "path does not exist"))
            continue
        scan_file(path, options, result)
    if ignore is not None:
        kept: List[Finding] = []
        for f in result.findings:
            (result.ignored if ignore.matches(f.fingerprint) else kept).append(f)
        result.findings = kept
    result.findings.sort(key=lambda f: (-int(f.severity), f.file, f.rule_id, f.location))
    return result

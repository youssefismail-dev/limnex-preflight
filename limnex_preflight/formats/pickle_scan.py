"""Static pickle analysis. Nothing is ever unpickled or called.

``pickletools.genops`` only *decodes* opcodes. On top of it a tiny symbolic stack tracks plain
data (strings, ints, tuples, dicts) and *references* to globals, so ``STACK_GLOBAL`` operands that
travel through the memo or a tuple are still resolved. Callables are never invoked.

Handled specially:
  * several pickles back to back in one file (the legacy PyTorch layout has five)
  * joblib's NumpyArrayWrapper, whose array bytes sit inline between opcodes
  * a stream that stops parsing: reported as incomplete and the rest is carved heuristically
"""

from __future__ import annotations

import io
import pickletools
import re
from typing import Dict, List, Optional, Tuple

from ..context import ScanContext, MB
from ..detectors import MODE_BINARY, MODE_FULL, MODE_SECRETS
from ..masking import clean
from .fastai_pairs import FASTAI_SOURCE_PAIRS

# --------------------------------------------------------------------------------------
# Deny-list of unsafe globals (module, name). A deny-list is never complete: see README.
# --------------------------------------------------------------------------------------
EXEC_MODULES = {
    "os", "posix", "nt", "subprocess", "pty", "runpy", "commands", "popen2", "code", "codeop", "timeit", "pdb", "bdb",
    "ctypes", "_ctypes", "multiprocessing", "importlib", "imp", "pkgutil", "distutils", "setuptools", "pip", "ensurepip",
    "venv", "cProfile", "profile", "trace", "pydoc", "idlelib", "webbrowser", "asyncio", "_posixsubprocess", "nturl2path",
}
EXEC_BUILTINS = {"eval", "exec", "compile", "__import__", "breakpoint", "execfile", "input", "reload"}
IO_BUILTINS = {"open", "setattr", "delattr", "vars", "globals", "locals", "dir", "__build_class__"}
# getattr(x, name) is routine in real pickles (an unbound method is stored as getattr(Class, "forward")), so it cannot be
# banned by import. It is judged at the call site by the attribute name, FAIL-CLOSED:
#   * name in GETATTR_ALLOWED            -> quiet
#   * name in DANGEROUS_ATTRS / dunder   -> HIGH   (unsafe-pickle-io)
#   * anything else, or a computed name  -> reported for review (pickle-getattr-review, MEDIUM) - never dropped silently
# GETATTR_ALLOWED is derived from data: the 80 public repositories of the evaluation contain exactly three getattr calls,
# all getattr(<a detection-model head class>, "forward"), in the tuning set (none in the held-out set).
# It is deliberately tiny. Add names only with an observed, benign checkpoint to point at.
GETATTR_ALLOWED = frozenset({"forward"})
# fastai stores some of its own bound methods on its objects (FilteredBase.__init__ keeps `self.dataloaders`, transforms
# keep `before_call`), so every fastai export holds getattr(<fastai object or class>, <method>). These are quiet only as
# EXACT (module, class, name) triples, and only when the object was made from that class in a way that names it (see
# _getattr_owner): fastcore's GetAttr forwards names a class does not define to another object, which a pickle chooses, so
# "any name on a fastai class" would not be safe. Each name resolves by normal lookup to a method in fastai's own source
# (checked at fastai 2.8.12). This adds no new power: fastai is not deny-listed, so a pickle could already name the same
# function directly. Calling what getattr returned is still pickle-computed-callable. A user's own subclass (__main__.X)
# is never on the list. Two parts (CHANGELOG, 0.1.1): the 11 pairs read by hand after the 0.1.1 study, and
# FASTAI_SOURCE_PAIRS, generated from fastai's source by method and inheritance (every fastai method fastai's own code uses
# as a value, on every fastai class that reaches it by normal lookup; fastai 2.8.12 and 2.7.19).
_FASTAI_STUDY_PAIRS = frozenset({
    ("fastai.data.core", "TfmdLists", "dataloaders"),
    ("fastai.data.core", "Datasets", "dataloaders"),
    ("fastai.tabular.core", "TabularPandas", "dataloaders"),
    ("fastai.vision.augment", "Flip", "before_call"),
    ("fastai.vision.augment", "Brightness", "before_call"),
    ("fastai.vision.augment", "Resize", "before_call"),
    ("fastai.vision.augment", "RandomResizedCrop", "before_call"),
    ("fastai.vision.augment", "RandomResizedCropGPU", "before_call"),
    ("fastai.vision.core", "PILImage", "create"),
    ("fastai.torch_core", "TensorImage", "lighting"),
    ("fastai.text.models.awdlstm", "WeightDropout", "_do_nothing"),
})
FASTAI_GETATTR_ALLOWED = _FASTAI_STUDY_PAIRS | FASTAI_SOURCE_PAIRS
# fastai 2.8 saves learners with cloudpickle, which writes the same bound methods as
# _builtin_type('MethodType')(<function>, <object>): a call on what a call returned. fastai_method_binding() keeps that
# quiet only for listed triples (CHANGELOG 0.1.1); stored code (_builtin_type('CodeType'), _make_function) stays reported.
_CLOUDPICKLE_BUILTIN_TYPE = ("cloudpickle.cloudpickle", "_builtin_type")
# Opcodes that create an object FROM the class they name (copyreg._reconstructor(cls, ...) is handled apart).
_CLASS_NAMING_OPS = frozenset({"NEWOBJ", "NEWOBJ_EX", "OBJ", "INST"})
DANGEROUS_ATTRS = frozenset({
    # process execution (os / subprocess / pty / posix families)
    "system", "popen", "popen2", "popen3", "popen4", "fork", "forkpty", "kill", "killpg", "startfile", "abort", "_exit",
    "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe",
    "spawn", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe",
    "posix_spawn", "posix_spawnp", "Popen", "run", "call", "check_call", "check_output", "getoutput", "getstatusoutput",
    # code loading / evaluation
    "eval", "exec", "compile", "input", "breakpoint", "import_module", "__import__", "run_path", "run_module", "run_code",
    "exec_module", "load_module", "load_source", "find_module", "load", "loads", "modules", "interact", "runsource", "runcode",
    "lambda_handler",
    # file system: open/write/delete/permissions
    "open", "write", "writelines", "write_text", "write_bytes", "touch", "mkdir", "makedirs", "rename", "renames", "replace",
    "symlink", "symlink_to", "hardlink_to", "link", "remove", "removedirs", "unlink", "rmdir", "rmtree", "truncate", "chmod",
    "chown", "copy", "copyfile", "copytree", "move", "dump", "dumps", "save", "savez", "savez_compressed", "savetxt", "tofile",
    # network
    "urlopen", "urlretrieve", "create_connection", "connect", "sendall", "sendto", "send", "bind", "listen", "request",
    # environment
    "putenv", "unsetenv", "setuid", "setgid", "setattr", "delattr", "globals", "locals", "vars",
})
SPECIFIC_EXEC = {
    "types.FunctionType", "types.CodeType", "types.LambdaType",
    "numpy.testing._private.utils.runstring", "numpy.testing.utils.runstring",
    "torch.utils.bottleneck.__main__.run_cprofile", "torch.utils.bottleneck.__main__.run_autograd_prof",
    "torch._inductor.codecache.compile_file",
    # logging.config gadgets: listen() runs a socket server that execs received config; fileConfig/dictConfig
    # instantiate and call arbitrary objects named in a config carried by the pickle (a documented RCE family).
    "logging.config.listen", "logging.config.fileConfig", "logging.config.dictConfig",
    # torch native-code loaders: load a shared object, or compile+load C++/CUDA, both running arbitrary machine code.
    "torch.ops.load_library", "torch._C._load_library", "torch.utils.cpp_extension.load",
    "torch.utils.cpp_extension.load_inline",
}
NESTED = {
    "pickle.loads", "pickle.load", "pickle.Unpickler", "_pickle.loads", "_pickle.load", "_pickle.Unpickler",
    "dill.loads", "dill.load", "cloudpickle.loads", "cloudpickle.load", "marshal.loads", "marshal.load",
    "shelve.open", "torch.load", "torch.serialization.load", "torch.serialization._load", "torch.jit.load",
    "torch.storage._load_from_bytes", "joblib.load", "numpy.load", "pandas.read_pickle", "pandas.io.pickle.read_pickle",
    "torch.hub.load", "torch.hub.load_state_dict_from_url",
    "torch.jit._load_for_lite_interpreter", "torch.jit.mobile._load_for_lite_interpreter",
}
IO_MODULES = {
    "socket", "_socket", "ssl", "http", "urllib", "urllib2", "urllib3", "requests", "httpx", "aiohttp", "ftplib",
    "smtplib", "telnetlib", "xmlrpc", "paramiko", "shutil", "tempfile", "fileinput", "tarfile", "zipfile", "sys",
    "platform", "webbrowser", "smtpd", "poplib", "imaplib", "nntplib", "socketserver", "select", "selectors", "signal",
}
IO_SPECIFIC = {
    "io.open", "_io.open", "io.FileIO", "_io.FileIO", "io.open_code", "_io.open_code", "codecs.open", "operator.attrgetter",
    "operator.methodcaller", "_operator.attrgetter", "_operator.methodcaller", "torch.hub.download_url_to_file", "torch.utils.data._utils.worker._worker_loop",
    "numpy.memmap", "numpy.fromfile", "numpy.savetxt", "numpy.save", "numpy.savez", "os.path.getsize",
}
_ALIAS_TOKENS = {"os", "posix", "nt", "subprocess"}

Classification = Tuple[str, str]  # (rule id, reason)


def classify_global(module: str, name: str) -> Optional[Classification]:
    mod = module.strip()
    nm = name.strip()
    full = f"{mod}.{nm}"
    root = mod.split(".")[0]
    if full in SPECIFIC_EXEC:
        return "unsafe-pickle-exec", "known code-execution helper"
    if full in NESTED:
        return "unsafe-pickle-nested", "starts another deserialisation that is not visible to a static scan"
    if full in IO_SPECIFIC:
        return "unsafe-pickle-io", "file-system, network or indirect attribute access"
    if root in ("builtins", "__builtin__", "__builtins__"):
        first = nm.split(".")[0]
        if first in EXEC_BUILTINS:
            return "unsafe-pickle-exec", f"builtins.{first} evaluates or imports arbitrary code"
        if first in IO_BUILTINS:
            return "unsafe-pickle-io", f"builtins.{first} gives file or indirect attribute access"
        return None
    if root in EXEC_MODULES:
        if root == "os" and (mod == "os.path" or nm.startswith("path.")):
            return None
        if root == "os" and mod == "os" and nm in ("path", "sep", "linesep", "fspath", "fsencode", "fsdecode", "PathLike"):
            return None
        return "unsafe-pickle-exec", f"{root} can run operating-system commands or import code"
    if root in IO_MODULES:
        return "unsafe-pickle-io", f"{root} gives network or file-system access"
    # Re-exports: numpy.os.system, some.module.subprocess.Popen, name "os.system" ...
    tokens = set(mod.split(".")) | set(nm.split("."))
    hit = tokens & _ALIAS_TOKENS
    if hit and not (root == "os" or mod.startswith("os.path")):
        return "unsafe-pickle-exec", f"path goes through {sorted(hit)[0]} (alias of a command-execution module)"
    # Protocol 4 resolves dotted names through attribute access (pathlib.Path.unlink): judge the method reached, not only
    # the module, by the same attribute list getattr is judged by.
    if "." in nm and nm.rsplit(".", 1)[-1] in DANGEROUS_ATTRS:
        return "unsafe-pickle-io", f"dotted name reaches the method {nm.rsplit('.', 1)[-1]!r} through a class"
    return None


# --------------------------------------------------------------------------------------
# Symbolic stack values
# --------------------------------------------------------------------------------------
class _Mark:
    pass


class _Unknown:
    pass


_MARK = _Mark()
_UNK = _Unknown()


class _G:
    """Reference to ``module.name`` (never resolved to a real object)."""

    __slots__ = ("module", "name")

    def __init__(self, module: str, name: str):
        self.module, self.name = module, name


class _Bytes:
    """A bytes constant. ``analysed`` is True when its content was itself scanned (nested pickle or zip)."""

    __slots__ = ("analysed",)

    def __init__(self, analysed: bool):
        self.analysed = analysed


class _Obj:
    __slots__ = ("callable", "args", "state", "op")

    def __init__(self, callable_, args, op: Optional[str] = None):
        self.callable, self.args, self.state, self.op = callable_, args, None, op  # op: the opcode that made it


_STRING_OPS = {"UNICODE", "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "STRING", "BINSTRING", "SHORT_BINSTRING"}
_BYTES_OPS = {"BINBYTES", "SHORT_BINBYTES", "BINBYTES8", "BYTEARRAY8"}
_INT_OPS = {"INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4"}
_MAX_CONTAINER = 4096
_TORCH_MAGIC = 0x1950A86A20F9469CFC6C

_ITEM_RE = re.compile(r"^[<>|=]?([a-zA-Z])(\d*)")
MAX_STRING_ARRAY = 64 * MB   # as for .npy string arrays
UTF32_NOTE = " (read as UTF-32 text: a NumPy string array)"


def utf32_text(data: bytes) -> Optional[str]:
    """The text in bytes that hold UTF-32 (a NumPy 'U' string array pickled without joblib, whose dtype the walk does not
    follow), little or big endian, or None. Each run of NUL padding becomes a line break, so elements do not run
    together. Pre-release test 5, T1: a classifier's class labels (an e-mail, a home path) were never read; most binary
    data fails the decode at once (code points beyond U+10FFFF), and control characters fail the printable check."""
    if len(data) < 8 or len(data) % 4:
        return None
    for enc in ("utf-32-le", "utf-32-be"):
        try:
            text = data.decode(enc)
        except UnicodeDecodeError:
            continue
        chars = text.replace("\x00", "")
        if len(chars) >= 2 and sum(1 for ch in chars if ch.isprintable() or ch in "\t\n\r") / len(chars) >= 0.9:
            return re.sub("\x00+", "\n", text)
    return None


def _array_strings(data: bytes, kind: str, width: int, big_endian: bool) -> str:
    """A fixed-width NumPy string array, element by element (as npy.py reads one): neighbours never run together."""
    if kind == "U":
        text = data.decode("utf-32-be" if big_endian else "utf-32-le", "replace")
    else:
        text = data.decode("latin-1")
    elements = [text[i: i + width] for i in range(0, len(text), width)] if width else []
    return "\n".join(e.rstrip("\x00") for e in elements)


def _dtype_itemsize(spec: object) -> Optional[Tuple[int, bool]]:
    """(itemsize, has_object) for a numpy typestring such as 'f8' or '<U10'; None if unknown."""
    if not isinstance(spec, str):
        return None
    m = _ITEM_RE.match(spec)
    if not m:
        return None
    kind, digits = m.group(1), m.group(2)
    if kind == "O":
        return 8, True
    if not digits:
        return None
    n = int(digits)
    if kind == "U":
        return 4 * n, False
    if kind in "biufcSVmM":
        return n, False
    return None


_RECONSTRUCTORS = {("copyreg", "_reconstructor"), ("copy_reg", "_reconstructor")}


def _is_array_wrapper(obj: "_Obj") -> bool:
    """joblib's NumpyArrayWrapper, after which joblib writes the array's raw bytes. Protocol 2+ creates it with NEWOBJ (the
    class is the callable). Protocols 0 and 1 have no NEWOBJ, so Python writes copyreg._reconstructor(NumpyArrayWrapper,
    object, None) and BUILD sets the state; joblib's loader reads the array after it all the same (pre-release test 4,
    S1: the array bytes were read as instructions, the walk broke, and the file was "not scanned")."""
    c = obj.callable
    if not isinstance(c, _G):
        return False
    if c.name == "NumpyArrayWrapper":
        return True
    if (c.module, c.name) in _RECONSTRUCTORS and isinstance(obj.args, tuple) and obj.args:
        return isinstance(obj.args[0], _G) and obj.args[0].name == "NumpyArrayWrapper"
    return False


class _Aborted(Exception):
    pass


class _Bounded:
    """A view of ``f`` whose reads never ask for more than what is left before ``end``.

    pickletools reads a length-prefixed argument (BINBYTES8, BINUNICODE8, ...) with ``f.read(n)``, ``n`` taken from the
    file. On a real file CPython allocates ``n`` bytes before reading, so a few damaged bytes could demand gigabytes.
    Clamped, the read comes back short and pickletools raises its usual "only N bytes remain" error, which is reported as
    an incomplete analysis. Found by evaluation/fuzz_formats.py.
    """

    def __init__(self, f, end: int):
        self._f = f
        self._end = end

    def _left(self) -> int:
        return max(0, self._end - self._f.tell())

    def read(self, n: int = -1) -> bytes:
        left = self._left()
        return self._f.read(left if n is None or n < 0 else min(n, left))

    def readline(self, limit: int = -1) -> bytes:
        left = self._left()
        return self._f.readline(left if limit is None or limit < 0 else min(limit, left))

    def tell(self) -> int:
        return self._f.tell()

    def seek(self, *args) -> int:
        return self._f.seek(*args)


PLAIN_READING = " (as a plain pickle loader reads it)"


_RECLASSED = "BUILD:__class__"   # the op of an object whose BUILD state may change its class: no certain owner


def _may_reclass(state: object) -> bool:
    """Could this BUILD state change the object's class? The loader applies a (state, slotstate) pair's slot state with
    setattr, so {'__class__': X} there makes the object an X (review of 0.1.1, finding 3: a Flip re-classed this way
    handed out another class's method). A slot state that cannot be seen, or a full one (keys past the container limit
    are not kept), may hold it too. In the main state '__class__' only lands in __dict__ with the default loader, but a
    class's own __setstate__ could use it, so it counts as well. A state the scan cannot see at all (what a call returned,
    a module attribute, anything unknown) may be such a pair, so it counts too (tester of 0.1.1, L1: it fell through to
    "cannot change")."""
    unseen = (_Obj, _Unknown, _G)

    def may_hold(d: object) -> bool:   # a seen dict that holds '__class__', or a full one, or one not seen at all
        return isinstance(d, unseen) or (isinstance(d, dict) and ("__class__" in d or len(d) >= _MAX_CONTAINER))

    if isinstance(state, unseen) or isinstance(state, dict):
        return may_hold(state)
    if isinstance(state, tuple) and len(state) == 2:
        main, slots = state
        if slots is not None and not isinstance(slots, dict):
            return True
        return may_hold(slots) or may_hold(main)
    return False


def getattr_name_is_high(attr: str) -> bool:
    """True for an attribute name that getattr may never take quietly (HIGH, unsafe-pickle-io), whatever the object:
    special __names__, the dangerous-name list, and module names. Checked before any allow-list."""
    bare = attr.lstrip("_")
    return (attr.startswith("__") and attr.endswith("__")) or attr in DANGEROUS_ATTRS or attr in EXEC_MODULES \
        or attr in IO_MODULES or bare in EXEC_MODULES or bare in IO_MODULES or bare in ("builtins", "builtin")


def _getattr_owner(target: object) -> Optional[Tuple[str, str]]:
    """(module, class) that getattr's first argument certainly belongs to, or None. Certain only for the class itself, an
    object made from it by an opcode that names the class (NEWOBJ, NEWOBJ_EX, OBJ, INST), or copyreg._reconstructor(cls,
    ...), and only while no BUILD state may have changed its class (_may_reclass). Anything a plain call returned (REDUCE
    of some function or class) has no known class."""
    if isinstance(target, _G):
        return target.module, target.name
    if not isinstance(target, _Obj) or not isinstance(target.callable, _G):
        return None
    if target.op in _CLASS_NAMING_OPS:
        return target.callable.module, target.callable.name
    if target.op == "REDUCE" and (target.callable.module, target.callable.name) in _RECONSTRUCTORS \
            and isinstance(target.args, tuple) and target.args and isinstance(target.args[0], _G):
        return target.args[0].module, target.args[0].name
    return None


def _function_by_class(f: object) -> Optional[Tuple[str, str, str]]:
    """(module, class, name) of a function named through its class: getattr(<global C>, 'm') (protocol 2) or the global
    'C.m' (protocol 4 and later). None for anything else: code stored in the file, what another call returned, ..."""
    if isinstance(f, _G):
        parts = f.name.split(".")
        return (f.module, parts[0], parts[1]) if len(parts) == 2 and all(parts) else None
    if (isinstance(f, _Obj) and f.op == "REDUCE" and isinstance(f.callable, _G)
            and f.callable.module in ("builtins", "__builtin__") and f.callable.name == "getattr"
            and isinstance(f.args, tuple) and len(f.args) == 2 and isinstance(f.args[0], _G)
            and "." not in f.args[0].name and isinstance(f.args[1], str)):
        return f.args[0].module, f.args[0].name, f.args[1]
    return None


def fastai_method_binding(obj: "_Obj") -> bool:
    """Amendment 3 (tester of 0.1.1, B2): is this call cloudpickle making a bound method of fastai's own method on fastai's
    own object? cloudpickle writes a bound method as _builtin_type('MethodType')(<function>, <object>). True only when
    the callable is exactly what cloudpickle.cloudpickle._builtin_type('MethodType') returned, there are exactly two
    arguments, the function is named through its class with a listed (module, class, name) triple and a name that is not
    HIGH, and the object's class is certain (_getattr_owner) with its own triple listed for that name. Making a bound
    method runs no code; calling it later is still a call on a call's result."""
    fn = obj.callable
    if not (obj.op == "REDUCE" and isinstance(fn, _Obj) and fn.op == "REDUCE" and isinstance(fn.callable, _G)
            and (fn.callable.module, fn.callable.name) == _CLOUDPICKLE_BUILTIN_TYPE and fn.args == ("MethodType",)):
        return False
    if not (isinstance(obj.args, tuple) and len(obj.args) == 2):
        return False
    triple = _function_by_class(obj.args[0])
    if triple is None or getattr_name_is_high(triple[2]) or triple not in FASTAI_GETATTR_ALLOWED:
        return False
    owner = _getattr_owner(obj.args[1])
    return owner is not None and (owner[0], owner[1], triple[2]) in FASTAI_GETATTR_ALLOWED


class _Walker:
    """Walks one pickle.

    ``plain`` (review 2026-11-11, finding 1): joblib's loader treats the bytes after a NumpyArrayWrapper as array data,
    which the normal walk skips; a plain pickle loader reads the same bytes as instructions. The plain walk is that second
    reading. It reports only from ``report_from`` (the first skipped block) on, and only definite findings (deny-listed
    imports, getattr judgements, nested loaders): genuine array bytes are noise as instructions, so the "cannot tell"
    findings and text scans are left out of it.
    """

    def __init__(self, ctx: ScanContext, label: str, stream_size: int, plain: bool = False, report_from: int = 0):
        self.ctx = ctx
        self.label = label + (PLAIN_READING if plain else "")
        self.size = stream_size
        self.plain = plain
        self.report_from = report_from
        self.skipped_ranges: List[int] = []   # start offsets of array blocks skipped as data (normal walk only)
        self.stack: List[object] = []
        self.memo: Dict[int, object] = {}
        self.ops = 0
        self.skipped_blobs = 0
        self.incomplete: Optional[str] = None
        self.depth = 0
        self.pending: Dict[int, tuple] = {}
        # UTF-32-like constants (a NumPy 'U' array's data), held until its dtype is known: id -> (pos, op, bytes, the object)
        self.text_data: Dict[int, Tuple[int, str, bytes, object]] = {}

    def quiet(self, pos: Optional[int]) -> bool:
        """In the plain walk, nothing before the first skipped block is reported (the normal walk covered it)."""
        return self.plain and (pos is None or pos < self.report_from)

    # -- stack helpers ------------------------------------------------------------------
    def push(self, v: object) -> None:
        self.stack.append(v)

    def pop(self) -> object:
        return self.stack.pop() if self.stack else _UNK

    def pop_mark(self) -> List[object]:
        items: List[object] = []
        while self.stack:
            v = self.stack.pop()
            if v is _MARK:
                items.reverse()
                return items
            items.append(v)
        items.reverse()
        return items

    def top(self) -> object:
        return self.stack[-1] if self.stack else _UNK

    # -- reporting ----------------------------------------------------------------------
    def loc(self, pos: Optional[int], op: str = "") -> str:
        where = f"{self.label} offset {pos}" if pos is not None else self.label
        return f"{where} ({op})" if op else where

    def check_global(self, module: str, name: str, pos: Optional[int], op: str) -> None:
        if self.quiet(pos):
            return
        verdict = classify_global(module, name)
        if verdict is None or verdict[0] == "unsafe-pickle-nested":
            return  # nested loaders are judged where they are called (on_reduce): is their input analysable?
        rule_id, reason = verdict
        value = f"{module}.{name}"
        self.ctx.add(rule_id, value, clean(value, 100), self.loc(pos, op), message=f"{_RULE_TITLES[rule_id]}: {value} ({reason})")

    def unresolved(self, pos: Optional[int], op: str, why: str) -> None:
        if self.plain:
            return
        self.ctx.add("pickle-unresolved-global", f"{op}@{pos}", why, self.loc(pos, op))

    # -- main loop ----------------------------------------------------------------------
    def run(self, f) -> str:
        """Walk one pickle. Returns 'stop' when STOP was reached, 'error' otherwise."""
        status = self._run_ops(f)
        self.flush_unconsumed()
        self.flush_text_data()
        return status

    def flush_text_data(self) -> None:
        """UTF-32-like data no dtype claimed: scanned as text, a NUL run being the line break between elements."""
        for pos, op, data, _ in self.text_data.values():
            text = utf32_text(data)
            if text:
                self.ctx.scan_text(text, self.loc(pos, op) + UTF32_NOTE)
        self.text_data.clear()

    def string_array(self, raw: object, dtype: object, pos: Optional[int]) -> None:
        """An ndarray rebuilt by numpy (_reconstruct + BUILD in protocols 0-4, _frombuffer in protocol 5) whose data the
        walk held back: split by the dtype's width, element by element, as npy.py reads a string array."""
        if isinstance(raw, _Obj) and isinstance(raw.callable, _G) and raw.callable.name == "encode" \
                and isinstance(raw.args, tuple) and raw.args:
            raw = raw.args[0]             # protocols 0-2 carry the data as a latin-1 string through _codecs.encode
        held = self.text_data.get(id(raw))
        spec = dtype.args[0] if isinstance(dtype, _Obj) and isinstance(dtype.args, tuple) and dtype.args else None
        m = _ITEM_RE.match(spec) if isinstance(spec, str) else None
        if held is None or held[3] is not raw or not m or m.group(1) != "U" or not m.group(2):
            return
        order = dtype.state[1] if isinstance(dtype.state, tuple) and len(dtype.state) > 1 else "<"
        del self.text_data[id(raw)]
        hpos, hop, data, _ = held
        self.ctx.scan_text(_array_strings(data, "U", int(m.group(2)), order == ">"),
                           self.loc(hpos, hop) + " (a NumPy string array)")

    def flush_unconsumed(self) -> None:
        """getattr / loaders that were imported but never directly called (e.g. handed to functools.partial or map)."""
        for g, pos, op in self.pending.values():
            if self.quiet(pos):
                continue
            full = f"{g.module}.{g.name}"
            if g.name == "getattr":
                self._getattr_finding("pickle-getattr-review", "builtins.getattr (referenced, not called directly)", pos, op)
            else:
                self.ctx.add("unsafe-pickle-nested", f"{full} (referenced)", clean(f"{full} (referenced)", 100), self.loc(pos, op),
                             message=f"{_RULE_TITLES['unsafe-pickle-nested']}: {full} is referenced but not called directly, so its input was not analysed")
        self.pending.clear()

    def track(self, g: "_G", pos: int, op: str) -> None:
        is_getattr = g.module in ("builtins", "__builtin__", "__builtins__") and g.name == "getattr"
        verdict = classify_global(g.module, g.name)
        if is_getattr or (verdict is not None and verdict[0] == "unsafe-pickle-nested"):
            self.pending[id(g)] = (g, pos, op)

    def _run_ops(self, f) -> str:
        opts = self.ctx.options
        try:
            for opcode, arg, pos in pickletools.genops(_Bounded(f, self.size)):
                self.ops += 1
                name = opcode.name
                if name in _STRING_OPS:
                    self.push(arg)
                    if isinstance(arg, str) and len(arg) >= 6 and not self.plain:
                        self.ctx.scan_text(arg, self.loc(pos, name))
                        if "\x00" in arg and len(arg) <= MAX_STRING_ARRAY:   # array data in protocols 0-2: a latin-1 string
                            try:
                                raw = arg.encode("latin-1")
                            except UnicodeEncodeError:
                                raw = b""
                            if utf32_text(raw):
                                self.text_data[id(arg)] = (pos, name, raw, arg)
                elif name in _BYTES_OPS:
                    b = _Bytes(False if self.plain else self.scan_bytes(arg, pos, name))
                    if not self.plain and b.analysed is None:         # UTF-32-like: held until a dtype claims it
                        b.analysed = False
                        self.text_data[id(b)] = (pos, name, bytes(arg), b)
                    self.push(b)
                elif name in _INT_OPS:
                    self.push(arg)
                elif name in ("NONE",):
                    self.push(None)
                elif name == "NEWTRUE":
                    self.push(True)
                elif name == "NEWFALSE":
                    self.push(False)
                elif name in ("FLOAT", "BINFLOAT"):
                    self.push(_UNK)
                elif name == "EMPTY_LIST":
                    self.push([])
                elif name == "EMPTY_TUPLE":
                    self.push(())
                elif name == "EMPTY_DICT":
                    self.push({})
                elif name == "EMPTY_SET":
                    self.push(_UNK)
                elif name == "MARK":
                    self.push(_MARK)
                elif name == "POP":
                    self.pop()
                elif name == "POP_MARK":
                    self.pop_mark()
                elif name == "DUP":
                    self.push(self.top())
                elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
                    n = int(name[-1])
                    items = [self.pop() for _ in range(n)][::-1]
                    self.push(tuple(items))
                elif name == "TUPLE":
                    self.push(tuple(self.pop_mark()))
                elif name == "LIST":
                    self.push(self.pop_mark()[:_MAX_CONTAINER])
                elif name == "DICT":
                    items = self.pop_mark()
                    d: Dict[object, object] = {}
                    for i in range(0, len(items) - 1, 2):
                        if len(d) < _MAX_CONTAINER and isinstance(items[i], (str, int, bytes)):
                            d[items[i]] = items[i + 1]
                    self.push(d)
                elif name == "APPEND":
                    v = self.pop()
                    t = self.top()
                    if isinstance(t, list) and len(t) < _MAX_CONTAINER:
                        t.append(v)
                elif name == "APPENDS":
                    items = self.pop_mark()
                    t = self.top()
                    if isinstance(t, list) and len(t) < _MAX_CONTAINER:
                        t.extend(items[: _MAX_CONTAINER - len(t)])
                elif name == "SETITEM":
                    v = self.pop()
                    k = self.pop()
                    t = self.top()
                    if isinstance(t, dict) and isinstance(k, (str, int, bytes)) and len(t) < _MAX_CONTAINER:
                        t[k] = v
                elif name == "SETITEMS":
                    items = self.pop_mark()
                    t = self.top()
                    if isinstance(t, dict):
                        for i in range(0, len(items) - 1, 2):
                            if len(t) < _MAX_CONTAINER and isinstance(items[i], (str, int, bytes)):
                                t[items[i]] = items[i + 1]
                elif name == "ADDITEMS":
                    self.pop_mark()
                elif name == "FROZENSET":
                    self.pop_mark()
                    self.push(_UNK)
                elif name == "MEMOIZE":
                    self.memo[len(self.memo)] = self.top()
                elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
                    self.memo[arg] = self.top()
                elif name in ("GET", "BINGET", "LONG_BINGET"):
                    self.push(self.memo.get(arg, _UNK))
                elif name == "GLOBAL":
                    module, _, nm = str(arg).partition(" ")
                    g = _G(module, nm)
                    self.push(g)
                    self.check_global(module, nm, pos, name)
                    self.track(g, pos, name)
                elif name == "STACK_GLOBAL":
                    nm = self.pop()
                    module = self.pop()
                    if isinstance(module, str) and isinstance(nm, str):
                        g = _G(module, nm)
                        self.push(g)
                        self.check_global(module, nm, pos, name)
                        self.track(g, pos, name)
                    else:
                        self.push(_UNK)
                        self.unresolved(pos, name, "STACK_GLOBAL operands are not constant strings")
                elif name == "INST":
                    module, _, nm = str(arg).partition(" ")
                    args = self.pop_mark()
                    inst = _Obj(_G(module, nm), tuple(args), "INST")
                    self.push(inst)
                    self.check_global(module, nm, pos, name)
                    self.on_reduce(inst, pos, name)
                elif name == "OBJ":
                    items = self.pop_mark()
                    cls = items[0] if items else _UNK
                    inst = _Obj(cls, tuple(items[1:]), "OBJ")
                    self.push(inst)
                    self.on_reduce(inst, pos, name)
                elif name == "REDUCE":
                    a = self.pop()
                    fn = self.pop()
                    obj = _Obj(fn, a, "REDUCE")
                    self.push(obj)
                    self.on_reduce(obj, pos, name)
                elif name == "NEWOBJ":
                    a = self.pop()
                    cls = self.pop()
                    inst = _Obj(cls, a, "NEWOBJ")
                    self.push(inst)
                    self.on_reduce(inst, pos, name)
                elif name == "NEWOBJ_EX":
                    self.pop()
                    a = self.pop()
                    cls = self.pop()
                    inst = _Obj(cls, a, "NEWOBJ_EX")
                    self.push(inst)
                    self.on_reduce(inst, pos, name)
                elif name == "BUILD":
                    state = self.pop()
                    obj = self.top()
                    if isinstance(obj, _Obj):
                        obj.state = state
                        if _may_reclass(state):
                            obj.op = _RECLASSED      # its class is no longer certain (_getattr_owner)
                        if not self.plain and _is_array_wrapper(obj) and isinstance(state, dict):
                            self.joblib_array(f, state, pos)
                        elif (self.text_data and isinstance(obj.callable, _G) and obj.callable.name == "_reconstruct"
                              and isinstance(state, tuple) and len(state) >= 5):
                            self.string_array(state[4], state[2], pos)
                elif name in ("EXT1", "EXT2", "EXT4"):
                    self.push(_UNK)
                    self.unresolved(pos, name, f"copyreg extension code {arg} cannot be resolved statically")
                elif name == "PERSID":
                    self.push(_UNK)
                elif name == "BINPERSID":
                    self.pop()
                    self.push(_UNK)
                elif name in ("NEXT_BUFFER",):
                    self.push(_UNK)
                elif name == "STOP":
                    return "stop"
                # PROTO, FRAME, READONLY_BUFFER: no stack effect we care about
                if self.ops % 2_000_000 == 0 and not self.ctx.charge(0):
                    self.incomplete = "scan budget exhausted"
                    return "error"
        except Exception as exc:  # genops raises ValueError, EOF-ish errors, struct errors, OverflowError ...
            self.incomplete = f"{type(exc).__name__}: {clean(str(exc), 120)}"
            return "error"
        self.incomplete = "stream ended before STOP"
        return "error"

    def on_reduce(self, obj: "_Obj", pos: int, op: str = "REDUCE") -> None:
        """Called for every opcode that CALLS something with arguments: REDUCE, INST, OBJ, NEWOBJ, NEWOBJ_EX."""
        fn = obj.callable
        if self.quiet(pos):
            if isinstance(fn, _G):
                self.pending.pop(id(fn), None)
            return
        if isinstance(fn, _Obj):
            if self.plain:
                return  # array bytes read as instructions call whatever is on the stack: noise, not a finding
            if fastai_method_binding(obj):
                return  # cloudpickle binding fastai's own listed method to its own object (fastai 2.8 exports)
            # the callable is the RESULT of another call (a resolver, a factory): what it is cannot be known statically,
            # so a human looks. Fail-closed, like getattr with an unknown attribute name.
            self.ctx.add("pickle-computed-callable", f"{op}@{pos}", "call on the result of another call", self.loc(pos, op),
                         message=f"{_RULE_TITLES['pickle-computed-callable']}")
            return
        if not isinstance(fn, _G):
            return
        self.pending.pop(id(fn), None)
        if self.text_data and fn.name == "_frombuffer" and isinstance(obj.args, tuple) and len(obj.args) >= 2:
            self.string_array(obj.args[0], obj.args[1], pos)
        verdict = classify_global(fn.module, fn.name)
        args = obj.args if isinstance(obj.args, tuple) else ()
        if verdict is not None and verdict[0] == "unsafe-pickle-nested":
            if args and isinstance(args[0], _Bytes) and args[0].analysed:
                return  # the embedded payload was scanned as a child file; its own findings are reported there
            self.ctx.add("unsafe-pickle-nested", f"{fn.module}.{fn.name}", clean(f"{fn.module}.{fn.name}", 100), self.loc(pos, op),
                         message=f"{_RULE_TITLES['unsafe-pickle-nested']}: {fn.module}.{fn.name} on data that could not be analysed")
            return
        if fn.module in ("builtins", "__builtin__", "__builtins__") and fn.name == "getattr":
            self.check_getattr(args, pos, op)

    def check_getattr(self, args: tuple, pos: int, op: str) -> None:
        attr = args[1] if len(args) > 1 else None
        if not isinstance(attr, str):
            self._getattr_finding("unsafe-pickle-io", "builtins.getattr(<computed attribute name>)", pos, op)
            return
        # dangerous names first: no allow-list (the fastai one included) can make them quiet
        if getattr_name_is_high(attr):
            self._getattr_finding("unsafe-pickle-io", f"builtins.getattr({attr!r})", pos, op)
            return
        if attr in GETATTR_ALLOWED:
            return
        owner = _getattr_owner(args[0]) if args else None
        if owner is not None and (owner[0], owner[1], attr) in FASTAI_GETATTR_ALLOWED:
            return
        # not known-dangerous, not known-benign: a human decides. Never silently dropped.
        self._getattr_finding("pickle-getattr-review", f"builtins.getattr({attr!r})", pos, op)

    def _getattr_finding(self, rule_id: str, value: str, pos: int, op: str) -> None:
        self.ctx.add(rule_id, value, clean(value, 100), self.loc(pos, op), message=f"{_RULE_TITLES[rule_id]}: {value}")

    # -- bytes carried inside a pickle --------------------------------------------------
    def scan_bytes(self, data, pos: int, op: str) -> bool:
        """Scan a bytes constant. Returns True when it was a nested pickle/zip that was analysed as a child file."""
        n = len(data)
        if n < 6:
            return False
        head = bytes(data[:4096])
        nested = (head[:4] == b"PK\x03\x04") or (head[:1] == b"\x80" and len(head) > 8 and 2 <= head[1] <= 5)
        if nested and self.ctx.depth < self.ctx.options.max_depth and n <= 2048 * MB:
            from ..scanner import scan_stream  # lazy: scanner imports this module

            scan_stream(io.BytesIO(bytes(data)), self.ctx.child(f"bytes@{pos}"), f"nested@{pos}.bin", n)
            return True
        limit = 1 * MB
        if n > limit and not self.ctx.options.deep and not (n <= MAX_STRING_ARRAY and utf32_text(bytes(data[:4096]))):
            self.skipped_blobs += 1
            return False
        loc = self.loc(pos, op)
        printable = sum(1 for b in head if 32 <= b < 127 or b in (9, 10, 13))
        if printable / max(1, len(head)) >= 0.9 and n <= limit:
            self.ctx.scan_text(bytes(data).decode("utf-8", "replace"), loc)
        else:
            from .binary import scan_blob_secrets

            scan_blob_secrets(self.ctx, bytes(data), loc)
            if n <= MAX_STRING_ARRAY and utf32_text(bytes(data)):
                return None        # read as text later, split by the array's dtype when one claims it (_Walker.string_array)
        return False

    # -- joblib ---------------------------------------------------------------------------
    def joblib_array(self, f, state: dict, pos: int) -> None:
        """Skip the raw array bytes joblib writes right after a NumpyArrayWrapper."""
        dtype = state.get("dtype")
        shape = state.get("shape")
        spec = dtype.args[0] if isinstance(dtype, _Obj) and isinstance(dtype.args, tuple) and dtype.args else None
        info = _dtype_itemsize(spec)
        if info is None or not isinstance(shape, tuple) or not all(isinstance(x, int) for x in shape):
            raise _Aborted("joblib array layout not understood")
        itemsize, has_object = info
        if has_object:
            # joblib writes a nested protocol-5 pickle for object arrays
            _scan_one(f, self.ctx, f"{self.label}>array", self.size)
            return
        count = 1
        for x in shape:
            count *= x
        if state.get("numpy_array_alignment_bytes") is not None:
            pad = f.read(1)
            if not pad:
                raise _Aborted("truncated joblib array padding")
            f.seek(pad[0], io.SEEK_CUR)
        payload = count * itemsize
        here = f.tell()
        if here + payload > self.size:
            raise _Aborted("joblib array runs past end of file")
        if payload > 0:
            self.skipped_ranges.append(here)
        kind = _ITEM_RE.match(spec).group(1)
        if kind in "US" and 0 < payload <= MAX_STRING_ARRAY:
            # a string array is text the scanner reads, as in a .npy file (pre-release test 5, T1: a classifier's class
            # labels were skipped with the weights). The byte order is in the dtype's state: (3, '<', ...)
            order = dtype.state[1] if isinstance(dtype.state, tuple) and len(dtype.state) > 1 else "<"
            data = f.read(payload)
            width = itemsize // 4 if kind == "U" else itemsize
            self.ctx.scan_text(_array_strings(data, kind, width, order == ">" or spec.startswith(">")),
                               f"{self.label} joblib string array at offset {here}")
            return
        if self.ctx.options.deep and 0 < payload <= 256 * MB:
            from .binary import scan_blob_secrets

            data = f.read(payload)
            scan_blob_secrets(self.ctx, data, f"{self.label} joblib array at offset {here}")
        else:
            f.seek(payload, io.SEEK_CUR)


_RULE_TITLES = {
    "unsafe-pickle-exec": "Pickle can execute code when loaded",
    "unsafe-pickle-io": "Pickle uses a file/network/indirect-access callable",
    "unsafe-pickle-nested": "Pickle starts a nested deserialisation",
    "pickle-getattr-review": "Pickle uses getattr with an attribute name that is neither known-benign nor known-dangerous (review)",
    "pickle-computed-callable": "Pickle calls an object produced by another call, which a static scan cannot identify (review)",
}

_RAW_GLOBAL = re.compile(rb"c([A-Za-z_][\w.]{0,60})\n([A-Za-z_][\w.]{0,60})\n")
_RAW_SHORT = re.compile(rb"\x8c([\x01-\x40])([A-Za-z_][\w.]*)\x94?\x8c([\x01-\x40])([A-Za-z_][\w.]*)\x94?\x93")


def _scan_one(f, ctx: ScanContext, label: str, size: int, depth: int = 0, plain: bool = False,
              report_from: int = 0) -> Tuple[str, Optional[str], _Walker]:
    w = _Walker(ctx, label, size, plain=plain, report_from=report_from)
    w.depth = depth
    try:
        status = w.run(f)
    except _Aborted as exc:
        w.incomplete = str(exc)
        status = "error"
    return status, w.incomplete, w


def heuristic_tail_scan(data: bytes, ctx: ScanContext, label: str, base: int) -> None:
    """Opcode-free scan of bytes the parser could not walk: imports and provider-specific secrets, and every text rule when
    the region is text."""
    for m in _RAW_GLOBAL.finditer(data):
        _report_raw_global(ctx, label, base + m.start(), m.group(1).decode("latin-1"), m.group(2).decode("latin-1"))
    for m in _RAW_SHORT.finditer(data):
        if len(m.group(2)) == m.group(1)[0] and len(m.group(4)) == m.group(3)[0]:
            _report_raw_global(ctx, label, base + m.start(), m.group(2).decode("latin-1"), m.group(4).decode("latin-1"))
    from .binary import scan_blob_secrets

    scan_blob_secrets(ctx, data, f"{label} unparsed region at offset {base}")
    # A region that is text (a protocol-0 pickle the walk could not finish is mostly text; so is JSON inside joblib's zlib
    # wrapper) also gets every text rule, not only the provider-specific ones (pre-release test 4, S1: planted values in
    # such a region were missed).
    from ..scanner import SNIFF_TEXT_MAX, _printable_ratio  # lazy: scanner imports this module

    if data and len(data) <= SNIFF_TEXT_MAX and _printable_ratio(data[:65536]) >= 0.95 and b"\x00" not in data[:4096]:
        ctx.scan_text(data.decode("utf-8", "replace"), f"{label} unparsed region at offset {base} (read as text)",
                      line_numbers=True)


def _report_raw_global(ctx: ScanContext, label: str, pos: int, module: str, name: str) -> None:
    verdict = classify_global(module, name)
    if verdict:
        rule_id, reason = verdict
        value = f"{module}.{name}"
        ctx.add(rule_id, value, clean(value, 100), f"{label} unparsed region near offset {pos}",
                message=f"{_RULE_TITLES[rule_id]}: {value} ({reason}; found by raw byte scan)")


def _is_whole_pickle(f, ctx: ScanContext, size: int) -> bool:
    """Review 2026-11-11, finding 4: does a complete pickle start here, even without the 0x80 of protocol 2+? (Protocols 0
    and 1 have no marker byte.) Walked once with a throwaway context, so that junk read as instructions reports nothing;
    only a stream that reaches STOP is then analysed for real. The position is restored."""
    start = f.tell()
    try:
        status, _, _ = _scan_one(f, ScanContext(ctx.top_path, ctx.options), "probe", size)
    finally:
        f.seek(start)
    return status == "stop"


def is_whole_legacy_pickle(f, size: int, name: str = "probe") -> bool:
    """Pre-release test 2026-11-19, finding F3: is the whole stream, from its first byte to its end, ONE complete pickle?
    Protocols 0 and 1 have no marker byte, so a .bin/.p/.model/.state/.jl file in those protocols was taken for text or an
    unknown format and never analysed. Walked once with a throwaway context (nothing is reported); the stream must reach
    STOP and end there (trailing whitespace allowed), so text or another binary format that merely starts like a pickle
    is not taken for one. The position is restored."""
    if size < 2:
        return False
    start = f.tell()
    try:
        f.seek(0)
        status, _, _ = _scan_one(f, ScanContext(name), "probe", size)
        if status != "stop":
            return False
        end = f.tell()
        return end == size or (size - end <= 64 and not f.read(64).strip())
    except Exception:
        return False
    finally:
        f.seek(start)


def scan_pickle_stream(f, ctx: ScanContext, label: str = "pickle") -> None:
    """Scan every pickle in the seekable binary stream ``f``."""
    try:
        origin = f.tell()
        f.seek(0, io.SEEK_END)
        size = f.tell()
        f.seek(origin)
    except Exception:
        data = f.read()
        f = io.BytesIO(data)
        size = len(data)
    legacy_torch = False
    index = 0
    expected = None
    while True:
        start = f.tell()
        if start >= size:
            return
        if index > 0:
            b = f.read(1)
            f.seek(start)
            if not legacy_torch and b != b"\x80" and not _is_whole_pickle(f, ctx, size):
                break
            if legacy_torch and expected is not None and index >= expected:
                break
        status, why, walker = _scan_one(f, ctx, label, size)
        if walker.skipped_ranges:
            # the same pickle as a plain loader reads it: the skipped array bytes become instructions
            end = f.tell()
            f.seek(start)
            _scan_one(f, ctx, label, size, plain=True, report_from=walker.skipped_ranges[0])
            f.seek(end)
        if walker.skipped_blobs:
            ctx.note("info", f"{walker.skipped_blobs} large bytes object(s) inside the pickle not scanned (use --deep)")
        if status != "stop":
            ctx.add("pickle-incomplete-analysis", f"{start}:{why}", clean(why or "", 100),
                    f"{label} offset {f.tell()}", message=f"Pickle could not be analysed to its end ({why}); remainder carved heuristically")
            rest_start = start
            f.seek(rest_start)
            remaining = f.read(min(size - rest_start, 64 * MB if not ctx.options.deep else 512 * MB))
            heuristic_tail_scan(remaining, ctx, label, rest_start)
            return
        # first pickle of the legacy PyTorch format holds only the magic number
        if index == 0 and walker.stack and walker.ops <= 5:
            top = walker.stack[-1] if walker.stack else None
            if top == _TORCH_MAGIC:
                legacy_torch = True
                expected = 5  # magic, protocol version, sys_info, main object, storage keys
        index += 1
        if index > 10000:
            ctx.note("skipped", "more than 10000 concatenated pickles; remainder not scanned")
            return
    # data after the last pickle
    pos = f.tell()
    if pos < size:
        if legacy_torch:
            if ctx.options.deep:
                from .binary import scan_blob_secrets

                scan_blob_secrets(ctx, f.read(min(size - pos, 512 * MB)), f"{label} tensor storage at offset {pos}")
            return
        trailing = size - pos
        ctx.add("pickle-trailing-data", f"{trailing}", f"{trailing} bytes",
                f"{label} offset {pos}", message=f"{trailing} bytes of non-pickle data follow the pickle stream")
        rest = f.read(min(trailing, 64 * MB))
        heuristic_tail_scan(rest, ctx, label, pos)

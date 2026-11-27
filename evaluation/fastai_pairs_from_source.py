"""Derive the fastai getattr allow-list from fastai's own source (0.1.1; see CHANGELOG).

Reads the wheels as zip files and parses their .py files with ``ast``: nothing is imported, installed or run.

    python -I evaluation/fastai_pairs_from_source.py --wheels DIR [--out limnex_preflight/formats/fastai_pairs.py]
                                                     [--report REPORT.json]

The rule (written down before the list was first generated):
1. Seeds: a method M used as a value (assigned, passed as an argument, returned, put in a container, used as a default;
   never called right there) through ``self.M`` / ``cls.M`` inside a class, ``<fastai class>.M``, or ``p.M`` for a
   parameter p annotated with (or defaulting to) a fastai class. M must be a function defined in fastai's source on that
   class or a base: a def in the class body (plain, classmethod, staticmethod) or a fastcore @patch / @patch_to.
   Properties are not methods.
2. Inheritance: a seed (D, M) allows (C, M) for D and every fastai class C inheriting from D, when normal lookup of M
   on C (C3 order) ends at a function from fastai's source. If it could stop first at a class outside fastai (fastcore,
   fasttransform, or an unknown class such as torch's), the pair is left out.
3. Classes are named where fastai defines them. Also (tester of 0.1.1, H1): Python 3.13 and later name the class of a
   function taken from a class through the function's own module, and a function that fastcore's @patch / @patch_to
   adds keeps the module it was written in. So a method patched from another fastai module P onto a class C, with C
   written there by its plain name (P binds that name), is also allowed as (P, C, M). Only C itself gets this second
   name: a function's qualified name holds the class it was patched onto, never a subclass. Two versions are read; a
   pair holds if it holds at either.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # this repository's own scanner (our code, not fastai's)
from limnex_preflight.formats.pickle_scan import getattr_name_is_high  # noqa: E402

# wheel name -> full sha256 published on PyPI, checked before reading (tester of 0.1.1, N1: was a 16-digit prefix)
WHEELS = {
    "fastai-2.8.12-py3-none-any.whl": "50474575768679ba7352bebd0fdf8e7d2c4ddc10242455bec79a19c964f03fd0",
    "fastcore-2.2.23-py3-none-any.whl": "7f65dbe5060c048bc973af041104eab9cb43b41e33a83c9e3d1918604a518290",
    "fasttransform-0.0.5-py3-none-any.whl": "c9ccda488c1f95900934e89e748ac4ec3742e39999893637b1481f0fc646474e",
    "fastai-2.7.19-py3-none-any.whl": "8c47d72ac28cd09671ce85f42100436b99af44e4147bd9df0495049b50708d3b",
    "fastcore-1.7.29-py3-none-any.whl": "76fd4815eabbed704faca3abfea4b7e1f98b6351ba6c869a2d405f37bc4b0074",
}
VERSIONS = {
    "2.8.12": ["fastai-2.8.12-py3-none-any.whl", "fastcore-2.2.23-py3-none-any.whl", "fasttransform-0.0.5-py3-none-any.whl"],
    "2.7.19": ["fastai-2.7.19-py3-none-any.whl", "fastcore-1.7.29-py3-none-any.whl"],
}
TOPS = ("fastai", "fastcore", "fasttransform")
FUNC_KINDS = ("method", "classmethod", "staticmethod")

# self-check: the 11 pairs read by hand after the first study, and the 6 that the independent review and the first
# held-out set named; the rule must reproduce them (it is not changed to fit them)
SECTION8 = {
    ("fastai.data.core", "TfmdLists", "dataloaders"), ("fastai.data.core", "Datasets", "dataloaders"),
    ("fastai.tabular.core", "TabularPandas", "dataloaders"), ("fastai.vision.augment", "Flip", "before_call"),
    ("fastai.vision.augment", "Brightness", "before_call"), ("fastai.vision.augment", "Resize", "before_call"),
    ("fastai.vision.augment", "RandomResizedCrop", "before_call"),
    ("fastai.vision.augment", "RandomResizedCropGPU", "before_call"), ("fastai.vision.core", "PILImage", "create"),
    ("fastai.torch_core", "TensorImage", "lighting"), ("fastai.text.models.awdlstm", "WeightDropout", "_do_nothing"),
}
NAMED_BY_REVIEW = {
    ("fastai.callback.hook", "Hook", "hook_fn"), ("fastai.vision.core", "PILMask", "create"),
    ("fastai.collab", "TabularCollab", "dataloaders"), ("fastai.vision.core", "TensorPoint", "create"),
    ("fastai.callback.progress", "ProgressCallback", "_write_stats"), ("fastai.data.core", "FilteredBase", "dataloaders"),
}


@dataclass
class Cls:
    module: str
    name: str
    file: str
    line: int
    base_nodes: list
    members: Dict[str, Tuple[str, str, int]] = field(default_factory=dict)  # name -> (kind, file, line)
    bases: list = field(default_factory=list)                               # resolved: Cls or "ext:<text>"

    @property
    def key(self) -> Tuple[str, str]:
        return self.module, self.name

    @property
    def is_fastai(self) -> bool:
        return self.module == "fastai" or self.module.startswith("fastai.")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_sources(wheel: Path) -> Dict[str, Tuple[str, str]]:
    """module -> (file name inside the wheel, source text), for the fastai/fastcore/fasttransform packages only."""
    out = {}
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            if not name.endswith(".py") or name.split("/")[0] not in TOPS:
                continue
            mod = name[:-3].replace("/", ".")
            if mod.endswith(".__init__"):
                mod = mod[: -len(".__init__")]
            out[mod] = (name, z.read(name).decode("utf-8"))
    return out


def dec_name(d: ast.expr) -> str:
    """'patch', 'patch_to', 'classmethod', 'x.setter', ... for a decorator node (call or not)."""
    if isinstance(d, ast.Call):
        d = d.func
    if isinstance(d, ast.Name):
        return d.id
    if isinstance(d, ast.Attribute):
        return d.attr if not isinstance(d.value, ast.Name) else f"{d.value.id}.{d.attr}"
    return ""


def def_kind(fn: ast.FunctionDef) -> str:
    names = [dec_name(d) for d in fn.decorator_list]
    if any(n in ("property", "cached_property", "functools.cached_property") or n.endswith((".setter", ".getter", ".deleter"))
           for n in names):
        return "property"
    if "classmethod" in names:
        return "classmethod"
    if "staticmethod" in names:
        return "staticmethod"
    return "method"


def kw_true(call: ast.expr, key: str) -> bool:
    return isinstance(call, ast.Call) and any(k.arg == key and isinstance(k.value, ast.Constant) and k.value.value is True
                                              for k in call.keywords)


def class_names_in(node: Optional[ast.expr]) -> List[str]:
    """Class names in an annotation or patch_to target: X, a.X, (X, Y), X | Y, Union[X, Y]."""
    if node is None:
        return []
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [ast.unparse(node)]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [n for e in node.elts for n in class_names_in(e)]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return class_names_in(node.left) + class_names_in(node.right)
    if isinstance(node, ast.Subscript) and dec_name(node.value) in ("Union", "typing.Union"):
        return class_names_in(node.slice)
    return []


class Version:
    def __init__(self, label: str, sources: Dict[str, Tuple[str, str]]):
        self.label = label
        self.sources = sources
        self.trees = {m: ast.parse(src, filename=f) for m, (f, src) in sources.items()}
        self.classes: Dict[Tuple[str, str], Cls] = {}
        self.by_name: Dict[str, List[Cls]] = defaultdict(list)
        self.ext_members: Dict[str, Dict[str, Tuple[str, str, int]]] = defaultdict(dict)  # fastai patches on outside classes
        self.patched_from: Dict[Tuple[Tuple[str, str], str], str] = {}  # (class key, method) -> the patch's module (rule 3)
        self.notes: List[str] = []
        self._collect_classes()
        self._resolve_bases()
        self._collect_patches()
        self._mro: Dict[Tuple[str, str], list] = {}

    # -- classes -------------------------------------------------------------------------------------------------
    def _top_level(self, body):
        """Statements at module level, looking inside if/try/with blocks but not into functions or classes."""
        for st in body:
            yield st
            if isinstance(st, (ast.If, ast.Try, ast.With)):
                for part in ("body", "orelse", "finalbody"):
                    yield from self._top_level(getattr(st, part, []) or [])
                for h in getattr(st, "handlers", []) or []:
                    yield from self._top_level(h.body)

    def _collect_classes(self):
        for mod, tree in self.trees.items():
            fname = self.sources[mod][0]
            for st in self._top_level(tree.body):
                if not isinstance(st, ast.ClassDef):
                    continue
                c = Cls(mod, st.name, fname, st.lineno, list(st.bases))
                for b in st.body:
                    if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        kind = def_kind(b)
                        prev = c.members.get(b.name)
                        if prev and prev[0] == "property":   # x.setter after @property x: still a property
                            kind = "property"
                        c.members[b.name] = (kind, fname, b.lineno)
                    elif isinstance(b, (ast.Assign, ast.AnnAssign)):
                        targets = b.targets if isinstance(b, ast.Assign) else [b.target]
                        for t in targets:
                            for n in (t.elts if isinstance(t, ast.Tuple) else [t]):
                                if isinstance(n, ast.Name):
                                    c.members[n.id] = ("attr", fname, b.lineno)
                self.classes[c.key] = c
                self.by_name[c.name].append(c)

    def resolve(self, text: str, module: str, note: bool = True) -> List[object]:
        """A class name as written in `module` -> known classes, or ['ext:<text>'] when it is not one of ours."""
        if "." in text:
            return ["ext:" + text]
        cands = self.by_name.get(text, [])
        if len(cands) == 1:
            return cands
        same = [c for c in cands if c.module == module]
        if same:
            return same[:1]
        if cands:
            if note:
                self.notes.append(f"ambiguous class name {text!r} in {module}: {[c.key for c in cands]}")
            return list(cands)
        return ["ext:" + text]

    def _resolve_bases(self):
        for c in self.classes.values():
            for b in c.base_nodes:
                if isinstance(b, (ast.Name, ast.Attribute)):
                    got = self.resolve(ast.unparse(b), c.module)
                    c.bases.append(got[0] if len(got) == 1 else "ext:ambiguous:" + ast.unparse(b))
                else:
                    c.bases.append("ext:" + ast.unparse(b))

    def _collect_patches(self):
        for mod, tree in self.trees.items():
            fname = self.sources[mod][0]
            for st in self._top_level(tree.body):
                # fastcore's properties(Class, 'a', 'b') turns those methods into properties
                if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) and dec_name(st.value) == "properties" \
                        and st.value.args and isinstance(st.value.args[0], ast.Name):
                    for c in self.resolve(st.value.args[0].id, mod):
                        for a in st.value.args[1:]:
                            if isinstance(c, Cls) and isinstance(a, ast.Constant) and a.value in c.members:
                                c.members[a.value] = ("property",) + c.members[a.value][1:]
                    continue
                if not isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for d in st.decorator_list:
                    n = dec_name(d)
                    if n == "patch":
                        args = st.args.posonlyargs + st.args.args
                        targets = class_names_in(args[0].annotation) if args else []
                    elif n == "patch_to" and isinstance(d, ast.Call) and d.args:
                        targets = class_names_in(d.args[0])
                    else:
                        continue
                    kind = "property" if kw_true(d, "as_prop") else ("classmethod" if kw_true(d, "cls_method") else "method")
                    for t in targets:
                        for c in self.resolve(t, mod):
                            if isinstance(c, Cls):
                                c.members[st.name] = (kind, fname, st.lineno)
                                if mod != c.module and t == c.name:      # rule 3: P binds C by its plain name
                                    self.patched_from[(c.key, st.name)] = mod
                                else:
                                    self.patched_from.pop((c.key, st.name), None)
                            else:
                                self.ext_members[c][st.name] = (kind, fname, st.lineno)

    # -- lookup --------------------------------------------------------------------------------------------------
    def mro(self, c) -> list:
        if isinstance(c, str):
            return [c]
        if c.key in self._mro:
            return self._mro[c.key]
        self._mro[c.key] = [c]                     # guards against a cycle while computing
        seqs = [self.mro(b) for b in c.bases] + [list(c.bases)]
        out = [c]
        seqs = [list(s) for s in seqs if s]
        while seqs:
            for s in seqs:
                head = s[0]
                if not any(head is t or head == t for q in seqs for t in q[1:]):
                    break
            else:   # inconsistent order: fall back to depth-first, left to right
                self.notes.append(f"C3 failed for {c.key}; depth-first order used")
                rest = []
                for q in seqs:
                    for t in q:
                        if not any(t is r or t == r for r in rest + out):
                            rest.append(t)
                out += rest
                break
            out.append(head)
            seqs = [[t for t in q if not (t is head or t == head)] for q in seqs]
            seqs = [q for q in seqs if q]
        self._mro[c.key] = out
        return out

    def lookup(self, c: Cls, name: str):
        """(where, kind, file, line) of the first class in c's order that holds `name`; where = Cls, or 'ext:...' when an
        outside class comes first (it may hold the name: unknown), or None if nothing holds it."""
        for k in self.mro(c):
            if isinstance(k, str):
                if name in self.ext_members.get(k, {}):
                    kind, f, line = self.ext_members[k][name]
                    return k, kind, f, line
                return k, "unknown", "", 0
            if name in k.members:
                kind, f, line = k.members[name]
                return k, kind, f, line
        return None

    def descendants(self, d: Cls) -> List[Cls]:
        return [c for c in self.classes.values() if any(k is d for k in self.mro(c))]

    # -- seeds ---------------------------------------------------------------------------------------------------
    def seeds(self) -> List[dict]:
        """Seeds from fastai's own modules only (fastcore and fasttransform are read for the class order, not seeds)."""
        out = []
        for mod, tree in self.trees.items():
            if not (mod == "fastai" or mod.startswith("fastai.")):
                continue
            fname = self.sources[mod][0]
            parents = {}
            for p in ast.walk(tree):
                for ch in ast.iter_child_nodes(p):
                    parents[ch] = p
            self._each(tree.body, (mod, fname, parents, out), "module", None, {})
        return out

    def _param_classes(self, fn, mod) -> Dict[str, List[Cls]]:
        args = fn.args
        pos = args.posonlyargs + args.args
        defaults = [None] * (len(pos) - len(args.defaults)) + list(args.defaults)
        pairs = list(zip(pos, defaults)) + list(zip(args.kwonlyargs, args.kw_defaults))
        out = {}
        for a, dflt in pairs:
            got = [c for t in class_names_in(a.annotation) for c in self.resolve(t, mod) if isinstance(c, Cls)]
            if isinstance(dflt, ast.Name):
                got += [c for c in self.resolve(dflt.id, mod) if isinstance(c, Cls)]
            out[a.arg] = got            # an empty list still shadows a class of the same name
        return out

    _BLOCKS = (ast.If, ast.Try, ast.With, ast.ExceptHandler)

    def _each(self, children, ctx, level, cls, scope):
        """Visit nodes. level: 'module' / 'class' (cls = that class, or None if not a top-level one) / 'func' / 'expr'.
        scope: receiver name -> classes its attribute lookup starts from (self/cls of a method, typed parameters; an
        empty list shadows a class of the same name)."""
        mod = ctx[0]
        for ch in children:
            if isinstance(ch, ast.ClassDef):
                self._each(ch.decorator_list + ch.bases + [k.value for k in ch.keywords], ctx, "expr", None, scope)
                c = self.classes.get((mod, ch.name)) if level == "module" else None
                self._each(ch.body, ctx, "class", c, scope)
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                outer = ch.decorator_list + ch.args.defaults + [k for k in ch.args.kw_defaults if k is not None]
                self._each(outer, ctx, "expr", None, scope)
                inner = dict(scope)
                inner.update(self._param_classes(ch, mod))
                pos = ch.args.posonlyargs + ch.args.args
                if level == "class" and cls is not None and pos and \
                        "staticmethod" not in [dec_name(d) for d in ch.decorator_list]:
                    inner[pos[0].arg] = [cls]                  # self / cls of a method
                self._each(ch.body, ctx, "func", None, inner)
            elif isinstance(ch, ast.Lambda):
                self._each(ch.args.defaults + [k for k in ch.args.kw_defaults if k is not None], ctx, "expr", None, scope)
                inner = dict(scope)
                inner.update({a.arg: [] for a in ch.args.posonlyargs + ch.args.args + ch.args.kwonlyargs})
                self._each([ch.body], ctx, "func", None, inner)
            else:
                self._check(ch, ctx, scope)
                keep = level in ("module", "class") and isinstance(ch, self._BLOCKS)
                sub = level if keep else ("func" if level == "func" else "expr")
                self._each(list(ast.iter_child_nodes(ch)), ctx, sub, cls if keep else None, scope)

    @staticmethod
    def _is_value_use(node, parents) -> bool:
        p = parents.get(node)
        if isinstance(p, ast.Call):
            return node is not p.func
        if isinstance(p, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            return node is p.value
        if isinstance(p, (ast.Return, ast.Yield, ast.YieldFrom, ast.Tuple, ast.List, ast.Set, ast.Dict, ast.keyword,
                          ast.arguments, ast.Starred, ast.BoolOp)):
            return True
        if isinstance(p, ast.Lambda):
            return node is p.body
        if isinstance(p, ast.IfExp):
            return node is not p.test
        return False

    def _check(self, node, ctx, scope):
        mod, fname, parents, out = ctx
        if not (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name)):
            return
        if not self._is_value_use(node, parents):
            return
        recv, name = node.value.id, node.attr
        if recv in scope:
            starts, how = scope[recv], "self/cls or parameter"
        else:
            starts, how = [c for c in self.resolve(recv, mod, note=False) if isinstance(c, Cls)], "class name"
        for d in starts:
            hit = self.lookup(d, name)
            if hit is None:
                continue
            where, kind, f, line = hit
            if kind not in FUNC_KINDS:
                continue
            out.append({"class": list(d.key), "name": name, "file": fname, "line": node.lineno, "how": how,
                        "defined_in": where if isinstance(where, str) else list(where.key), "def_line": line,
                        "def_file": f, "fastai_def": isinstance(where, Cls) and where.is_fastai})

    # -- pairs ---------------------------------------------------------------------------------------------------
    def pairs(self, seeds: List[dict]):
        allowed, left_out = {}, []
        for s in seeds:
            if not s["fastai_def"]:
                left_out.append({"seed": s, "why": "defined outside fastai"})
                continue
            if getattr_name_is_high(s["name"]):
                left_out.append({"seed": s, "why": "HIGH name: checked first, can never be quiet"})
                continue
            d = self.classes[tuple(s["class"])]
            for c in self.descendants(d):
                if not c.is_fastai:
                    continue
                where, kind, f, line = self.lookup(c, s["name"])
                if isinstance(where, Cls) and where.is_fastai and kind in FUNC_KINDS:
                    allowed.setdefault((c.module, c.name, s["name"]), (where.key, f, line))
                    p = self.patched_from.get((c.key, s["name"])) if where is c else None
                    if p and (p == "fastai" or p.startswith("fastai.")):   # rule 3: the name Python 3.13+ writes
                        allowed.setdefault((p, c.name, s["name"]), (where.key, f, line))
                else:
                    left_out.append({"pair": [c.module, c.name, s["name"]], "why": f"lookup stops at {where} ({kind})"})
        return allowed, left_out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wheels", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--report", type=Path)
    a = ap.parse_args(argv)

    sums = {}
    for w, short in WHEELS.items():
        data = (a.wheels / w).read_bytes()
        sums[w] = sha256(data)
        if sums[w] != short:
            print(f"{w}: sha256 {sums[w]} does not match PyPI's {short}", file=sys.stderr)
            return 1

    report = {"wheels": sums, "versions": {}}
    union: Dict[Tuple[str, str, str], dict] = {}
    for label, wheels in VERSIONS.items():
        sources = {}
        for w in wheels:
            sources.update(read_sources(a.wheels / w))
        v = Version(label, sources)
        seeds = v.seeds()
        allowed, left_out = v.pairs(seeds)
        seed_keys = sorted({(tuple(s["class"]), s["name"]) for s in seeds if s["fastai_def"]})
        report["versions"][label] = {
            "classes": len(v.classes), "seeds": seeds, "seed_count": len(seed_keys), "pairs": len(allowed),
            "left_out": left_out, "notes": sorted(set(v.notes)),
        }
        for k, (definer, f, line) in allowed.items():
            union.setdefault(k, {"versions": [], "defined_in": list(definer), "file": f, "line": line})
            union[k]["versions"].append(label)
        print(f"fastai {label}: {len(v.classes)} classes, {len(seed_keys)} seeds, {len(allowed)} pairs, "
              f"{len(left_out)} left out, {len(set(v.notes))} notes")

    pairs = set(union)
    missing8 = sorted(SECTION8 - pairs)
    missing_review = sorted(NAMED_BY_REVIEW - pairs)
    report["union_pairs"] = {"|".join(k): v for k, v in sorted(union.items())}
    report["self_check"] = {"section8_missing": missing8, "review_missing": missing_review}
    print(f"union: {len(pairs)} pairs; section-8 pairs not produced: {missing8}; review pairs not produced: {missing_review}")

    if a.report:
        a.report.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    if a.out:
        lines = [
            '"""fastai methods that fastai\'s own code uses as values (kept bound on its objects, or taken from a class),',
            "by (module, class, name), for every fastai class that reaches them by normal attribute lookup.",
            "",
            "GENERATED by evaluation/fastai_pairs_from_source.py from fastai's source (read with ast; nothing imported or",
            "run), versions 2.8.12 and 2.7.19. The rule is in that script's docstring. Do not edit by hand.",
            "Wheels read (sha256 as published on PyPI):",
        ]
        for w, s in sums.items():
            lines += [f"    {w}", f"        {s}"]
        lines += [
            '"""',
            "",
            "FASTAI_SOURCE_PAIRS = frozenset({",
        ]
        lines += [f"    ({m!r}, {c!r}, {n!r})," for m, c, n in sorted(pairs)]
        lines += ["})", ""]
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:   # tester of 0.1.1, L4: write_text(newline=) is 3.10+
            fh.write("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Plain-text formats: .json/.jsonl, .yaml/.yml, .txt/.cfg/.ini/.toml/.md/.env, and Jupyter notebooks."""

from __future__ import annotations

import json
import re
from typing import Any, BinaryIO, Iterator, List, Optional, Tuple

from ..context import ScanContext
from ..detectors import MODE_FULL
from ..masking import clean, mask_secret
from .binary import read_limited

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_B64_MIME_PREFIXES = ("image/", "audio/", "video/", "application/pdf", "application/octet-stream")


def decode_text(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", "replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", "replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _fmt_path(path: Tuple[Any, ...], root: str) -> str:
    out = [root]
    for p in path:
        if isinstance(p, int):
            out.append(f"[{p}]")
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\-]*", p):
            out.append("." + p)
        else:
            out.append("[" + json.dumps(p[:60]) + "]")
    s = "".join(out)
    return s if len(s) <= 160 else s[:157] + "..."


def walk_json(ctx: ScanContext, obj: Any, root: str = "$", *, skip_mime_data: bool = True, mode: str = MODE_FULL) -> int:
    """Scan every string value and key of a parsed JSON document. Locations are JSON paths."""
    n = 0
    stack: List[Tuple[Any, Tuple[Any, ...], Optional[str]]] = [(obj, (), None)]
    while stack:
        node, path, key = stack.pop()
        if isinstance(node, str):
            n += ctx.scan_text(node, _fmt_path(path, root), key=key, mode=mode)
        elif isinstance(node, dict):
            _mapping_checks(ctx, node, path, root)
            items = list(node.items())
            for k, v in reversed(items):
                ks = str(k)
                if len(ks) >= 6:
                    n += ctx.scan_text(ks, _fmt_path(path + (ks,), root) + " (key)", mode=mode)
                if skip_mime_data and isinstance(v, str) and len(v) > 256 and ks.startswith(_B64_MIME_PREFIXES):
                    continue
                if isinstance(v, (str, dict, list)):
                    stack.append((v, path + (ks,), ks))
        elif isinstance(node, list):
            for i in range(len(node) - 1, -1, -1):
                v = node[i]
                if isinstance(v, (str, dict, list)):
                    stack.append((v, path + (i,), key))
    return n


def _mapping_checks(ctx: ScanContext, d: dict, path: Tuple[Any, ...], root: str) -> None:
    """Rules that need two sibling keys at once."""
    # Keras Lambda layers carry serialised Python code (marshalled bytecode) that runs when the model is loaded
    # without safe mode. Keras 2 stores class_name "Lambda"; Keras 3 stores a nested "__lambda__" object.
    cls = d.get("class_name")
    if cls == "__lambda__" or (cls == "Lambda" and isinstance(d.get("config"), dict) and "function" in d["config"]):
        name = (d.get("config") or {}).get("name") if isinstance(d.get("config"), dict) else None
        ctx.add("unsafe-keras-lambda", f"{cls}:{name}", "Lambda layer", _fmt_path(path, root))
    if "username" in d and isinstance(d.get("key"), str) and re.fullmatch(r"[0-9a-f]{32}", d["key"]):
        ctx.add("kaggle-credentials-json", d["key"], mask_secret(d["key"]), _fmt_path(path + ("key",), root))


def scan_json_bytes(data: bytes, ctx: ScanContext, name: str) -> None:
    text = decode_text(data)
    try:
        obj = json.loads(text)
    except ValueError:
        ctx.note("info", f"{name}: not valid JSON; scanned as plain text")
        ctx.scan_text(text, "", line_numbers=True)
        return
    walk_json(ctx, obj)


def scan_jsonl_text(text: str, ctx: ScanContext) -> None:
    for i, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            ctx.scan_text(line, f"line {i}")
            continue
        walk_json(ctx, obj, root=f"line {i}: $")


def scan_plain_text(f: BinaryIO, ctx: ScanContext, name: str) -> None:
    cap = ctx.options.max_text_bytes
    data = read_limited(f, cap)
    if not ctx.charge(len(data)):
        return
    if len(data) == cap:
        ctx.note("skipped", f"text larger than {cap} bytes; only the first part was scanned")
    lower = name.lower()
    if lower.endswith(".json"):
        scan_json_bytes(data, ctx, name)
    elif lower.endswith((".jsonl", ".ndjson")):
        scan_jsonl_text(decode_text(data), ctx)
    elif lower.endswith(".ipynb"):
        scan_notebook_bytes(data, ctx, name)
    else:
        text = decode_text(data)
        ctx.scan_text(text, "", line_numbers=True)
        if lower.endswith((".yaml", ".yml")):
            check_yaml_tags(text, ctx)


# Review 2026-11-11: PyYAML's unsafe loaders turn these tags into Python objects (or imports); safe_load refuses them.
# Plain data tags (!!python/tuple, !!python/str ...) are left alone.
_YAML_OBJECT_TAG = re.compile(r"(?:!!|tag:yaml\.org,2002:)python/(?:object(?:/apply|/new)?|name|module)\b[^\s,\]}]{0,80}")


def check_yaml_tags(text: str, ctx: ScanContext) -> None:
    for m in _YAML_OBJECT_TAG.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        ctx.add("unsafe-yaml-python-tag", m.group(0), clean(m.group(0), 100), f"line {line}", line=line,
                message="YAML tag that makes Python's unsafe YAML loaders build or import Python objects "
                        "(yaml.safe_load refuses it)")


# ----------------------------------------------------------------------------------------
# Jupyter notebooks
# ----------------------------------------------------------------------------------------
def _as_text(v: Any) -> str:
    if isinstance(v, list):
        return "".join(x for x in v if isinstance(x, str))
    return v if isinstance(v, str) else ""


def _notebook_cells(nb: dict) -> Iterator[Tuple[int, dict]]:
    cells = nb.get("cells")
    if cells is None and isinstance(nb.get("worksheets"), list):  # nbformat 3
        cells = [c for ws in nb["worksheets"] if isinstance(ws, dict) for c in ws.get("cells", [])]
    for i, c in enumerate(cells or []):
        if isinstance(c, dict):
            yield i, c


def scan_notebook_bytes(data: bytes, ctx: ScanContext, name: str) -> None:
    try:
        nb = json.loads(decode_text(data))
    except ValueError:
        ctx.note("error", f"{name}: not valid notebook JSON; scanned as plain text")
        ctx.scan_text(decode_text(data), "", line_numbers=True)
        return
    if not isinstance(nb, dict):
        walk_json(ctx, nb)
        return
    # notebook-level metadata (kernelspec, colab/kaggle blocks, widgets state, authors)
    meta = nb.get("metadata")
    if meta is not None:
        walk_json(ctx, meta, root="metadata")
    for i, cell in _notebook_cells(nb):
        ctype = cell.get("cell_type", "?")
        label = f"cell {i + 1} ({ctype})"
        src = _as_text(cell.get("source") if "source" in cell else cell.get("input"))
        if src:
            ctx.scan_text(src, f"{label} source", line_numbers=True)
        if isinstance(cell.get("metadata"), (dict, list)):
            walk_json(ctx, cell["metadata"], root=f"{label} metadata")
        att = cell.get("attachments")
        if isinstance(att, dict):
            walk_json(ctx, att, root=f"{label} attachments")
        for j, out in enumerate(cell.get("outputs") or [], 1):
            if not isinstance(out, dict):
                continue
            otype = out.get("output_type", "?")
            olabel = f"{label} output {j} ({otype}"
            if otype == "stream":
                olabel += f" {out.get('name', '')}"
            olabel += ")"
            if "text" in out:
                ctx.scan_text(_ANSI.sub("", _as_text(out["text"])), olabel, line_numbers=True)
            data_blk = out.get("data")
            if isinstance(data_blk, dict):
                for mime, payload in data_blk.items():
                    if isinstance(mime, str) and mime.startswith(_B64_MIME_PREFIXES):
                        continue
                    if isinstance(payload, (str, list)):
                        ctx.scan_text(_ANSI.sub("", _as_text(payload)), f"{olabel} {mime}", line_numbers=True)
                    else:
                        walk_json(ctx, payload, root=f"{olabel} {mime}")
            if otype == "error":
                for k in ("ename", "evalue"):
                    if isinstance(out.get(k), str):
                        ctx.scan_text(_ANSI.sub("", out[k]), f"{olabel} {k}")
                tbk = out.get("traceback")
                # nbformat stores traceback lines WITHOUT line terminators, unlike source/text lists
                tb = "\n".join(x for x in tbk if isinstance(x, str)) if isinstance(tbk, list) else _as_text(tbk)
                if tb:
                    ctx.scan_text(_ANSI.sub("", tb), f"{olabel} traceback", line_numbers=True)
            # nbformat 3 style
            for k in ("stream", "text", "html", "latex", "json"):
                if k in out and k != "text" and isinstance(out[k], (str, list)):
                    ctx.scan_text(_as_text(out[k]), f"{olabel} {k}", line_numbers=True)
    # anything top-level that is not cells/metadata
    for k, v in nb.items():
        if k in ("cells", "metadata", "worksheets"):
            continue
        if isinstance(v, (dict, list, str)):
            walk_json(ctx, v, root=k)

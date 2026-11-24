""".preflightignore: accept individual findings by fingerprint. Rules can never be switched off."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

_FP = re.compile(r"^[0-9a-f]{16}$")


class IgnoreFileError(ValueError):
    pass


@dataclass
class IgnoreList:
    entries: Dict[str, str] = field(default_factory=dict)  # fingerprint -> comment
    used: Set[str] = field(default_factory=set)
    sources: List[str] = field(default_factory=list)

    def merge(self, other: "IgnoreList") -> None:
        for fp, c in other.entries.items():
            self.entries.setdefault(fp, c)
        self.sources.extend(other.sources)

    def matches(self, fingerprint: str) -> bool:
        if fingerprint in self.entries:
            self.used.add(fingerprint)
            return True
        return False

    @property
    def unused(self) -> List[str]:
        return sorted(set(self.entries) - self.used)


def parse_ignore_text(text: str, source: str = "<ignore>") -> IgnoreList:
    """One fingerprint per line. ``#`` starts a comment. Anything else is rejected on purpose."""
    out = IgnoreList(sources=[source])
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        token, _, comment = line.partition("#")
        token = token.strip()
        if not _FP.match(token):
            # the line itself is never repeated: in the Action this message is a public log annotation, and a pull request
            # could point the ignore file at any file on the runner (pre-release test 3, R2)
            raise IgnoreFileError(
                f"{source}:{n}: invalid entry (the line is not repeated here). Entries are 16-hex-digit finding "
                "fingerprints (shown in every report), optionally followed by '# comment'. Whole rules cannot be disabled."
            )
        out.entries[token] = comment.strip()
    return out


def decode_ignore_bytes(data: bytes, source: str) -> str:
    """UTF-8, with or without a byte-order mark; anything else is a clear usage error. Shared by the CLI and the Action's
    pull-request path, which reads the file from the base commit (pre-release test 2, N3: there a byte-order mark failed
    every pull-request run while the CLI accepted the same file)."""
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise IgnoreFileError(f"{source}: not UTF-8 text; save the ignore file as UTF-8") from None


def link_on_the_way(path: str, root: str) -> Optional[str]:
    """The first part of ``path`` inside ``root`` that is a symbolic link (the file itself included); for a path outside
    ``root``, the file itself if it is a link; else None."""
    root, target = os.path.abspath(root), os.path.abspath(path)
    try:
        rel = os.path.relpath(target, root)
    except ValueError:  # another drive (Windows)
        rel = os.pardir
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return target if os.path.islink(target) else None
    cur = root
    for part in rel.split(os.sep):
        cur = os.path.join(cur, part)
        if os.path.islink(cur):
            return cur
    return None


def load_ignore_file(path: str, confine_to: Optional[str] = None) -> IgnoreList:
    """UTF-8, with or without a byte-order mark. Pre-release test 2026-11-19, finding F12: a UTF-16 file (what Windows
    PowerShell 5 writes by default) crashed with a traceback, and a byte-order mark made the first entry invalid.

    ``confine_to`` (the Action's workspace): the file, and every folder on the way to it inside the workspace, must not be
    a symbolic link. A pull request controls what is on disk there; through a committed link it could make the runner
    read any of its own files (pre-release test 3, R2). Everywhere, the file must be a regular file: a named pipe would
    block the read forever."""
    if confine_to:
        link = link_on_the_way(path, confine_to)
        if link:
            raise IgnoreFileError(f"{path}: is a symbolic link or inside a linked folder; in the Action an ignore file is "
                                  "never read through a link")
    try:
        mode = os.stat(path).st_mode
    except OSError as exc:
        raise IgnoreFileError(f"{path}: cannot be read ({exc.strerror or type(exc).__name__})") from None
    if not stat.S_ISREG(mode):
        raise IgnoreFileError(f"{path}: not a regular file")
    with open(path, "rb") as fh:
        text = decode_ignore_bytes(fh.read(), path)
    return parse_ignore_text(text, path)

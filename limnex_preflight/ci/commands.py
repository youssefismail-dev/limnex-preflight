"""GitHub workflow commands (annotations, outputs, summary) with the escaping they need.

Why this matters: a repository can contain a file whose NAME holds a newline followed by ``::error::...`` or
``::set-output``-style text. If such a name is printed raw, GitHub parses it as a command (annotation forgery, output
injection). Everything that reaches stdout, ``$GITHUB_OUTPUT`` or ``$GITHUB_STEP_SUMMARY`` goes through this module.
"""

from __future__ import annotations

import os
import posixpath
import re
import secrets
from typing import Dict, List, Optional

from ..detectors import RULES_BY_ID
from ..models import Finding, Severity, first_of, grouped
from ..sanitize import no_surrogates as visible_bytes
from ..sanitize import visible

MAX_PER_LEVEL = 10  # GitHub shows at most 10 error + 10 warning + 10 notice annotations per step; 50 per job
_LEVEL = {Severity.CRITICAL: "error", Severity.HIGH: "error", Severity.MEDIUM: "warning", Severity.LOW: "notice", Severity.INFO: "notice"}


def escape_data(s: str) -> str:
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def escape_property(s: str) -> str:
    return escape_data(s).replace(":", "%3A").replace(",", "%2C")


def annotation(level: str, message: str, *, file: Optional[str] = None, line: Optional[int] = None, title: Optional[str] = None) -> str:
    props = []
    if file:
        props.append(f"file={escape_property(file)}")
    if line:
        props.append(f"line={int(line)}")
    if title:
        props.append(f"title={escape_property(title)}")
    return visible_bytes(f"::{level}{' ' + ','.join(props) if props else ''}::{escape_data(message)}")


def finding_annotations(findings: List[Finding], workspace: str, cap: int = MAX_PER_LEVEL, report_only: bool = False) -> List[str]:
    """Masked annotations, most severe first, at most ``cap`` per level, then one '+N more' line. ``findings`` must be sanitised.

    ``report_only`` (fail-on none): the job is going to be green, so nothing is shown as an error."""
    out: List[str] = []
    used: Dict[str, int] = {}
    omitted = 0
    for members in grouped(sorted(findings, key=lambda x: (-int(x.severity), x.file, x.rule_id))):
        f = first_of(members)
        level = _LEVEL[f.severity]
        if report_only and level == "error":
            level = "warning"
        if used.get(level, 0) >= cap:
            omitted += len(members)
            continue
        used[level] = used.get(level, 0) + 1
        top = f.file.split("!/", 1)[0]
        inner = f" in {f.file}" if "!/" in f.file else ""
        if len(members) > 1:   # one annotation for a group (same file, rule and severity, 4 or more)
            text = f"{len(members)} x {RULES_BY_ID[f.rule_id].title}{inner}; first [{f.masked}] at {f.location}"
        else:
            text = f"{f.message} [{f.masked}] at {f.location}{inner}"
        out.append(annotation(
            level,
            visible(text),
            file=workspace_relative(top, workspace),
            line=f.line,
            title=f"{f.rule_id} ({f.severity.label})",
        ))
    if omitted:
        out.append(annotation("notice", f"+{omitted} more finding(s) not annotated; see the job summary and the full report", title="more findings"))
    return out


def workspace_relative(path: str, workspace: str) -> str:
    p = path
    if os.path.isabs(p):
        try:
            rel = os.path.relpath(p, workspace)
            if not rel.startswith(".."):
                p = rel
        except ValueError:
            pass
    if os.sep == "\\":            # on Linux a backslash is part of a name (a \\xNN escape too: pre-release test 5, T11)
        p = p.replace("\\", "/")
    return posixpath.normpath(p) if not os.path.isabs(path) else p  # "./a/b.json" -> "a/b.json"


def write_output(name: str, value: str, path: Optional[str]) -> None:
    """Append ``name=value`` to $GITHUB_OUTPUT using a random heredoc delimiter, so no value can inject further outputs."""
    if not path:
        return
    if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
        raise ValueError(f"bad output name {name!r}")
    delim = "ghadelim_" + secrets.token_hex(16)
    with open(path, "a", encoding="utf-8", errors="backslashreplace", newline="\n") as fh:
        fh.write(f"{name}<<{delim}\n{value}\n{delim}\n")


def append_summary(markdown: str, path: Optional[str]) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8", errors="backslashreplace", newline="\n") as fh:
        fh.write(markdown if markdown.endswith("\n") else markdown + "\n")


def guarded(text: str) -> str:
    """Wrap untrusted text so that workflow commands inside it are inert (``::stop-commands::``)."""
    token = "stop_" + secrets.token_hex(16)
    return f"::stop-commands::{token}\n{text}{'' if text.endswith(chr(10)) else chr(10)}::{token}::\n"

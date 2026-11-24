"""Which files did this push / pull request change?

Rules (each one has a test in tests/test_ci_changed.py):

* pull_request: files changed between the merge base of (base, HEAD) and HEAD.
* pull_request_target: the same, but GITHUB_SHA is the base branch's tip on this event, not the pull request, so the head is
  the checkout's HEAD, and only when it IS the pull request's head (its files are on disk only then); otherwise everything
  is scanned and the log says why (pre-release test 4, S2: base was compared with base, 0 files were scanned, exit 0).
* push: ``event.before .. GITHUB_SHA``.
* first push of a branch (``before`` is all zeros), a ``before`` commit that is not in the checkout (shallow clone, forced
  push), no merge base, other events, or any git failure: return ``files=None`` plus a reason, and the caller scans everything
  and says so. We never silently scan less than was asked for.
* ``git diff -z``: file names are NUL-separated, so spaces, newlines, quotes and unicode survive.
* deleted files are skipped; renames report the new name; the WHOLE file is scanned, not the diff.
* git is called with an allow-listed command line; revisions are validated (never start with ``-``) and resolved to full
  SHAs first, so a hostile ref name cannot become a git option.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from typing import List, Mapping, NamedTuple, Optional

ZEROS = re.compile(r"^0{40}$|^0{64}$")
SHA = re.compile(r"^[0-9a-fA-F]{40}$|^[0-9a-fA-F]{64}$")
REV = re.compile(r"^[0-9A-Za-z_./~^@{}:+%=,-]{1,200}$")  # refs and SHAs; checked for a leading dash separately


class Changed(NamedTuple):
    files: Optional[List[str]]  # paths relative to the current directory; None = could not tell, scan everything
    note: str  # human-readable explanation for the log / job summary


class GitError(Exception):
    pass


def _git(root: str, *args: str) -> bytes:
    cmd = ["git", "-C", root, "-c", "core.quotepath=off", "-c", "diff.external=", *args]
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=120,
                           env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"})
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitError(f"git could not be run: {exc}") from exc
    if p.returncode != 0:
        raise GitError(p.stderr.decode("utf-8", "replace").strip() or f"git exited with {p.returncode}")
    return p.stdout


def _resolve(root: str, rev: str) -> str:
    """Revision -> full commit SHA, or GitError. Refuses anything that could be read as an option."""
    if not rev or rev.startswith("-") or not REV.match(rev):
        raise GitError(f"refusing revision {rev[:60]!r}")
    try:
        return _git(root, "rev-parse", "--verify", "--quiet", "--end-of-options", rev + "^{commit}").decode().strip()
    except GitError:
        raise GitError(f"revision {rev[:60]!r} is not in this checkout") from None


def _list_changed(root: str, base: str, head: str) -> List[str]:
    raw = _git(root, "diff", "-z", "--name-only", "--no-ext-diff", "--no-textconv", "-M", "--diff-filter=ACMRT", base, head, "--")
    return [os.fsdecode(x) for x in raw.split(b"\0") if x]


def _is_shallow(root: str) -> bool:
    try:
        return _git(root, "rev-parse", "--is-shallow-repository").decode().strip() == "true"
    except GitError:
        return False


def _bases(env: Mapping[str, str], event: dict, root: str, override_base: str, override_head: str):
    """Return (base_sha, head_sha, description) or raise GitError with the reason for falling back."""
    name = env.get("GITHUB_EVENT_NAME", "")
    sha = env.get("GITHUB_SHA", "")
    if override_base or override_head:
        if not override_base:
            raise GitError("'head' was given without 'base'")
        head = _resolve(root, override_head or "HEAD")
        return _resolve(root, override_base), head, f"explicit range {override_base[:40]}..{(override_head or 'HEAD')[:40]}"
    if name in ("pull_request", "pull_request_target"):
        base_sha = (((event.get("pull_request") or {}).get("base") or {}).get("sha")) or ""
        if not SHA.match(base_sha):
            raise GitError("the pull_request event has no base SHA")
        try:
            base = _resolve(root, base_sha)
        except GitError:
            raise GitError("the pull request's base commit is not in this checkout (shallow clone: use fetch-depth: 0)")
        if name == "pull_request_target":
            pr_head = (((event.get("pull_request") or {}).get("head") or {}).get("sha")) or ""
            head = _resolve(root, "HEAD")
            if not SHA.match(pr_head) or head.lower() != pr_head.lower():
                raise GitError("on pull_request_target the checkout is the base branch, not the pull request's head "
                               "(check out github.event.pull_request.head.sha to scan only the changed files)")
        else:
            head = _resolve(root, sha if SHA.match(sha) else "HEAD")
        try:
            mb = _git(root, "merge-base", base, head).decode().strip()
        except GitError:
            raise GitError("no merge base found (shallow clone: use fetch-depth: 0)")
        return mb, head, f"pull request: merge base {mb[:10]}..{head[:10]}"
    if name == "push":
        before = event.get("before") or ""
        if ZEROS.match(before):
            raise GitError("first push of this branch (before is all zeros): there is no previous commit to compare with")
        if not SHA.match(before):
            raise GitError("the push event has no usable 'before' SHA")
        try:
            b = _resolve(root, before)
        except GitError:
            raise GitError("the previous commit is not in this checkout (shallow clone or forced push: use fetch-depth: 0)")
        h = _resolve(root, sha if SHA.match(sha) else "HEAD")
        return b, h, f"push: {b[:10]}..{h[:10]}"
    raise GitError(f"changed-only is defined for push, pull_request and pull_request_target events, not "
                   f"{name or 'this run'!r}")


def changed_files(env: Mapping[str, str], scan_path: str, *, base: str = "", head: str = "") -> Changed:
    """Changed files that lie under ``scan_path``, as paths relative to the current directory."""
    # realpath on BOTH sides: git reports the physical top-level (macOS: /private/var/..., Windows: long names), while the
    # workspace path may be reached through a symlink or an 8.3 short name
    scan_abs = os.path.realpath(scan_path)
    root_dir = scan_abs if os.path.isdir(scan_abs) else os.path.dirname(scan_abs)
    try:
        root = os.path.realpath(_git(root_dir, "rev-parse", "--show-toplevel").decode().strip())
    except GitError as exc:
        return Changed(None, f"changed-only: not a git checkout ({_first_line(exc)})")
    event: dict = {}
    event_path = env.get("GITHUB_EVENT_PATH")
    if event_path and os.path.isfile(event_path):
        try:
            with open(event_path, "r", encoding="utf-8") as fh:
                event = json.load(fh)
        except (OSError, ValueError):
            event = {}
    try:
        b, h, what = _bases(env, event, root, base.strip(), head.strip())
        names = _list_changed(root, b, h)
    except GitError as exc:
        shallow = " (the checkout is shallow)" if _is_shallow(root) else ""
        return Changed(None, f"changed-only: {_first_line(exc)}{shallow}")
    cwd = os.path.realpath(os.getcwd())
    out: List[str] = []
    for rel in names:
        full = os.path.join(root, rel)
        if not os.path.lexists(full) or (os.path.isdir(full) and not os.path.islink(full)):
            continue  # deleted in the working tree, or a submodule (gitlink) / directory entry; a link to a folder is kept
            #           so that the Action reports it as a full scan does (pre-release test 3, R12)
        if not _inside(os.path.realpath(full) if not os.path.islink(full) else os.path.abspath(full), scan_abs):
            continue
        try:
            shown = os.path.relpath(full, cwd)
        except ValueError:  # different drive on Windows
            shown = full
        out.append(shown)
    out.sort()
    return Changed(out, f"changed-only: {what}; {len(names)} changed file(s), {len(out)} under the scan path")


def _inside(path: str, scope: str) -> bool:
    path, scope = os.path.normcase(path), os.path.normcase(scope)
    return path == scope or path.startswith(scope.rstrip(os.sep) + os.sep)


def _first_line(exc: Exception) -> str:
    return (str(exc).splitlines() or [""])[0][:200]

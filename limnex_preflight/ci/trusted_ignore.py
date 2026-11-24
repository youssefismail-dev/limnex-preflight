"""Which ignore entries the Action may apply.

On a pull request the files being scanned are the pull request's, and so is its ``.preflightignore``. Read from the checkout,
a pull request could silence its own finding by adding that finding's fingerprint to the ignore file. So on
``pull_request`` / ``pull_request_target`` events the ignore files are read from the pull request's BASE commit, the
trusted side: an entry that only the pull request adds is not applied, and the log says how many were left out.

If the base commit is not in the checkout (``actions/checkout`` clones shallow by default), that one commit is fetched.
If it still cannot be read, NO ignore entry is applied and the log says so loudly (fix: ``fetch-depth: 0``). Fail closed:
until the pre-release test of 2026-11-19 (finding F2) the checkout's own files were used here, so in the default shallow
setup a pull request could silence its own finding. Other events (push, schedule, workflow_dispatch) scan trusted history
and use the checkout's files, exactly as the CLI does.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Mapping, Optional, Tuple

from ..branding import IGNORE_FILE_NAME
from ..cli import discover_ignores
from ..ignore import IgnoreFileError, IgnoreList, decode_ignore_bytes, parse_ignore_text
from .changed import GitError, _git, _resolve

PR_EVENTS = ("pull_request", "pull_request_target")


def _candidates(path: str) -> List[str]:
    """The ignore files a scan of ``path`` applies (cli.discover_ignores): ./<name> and one in the scanned directory."""
    out: List[str] = []
    for c in (os.path.join(os.getcwd(), IGNORE_FILE_NAME),
              os.path.join(path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path)), IGNORE_FILE_NAME)):
        real = os.path.realpath(c)
        if real not in out:
            out.append(real)
    return out


def _base_sha(env: Mapping[str, str]) -> str:
    event_path = env.get("GITHUB_EVENT_PATH")
    if not event_path or not os.path.isfile(event_path):
        return ""
    try:
        with open(event_path, "r", encoding="utf-8") as fh:
            event = json.load(fh)
    except (OSError, ValueError):
        return ""
    return str((((event.get("pull_request") or {}).get("base") or {}).get("sha")) or "")


_SHA = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")


def _base_commit(root: str, sha: str) -> str:
    """The base commit's full SHA: from the checkout, else fetched (only that commit, from origin); '' if neither works."""
    if not sha:
        return ""
    try:
        return _resolve(root, sha)
    except GitError:
        pass
    if not _SHA.match(sha):          # only a full commit id from the event is ever fetched
        return ""
    try:
        _git(root, "fetch", "--quiet", "--no-tags", "--depth=1", "origin", sha)
        return _resolve(root, sha)
    except GitError:
        return ""


def trusted_ignores(env: Mapping[str, str], path: str, workspace: Optional[str] = None) -> Tuple[IgnoreList, str]:
    """(the ignore list to apply, a note for the log; '' when nothing needs saying, 'WARNING: ...' when it must be seen).
    No ignore file is read through a symbolic link: not in the checkout (``workspace``) and not in the base commit
    (pre-release test 3, R2)."""
    if env.get("GITHUB_EVENT_NAME", "") not in PR_EVENTS:
        return discover_ignores([], False, [path], confine_to=workspace or os.getcwd()), ""
    # On a pull request the checkout's own ignore file is never applied (the base commit's is), so a problem in it (an
    # invalid line, a link) must not stop the run BEFORE the scan and hide the findings (pre-release test 5, T4). The note
    # starts with "ERROR:": the Action scans, reports everything, then exits 2 as before (pre-release test 3, R2). The file
    # is still never read through a link (load_ignore_file refuses before opening it).
    own_problem = ""
    try:
        own = discover_ignores([], False, [path], confine_to=workspace or os.getcwd())
    except IgnoreFileError as exc:
        own = IgnoreList()
        own_problem = f"the pull request's own ignore file is not applied on pull requests, and it has a problem: {exc}"
    closed = ("WARNING: the pull request's base commit is not in this checkout and could not be fetched, so NO ignore "
              "entries were applied (a pull request must not silence its own findings); add fetch-depth: 0 to "
              "actions/checkout to apply the base branch's entries")
    base = ""
    try:
        start = os.path.realpath(path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path)))
        root = os.path.realpath(_git(start, "rev-parse", "--show-toplevel").decode().strip())
        base = _base_commit(root, _base_sha(env))
    except GitError:
        base = ""
    if not base:
        if own_problem:
            return IgnoreList(), "ERROR: " + own_problem
        return IgnoreList(), closed if own.entries else ""
    trusted = IgnoreList()
    for real in _candidates(path):
        try:
            rel = os.path.relpath(real, root).replace(os.sep, "/")
        except ValueError:  # another drive (Windows): not part of this repository
            continue
        if rel.startswith("../") or rel == "..":
            continue
        try:
            data = _git(root, "show", f"{base}:{rel}")
        except GitError:
            continue  # the file does not exist in the base commit: one the pull request added is not applied
        source = f"{rel} (base {base[:10]})"
        try:
            mode = _git(root, "ls-tree", base, "--", rel).split(b" ", 1)[0]
        except GitError:
            mode = b""
        if mode == b"120000":
            raise IgnoreFileError(f"{source}: is a symbolic link; an ignore file is never read through a link")
        trusted.merge(parse_ignore_text(decode_ignore_bytes(data, source), source))
    left_out = len(set(own.entries) - set(trusted.entries))
    note = f"pull request: ignore entries read from the base commit {base[:10]}"
    if left_out:
        note += f"; {left_out} entr{'y' if left_out == 1 else 'ies'} added by this pull request NOT applied"
    if own_problem:
        note = f"ERROR: {own_problem}; {note}"
    return trusted, note

"""Entry point of the GitHub Action: ``python -m limnex_preflight.ci``.

Input arrives ONLY through environment variables (see branding.ACTION_ENV_PREFIX) and the event payload file, never through
a shell command line, so no input value is ever interpreted by a shell.

Exit status (the contract documented in the README and tested in tests/test_ci_action.py):
  0  nothing at or above the fail-on threshold (and, with fail-on-incomplete, every file was fully scanned)
  1  findings at or above the threshold
  2  scanner error: bad input, an uncaught exception, a full scan that found files but could read none of them ("nothing
     was checked", like the CLI's exit 3; not with changed-only, where a change touching no scannable file is normal),
     or (only with fail-on-incomplete) a file that was not fully scanned
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import IO, Mapping, Optional, Tuple

from .. import __version__
from ..branding import ACTION_ENV_PREFIX as P
from ..branding import DISPLAY_NAME, TOOL_NAME
from ..cli import EXIT_ERRORS, EXIT_FINDINGS, EXIT_USAGE, decide_exit, incomplete_notes, nothing_read
from ..context import ScanOptions
from ..ignore import IgnoreFileError
from ..models import Note, Severity
from ..report import file_counts, severity_counts, to_human, to_sarif
from ..sanitize import safe_result, visible
from ..scanner import SKIP_DIRS, iter_files, scan_paths
from .changed import changed_files
from .commands import annotation, append_summary, finding_annotations, guarded, write_output
from .summary import build_summary
from .trusted_ignore import trusted_ignores

FAIL_ON_CHOICES = {"high": Severity.HIGH, "medium": Severity.MEDIUM, "low": Severity.LOW, "none": None}


class InputError(Exception):
    pass


def _bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = env.get(P + name, "").lower()
    if raw == "":
        return default
    # only 'true' / 'false' (any case), as the message says: action.yml's own conditions compare with 'true' (GitHub compares
    # strings ignoring case), so 'yes' or '1' would mean true here but false there (pre-release test 2, N16). Surrounding
    # spaces are not stripped for the same reason: " true " failed action.yml's check, so nothing was uploaded and
    # nothing was said (pre-release test 3, R13).
    if raw == "true":
        return True
    if raw == "false":
        return False
    raise InputError(f"input {name.lower().replace('_', '-')!r} must be 'true' or 'false', got {visible(raw)[:30]!r}")


def _check_report_path(path: str, workspace: str) -> None:
    """The SARIF report is never written through a symbolic link (pre-release test 2, N2). The configured path is trusted
    (it comes from the workflow), but what is on disk under the workspace comes from the checked-out commit, which a pull
    request controls: a link committed at the configured path, or at a folder on the way to it, would make the runner
    overwrite a file outside the workspace. Inside the workspace no part of the path may be a link; outside it (for
    example the runner's temp folder) the file itself may not be one."""
    ws = os.path.abspath(workspace)
    target = os.path.abspath(path)
    rel = os.path.relpath(target, ws) if os.path.splitdrive(target)[0] == os.path.splitdrive(ws)[0] else os.pardir
    if rel != os.pardir and not rel.startswith(os.pardir + os.sep):
        cur = ws
        for part in rel.split(os.sep):
            cur = os.path.join(cur, part)
            if os.path.islink(cur):
                raise InputError(f"sarif-file: {visible(os.path.relpath(cur, ws))[:200]!r} is a symbolic link; the report is "
                                 "never written through a link")
    elif os.path.islink(target):
        raise InputError(f"sarif-file: {visible(target)[:200]!r} is a symbolic link; the report is never written through a link")


def _open_report(path: str, workspace: str):
    _check_report_path(path, workspace)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    _check_report_path(path, workspace)  # again: the folders now exist
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    return os.fdopen(os.open(path, flags, 0o644), "w", encoding="utf-8", errors="backslashreplace", newline="\n")


def _key(path: str) -> str:
    """One spelling of a path for comparing: its folder resolved, its own name kept (a link stays the link)."""
    a = os.path.abspath(path)
    return os.path.normcase(os.path.join(os.path.realpath(os.path.dirname(a)), os.path.basename(a)))


def _changed_entries(files, path: str, options: ScanOptions) -> Tuple[list, list]:
    """(entries for scan_paths, notes) for changed-only: the changed files, decided exactly as a full scan of ``path``
    decides them (pre-release test 2, N4; test 3, R12; test 4, R12 again: a changed link out of the scanned folder was
    scanned, a full scan does not follow it, and a link got another fingerprint than in a full scan). The full scan's walk
    is listed (nothing is read) and only what concerns a changed file is kept: the changed path's own entry (a file to
    scan, or a link that is noted, left out or not followed), and any file scanned under a name whose content is a changed
    file. A changed file the walk does not reach gets the reason of the folder that stopped it."""
    changed = {}
    for f in files:
        changed.setdefault(_key(f), f)
    changed_real = {os.path.normcase(os.path.realpath(f)) for f in files}
    entries, stops, seen = [], {}, set()
    for p, level, reason in iter_files([path], options.follow_symlinks, options.exclude, options.confine_to):
        k = _key(p)
        if level is not None:
            stops[k] = (level, reason)
        if k in changed:
            seen.add(k)
            entries.append((p, level, reason))
        elif level is None and os.path.normcase(os.path.realpath(p)) in changed_real:
            entries.append((p, None, None))
    notes = []
    for k, f in sorted(changed.items()):
        if k not in seen:
            level, reason = _not_reached(f, stops)
            notes.append(Note(level, f.replace("\\", "/"), reason))
    return entries, notes


def _not_reached(path: str, stops: dict) -> Tuple[str, str]:
    a = os.path.abspath(path)
    while True:
        parent = os.path.dirname(a)
        if parent == a:
            return "skipped", "changed, but a full scan of the path does not reach it; not scanned"
        a = parent
        if os.path.basename(a) in SKIP_DIRS:
            return "skipped", f"inside {os.path.basename(a)} (version-control data is never scanned); not scanned"
        if _key(a) in stops:
            return stops[_key(a)]


def main(env: Optional[Mapping[str, str]] = None, stdout: Optional[IO[str]] = None) -> int:
    env = os.environ if env is None else env
    stdout = stdout or sys.stdout
    out_file = env.get("GITHUB_OUTPUT")
    code = EXIT_USAGE
    try:
        code = _run(env, stdout)
    except (InputError, IgnoreFileError) as exc:
        print(annotation("error", f"{TOOL_NAME}: {exc}", title="invalid input"), file=stdout)
        _publish_failure(out_file)
        code = EXIT_USAGE
    except Exception as exc:  # never let a crash look like "findings" (Python exits 1 on uncaught exceptions)
        print(annotation("error", f"{DISPLAY_NAME} crashed ({type(exc).__name__}). This is a bug in the scanner, not a finding.",
                         title="scanner error"), file=stdout)
        print(guarded(traceback.format_exc()), file=stdout, end="")
        _publish_failure(out_file)
        code = EXIT_USAGE
    write_output("exit-code", str(code), out_file)
    return code


def _publish_failure(out_file: Optional[str]) -> None:
    for name, value in (("findings-total", "0"), ("findings-critical", "0"), ("findings-high", "0"), ("findings-medium", "0"),
                        ("findings-low", "0"), ("findings-info", "0"), ("highest-severity", "none"), ("files-scanned", "0"),
                        ("files-not-scanned", "0"), ("sarif-path", "")):
        write_output(name, value, out_file)


def _run(env: Mapping[str, str], stdout: IO[str]) -> int:
    path = env.get(P + "PATH", "") or "."
    fail_name = (env.get(P + "FAIL_ON", "") or "medium").strip().lower()
    if fail_name not in FAIL_ON_CHOICES:
        raise InputError(f"input 'fail-on' must be one of high, medium, low, none; got {visible(fail_name)[:30]!r}")
    fail_on = FAIL_ON_CHOICES[fail_name]
    changed_only = _bool(env, "CHANGED_ONLY")
    deep = _bool(env, "DEEP")
    fail_incomplete = _bool(env, "FAIL_ON_INCOMPLETE")
    exclude = tuple(line.strip() for line in env.get(P + "EXCLUDE", "").splitlines() if line.strip())
    sarif_file = env.get(P + "SARIF_FILE", "")
    # pre-release test 2026-11-19, finding F21: inputs that do nothing on their own were silently ignored
    if (env.get(P + "BASE", "").strip() or env.get(P + "HEAD", "").strip()) and not changed_only:
        print(annotation("warning", f"{TOOL_NAME}: the inputs base and head are used only with changed-only: true; "
                                    "they were ignored", title="unused input"), file=stdout)
    if _bool(env, "UPLOAD_SARIF") and not sarif_file.strip():
        print(annotation("warning", f"{TOOL_NAME}: upload-sarif is true but sarif-file is empty, so nothing is uploaded; "
                                    "set sarif-file", title="unused input"), file=stdout)
    if not os.path.lexists(path):
        raise InputError(f"path does not exist: {visible(path)[:200]}")
    workspace = env.get("GITHUB_WORKSPACE") or os.getcwd()
    if sarif_file:
        _check_report_path(os.path.abspath(sarif_file), workspace)  # before scanning, so a bad path fails fast

    # ---- what to scan (no symbolic link is ever followed out of the workspace: pre-release test F14, scanner.symlink_decision)
    options = ScanOptions(deep=deep, exclude=exclude, confine_to=os.path.realpath(workspace))
    entries = None
    scope_note = ""
    left_out: list = []
    if changed_only:
        ch = changed_files(env, path, base=env.get(P + "BASE", ""), head=env.get(P + "HEAD", ""))
        scope_note = ch.note
        if ch.files is None:
            scope_note += " -> falling back to scanning EVERYTHING under the path"
            print(f"{TOOL_NAME}: {visible(scope_note)}", file=stdout)
        else:
            entries, left_out = _changed_entries(ch.files, path, options)
            print(f"{TOOL_NAME}: {visible(scope_note)}", file=stdout)
            if not any(level is None for _, level, _ in entries):
                print(f"{TOOL_NAME}: nothing to scan", file=stdout)

    # ---- scan (ignore files are those of the PATH input, exactly as in a full scan, also when only some files are scanned;
    # on a pull request they are read from the BASE commit, so a pull request cannot silence its own finding)
    ignore, ignore_note = trusted_ignores(env, path, workspace)
    ignore_error = ignore_note.startswith("ERROR")      # the pull request's own ignore file: scan and report, then exit 2
    if ignore_note:
        if ignore_error:
            print(annotation("error", f"{TOOL_NAME}: {ignore_note[len('ERROR: '):]}", title="ignore file"), file=stdout)
        elif ignore_note.startswith("WARNING"):
            print(annotation("warning", f"{TOOL_NAME}: {ignore_note}", title="ignore file"), file=stdout)
        else:
            print(f"{TOOL_NAME}: {visible(ignore_note)}", file=stdout)
        scope_note = (scope_note + "; " if scope_note else "") + ignore_note
    result = scan_paths([path], options, ignore, entries=entries)
    result.notes.extend(left_out)

    # ---- reports: everything below only ever sees the sanitised (masked) copy
    safe = safe_result(result)
    incomplete = incomplete_notes(safe)
    legacy = decide_exit(safe, fail_on, no_error_exit=True, fail_on_incomplete=fail_incomplete)
    code = 1 if legacy == EXIT_FINDINGS else (2 if legacy == EXIT_ERRORS else 0)  # incomplete + fail-on-incomplete -> 2
    if ignore_error:
        code = 2
    # retest of release candidate 11: files found but none read passed as a green check ("nothing was checked" only warned)
    checked_nothing = not changed_only and nothing_read(safe)
    if checked_nothing and code == 0:
        code = 2
    counts = severity_counts(safe.findings)

    sarif_path = ""
    if sarif_file:
        sarif_path = os.path.abspath(sarif_file)
        with _open_report(sarif_path, workspace) as fh:
            fh.write(to_sarif(result, __version__, base_dir=workspace))   # it sanitises; URIs need the real paths

    print(guarded(to_human(safe, __version__)), file=stdout, end="")
    for line in finding_annotations(safe.findings, workspace, report_only=fail_on is None):
        print(line, file=stdout)
    if incomplete:
        level = "error" if (fail_incomplete and code == 2) else "warning"
        print(annotation(level, f"{len(incomplete)} file(s) were not fully scanned (not scanned, unreadable or reduced coverage); see the job summary",
                         title="incomplete scan"), file=stdout)
    if file_counts(safe)[0] == 0 and not changed_only:
        # a full scan that read no file is almost always a wrong path (pre-release test 3, R8); with changed-only, a pull
        # request that changes no scannable file is normal and only the summary says so. Files found but none read fail
        # (exit 2); an empty folder only warns, as in the CLI.
        level = "error" if checked_nothing else "warning"
        print(annotation(level, f"{DISPLAY_NAME}: nothing was checked (no file under the path could be scanned)"
                         + (" (exit 2)" if checked_nothing else ""), title="nothing checked"), file=stdout)
    if code == 1:
        print(annotation("error", f"{DISPLAY_NAME}: findings at or above '{fail_name}' (exit 1)", title="findings"), file=stdout)
    elif code == 2 and ignore_error:
        print(annotation("error", f"{DISPLAY_NAME}: the ignore file has a problem (see above); everything was scanned and "
                                  "reported, and the run exits 2", title="ignore file"), file=stdout)
    elif code == 2 and not checked_nothing:
        print(annotation("error", f"{DISPLAY_NAME}: fail-on-incomplete is on and {len(incomplete)} file(s) were not fully scanned (exit 2)",
                         title="incomplete scan"), file=stdout)

    append_summary(build_summary(safe, version=__version__, fail_on=fail_on, exit_code=code, scope_note=scope_note,
                                 incomplete=len(incomplete)), env.get("GITHUB_STEP_SUMMARY"))

    out_file = env.get("GITHUB_OUTPUT")
    highest = max((f.severity for f in safe.findings), default=None)
    write_output("findings-total", str(len(safe.findings)), out_file)
    for label in ("critical", "high", "medium", "low", "info"):
        write_output(f"findings-{label}", str(counts[label]), out_file)
    write_output("highest-severity", highest.label if highest is not None else "none", out_file)
    scanned, not_scanned = file_counts(safe)   # pre-release test 2, N10: the same count as the log and the summary
    write_output("files-scanned", str(scanned), out_file)
    write_output("files-not-scanned", str(not_scanned), out_file)
    write_output("sarif-path", sarif_path, out_file)
    return code

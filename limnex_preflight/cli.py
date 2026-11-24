"""Command-line interface: ``limnex-preflight PATH...``"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import traceback
from typing import List, Optional, Sequence

from . import __version__
from .branding import DISPLAY_NAME, IGNORE_FILE_NAME, TOOL_NAME
from .context import ScanOptions
from .ignore import IgnoreFileError, IgnoreList, load_ignore_file
from .models import Note, Severity
from .report import list_rules_text, to_human, to_json, to_sarif
from .scanner import ScanResult, scan_paths

EXIT_OK, EXIT_FINDINGS, EXIT_USAGE, EXIT_ERRORS = 0, 1, 2, 3
INCOMPLETE_KINDS = ("error", "skipped", "degraded")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description="Static scan of ML model files for secrets, personal data and unsafe pickles. "
        "Files are never loaded or executed; there is no network access and no telemetry.",
        epilog="Exit codes: 0 clean, 1 findings at/above --fail-on, 2 usage or internal error, 3 scan problems only "
        "(unreadable/unsupported/budget-exceeded files, or files found but not one read; see --no-error-exit and "
        "--fail-on-incomplete). "
        "Put -- before a PATH that starts with a dash.",
    )
    p.add_argument("paths", nargs="*", metavar="PATH", help="files or directories to scan")
    p.add_argument("--format", choices=["human", "json", "sarif"], default="human",
                   help="report format (default: human). Goes to --output, or to stdout if there is none")
    p.add_argument("--output", "-o", metavar="FILE",
                   help="write the --format report to FILE (the human report is then also printed unless --format human); "
                        "'-' means stdout")
    p.add_argument("--json", nargs="?", const="-", metavar="FILE", help="also write a JSON report (to stdout if no FILE)")
    p.add_argument("--sarif", nargs="?", const="-", metavar="FILE", help="also write a SARIF 2.1.0 report (to stdout if no FILE)")
    p.add_argument("--fail-on", default="medium", choices=["info", "low", "medium", "high", "critical", "none", "never"],
                   help="exit 1 when a finding at or above this severity exists (default: medium; 'none' and 'never' disable)")
    p.add_argument("--fail-on-incomplete", action="store_true",
                   help="exit 3 when any file could not be fully scanned (error, 'Not scanned', or reduced coverage)")
    p.add_argument("--ignore-file", metavar="FILE", action="append", default=[],
                   help=f"fingerprint ignore file (default: ./{IGNORE_FILE_NAME} and one in each folder named on the "
                        "command line)")
    p.add_argument("--no-ignore", action="store_true", help="do not read any ignore file")
    p.add_argument("--show-ignored", action="store_true", help="list findings suppressed by the ignore file")
    p.add_argument("--deep", action="store_true",
                   help="also carve provider-specific secrets out of raw tensor/binary payloads (slow on big files)")
    p.add_argument("--exclude", action="append", default=[], metavar="PATTERN",
                   help="leave out what matches inside the scanned folders (repeatable). A name without '/' matches a "
                        "folder or file of that name anywhere (.venv, *.log); a pattern with '/' matches the path from the "
                        "scanned folder (data/*.csv). Nothing is left out by default; every exclusion is reported.")
    p.add_argument("--follow-symlinks", action="store_true",
                   help="also follow symbolic links that point outside the scanned folders (a Hugging Face cache "
                        "snapshot is such a tree). Links named on the command line are always followed.")
    p.add_argument("--max-depth", type=_at_least_one, default=6, metavar="N", help="maximum nested archive depth (default 6)")
    p.add_argument("--no-error-exit", action="store_true", help="do not exit with 3 when files could not be read or parsed")
    p.add_argument("--no-color", action="store_true", help="disable coloured output")
    p.add_argument("-v", "--verbose", action="store_true", help="list every skipped file, extra locations and notes")
    p.add_argument("--list-rules", action="store_true", help="print the rule catalogue and exit")
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {__version__}")
    return p


# ---------------------------------------------------------------------------------------------------------------
# Helpers shared with the GitHub Action (limnex_preflight.ci)
# ---------------------------------------------------------------------------------------------------------------
def discover_ignores(ignore_files: Sequence[str], no_ignore: bool, roots: Sequence[str],
                     confine_to: Optional[str] = None) -> IgnoreList:
    """Ignore files that apply to a scan of ``roots``: explicit ones, else ./<name> plus one in each scanned directory.
    ``confine_to`` (the Action's workspace): an ignore file that is a link, or anything but a regular file, is an error."""
    merged = IgnoreList()
    if no_ignore:
        return merged
    candidates: List[str] = list(ignore_files)
    if not ignore_files:
        candidates.append(os.path.join(os.getcwd(), IGNORE_FILE_NAME))
        for p in roots:
            d = p if os.path.isdir(p) else os.path.dirname(os.path.abspath(p))
            candidates.append(os.path.join(d, IGNORE_FILE_NAME))
    seen = set()
    for c in candidates:
        real = os.path.realpath(c)
        if real in seen:
            continue
        seen.add(real)
        if confine_to and os.path.lexists(c):
            merged.merge(load_ignore_file(c, confine_to))
        elif os.path.isfile(c):
            merged.merge(load_ignore_file(c))
        elif os.path.lexists(c) and not os.path.isdir(c) and not os.path.islink(c):
            raise IgnoreFileError(f"{c}: not a regular file")      # a named pipe would block the read (pre-release test 3)
        elif c in ignore_files:
            raise IgnoreFileError(f"ignore file not found: {c}")
    return merged


def incomplete_notes(result: ScanResult) -> List[Note]:
    """Files that were not (fully) scanned: read/parse errors, 'Not scanned' entries (incl. budget exhausted), reduced coverage."""
    return [n for n in result.notes if n.kind in INCOMPLETE_KINDS]


def decide_exit(result: ScanResult, fail_on: Optional[Severity], *, no_error_exit: bool = False,
                fail_on_incomplete: bool = False) -> int:
    if fail_on is not None and result.failing(fail_on):
        return EXIT_FINDINGS
    if fail_on_incomplete and incomplete_notes(result):
        return EXIT_ERRORS
    if result.errors and not no_error_exit:
        return EXIT_ERRORS
    if not no_error_exit and nothing_read(result):
        return EXIT_ERRORS
    return EXIT_OK


def nothing_read(result: ScanResult) -> bool:
    """Files were found but not one could be read (all of an unsupported type, all links not followed ...): "nothing was
    checked" must not pass as clean (user review before launch: a Hugging Face cache snapshot exited 0). An empty
    folder, or one whose files were all left out with --exclude, is not this case."""
    from .report import file_counts      # report imports scanner, which this module imports too
    scanned, not_scanned = file_counts(result)
    return scanned == 0 and not_scanned > 0


def parse_fail_on(name: str) -> Optional[Severity]:
    return None if name in ("never", "none") else Severity.parse(name)


# ---------------------------------------------------------------------------------------------------------------
def _at_least_one(text: str) -> int:
    """--max-depth (pre-release test F19: 0 and negative numbers were accepted)."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)  # argparse exits with 2 on usage errors
    try:   # a safety net: reports hold no lone surrogate (sanitize.no_surrogates; pre-release test 4, S3)
        sys.stdout.reconfigure(errors="backslashreplace")
    except (AttributeError, ValueError, io.UnsupportedOperation):
        pass
    try:
        return _run(parser, args)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print(f"{TOOL_NAME}: interrupted", file=sys.stderr)
        return EXIT_USAGE
    except Exception:  # an uncaught exception must not look like "findings" (Python would exit 1)
        print(f"{TOOL_NAME}: internal error (this is a bug in {DISPLAY_NAME}, not a finding):", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return EXIT_USAGE


def _run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.list_rules:
        sys.stdout.write(list_rules_text())
        return EXIT_OK
    if not args.paths:
        parser.error("at least one PATH is required")
    # pre-release test F10: a typo is a usage error, never "a file could not be scanned" (which can pass); test 2, N20: so
    # is a broken symbolic link named here (it used to give "scanned 0 file(s): no findings", exit 0)
    missing = [p for p in args.paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            why = "broken symbolic link" if os.path.lexists(p) else "path does not exist"
            print(f"{TOOL_NAME}: {p}: {why}", file=sys.stderr)
        return EXIT_USAGE
    if args.output == "-":   # pre-release test 2, N11: "-o -" is stdout, so the one-report rule applies to it too
        args.output = None
    to_stdout = int(args.format != "human" and not args.output) + int(args.json == "-") + int(args.sarif == "-")
    if to_stdout > 1:  # pre-release test F13: two JSON documents glued together on stdout are neither one
        print(f"{TOOL_NAME}: only one report can go to stdout; write the others to files", file=sys.stderr)
        return EXIT_USAGE
    for dest in (args.output, args.json, args.sarif):   # pre-release test 2, N15: fail before scanning, not as a "bug"
        if dest and dest != "-" and not os.path.isdir(os.path.dirname(os.path.abspath(dest))):
            print(f"{TOOL_NAME}: cannot write the report to {dest}: the folder does not exist", file=sys.stderr)
            return EXIT_USAGE
    for option, dest in (("--output", args.output), ("--json", args.json), ("--sarif", args.sarif)):
        why = _unsafe_report_path(dest, args.paths) if dest and dest != "-" else None
        if why:
            print(f"{TOOL_NAME}: {option} {dest}: {why}; nothing was written. A FILE right after --json or --sarif is the "
                  "report, not a path to scan (write --json=FILE to make that clear)", file=sys.stderr)
            return EXIT_USAGE
    try:
        ignore = discover_ignores(args.ignore_file, args.no_ignore, args.paths)
    except (IgnoreFileError, OSError) as exc:
        print(f"{TOOL_NAME}: {exc}", file=sys.stderr)
        return EXIT_USAGE
    options = ScanOptions(deep=args.deep, max_depth=args.max_depth, exclude=tuple(args.exclude),
                          follow_symlinks=args.follow_symlinks)
    result = scan_paths(args.paths, options, ignore)
    fail_on = parse_fail_on(args.fail_on)
    colour = sys.stdout.isatty() and not args.no_color and not os.environ.get("NO_COLOR")

    def render(fmt: str) -> str:
        if fmt == "json":
            return to_json(result, __version__, fail_on)
        if fmt == "sarif":
            return to_sarif(result, __version__)
        return to_human(result, __version__, colour=colour and not args.output, verbose=args.verbose,
                        show_ignored=args.show_ignored)

    code = decide_exit(result, fail_on, no_error_exit=args.no_error_exit, fail_on_incomplete=args.fail_on_incomplete)
    stdout_taken = args.json == "-" or args.sarif == "-" or (args.format != "human" and not args.output)
    try:
        if args.format != "human" or args.output:
            _emit(render(args.format), args.output or "-")
        if args.json:
            _emit(render("json"), args.json)
        if args.sarif:
            _emit(render("sarif"), args.sarif)
        if not stdout_taken and not (args.format == "human" and args.output):
            sys.stdout.write(render("human"))
            sys.stdout.flush()
        elif result.errors:
            print(f"{TOOL_NAME}: {len(result.errors)} file(s) could not be fully scanned (see report notes)", file=sys.stderr)
    except BrokenPipeError:  # pre-release test F13: `... | head` closed the pipe; that is not an internal error
        _silence_stdout()
    except OSError as exc:   # pre-release test 2, N15: a report file that cannot be written is not a bug in the scanner
        print(f"{TOOL_NAME}: cannot write the report: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return code


_REPORT_MARKS = (f'"name": "{TOOL_NAME}"',)          # in the first lines of a JSON or SARIF report this tool wrote
# the last line of a text report (to_human), maybe followed by the note about stale ignore entries
_SUMMARY_LINE = re.compile(rf"^{re.escape(TOOL_NAME)} \d+\.\d+\.\d+\S*: scanned \d+ file\(s\)")


def _unsafe_report_path(dest: str, paths: Sequence[str]) -> Optional[str]:
    """Why ``dest`` must not be written, or None. --json and --sarif take an optional FILE, so in
    "--sarif model.pkl other/" the path meant to be scanned became the report and was overwritten without a word
    (pre-release test 4, S4). A path to scan is never a report, and an existing regular file is replaced only when it is
    a report this tool wrote (JSON or SARIF: the tool's name near the start; text: its summary line at the end), or empty
    and outside the scanned folders. Inside a scanned folder a text report is not enough either: a model card that quotes
    a scan's last line looks like one (pre-release test 5, T6). Devices and pipes (/dev/null) are left alone."""
    target = os.path.abspath(dest)
    if any(target == os.path.abspath(p) for p in paths):
        return "it is also a path to scan"
    inside = any(os.path.isdir(p) and _inside_folder(target, p) for p in paths)
    try:
        if not os.path.isfile(dest):
            return None
        if os.path.getsize(dest) == 0:
            return f"{dest} is an empty file inside a folder being scanned" if inside else None
        with open(dest, "rb") as fh:
            head = fh.read(4096).decode("utf-8", "replace")
            fh.seek(max(0, os.path.getsize(dest) - 4096))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return None            # the write itself then fails, and says why
    if any(m in head for m in _REPORT_MARKS):
        return None
    last = [line for line in tail.splitlines() if line.strip()][-2:]
    if not inside and any(_SUMMARY_LINE.match(line) for line in last):
        return None
    return f"{dest} exists and is not a report written by {TOOL_NAME}" + (" (inside a folder being scanned, only a JSON "
                                                                          "or SARIF report is replaced)" if inside else "")


def _inside_folder(path: str, folder: str) -> bool:
    try:
        f = os.path.normcase(os.path.abspath(folder))
        return os.path.commonpath([os.path.normcase(path), f]) == f
    except ValueError:      # another drive (Windows)
        return False


def _silence_stdout() -> None:
    """After a closed pipe, send what is left to the null device, so Python's own flush at exit does not fail again."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError, AttributeError, io.UnsupportedOperation):
        pass


def _emit(text: str, dest: str) -> None:
    if dest == "-":
        sys.stdout.write(text)
    else:
        with open(dest, "w", encoding="utf-8", errors="backslashreplace", newline="\n") as fh:   # a safety net: reports
            fh.write(text)                                                                          # hold no lone surrogate


if __name__ == "__main__":
    raise SystemExit(main())

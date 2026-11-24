"""Human, JSON and SARIF 2.1.0 reports. Only masked values ever reach a report."""

from __future__ import annotations

import json
import os
import posixpath
import re
from urllib.parse import quote
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .branding import REPO_URL, SARIF_FINGERPRINT_KEY, TOOL_NAME
from .detectors import ALL_REGEX_RULES, EXTRA_RULES, RULES_BY_ID
from .models import Finding, Note, Severity
from .sanitize import path_bytes, safe_result, visible
from .scanner import ScanResult

# File kinds that were not scanned (scanner.scan_file / scan_stream): not counted as "scanned" in the summary.
NOT_SCANNED_KINDS = ("unsupported", "error")

SARIF_SCHEMA = "https://json.schemastore.org/sarif-2.1.0.json"
_SECURITY_SEVERITY = {
    Severity.CRITICAL: "9.5", Severity.HIGH: "8.0", Severity.MEDIUM: "5.5", Severity.LOW: "3.0", Severity.INFO: "0.5",
}
_SARIF_LEVEL = {
    Severity.CRITICAL: "error", Severity.HIGH: "error", Severity.MEDIUM: "warning", Severity.LOW: "note", Severity.INFO: "note",
}


def severity_counts(findings: List[Finding]) -> Dict[str, int]:
    c = Counter(f.severity.label for f in findings)
    return {s.label: c.get(s.label, 0) for s in sorted(Severity, reverse=True)}


def not_scanned_entries(result: ScanResult) -> List[Note]:
    """The "Not scanned" list, the same in every report: each "skipped" note (a file of an unsupported type, a link that is
    not followed, a named pipe, an archive member left out, a budget reached ...) and one line for each file whose scan
    failed (its error is under "Scan problems")."""
    entries = [n for n in result.notes if n.kind == "skipped"]
    entries += [Note("skipped", f.path, "the scan failed; see Scan problems") for f in result.files if f.kind == "error"]
    return entries


def file_counts(result: ScanResult) -> Tuple[int, int]:
    """(files scanned, entries not scanned): one count for every report and output (pre-release test 2, N10). The second
    number is the length of the "Not scanned" list (pre-release test 4, S12: the list said 3 where the count said 2,
    because a link that is not followed was listed but not counted)."""
    scanned = sum(1 for f in result.files if f.kind not in NOT_SCANNED_KINDS)
    return scanned, len(not_scanned_entries(result))


def _top_file(f: Finding) -> str:
    return f.file.split("!/", 1)[0]


def to_json(result: ScanResult, version: str, fail_on: Optional[Severity]) -> str:
    result = safe_result(result)
    scanned, not_scanned = file_counts(result)
    doc: Dict[str, Any] = {
        "tool": {"name": TOOL_NAME, "version": version},
        "summary": {
            "files_scanned": scanned,
            "files_not_scanned": not_scanned,
            "excluded": len([n for n in result.notes if n.kind == "excluded"]),
            "findings": len(result.findings),
            "ignored": len(result.ignored),
            "by_severity": severity_counts(result.findings),
            "errors": len(result.errors),
            "degraded": len([n for n in result.notes if n.kind == "degraded"]),
            "fail_on": fail_on.label if fail_on is not None else None,
        },
        "findings": [f.to_dict() for f in result.findings],
        "ignored_fingerprints": sorted({f.fingerprint for f in result.ignored}),
        "unused_ignore_entries": result.ignore.unused,
        "files": [{"path": s.path, "kind": s.kind, "size": s.size} for s in result.files],
        "notes": [n.to_dict() for n in result.notes],
    }
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


_HELP = {
    "secret": "Remove the secret from the file, rotate it (assume it is compromised), and republish. Accept a deliberate "
              "finding by adding its fingerprint to the ignore file.",
    "personal": "Remove or anonymise the value before publishing (re-save the file without the path/host/e-mail), or accept "
                "it deliberately by adding its fingerprint to the ignore file.",
    "unsafe-code": "Do not load this file with pickle-based loaders. Re-export the weights in a safe format (safetensors) from a "
                   "trusted source, or review the flagged call by hand.",
    "scan": "The file could not be fully scanned.",
}
_BYTE_OFFSET = re.compile(r"\boffset (\d+)")


def _sarif_uri(path: str, base_dir: Optional[str]) -> str:
    """Repository-relative, forward-slash, percent-encoded URI (GitHub resolves it against the checkout)."""
    p = path
    if os.path.isabs(p) and base_dir:
        try:
            rel = os.path.relpath(p, base_dir)
        except ValueError:  # other drive on Windows
            rel = p
        if not rel.startswith(".."):
            p = rel
    p = p.replace("\\", "/")
    if not os.path.isabs(path):
        p = posixpath.normpath(p)  # "./models/a.pt" -> "models/a.pt"
    # quoted from the name's own bytes, so a name that is not UTF-8 still points at its file (pre-release test 4, S3)
    if os.path.isabs(path) and p == path.replace("\\", "/"):
        return "file:///" + quote(path_bytes(p.lstrip("/")), safe="/:")
    return quote(path_bytes(p), safe="/")


def to_sarif(result: ScanResult, version: str, base_dir: Optional[str] = None) -> str:
    """SARIF 2.1.0. Only masked values: see sanitize.py. ``base_dir`` is what artifact URIs are made relative to. Pass the
    result as the scan returned it: each URI is made from the file's own path, the text from the sanitised copy."""
    raw = result
    result = safe_result(result)
    base_dir = base_dir if base_dir is not None else os.getcwd()
    rule_ids = []
    seen = set()
    for r in ALL_REGEX_RULES + EXTRA_RULES:
        if r.id not in seen:
            seen.add(r.id)
            rule_ids.append(r.id)
    index = {rid: i for i, rid in enumerate(rule_ids)}
    rules = []
    for rid in rule_ids:
        r = RULES_BY_ID[rid]
        rules.append({
            "id": rid,
            "name": rid,
            "shortDescription": {"text": r.title},
            "fullDescription": {"text": r.title},
            "help": {"text": _HELP.get(r.category, _HELP["scan"])},
            "helpUri": REPO_URL + "#what-it-finds",
            "defaultConfiguration": {"level": _SARIF_LEVEL[r.severity]},
            "properties": {
                "tags": ["security", r.category],
                "security-severity": _SECURITY_SEVERITY[r.severity],
            },
        })
    results = []
    for raw_f, f in zip(raw.findings, result.findings):
        top = _top_file(raw_f)
        region: Dict[str, Any] = {"startLine": f.line or 1}
        offset = _BYTE_OFFSET.search(f.location)
        if not f.line and offset and "!/" not in f.file:
            # binary file: the line is a placeholder (GitHub needs one); the byte offset is the real position
            region.update({"byteOffset": int(offset.group(1)), "byteLength": 1})
        uri = _sarif_uri(top, base_dir)
        artifact: Dict[str, Any] = {"uri": uri}
        if not uri.startswith("file:"):
            artifact["uriBaseId"] = "%SRCROOT%"  # relative URIs are resolved against the checkout
        loc: Dict[str, Any] = {
            "physicalLocation": {
                "artifactLocation": artifact,
                "region": region,
            },
            "logicalLocations": [{"name": f.location, "fullyQualifiedName": f.file + " :: " + f.location}],
        }
        results.append({
            "ruleId": f.rule_id,
            "ruleIndex": index.get(f.rule_id, 0),
            "level": _SARIF_LEVEL[f.severity],
            "message": {"text": f"{f.message} [{f.masked}] at {f.location}" + (f" in {f.file}" if "!/" in f.file else "")},
            "locations": [loc],
            "partialFingerprints": {SARIF_FINGERPRINT_KEY: f.fingerprint},
            "properties": {"severity": f.severity.label, "maskedValue": f.masked, "occurrences": f.count,
                           "location": f.location},
        })
    notifications = []
    for n in result.notes:
        if n.kind in ("error", "degraded", "skipped", "excluded"):
            notifications.append({
                "level": {"error": "error", "degraded": "warning"}.get(n.kind, "note"),
                "message": {"text": f"{n.file}: {n.message}"},
            })
    invocation: Dict[str, Any] = {"executionSuccessful": not result.errors}
    if notifications:
        invocation["toolExecutionNotifications"] = notifications[:200]
    doc = {
        "$schema": SARIF_SCHEMA,
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {
                "name": TOOL_NAME,
                "version": version,
                "informationUri": REPO_URL,
                "rules": rules,
            }},
            "results": results,
            "invocations": [invocation],
        }],
    }
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


_COLOURS = {
    Severity.CRITICAL: "\x1b[1;31m", Severity.HIGH: "\x1b[31m", Severity.MEDIUM: "\x1b[33m",
    Severity.LOW: "\x1b[36m", Severity.INFO: "\x1b[2m",
}


def to_human(result: ScanResult, version: str, *, colour: bool = False, verbose: bool = False, show_ignored: bool = False) -> str:
    def paint(sev: Severity, text: str) -> str:
        return f"{_COLOURS[sev]}{text}\x1b[0m" if colour else text

    result = safe_result(result)
    out: List[str] = []
    by_file: Dict[str, List[Finding]] = {}
    for f in result.findings:
        by_file.setdefault(f.file, []).append(f)
    for file, items in by_file.items():
        out.append(visible(file))
        for f in items:
            tag = paint(f.severity, f"[{f.severity.label.upper():8}]")
            more = f"  (x{f.count})" if f.count > 1 else ""
            out.append(f"  {tag} {f.rule_id}{more}")
            out.append(f"      {f.message}")
            out.append(f"      where: {visible(f.location)}")
            out.append(f"      value: {f.masked}    fingerprint: {f.fingerprint}")
            if verbose and f.more_locations:
                for loc in f.more_locations:
                    out.append(f"      also:  {visible(loc)}")
        out.append("")
    if show_ignored and result.ignored:
        out.append("Ignored by .preflightignore:")
        for f in result.ignored:
            out.append(f"  {f.fingerprint}  {f.rule_id}  {visible(f.file)}  [{f.masked}]")
        out.append("")
    errors = result.errors
    degraded = [n for n in result.notes if n.kind == "degraded"]
    skipped = not_scanned_entries(result)
    excluded = [n for n in result.notes if n.kind == "excluded"]
    infos = [n for n in result.notes if n.kind == "info"]
    if errors:
        out.append(f"Scan problems ({len(errors)}) - these files were NOT fully checked:")
        for n in errors[:50]:
            out.append(f"  ! {visible(n.file)}: {visible(n.message)}")
        out.append("")
    if degraded:
        out.append(f"Reduced coverage ({len(degraded)}) - scanned with a weaker method:")
        for n in degraded[:20]:
            out.append(f"  ~ {visible(n.file)}: {visible(n.message)}")
        out.append("")
    if skipped:
        out.append(f"Not scanned ({len(skipped)}):")
        shown = skipped if verbose else skipped[:10]
        for n in shown:
            out.append(f"  - {visible(n.file)}: {visible(n.message)}")
        if len(skipped) > len(shown):
            out.append(f"  ... and {len(skipped) - len(shown)} more (use -v)")
        out.append("")
    if excluded:   # a choice, not a gap in coverage (pre-release test 2, N5); still always reported
        out.append(f"Left out on purpose ({len(excluded)}):")
        shown = excluded if verbose else excluded[:10]
        for n in shown:
            out.append(f"  - {visible(n.file)}: {visible(n.message)}")
        if len(excluded) > len(shown):
            out.append(f"  ... and {len(excluded) - len(shown)} more (use -v)")
        out.append("")
    if verbose and infos:
        out.append("Notes:")
        for n in infos:
            out.append(f"  . {visible(n.file)}: {visible(n.message)}")
        out.append("")
    counts = severity_counts(result.findings)
    parts = [f"{counts[s.label]} {s.label}" for s in sorted(Severity, reverse=True) if counts[s.label]]
    scanned, not_scanned = file_counts(result)   # pre-release test F13
    # pre-release test 2, N7: "scanned 0 file(s): no findings" read like a clean result when nothing was checked
    summary = ", ".join(parts) if parts else ("no findings" if scanned else "nothing was checked")
    out.append(
        f"{TOOL_NAME} {version}: scanned {scanned} file(s)"
        + (f" ({not_scanned} not scanned)" if not_scanned else "")
        + f": {summary}"
        + (f"; {len(result.ignored)} ignored" if result.ignored else "")
    )
    stale = result.ignore.unused
    if stale:
        out.append(f"note: {len(stale)} entr{'y' if len(stale) == 1 else 'ies'} in the ignore file matched nothing: " + ", ".join(stale[:5]) + (" ..." if len(stale) > 5 else ""))
    return "\n".join(out) + "\n"


def list_rules_text() -> str:
    seen = set()
    rows = []
    for r in ALL_REGEX_RULES + EXTRA_RULES:
        if r.id in seen:
            continue
        seen.add(r.id)
        rows.append((int(r.severity), r.id, r.category, r.severity.label, RULES_BY_ID[r.id].title))   # the general title
    rows.sort(key=lambda x: (-x[0], x[1]))
    w = max(len(r[1]) for r in rows)
    return "\n".join(f"{r[3]:8} {r[1]:{w}}  {r[2]:11} {r[4]}" for r in rows) + "\n"

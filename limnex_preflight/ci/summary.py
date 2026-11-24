"""Markdown job summary ($GITHUB_STEP_SUMMARY). Masked values only; everything is escaped for Markdown tables."""

from __future__ import annotations

import re
from typing import List, Optional

from ..branding import DISPLAY_NAME
from ..models import Severity
from ..report import file_counts, not_scanned_entries, severity_counts
from ..sanitize import safe_result, visible
from ..scanner import ScanResult

MAX_ROWS = 50
MAX_NOTES = 25
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]<>()#+\-!|~])")


def md(text: str, limit: int = 160) -> str:
    """Escape for use inside a Markdown table cell."""
    text = visible(text)
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return _MD_SPECIAL.sub(r"\\\1", text).replace("&", "&amp;")


def code(text: str, limit: int = 160) -> str:
    text = visible(text)
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    # characters Markdown/HTML could act on are swapped for look-alikes; a code span already renders them literally
    return "`" + text.replace("`", "'").replace("|", "∣").replace("<", "‹").replace(">", "›") + "`"


def build_summary(result: ScanResult, *, version: str, fail_on: Optional[Severity], exit_code: int, scope_note: str,
                  incomplete: int) -> str:
    result = safe_result(result)
    counts = severity_counts(result.findings)
    lines: List[str] = []
    verdict = {0: "passed", 1: "FAILED (findings at or above the threshold)", 2: "FAILED (scan did not complete)"}.get(exit_code, "finished")
    scanned, not_scanned = file_counts(result)   # pre-release test 2, N10: the same count as the log and the outputs
    if exit_code == 0 and scanned == 0:
        verdict = "nothing was checked"          # pre-release test 3, R8: "passed" with 0 files read like a clean result
    lines.append(f"## {DISPLAY_NAME} - {verdict}")
    lines.append("")
    threshold = "none (report only)" if fail_on is None else f"{fail_on.label} and above"
    lines.append(f"Scanned **{scanned}** file(s)" + (f" ({not_scanned} not scanned)" if not_scanned else "")
                 + f" · fail threshold: **{threshold}** · version {md(version)}")
    if scope_note:
        lines.append("")
        lines.append(f"> {md(scope_note, 400)}")
    lines.append("")
    lines.append("| critical | high | medium | low | info |")
    lines.append("|---:|---:|---:|---:|---:|")
    lines.append("| " + " | ".join(str(counts[s]) for s in ("critical", "high", "medium", "low", "info")) + " |")
    if result.ignored:
        lines.append("")
        lines.append(f"{len(result.ignored)} finding(s) accepted through the ignore file.")
    if result.findings:
        lines.append("")
        lines.append("### Findings (values are masked)")
        lines.append("")
        lines.append("| severity | rule | file | where | value | fingerprint |")
        lines.append("|---|---|---|---|---|---|")
        for f in result.findings[:MAX_ROWS]:
            lines.append(f"| {f.severity.label} | {code(f.rule_id)} | {code(f.file, 80)} | {code(f.location, 80)} | {code(f.masked, 60)} | {code(f.fingerprint)} |")
        if len(result.findings) > MAX_ROWS:
            lines.append("")
            lines.append(f"… and {len(result.findings) - MAX_ROWS} more finding(s); use the SARIF/JSON report for the full list.")
    errors = result.errors
    degraded = [n for n in result.notes if n.kind == "degraded"]
    skipped = not_scanned_entries(result)
    excluded = [n for n in result.notes if n.kind == "excluded"]
    for title, notes in (("Scan problems - these files were NOT fully checked", errors),
                         ("Reduced coverage - scanned with a weaker method", degraded),
                         ("Not scanned", skipped),
                         ("Left out on purpose (exclude)", excluded)):
        if notes:
            lines.append("")
            lines.append(f"### {title} ({len(notes)})")
            lines.append("")
            for n in notes[:MAX_NOTES]:
                lines.append(f"- {code(n.file, 100)}: {md(n.message, 200)}")
            if len(notes) > MAX_NOTES:
                lines.append(f"- … and {len(notes) - MAX_NOTES} more")
    if incomplete and exit_code == 2:
        lines.append("")
        lines.append(f"`fail-on-incomplete` is on: {incomplete} file(s) were not fully scanned.")
    stale = result.ignore.unused
    if stale:
        lines.append("")
        lines.append(f"{len(stale)} ignore-file entr{'y' if len(stale) == 1 else 'ies'} matched nothing and can be removed.")
    lines.append("")
    return "\n".join(lines)

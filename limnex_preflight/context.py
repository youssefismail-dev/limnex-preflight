"""Scan context: options, limits, finding aggregation and fingerprints."""

from __future__ import annotations

import bisect
import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .detectors import ENGINE, MAX_HITS_PER_RULE, MODE_FULL, RULES_BY_ID, Hit
from .models import CATEGORY_SCAN, Finding, Note, Severity

MB = 1024 * 1024


@dataclass
class ScanOptions:
    deep: bool = False  # also carve provider-specific secrets out of raw tensor/binary payloads
    max_depth: int = 6  # nested archive depth
    max_text_bytes: int = 256 * MB  # per text/JSON file
    max_decompressed_bytes: int = 512 * MB  # per compressed stream (zip-bomb guard)
    max_member_bytes: int = 8 * 1024 * MB  # per archive member
    max_members: int = 20000  # per archive
    max_total_bytes: int = 32 * 1024 * MB  # all data read from one top-level file
    max_locations: int = 5  # extra locations remembered per finding
    follow_symlinks: bool = False
    exclude: Tuple[str, ...] = ()  # --exclude patterns (scanner.excluded_by); nothing is left out by default
    confine_to: Optional[str] = None  # the Action: no symbolic link is followed out of this folder (the workspace)


def fingerprint(rule_id: str, file_key: str, value: str) -> str:
    """Stable id of one accepted finding: rule + file name + raw value (truncated SHA-256).

    The location inside the file is deliberately left out so re-saving a model does not
    invalidate an ignore entry; the directory is left out so moving a checkout does not either.
    """
    h = hashlib.sha256()
    h.update(rule_id.encode())
    h.update(b"\0")
    h.update(file_key.encode("utf-8", "replace"))
    h.update(b"\0")
    h.update(value.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


class _Shared:
    def __init__(self, options: ScanOptions):
        self.options = options
        self.findings: Dict[Tuple[str, str, str], Finding] = {}
        self.notes: List[Note] = []
        self.bytes_read = 0
        self.seen_strings: Set[int] = set()

    def over_budget(self) -> bool:
        return self.bytes_read > self.options.max_total_bytes


class ScanContext:
    """One file (or archive member) being scanned. Handlers report through this object."""

    def __init__(self, top_path: str, options: Optional[ScanOptions] = None, chain: Tuple[str, ...] = (), _shared: Optional[_Shared] = None):
        self.top_path = top_path
        self.chain = chain
        self.options = options or ScanOptions()
        self.shared = _shared or _Shared(self.options)
        self.depth = len(chain)

    # -- identity ----------------------------------------------------------------------
    @property
    def display(self) -> str:
        return self.top_path + "".join("!/" + c for c in self.chain)

    @property
    def file_key(self) -> str:
        return "!/".join((os.path.basename(self.top_path),) + self.chain)

    @property
    def findings(self) -> List[Finding]:
        return list(self.shared.findings.values())

    @property
    def notes(self) -> List[Note]:
        return self.shared.notes

    def child(self, member: str) -> "ScanContext":
        member = member.replace("\\", "/")
        return ScanContext(self.top_path, self.options, self.chain + (member,), self.shared)

    # -- reporting ---------------------------------------------------------------------
    def note(self, kind: str, message: str) -> None:
        self.shared.notes.append(Note(kind, self.display, message))

    def charge(self, nbytes: int) -> bool:
        """Account for data read. Returns False when the per-file budget is exhausted."""
        self.shared.bytes_read += nbytes
        if self.shared.over_budget():
            self.note("skipped", "scan budget exhausted (max_total_bytes); remaining content not scanned")
            return False
        return True

    def add(
        self,
        rule_id: str,
        value: str,
        masked: str,
        location: str,
        *,
        severity: Optional[Severity] = None,
        message: Optional[str] = None,
        line: Optional[int] = None,
    ) -> Finding:
        rule = RULES_BY_ID[rule_id]
        fp = fingerprint(rule_id, self.file_key, value)
        key = (self.display, rule_id, fp)
        existing = self.shared.findings.get(key)
        if existing is not None:
            existing.count += 1
            if len(existing.more_locations) < self.options.max_locations and location != existing.location \
                    and location not in existing.more_locations:
                existing.more_locations.append(location)
            return existing
        f = Finding(
            rule_id=rule_id,
            category=rule.category,
            severity=severity if severity is not None else rule.severity,
            message=message or rule.title,
            file=self.display,
            location=location,
            masked=masked,
            fingerprint=fp,
            line=line,
        )
        self.shared.findings[key] = f
        return f

    def scan_text(
        self,
        text: str,
        location: str,
        *,
        key: Optional[str] = None,
        mode: str = MODE_FULL,
        line_numbers: bool = False,
        line_offset: int = 0,
        dedupe: bool = True,
    ) -> int:
        """Run all detectors over ``text``. ``location`` describes where the text lives."""
        if len(text) < 6:
            return 0
        if dedupe:
            sig = hash((text, key, mode, location if line_numbers else None))
            if sig in self.shared.seen_strings:
                return 0
            if len(self.shared.seen_strings) < 3_000_000:
                self.shared.seen_strings.add(sig)
        capped: Set[str] = set()
        hits = ENGINE.scan(text, key=key, mode=mode, capped=capped)
        for rid in sorted(capped):
            where = f" in {location}" if location else ""
            self.note("degraded", f"more than {MAX_HITS_PER_RULE} matches of rule {rid}{where}; only the first "
                                  f"{MAX_HITS_PER_RULE} are reported")
        newlines = None
        if line_numbers and hits and len(text) > 200_000:
            newlines = [i for i, ch in enumerate(text) if ch == "\n"] if len(text) < 50_000_000 else None
        for h in hits:
            self._add_hit(h, text, location, line_numbers, line_offset, newlines)
        return len(hits)

    def _add_hit(self, h: Hit, text: str, location: str, line_numbers: bool, line_offset: int, newlines=None) -> None:
        loc = location
        line = None
        if line_numbers:
            before = bisect.bisect_left(newlines, h.start) if newlines is not None else text.count("\n", 0, h.start)
            line = before + 1 + line_offset
            loc = f"{location}, line {line}" if location else f"line {line}"
        self.add(h.rule.id, h.value, h.masked, loc, severity=h.severity, line=line, message=h.rule.title)

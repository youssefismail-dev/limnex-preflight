"""Core data types: severities and findings."""

from __future__ import annotations

import enum
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional


GROUP_MIN = 4   # this many findings of one rule, at one severity, in one file are shown as one group (0.1.1)


def groupable(f: "Finding") -> bool:
    """Only the noisy kinds are ever grouped: MEDIUM and below, and never unsafe code. Each HIGH/CRITICAL finding and each
    unsafe-code finding stays its own entry, because which secret or which callable it is matters (review of 0.1.1,
    finding 1: four dangerous callables in one pickle showed as one block naming only the first)."""
    return f.category != CATEGORY_UNSAFE and f.severity <= Severity.MEDIUM


def grouped(findings: List["Finding"], min_size: int = GROUP_MIN) -> List[List["Finding"]]:
    """The findings in their order, as a list of items: a single finding, or every groupable finding of one (file, rule,
    severity) that has at least ``min_size`` of them, collected at the place of its first finding. For display only: what
    is found, counted, fingerprinted and what the exit code is never depend on it (user review: 200 lines for one CSV)."""
    sizes = Counter((f.file, f.rule_id, f.severity) for f in findings if groupable(f))
    out: List[List["Finding"]] = []
    at: Dict[tuple, int] = {}
    for f in findings:
        k = (f.file, f.rule_id, f.severity)
        if not groupable(f) or sizes[k] < min_size:
            out.append([f])
        elif k in at:
            out[at[k]].append(f)
        else:
            at[k] = len(out)
            out.append([f])
    return out


_OFFSET = re.compile(r"\boffset (\d{1,20})\b")


def _offset(location: str) -> Optional[int]:
    """The byte offset the scanner wrote into a location ("... offset 123", always decimal). A location also carries text
    from the file (JSON keys, metadata keys, member names), so the last match is taken and read as base 10 (tester of
    0.1.1, B1: a JSON key "timezone offset 0100" crashed the report when this used int(x, 0))."""
    matches = _OFFSET.findall(location or "")
    return int(matches[-1], 10) if matches else None


def first_of(members: List["Finding"]) -> "Finding":
    """The member a group shows as its example: the earliest line; without lines, the smallest byte offset (a pickle's
    "offset N"; review of 0.1.1, finding 1: "first" was the first found, not the first in the file); then the earlier
    found. Findings with neither come last. Display only, so it never fails: on any problem the first found is shown."""
    def key(t):
        i, f = t
        if isinstance(f.line, int):
            return (0, f.line, i)
        off = _offset(f.location)
        return (1, off, i) if off is not None else (2, 0, i)
    try:
        return min(enumerate(members), key=key)[1]
    except Exception:   # noqa: BLE001 - choosing an example must never stop a report (tester of 0.1.1, B1)
        return members[0]


class Severity(enum.IntEnum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    @classmethod
    def parse(cls, name: str) -> "Severity":
        try:
            return cls[name.strip().upper()]
        except KeyError:
            raise ValueError(f"unknown severity: {name!r}") from None

    @property
    def label(self) -> str:
        return self.name.lower()


# Categories keep unsafe-code findings clearly separate from secrets and personal data.
CATEGORY_SECRET = "secret"
CATEGORY_PII = "personal"
CATEGORY_UNSAFE = "unsafe-code"
CATEGORY_SCAN = "scan"


@dataclass
class Finding:
    rule_id: str
    category: str
    severity: Severity
    message: str
    file: str  # display path, e.g. "models/a.zip!/inner/model.pt"
    location: str  # place inside the file: "pickle offset 31", "$.config.token", "cell 3 ..."
    masked: str  # the value, already masked; the raw value is never stored
    fingerprint: str
    line: Optional[int] = None
    count: int = 1
    more_locations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "category": self.category,
            "severity": self.severity.label,
            "message": self.message,
            "file": self.file,
            "location": self.location,
            "line": self.line,
            "value_masked": self.masked,
            "fingerprint": self.fingerprint,
            "occurrences": self.count,
            "other_locations": list(self.more_locations),
        }


@dataclass
class Note:
    """Something that is not a finding but the user must know: skipped or unparsable input."""

    kind: str  # "error" | "skipped" | "degraded" (incomplete) | "excluded" (left out on purpose) | "info"
    file: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {"kind": self.kind, "file": self.file, "message": self.message}

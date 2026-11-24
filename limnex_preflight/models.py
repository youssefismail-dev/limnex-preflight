"""Core data types: severities and findings."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
    location: str  # place inside the file: "pickle offset 0x1f", "$.config.token", "cell 3 ..."
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

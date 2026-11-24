"""Output hygiene shared by every report (human, JSON, SARIF, job summary, annotations).

Reports must contain masked values only. A finding's *value* is always masked, but text taken from the scanned file can
carry a raw secret into a report by another door: a JSON key, a tensor name, an archive member name, an HDF5 attribute
name, a notebook cell label ... ``location`` and the archive-member part of ``file`` are therefore re-scanned with the
detectors and every hit is replaced by its masked form before anything is written.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import List

from .detectors import ENGINE
from .models import Finding, Note
from .scanner import ScanResult, ScannedFile

_CTRL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_ESC = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
_SURROGATE = re.compile("[\ud800-\udfff]")


def no_surrogates(text: str) -> str:
    """A file name that is not valid UTF-8 reaches Python with each bad byte as a lone surrogate (os.fsdecode), which no
    UTF-8 output can hold: writing one crashed --format sarif, --json FILE and the Action's job summary, and JSON on
    stdout was not valid UTF-8 (pre-release test 4, S3). Each such byte is shown as \\xNN (another lone surrogate, which
    only a Windows name can hold, as \\uNNNN)."""
    def esc(m: "re.Match[str]") -> str:
        c = ord(m.group(0))
        return "\\x%02x" % (c - 0xDC00) if 0xDC80 <= c <= 0xDCFF else "\\u%04x" % c
    return _SURROGATE.sub(esc, text)


def path_bytes(path: str) -> bytes:
    """The bytes of a path as the file system has them (for a URI), whatever its name holds."""
    try:
        return path.encode("utf-8", "surrogateescape")
    except UnicodeEncodeError:
        return no_surrogates(path).encode("utf-8")


def scrub_text(text: str) -> str:
    """Mask every detectable secret / personal value inside ``text`` (a location, a member name, ...)."""
    text = no_surrogates(text)
    if len(text) < 6:
        return text
    hits = sorted(ENGINE.scan(text), key=lambda h: h.start, reverse=True)
    for h in hits:
        text = text[: h.start] + h.masked + text[h.end:]
    return text


def scrub_display(display: str) -> str:
    """Keep the scanned file's own path (the user passed it); scrub archive-member names, which come from file content."""
    top, sep, rest = no_surrogates(display).partition("!/")
    return top + sep + scrub_text(rest) if sep else top


def visible(text: str) -> str:
    """Make control characters visible so that a hostile file name cannot forge extra lines in a log (and bytes that are
    not UTF-8 printable, see no_surrogates)."""
    return _CTRL.sub(lambda m: _ESC.get(m.group(0), "\\x%02x" % ord(m.group(0))), no_surrogates(text))


def safe_finding(f: Finding) -> Finding:
    # message and masked too: a structural finding (pickle-incomplete-analysis, scan diagnostics ...) carries text taken
    # from the file in both, e.g. a parser error that quotes the content it choked on. A detector finding's value is
    # already masked, and scrubbing a masked value finds nothing more, so this only ever removes raw text.
    return replace(f, file=scrub_display(f.file), location=scrub_text(f.location), message=scrub_text(f.message),
                   masked=scrub_text(f.masked), more_locations=[scrub_text(x) for x in f.more_locations])


def safe_notes(notes: List[Note]) -> List[Note]:
    return [replace(n, file=scrub_display(n.file), message=scrub_text(n.message)) for n in notes]


def safe_result(result: ScanResult) -> ScanResult:
    """A copy of ``result`` that is safe to print anywhere."""
    return replace(
        result,
        findings=[safe_finding(f) for f in result.findings],
        ignored=[safe_finding(f) for f in result.ignored],
        notes=safe_notes(result.notes),
        files=[ScannedFile(scrub_display(s.path), s.kind, s.size) for s in result.files],
    )

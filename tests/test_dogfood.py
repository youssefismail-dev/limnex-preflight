"""The repository scanned with its own scanner. Only the planted fake values of the tests and fixtures may be accepted."""
import os
import re
import subprocess
import sys

import pytest

from limnex_preflight.context import ScanOptions
from limnex_preflight.detectors import ENGINE
from limnex_preflight.models import Severity
from limnex_preflight.scanner import scan_paths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_files():
    try:
        out = subprocess.run(["git", "-C", ROOT, "ls-files", "-z", "--cached", "--others", "--exclude-standard"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("needs a git checkout")
    return [f for f in out.decode("utf-8").split("\0") if f and os.path.isfile(os.path.join(ROOT, f))]


def test_the_dogfood_ignore_file_is_current():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tests", "make_dogfood_ignore.py"), "--check"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, "regenerate with: python tests/make_dogfood_ignore.py\n" + r.stderr


def test_the_dogfood_ignore_file_only_covers_fixture_files_and_never_rules():
    entries = [l for l in open(os.path.join(ROOT, ".preflightignore"), encoding="utf-8").read().splitlines() if l and not l.startswith("#")]
    assert entries
    for line in entries:
        m = re.match(r"^([0-9a-f]{16})  # (\S+): ([a-z0-9-]+)$", line)
        assert m, f"unexpected ignore entry (must be fingerprint + file + rule comment): {line!r}"
        assert m.group(2).startswith("tests/") or m.group(2) == "evaluation/fixtures.py", f"{m.group(2)} is not a fixture file"


def test_the_repository_is_clean_under_its_own_ignore_file(monkeypatch):
    monkeypatch.chdir(ROOT)
    from limnex_preflight.cli import discover_ignores

    files = git_files()
    res = scan_paths(files, ScanOptions(), discover_ignores([], False, ["."]))
    assert [(f.file, f.rule_id, f.severity.label) for f in res.findings if f.severity >= Severity.LOW] == []
    assert res.ignore.unused == [], "stale entries in .preflightignore: regenerate it"
    assert not res.errors


def test_documentation_does_not_trip_the_scanner_on_itself():
    for rel in ("README.md", "RELEASE_CHECKLIST.md", "action.yml", ".pre-commit-hooks.yaml", "limnex_preflight/detectors.py"):
        if not os.path.exists(os.path.join(ROOT, rel)):     # RELEASE_CHECKLIST.md stays in the private repository
            continue
        text = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        assert [h.rule.id for h in ENGINE.scan(text)] == [], rel


@pytest.mark.xfail(strict=True, reason="DETECTION BUG, reported not fixed in the CI phase: a Markdown code span "
                   "`api_key=self.api_key` is reported as a generic-credential (the backtick keeps the code-reference filter from matching)")
def test_a_markdown_code_span_with_an_attribute_reference_is_not_a_credential():
    assert [h.rule.id for h in ENGINE.scan("(code lines like `api_key=self.api_key`)")] == []

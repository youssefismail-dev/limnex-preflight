r"""File names that are not valid UTF-8 (pre-release test 4, S3).

On Linux such a name reaches Python with each bad byte as a lone surrogate. No UTF-8 report can hold one, so --format
sarif, --json FILE and the Action's job summary crashed (exit 2), and JSON on stdout was not valid UTF-8: any repository
with such a file could not use the Action. Every report now shows the byte as \xNN, and a SARIF URI is quoted from the
name's real bytes."""
import errno
import json
import os
import sys

import pytest

from limnex_preflight import cli
from limnex_preflight.ci.summary import build_summary
from limnex_preflight.models import Note
from limnex_preflight.report import to_human, to_json, to_sarif
from limnex_preflight.scanner import scan_paths

BAD = "bad\udcffname.txt"            # os.fsdecode(b"bad\xffname.txt") on Linux
SHOWN = r"bad\xffname.txt"


def _strict(text):
    text.encode("utf-8")              # raises on a lone surrogate
    return text.replace("\\\\", "\\")  # JSON and SARIF escape the backslash


def test_every_report_holds_a_name_that_is_not_utf8(tmp_path):
    """Simulated (Windows cannot hold such a name): a finding and a note on that name."""
    good = tmp_path / "x.txt"
    good.write_text("path=/home/alice/x\n", encoding="utf-8")
    res = scan_paths([str(good)])
    assert res.findings
    for f in res.findings:
        f.file = BAD                 # relative and short: the job summary shortens long paths
    res.notes.append(Note("skipped", BAD, "unsupported file type (not scanned)"))
    for text in (to_json(res, "0", None), to_sarif(res, "0"), to_human(res, "0"),
                 build_summary(res, version="0", fail_on=None, exit_code=0, scope_note="", incomplete=0)):
        assert SHOWN in _strict(text)
    uris = [r["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
            for r in json.loads(to_sarif(res, "0"))["runs"][0]["results"]]
    assert uris and all(u.endswith("bad%FFname.txt") for u in uris), uris


@pytest.mark.skipif(sys.platform == "win32" or sys.getfilesystemencoding().lower() != "utf-8",
                    reason="a name that is not UTF-8 needs a byte-oriented file system (Linux)")
def test_a_real_name_that_is_not_utf8(tmp_path, monkeypatch, capsys):
    d = tmp_path / "nu"
    d.mkdir()
    try:
        fh = open(os.path.join(os.fsencode(str(d)), b"bad\xffname.txt"), "w", encoding="utf-8")
    except OSError as exc:            # macOS (APFS) refuses such a name: errno EILSEQ, "Illegal byte sequence"
        if exc.errno != errno.EILSEQ:
            raise
        pytest.skip("this file system refuses names that are not UTF-8 (macOS), so no such file can exist here")
    with fh:
        fh.write("path=/home/alice/x\n")
    for args in (["--format", "sarif"], ["--format", "json"], ["--sarif", str(tmp_path / "r.sarif")],
                 ["--json", str(tmp_path / "r.json")]):
        capsys.readouterr()
        assert cli.main(args + ["--fail-on", "none", str(d)]) == 0, args
        out = capsys.readouterr().out
        assert SHOWN in _strict(out), args
    for name in ("r.sarif", "r.json"):
        assert SHOWN in _strict((tmp_path / name).read_text(encoding="utf-8"))
    from ci_helpers import run_action
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, str(tmp_path), path="nu", fail_on="low",
                                             sarif_file="a.sarif")
    assert code == 1 and SHOWN in summary and outputs["findings-total"] == "1", out

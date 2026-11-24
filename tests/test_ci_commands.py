import os

import pytest

from limnex_preflight.ci import commands as c
from limnex_preflight.models import Finding, Severity

from ci_helpers import parse_github_output


def test_escaping_follows_the_workflow_command_rules():
    assert c.escape_data("a%b\r\nc") == "a%25b%0D%0Ac"
    assert c.escape_property("a:b,c%\n") == "a%3Ab%2Cc%25%0A"
    assert c.annotation("error", "m\nx", file="d/a,b.json", line=3, title="t:1") == "::error file=d/a%2Cb.json,line=3,title=t%3A1::m%0Ax"


def mk(i, sev, rule="email-address", file="a.json"):
    return Finding(rule, "personal", sev, "msg", file, f"$.k{i}", "m***", f"{i:016x}", line=i + 1)


def test_annotations_are_capped_per_level_with_a_plus_n_more_line():
    findings = [mk(i, Severity.MEDIUM) for i in range(25)] + [mk(100 + i, Severity.HIGH) for i in range(12)] + [mk(200, Severity.LOW)]
    lines = c.finding_annotations(findings, ".")
    assert sum(l.startswith("::error ") for l in lines) == 10
    assert sum(l.startswith("::warning ") for l in lines) == 10
    assert sum(l.startswith("::notice file=") for l in lines) == 1
    more = [l for l in lines if "more finding" in l]
    assert len(more) == 1 and "+17 more" in more[0]
    assert lines[0].startswith("::error ")  # most severe first


def test_annotation_carries_file_and_line_but_only_masked_text():
    f = mk(4, Severity.HIGH, file="models/sub dir/cé.json")
    (line,) = c.finding_annotations([f], os.getcwd())
    assert line.startswith("::error file=models/sub dir/cé.json,line=5,title=email-address (high)::")
    assert "m***" in line


def test_hostile_file_name_cannot_forge_a_command():
    f = mk(1, Severity.HIGH, file="x\n::error::pwned\n::set-output name=a::b.json")
    (line,) = c.finding_annotations([f], ".")
    assert "\n" not in line and line.count("::error") == 1
    assert "%0A" in line


def test_output_values_cannot_inject_further_outputs(tmp_path):
    out = tmp_path / "out"
    c.write_output("sarif-path", "/tmp/x\nfindings-total=999\nexit-code=0", str(out))
    c.write_output("exit-code", "1", str(out))
    parsed = parse_github_output(str(out))
    assert parsed == {"sarif-path": "/tmp/x\nfindings-total=999\nexit-code=0", "exit-code": "1"}


def test_bad_output_names_are_refused(tmp_path):
    with pytest.raises(ValueError):
        c.write_output("Bad Name", "x", str(tmp_path / "o"))


def test_guard_token_is_random_and_wraps_text():
    a, b = c.guarded("x\n"), c.guarded("x\n")
    assert a != b and a.startswith("::stop-commands::") and a.rstrip().endswith("::")
    tok = a.split("\n")[0].split("::stop-commands::")[1]
    assert a.endswith(f"::{tok}::\n")

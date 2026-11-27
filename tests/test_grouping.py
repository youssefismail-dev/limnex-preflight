"""0.1.1: repeated findings (4 or more of one rule, at one severity, in one file) are shown as one group in the human
report, the Action's annotations and the job summary. Detection, counts, fingerprints, JSON, SARIF and exit codes do not
change. All e-mail addresses here are invented and assembled at run time."""
import json
import re

import pytest

from limnex_preflight import __version__, cli
from limnex_preflight.ci import commands as c
from limnex_preflight.ci.summary import build_summary
from limnex_preflight.ignore import IgnoreList
from limnex_preflight.models import Finding, Severity, grouped
from limnex_preflight.report import to_human, to_json, to_sarif
from limnex_preflight.scanner import scan_paths


def mk(i, sev=Severity.MEDIUM, rule="email-address", file="a.json"):
    return Finding(rule, "personal", sev, "msg", file, f"$.k{i}", "m***", f"{i:016x}", line=i + 1)


def emails(tmp_path, n, name="users.csv"):
    p = tmp_path / name
    p.write_text("name,mail\n" + "".join(f"p{i},person{i}" + "@" + "mailbox-test" + ".org\n" for i in range(n)),
                 encoding="utf-8")
    return p


# ---- the grouping itself -------------------------------------------------------------------------------------------------
def test_three_stay_single_and_four_become_one_group():
    assert [len(g) for g in grouped([mk(i) for i in range(3)])] == [1, 1, 1]
    assert [len(g) for g in grouped([mk(i) for i in range(4)])] == [4]


def test_groups_are_per_file_rule_and_severity_and_keep_the_order():
    items = ([mk(0, rule="ip-address")] + [mk(i) for i in range(1, 6)] + [mk(10 + i, file="b.json") for i in range(4)]
             + [mk(20 + i, Severity.HIGH) for i in range(2)] + [mk(30)])
    out = grouped(items)
    assert [(g[0].rule_id, g[0].file, g[0].severity, len(g)) for g in out] == [
        ("ip-address", "a.json", Severity.MEDIUM, 1),
        ("email-address", "a.json", Severity.MEDIUM, 6),      # the late one (k30) joins its group at the first place
        ("email-address", "b.json", Severity.MEDIUM, 4),
        ("email-address", "a.json", Severity.HIGH, 1),
        ("email-address", "a.json", Severity.HIGH, 1),
    ]
    assert sum(len(g) for g in out) == len(items)


def test_unsafe_code_and_high_findings_are_never_grouped():
    """Review of 0.1.1, finding 1: which callable or which secret it is matters, so these stay one entry each."""
    unsafe = [Finding("unsafe-pickle-exec", "unsafe-code", Severity.CRITICAL, "msg", "m.pkl", f"pickle offset {o} (GLOBAL)",
                      v, f"{o:016x}") for o, v in ((2, "os.system"), (14, "builtins.eval"), (30, "subprocess.Popen"),
                                                    (50, "runpy.run_path"))]
    assert [len(g) for g in grouped(unsafe)] == [1, 1, 1, 1]
    review = [Finding("pickle-getattr-review", "unsafe-code", Severity.MEDIUM, "msg", "m.pkl", f"pickle offset {i}",
                      "x", f"{i:016x}") for i in range(6)]
    assert [len(g) for g in grouped(review)] == [1] * 6              # unsafe code at MEDIUM too
    assert [len(g) for g in grouped([mk(i, Severity.HIGH, rule="aws-access-key-id") for i in range(5)])] == [1] * 5
    assert [len(g) for g in grouped([mk(i, Severity.LOW, rule="high-entropy-string") for i in range(5)])] == [5]


def test_first_without_lines_is_the_smallest_offset_not_the_first_found():
    from limnex_preflight.models import first_of
    members = [Finding("email-address", "personal", Severity.MEDIUM, "msg", "m.pkl", loc, "m***", f"{i:016x}")
               for i, loc in enumerate(["pickle offset 248014 (BINUNICODE)", "pickle offset 31 (BINUNICODE)",
                                        "pickle offset 548410 (BINUNICODE)", "pickle offset 40 (BINUNICODE)"])]
    assert first_of(members).location == "pickle offset 31 (BINUNICODE)"         # 31 < 40 < 248014
    assert first_of([mk(5), mk(2)] + members).line == 3                            # a line number still comes first


# tester of 0.1.1, B1: the offset was read out of any location text with int(x, 0), so a key "offset 0100" crashed the
# human report and the Action (exit 2). The scanner writes offsets in decimal; text from the file comes before them.
def _keys_json(tmp_path, keys):
    p = tmp_path / "tz.json"
    p.write_text('{"x": {' + ", ".join(f'"{k} {i}": "p{i}.person' + '@' + 'mailbox-test.org"'
                                         for i, k in enumerate(keys)) + "}}", encoding="utf-8")
    return p


@pytest.mark.parametrize("keys", [("timezone offset 0100",) * 4,
                                  ("offset 0100", "a", "b", "c"),
                                  ("contact offset 001", "contact offset 002", "contact offset 003", "contact offset 004"),
                                  ("offset 99999999999999999999999",) * 4])
def test_a_key_with_offset_and_a_leading_zero_does_not_crash_any_output(tmp_path, keys):
    p = _keys_json(tmp_path, keys)
    res = scan_paths([str(p)])
    assert sum(f.rule_id == "email-address" for f in res.findings) == 4
    assert "4 findings" in to_human(res, __version__)
    assert "4 findings" in to_human(res, __version__, verbose=True)
    lines = c.finding_annotations(res.findings, ".")
    assert len(lines) == 1 and "4 x" in lines[0]
    md = build_summary(res, version=__version__, fail_on=Severity.MEDIUM, exit_code=1, scope_note="", incomplete=0)
    assert "(4 findings)" in md
    assert cli.main([str(p)]) == 1 and cli.main([str(p), "-v"]) == 1


def test_the_offset_is_the_one_the_scanner_wrote_last_and_in_decimal():
    from limnex_preflight.models import _offset, first_of
    assert _offset("tar member offset 9/data.pkl offset 0100 (BINUNICODE)") == 100   # base 10, the last one
    assert _offset("$.timezone offset 0100") == 100
    assert _offset("$.nothing here") is None
    members = [Finding("email-address", "personal", Severity.MEDIUM, "msg", "m.pkl", loc, "m***", f"{i:016x}")
               for i, loc in enumerate(["x offset 5/data.pkl offset 300 (BINUNICODE)", "pickle offset 200 (BINUNICODE)"])]
    assert first_of(members).location.startswith("pickle offset 200")


def test_choosing_an_example_never_fails(monkeypatch):
    from limnex_preflight import models
    members = [Finding("email-address", "personal", Severity.MEDIUM, "msg", "m.pkl", f"pickle offset {o}", "m***",
                       f"{o:016x}") for o in (9, 3)]

    def broken(location):
        raise ValueError("any parsing problem")
    monkeypatch.setattr(models, "_offset", broken)
    assert models.first_of(members) is members[0]        # the first found, and the report still renders


def test_a_pickle_naming_four_dangerous_callables_shows_all_four(tmp_path):
    """The reviewer's case: GLOBAL + POP for each, nothing called; an inert file that is never loaded."""
    p = tmp_path / "four_callables.pkl"
    p.write_bytes(b"\x80\x02" + b"cos\nsystem\n0" + b"cbuiltins\neval\n0" + b"csubprocess\nPopen\n0"
                  + b"crunpy\nrun_path\n0" + b"N.")
    res = scan_paths([str(p)])
    text = to_human(res, __version__)
    for name in ("os.system", "builtins.eval", "subprocess.Popen", "runpy.run_path"):
        assert name in text, name
    assert "more in this file" not in text
    lines = c.finding_annotations(res.findings, ".")
    assert sum(l.startswith("::error ") for l in lines) == sum(f.severity >= Severity.HIGH for f in res.findings) >= 4


# ---- a real scan: the report groups, nothing else changes ----------------------------------------------------------------
def test_33_addresses_in_one_file_are_one_block_but_still_33_findings(tmp_path):
    p = emails(tmp_path, 33)
    res = scan_paths([str(p)])
    assert sum(f.rule_id == "email-address" for f in res.findings) == 33
    text = to_human(res, __version__)
    assert text.count("email-address") == 1 and "33 findings" in text
    assert "... and 32 more in this file" in text
    assert "first: line 2 " in text                              # the earliest line, not the first one found
    assert len(text.splitlines()) < 15
    assert "33 medium" in text                                   # the totals line counts every finding
    verbose = to_human(res, __version__, verbose=True)
    assert all(f.fingerprint in verbose for f in res.findings)  # -v still lists every fingerprint
    assert len(json.loads(to_json(res, __version__, None))["findings"]) == 33
    assert sum(len(r["results"]) for r in json.loads(to_sarif(res, __version__))["runs"]) == 33
    assert cli.main([str(p)]) == 1                               # MEDIUM still fails the default threshold


def test_ignored_findings_leave_the_group(tmp_path):
    p = emails(tmp_path, 10)
    first = scan_paths([str(p)])
    ignore = IgnoreList(entries={f.fingerprint: "" for f in first.findings[:7]})
    res = scan_paths([str(p)], ignore=ignore)
    assert len(res.findings) == 3 and len(res.ignored) == 7
    text = to_human(res, __version__)
    assert not re.search(r"email-address +\d+ findings", text) and text.count("email-address") == 3   # 3 left: one by one


def test_two_files_make_two_groups(tmp_path):
    a, b = emails(tmp_path, 5, "a.csv"), emails(tmp_path, 6, "b.csv")
    text = to_human(scan_paths([str(a), str(b)]), __version__)
    assert "5 findings" in text and "6 findings" in text


# ---- the Action ----------------------------------------------------------------------------------------------------------
def test_one_annotation_per_group():
    lines = c.finding_annotations([mk(i) for i in range(25)] + [mk(99, Severity.HIGH, rule="aws-access-key-id")], ".")
    assert len(lines) == 2
    assert lines[0].startswith("::error ")                       # most severe first
    assert "25 x" in lines[1] and lines[1].startswith("::warning file=a.json,line=1,")


def test_the_cap_counts_groups_and_the_plus_n_line_counts_findings():
    findings = [mk(100 * f + i, file=f"f{f}.json") for f in range(12) for i in range(5)]   # 12 groups of 5
    lines = c.finding_annotations(findings, ".")
    assert sum(l.startswith("::warning ") for l in lines) == 10
    more = [l for l in lines if "more finding" in l]
    assert len(more) == 1 and "+10 more" in more[0]              # 2 groups of 5 left out


def test_job_summary_has_one_row_per_group(tmp_path):
    p = emails(tmp_path, 33)
    res = scan_paths([str(p)])
    md = build_summary(res, version=__version__, fail_on=Severity.MEDIUM, exit_code=1, scope_note="", incomplete=0)
    rows = [l for l in md.splitlines() if l.startswith("| medium |")]
    assert len(rows) == 1 and "(33 findings)" in rows[0] and "33 fingerprints" in rows[0]
    assert "| 0 | 0 | 33 | 0 | 0 |" in md                       # the counts table is unchanged
    # review of 0.1.1, finding 5: every grouped fingerprint is in the summary, folded away
    assert "<details><summary>Fingerprints of the grouped findings" in md and "</details>" in md
    folded = md[md.index("<details>"):md.index("</details>")]
    assert all(f.fingerprint in folded for f in res.findings)


def test_the_folded_fingerprint_list_is_capped_per_group():
    from limnex_preflight.ci import summary as s
    from limnex_preflight.scanner import ScanResult
    res = ScanResult()
    res.findings = [mk(i) for i in range(s.MAX_GROUP_FINGERPRINTS + 5)]
    md = build_summary(res, version="0", fail_on=Severity.MEDIUM, exit_code=1, scope_note="", incomplete=0)
    folded = md[md.index("<details>"):md.index("</details>")]
    assert sum(f.fingerprint in folded for f in res.findings) == s.MAX_GROUP_FINGERPRINTS
    assert "… and 5 more in the SARIF/JSON report" in folded


def test_groups_past_the_table_rows_still_have_their_fingerprints():
    """Tester of 0.1.1, L3: 55 single findings and then a group of 5 - the group had no row and no fingerprints."""
    from limnex_preflight.ci import summary as s
    from limnex_preflight.scanner import ScanResult
    res = ScanResult()
    singles = [mk(i, file=f"f{i}.json") for i in range(s.MAX_ROWS + 5)]
    group = [mk(1000 + i, file="late.json") for i in range(5)]
    res.findings = singles + group
    md = build_summary(res, version="0", fail_on=Severity.MEDIUM, exit_code=1, scope_note="", incomplete=0)
    folded = md[md.index("<details>"):md.index("</details>")]
    assert all(f.fingerprint in folded for f in group)


def test_the_folded_list_has_a_total_cap(monkeypatch):
    from limnex_preflight.ci import summary as s
    from limnex_preflight.scanner import ScanResult
    monkeypatch.setattr(s, "MAX_LISTED_FINGERPRINTS", 10)
    res = ScanResult()
    res.findings = [mk(100 * g + i, file=f"g{g}.json") for g in range(4) for i in range(4)]   # 4 groups of 4
    md = build_summary(res, version="0", fail_on=Severity.MEDIUM, exit_code=1, scope_note="", incomplete=0)
    folded = md[md.index("<details>"):md.index("</details>")]
    assert sum(f.fingerprint in folded for f in res.findings) == 10
    assert "… and 2 more in the SARIF/JSON report" in folded and "this list stops at 10 fingerprints" in folded


def test_verbose_lists_the_extra_places_of_grouped_findings(tmp_path):
    """Tester of 0.1.1, L2: with -v, a grouped finding's other places ("also:") were dropped; 0.1.0 printed them."""
    p = tmp_path / "users.csv"
    rows = [f"p{i},person{i}" + "@" + "mailbox-test.org" for i in range(4)]
    p.write_text("name,mail\n" + "\n".join(rows + rows) + "\n", encoding="utf-8")    # each address twice
    res = scan_paths([str(p)])
    assert len(res.findings) == 4 and all(f.count == 2 and f.more_locations for f in res.findings)
    verbose = to_human(res, __version__, verbose=True)
    for f in res.findings:
        assert all(f"also:  {loc}" in verbose for loc in f.more_locations)

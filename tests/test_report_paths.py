"""A report never overwrites a file it should not (pre-release test 4, S4).

--json and --sarif take an optional FILE, so in "limnex-preflight --sarif victim.pkl m/README.md" the pickle meant to be
scanned became the SARIF report and was replaced without a word. A path to scan is never a report now, and an existing
file is replaced only when it is empty or a report this tool wrote."""
import pickle

import pytest

from limnex_preflight import cli


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "README.md").write_text("hello\n", encoding="utf-8")
    (tmp_path / "victim.pkl").write_bytes(pickle.dumps({"a": 1}))
    return tmp_path


@pytest.mark.parametrize("option", ["--sarif", "--json", "-o"])
def test_a_file_that_is_not_a_report_is_never_overwritten(tree, option, capsys):
    before = (tree / "victim.pkl").read_bytes()
    assert cli.main([option, "victim.pkl", "m/README.md"]) == 2
    assert (tree / "victim.pkl").read_bytes() == before
    assert "not a report" in capsys.readouterr().err


def test_a_path_to_scan_is_never_the_report(tree, capsys):
    (tree / "empty.json").write_text("", encoding="utf-8")
    assert cli.main(["empty.json", "--json", "empty.json"]) == 2
    assert "also a path to scan" in capsys.readouterr().err


@pytest.mark.parametrize("option", ["--sarif", "--json", "-o"])
def test_a_report_of_this_tool_is_replaced_by_the_next_run(tree, option):
    for _ in range(2):                                 # the second run finds the first run's report and replaces it
        assert cli.main(["m", option, "out.report", "--fail-on", "none"]) == 0
    assert (tree / "out.report").stat().st_size > 0


def test_an_empty_file_may_become_the_report(tree):
    (tree / "r.sarif").write_text("", encoding="utf-8")
    assert cli.main(["m", "--sarif=r.sarif"]) == 0
    assert '"version": "2.1.0"' in (tree / "r.sarif").read_text(encoding="utf-8")

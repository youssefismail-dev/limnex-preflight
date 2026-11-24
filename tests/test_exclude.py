"""--exclude (pre-release test 2026-11-19, finding F15: there was no way to leave out a folder such as a virtualenv, and
scanning `.` in a project with one inside flooded the report).

Explicit only: nothing is left out by default, because a default list (node_modules, .venv ...) would let anyone hide a
model file from the Action by putting it in a folder with such a name. Every exclusion is reported."""
import json
import os

from limnex_preflight import cli
from limnex_preflight.context import ScanOptions
from limnex_preflight.scanner import scan_paths

HOME = "/home/alice/work"


def _project(tmp_path):
    root = tmp_path / "proj"
    for rel in ("model/config.json", ".venv/lib/site/settings.json", "node_modules/pkg/package.json",
                "logs/run1.json", "model/logs/keep.json"):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"path": f"{HOME}/{rel}"}), encoding="utf-8")
    return root


def _files(res):
    return sorted({f.file.split("proj/", 1)[-1] for f in res.findings})


def test_nothing_is_left_out_by_default(tmp_path):
    res = scan_paths([str(_project(tmp_path))])
    assert ".venv/lib/site/settings.json" in _files(res) and "node_modules/pkg/package.json" in _files(res)


def test_a_name_without_a_slash_leaves_out_that_folder_anywhere(tmp_path):
    res = scan_paths([str(_project(tmp_path))], ScanOptions(exclude=(".venv", "node_modules")))
    assert _files(res) == ["logs/run1.json", "model/config.json", "model/logs/keep.json"]
    notes = [(n.kind, n.file.split("proj/", 1)[-1], n.message) for n in res.notes]
    assert ("excluded", ".venv", "excluded by --exclude .venv") in notes, notes
    assert ("excluded", "node_modules", "excluded by --exclude node_modules") in notes, notes


def test_a_pattern_with_a_slash_matches_the_path_from_the_scanned_folder(tmp_path):
    res = scan_paths([str(_project(tmp_path))], ScanOptions(exclude=("logs/*.json",)))
    assert "logs/run1.json" not in _files(res)
    assert "model/logs/keep.json" in _files(res)                  # only logs/ at the top of the scanned folder


def test_a_path_named_on_the_command_line_is_always_scanned(tmp_path):
    root = _project(tmp_path)
    res = scan_paths([str(root / ".venv")], ScanOptions(exclude=(".venv",)))
    assert _files(res) == [".venv/lib/site/settings.json"]


def test_the_cli_option_can_be_repeated(tmp_path, capsys):
    assert cli.main([str(_project(tmp_path)), "--exclude", ".venv", "--exclude", "node_modules", "--exclude", "logs",
                     "--fail-on", "critical"]) == 0
    out = capsys.readouterr().out
    assert "settings.json" not in out and "package.json" not in out and "excluded by --exclude" in out


def test_the_action_reads_newline_separated_patterns(tmp_path, monkeypatch):
    from ci_helpers import run_action
    root = _project(tmp_path)
    code, out, _, _ = run_action(tmp_path, monkeypatch, str(root), path=".", fail_on="none",
                                 env_extra={"LIMNEX_PREFLIGHT_EXCLUDE": ".venv\nnode_modules\n"})
    assert code == 0 and "excluded by --exclude .venv" in out and "settings.json" not in out.split("excluded")[0], out



# ---- pre-release test 2 (2026-11-20): N4, N5 ----------------------------------------------------------------------------
def test_exclusions_do_not_make_the_scan_incomplete(tmp_path, capsys):
    # N5: a deliberate exclusion is a choice, not a gap in coverage; strict mode used to exit 3 on a clean folder
    root = _project(tmp_path)
    assert cli.main([str(root), "--exclude", ".venv", "--exclude", "node_modules", "--fail-on", "none",
                     "--fail-on-incomplete"]) == 0
    assert "Left out on purpose (2)" in capsys.readouterr().out


def test_exclusions_do_not_fail_the_action_with_fail_on_incomplete(tmp_path, monkeypatch):
    from ci_helpers import run_action
    root = _project(tmp_path)
    code, out, _, summary = run_action(tmp_path, monkeypatch, str(root), path=".", fail_on="none", fail_on_incomplete="true",
                                       env_extra={"LIMNEX_PREFLIGHT_EXCLUDE": ".venv\nnode_modules\n"})
    assert code == 0, out
    assert "Left out on purpose" in summary


def test_excluded_path_checks_every_folder_on_the_way():
    from limnex_preflight.scanner import excluded_path
    assert excluded_path("models/.venv/env.txt", (".venv",)) == ".venv"
    assert excluded_path("models/run.log", ("*.log",)) == "*.log"
    assert excluded_path("data/x.csv", ("data/*.csv",)) == "data/*.csv"
    assert excluded_path("model/logs/keep.json", ("logs/*.json",)) is None
    assert excluded_path("./model/config.json", (".venv",)) is None


def test_changed_only_applies_exclude(tmp_path, monkeypatch):
    # N4: with changed-only the changed files are named paths, which the scanner always scans; exclude was ignored
    import pytest
    from ci_helpers import HAVE_GIT, commit_all, new_repo, run_action, write_event
    if not HAVE_GIT:
        pytest.skip("git not installed")
    repo = new_repo(tmp_path / "r")
    os.makedirs(os.path.join(repo, "models"))
    with open(os.path.join(repo, "models", "ok.json"), "w", encoding="utf-8") as fh:
        fh.write("{}")
    c1 = commit_all(repo, "base")
    os.makedirs(os.path.join(repo, "models", ".venv"))
    for rel in ("models/.venv/env.txt", "models/run.log"):
        with open(os.path.join(repo, rel), "w", encoding="utf-8") as fh:
            fh.write("path=/home/alice/secret-project\n")
    with open(os.path.join(repo, "models", "ok.json"), "w", encoding="utf-8") as fh:
        fh.write('{"a": 1}')
    c2 = commit_all(repo, "change")
    ev = {"GITHUB_EVENT_NAME": "push", "GITHUB_SHA": c2,
          "GITHUB_EVENT_PATH": write_event(tmp_path, {"before": c1, "after": c2})}
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, repo, env_extra=ev, path=".", changed_only="true",
                                             fail_on="low", exclude=".venv\n*.log")
    assert code == 0 and outputs["files-scanned"] == "1", out
    assert "excluded by --exclude .venv" in out and "excluded by --exclude *.log" in out, out
    assert "Left out on purpose" in summary and "models/.venv/env.txt" in summary, summary



# ---- pre-release test 3, R12: changed-only decides links as a full scan does (the filter itself: test_links_round4.py)
def test_changed_links_are_decided_as_in_a_full_scan(tmp_path, monkeypatch):
    import pytest
    from ci_helpers import HAVE_GIT, commit_all, new_repo, run_action, write_event
    if not HAVE_GIT:
        pytest.skip("git not installed")
    repo = new_repo(tmp_path / "r")
    for rel, text in (("models/ok.json", "{}"), ("models/.venv/x.json", '{"p": "/home/alice/secret-project"}'),
                      ("models/sub/ok2.json", "{}")):
        os.makedirs(os.path.dirname(os.path.join(repo, rel)), exist_ok=True)
        with open(os.path.join(repo, rel), "w", encoding="utf-8") as fh:
            fh.write(text)
    c1 = commit_all(repo, "base")
    try:
        os.symlink(os.path.join(".venv", "x.json"), os.path.join(repo, "models", "a_link.json"))
        os.symlink("sub", os.path.join(repo, "models", "dirlink"), target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks cannot be created here (Windows without Developer Mode)")
    c2 = commit_all(repo, "two links")
    ev = {"GITHUB_EVENT_NAME": "push", "GITHUB_SHA": c2,
          "GITHUB_EVENT_PATH": write_event(tmp_path, {"before": c1, "after": c2})}
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, repo, env_extra=ev, path=".", changed_only="true",
                                             fail_on="low", exclude=".venv")
    assert code == 0 and outputs["findings-total"] == "0", out        # the link into .venv was scanned before
    assert "symbolic link into a place left out by --exclude .venv" in out, out
    assert outputs["files-scanned"] == "0" and outputs["files-not-scanned"] == "0", out      # the folder link is not followed
    # it was dropped silently; now it gets the note a full scan gives it ("info": shown with -v, like every info note)
    from limnex_preflight.ci import action
    from limnex_preflight.context import ScanOptions
    entries, notes = action._changed_entries([os.path.join("models", "dirlink")], ".", ScanOptions())
    assert [(level, "not followed" in reason) for _, level, reason in entries] == [("info", True)] and not notes, entries

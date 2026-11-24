import json
import os
import subprocess
import sys

import pytest

from limnex_preflight.cli import main
from limnex_preflight.ignore import IgnoreFileError, parse_ignore_text


@pytest.fixture()
def leaky(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"saved_from": "/home/fx_zed_user/proj", "contact": "fx.person@gmail.com", "ip": "10.1.2.3"}))
    return p


@pytest.fixture()
def clean(tmp_path):
    p = tmp_path / "ok.json"
    p.write_text(json.dumps({"model_type": "bert", "hidden": 8}))
    return p


def test_exit_codes(leaky, clean, capsys):
    assert main([str(clean)]) == 0
    assert main([str(leaky)]) == 1  # username-in-path is MEDIUM; default --fail-on medium
    assert main([str(leaky), "--fail-on", "high"]) == 0
    assert main([str(leaky), "--fail-on", "never"]) == 0
    assert main([str(leaky), "--fail-on", "low"]) == 1


def test_missing_path_is_an_error_not_a_pass(tmp_path):
    # Pre-release test 2026-11-19 (F10): a usage error (2), no longer a scan problem (3) that --no-error-exit turned into 0
    assert main([str(tmp_path / "nope.pkl")]) == 2
    assert main([str(tmp_path / "nope.pkl"), "--no-error-exit"]) == 2


def test_usage_error_without_paths():
    with pytest.raises(SystemExit) as e:
        main([])
    assert e.value.code == 2


def test_json_report(leaky, capsys):
    assert main([str(leaky), "--json"]) == 1
    doc = json.loads(capsys.readouterr().out)
    assert doc["summary"]["findings"] == 3 and doc["tool"]["name"] == "limnex-preflight"
    f = doc["findings"][0]
    assert set(f) >= {"rule_id", "category", "severity", "file", "location", "value_masked", "fingerprint"}
    assert "fx_zed_user" not in json.dumps(doc) and "fx.person" not in json.dumps(doc)


def test_json_and_sarif_to_files_keep_human_output(leaky, tmp_path, capsys):
    j, s = tmp_path / "r.json", tmp_path / "r.sarif"
    assert main([str(leaky), "--json", str(j), "--sarif", str(s)]) == 1
    out = capsys.readouterr().out
    assert "username-in-path" in out
    assert json.loads(j.read_text())["summary"]["findings"] == 3
    sarif = json.loads(s.read_text())
    assert sarif["version"] == "2.1.0"
    run = sarif["runs"][0]
    assert run["tool"]["driver"]["name"] == "limnex-preflight"
    ids = [r["id"] for r in run["tool"]["driver"]["rules"]]
    for res in run["results"]:
        assert ids[res["ruleIndex"]] == res["ruleId"]
        assert res["level"] in ("error", "warning", "note")
        assert res["partialFingerprints"]["limnexFingerprint/v1"]
        assert res["locations"][0]["physicalLocation"]["artifactLocation"]["uri"].endswith("config.json")


def test_ignore_by_fingerprint(leaky, tmp_path, capsys):
    main([str(leaky), "--json", "--no-ignore"])
    doc = json.loads(capsys.readouterr().out)
    fps = {f["rule_id"]: f["fingerprint"] for f in doc["findings"]}
    ig = tmp_path / ".preflightignore"
    ig.write_text("# accepted\n" + fps["username-in-path"] + "  # my own demo path\n" + fps["email-address"] + "\n" + fps["ip-address"] + "\n")
    assert main([str(leaky), "--ignore-file", str(ig)]) == 0
    out = capsys.readouterr().out
    assert "3 ignored" in out


def test_ignore_is_exact_not_per_rule(leaky, tmp_path, capsys):
    # accepting one path must not hide a different path in the same file
    p = tmp_path / "two.json"
    p.write_text(json.dumps({"a": "/home/fx_one_user/x", "b": "/home/fx_two_user/x"}))
    main([str(p), "--json", "--no-ignore"])
    doc = json.loads(capsys.readouterr().out)
    first = doc["findings"][0]["fingerprint"]
    ig = tmp_path / "ig"
    ig.write_text(first + "\n")
    assert main([str(p), "--ignore-file", str(ig), "--json"]) == 1
    doc2 = json.loads(capsys.readouterr().out)
    assert doc2["summary"]["findings"] == 1 and doc2["summary"]["ignored"] == 1


def test_ignore_file_rejects_rule_level_entries():
    for bad in ["username-in-path", "rule:email-address", "*", "ab12", "ZZZZZZZZZZZZZZZZ"]:
        with pytest.raises(IgnoreFileError):
            parse_ignore_text(bad)


def test_default_ignore_file_discovered_next_to_scanned_dir(tmp_path, capsys):
    d = tmp_path / "model"
    d.mkdir()
    (d / "c.json").write_text(json.dumps({"p": "/home/fx_zed_user/x"}))
    main([str(d), "--json"])
    fp = json.loads(capsys.readouterr().out)["findings"][0]["fingerprint"]
    (d / ".preflightignore").write_text(fp + "\n")
    assert main([str(d)]) == 0


def test_stale_ignore_entries_are_reported(clean, tmp_path, capsys):
    ig = tmp_path / "ig"
    ig.write_text("0123456789abcdef\n")
    main([str(clean), "--ignore-file", str(ig)])
    assert "matched nothing" in capsys.readouterr().out


def test_directory_scan_skips_git_and_scans_a_linked_file_under_its_own_name(tmp_path, capsys):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("email = fx.person@gmail.com")
    (tmp_path / "ok.json").write_text("{}")
    # On Windows os.symlink raises OSError 1314 without admin rights: keep the .git check there, the symlink check where possible
    try:
        os.symlink(str(tmp_path / "ok.json"), str(tmp_path / "link.json"))
        have_symlink = True
    except (OSError, NotImplementedError):
        have_symlink = False
    assert main([str(tmp_path), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    paths = [f["path"] for f in doc["files"]]            # the report always uses forward slashes
    assert paths == [(tmp_path / "ok.json").as_posix()]
    if have_symlink:
        # A file with a name of its own is scanned under that name; the link to it is noted (pre-release test 3, R1: the
        # link's name used to decide how the file was read). Only a file without an extension, such as a Hugging Face
        # cache blob, is scanned under a link's name.
        assert any(n["kind"] == "info" and n["file"] == (tmp_path / "link.json").as_posix() and "own name" in n["message"]
                   for n in doc["notes"])


def test_list_rules(capsys):
    assert main(["--list-rules"]) == 0
    out = capsys.readouterr().out
    assert "unsafe-pickle-exec" in out and "huggingface-token" in out


def test_module_entry_point(leaky):
    r = subprocess.run([sys.executable, "-m", "limnex_preflight", str(leaky), "--fail-on", "high"], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert r.returncode == 0 and "scanned 1 file" in r.stdout


def test_no_network_imports():
    """The package must not import any networking module at scan time."""
    # h5py is blocked: it is an optional third-party dependency, and on Windows h5py -> numpy -> platform.uname() imports `socket`
    # just to read the host name (no connection is made). This test is about the package's own imports and the standard library
    # it uses. An audit hook records WHO imports a networking module so a failure names the culprit.
    code = ("import sys, traceback; sys.modules['h5py'] = None; before = set(sys.modules); seen = {}\n"
            "def hook(event, args):\n"
            "    if event == 'import' and args[0] in ('socket','http.client','urllib.request','requests','ssl') and args[0] not in seen:\n"
            "        seen[args[0]] = ''.join(traceback.format_stack(limit=14))\n"
            "sys.addaudithook(hook)\n"
            "import limnex_preflight.cli, limnex_preflight.scanner, limnex_preflight.ci.action\n"
            "bad=[m for m in ('socket','http.client','urllib.request','requests','ssl') if m in sys.modules and m not in before]\n"
            "print(bad)\n"
            "for m in bad: print(m, 'imported from:'); print(seen.get(m))\n")
    r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                       cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert r.stdout.splitlines()[0] == "[]", r.stdout + r.stderr


def test_ipynb_checkpoints_are_scanned_because_they_get_published(tmp_path, capsys):
    d = tmp_path / ".ipynb_checkpoints"
    d.mkdir()
    (d / "nb-checkpoint.json").write_text(json.dumps({"p": "/home/fx_zed_user/x"}))
    assert main([str(tmp_path)]) == 1


def test_format_json_to_a_file_keeps_the_human_report_on_stdout(leaky, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["--format", "json", "--output", "r.json", str(leaky)]) == 1
    assert "username-in-path" in capsys.readouterr().out
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))["summary"]["findings"] == 3


def test_format_json_without_output_prints_only_json(leaky, capsys):
    assert main(["--format", "json", str(leaky)]) == 1
    assert json.loads(capsys.readouterr().out)["summary"]["findings"] == 3


def test_fail_on_none_is_an_alias_of_never(leaky):
    assert main([str(leaky), "--fail-on", "none"]) == 0


def test_fail_on_incomplete(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "ok.json").write_text("{}")
    (d / "blob.weights").write_bytes(bytes(range(256)) * 40)
    assert main([str(d)]) == 0  # "Not scanned" alone is not an error by default
    assert main([str(d), "--fail-on-incomplete"]) == 3
    assert main([str(d), "--fail-on-incomplete", "--fail-on", "none"]) == 3
    (d / "ok.json").write_text(json.dumps({"p": "/home/fx_zed_user/x"}))
    assert main([str(d), "--fail-on-incomplete"]) == 1  # findings win


def test_path_starting_with_a_dash_after_double_dash(tmp_path, monkeypatch):
    d = tmp_path / "-weird"
    d.mkdir()
    (d / "c.json").write_text(json.dumps({"p": "/home/fx_zed_user/x"}))
    monkeypatch.chdir(tmp_path)
    assert main(["--", "-weird"]) == 1


def test_internal_errors_exit_2_never_1(leaky, monkeypatch, capsys):
    import limnex_preflight.cli as cli

    def boom(*a, **k):
        raise RuntimeError("bug")

    monkeypatch.setattr(cli, "scan_paths", boom)
    assert main([str(leaky)]) == 2
    assert "internal error" in capsys.readouterr().err



# ---- pre-release test 2 (2026-11-20): N10, N11, N14, N15 ----------------------------------------------------------------
from limnex_preflight import cli  # noqa: E402
def _one_scanned_one_not(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "ok.json").write_text("{}", encoding="utf-8")
    (tmp_path / "blob.weird").write_bytes(bytes(range(256)) * 8)       # an unsupported type: listed, not scanned
    return tmp_path


def test_json_counts_only_files_that_were_scanned(tmp_path, capsys):
    d = _one_scanned_one_not(tmp_path)
    cli.main([str(d), "--format", "json", "--no-ignore", "--fail-on", "none"])
    summary = json.loads(capsys.readouterr().out)["summary"]
    assert summary["files_scanned"] == 1 and summary["files_not_scanned"] == 1, summary   # N10: files_scanned was 2


def test_the_action_outputs_and_summary_use_the_same_count(tmp_path, monkeypatch):
    from ci_helpers import run_action
    d = _one_scanned_one_not(tmp_path / "w")
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, str(d), path=".", fail_on="none")
    assert outputs["files-scanned"] == "1" and outputs["files-not-scanned"] == "1", outputs
    assert "Scanned **1** file(s) (1 not scanned)" in summary and "scanned 1 file(s) (1 not scanned)" in out


def test_output_dash_is_stdout_and_counts_as_the_one_report_there(tmp_path, capsys):
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    assert cli.main([str(tmp_path / "a.json"), "--format", "json", "-o", "-", "--no-ignore"]) == 0
    json.loads(capsys.readouterr().out)                                  # N11: JSON followed by the human report
    assert cli.main([str(tmp_path / "a.json"), "--format", "sarif", "-o", "-", "--json"]) == 2
    assert "only one report can go to stdout" in capsys.readouterr().err


def test_help_describes_which_ignore_files_are_read(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "each folder named on the command line" in out and "each scanned directory" not in out   # N14


def test_an_unwritable_report_path_is_a_usage_error_not_a_bug(tmp_path, capsys):
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    assert cli.main([str(tmp_path / "a.json"), "--json", str(tmp_path / "no" / "such" / "x.json")]) == 2
    err = capsys.readouterr().err
    assert "folder does not exist" in err and "internal error" not in err and "Traceback" not in err     # N15


# ---- pre-release test 4, S12: "Not scanned (3)" next to "(2 not scanned)" ------------------------------------------------
def test_the_not_scanned_count_is_the_length_of_the_not_scanned_list(tmp_path, capsys):
    """A link that is not followed was listed under "Not scanned" but not counted; a file whose scan failed was counted
    but not listed there. Now the count is the list."""
    import re
    from limnex_preflight.models import Note
    from limnex_preflight.report import file_counts, to_human
    from limnex_preflight.scanner import ScannedFile, scan_paths
    from limnex_preflight.ci.summary import build_summary
    res = scan_paths([str(_one_scanned_one_not(tmp_path))])
    res.notes.append(Note("skipped", "d/outlink.txt", "symbolic link points outside the scanned folder; not followed"))
    res.files.append(ScannedFile("d/broken.pkl", "error", 0))
    res.notes.append(Note("error", "d/broken.pkl", "scan failed (OSError: x)"))
    assert file_counts(res)[1] == 3
    human = to_human(res, "0")
    assert "Not scanned (3):" in human and "(3 not scanned)" in human, human
    assert "the scan failed; see Scan problems" in human
    summary = build_summary(res, version="0", fail_on=None, exit_code=0, scope_note="", incomplete=0)
    assert "### Not scanned (3)" in summary and "(3 not scanned)" in summary

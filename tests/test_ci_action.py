"""The action entry point: exit-code contract, input handling, outputs, summary, annotations."""
import json
import os
import subprocess
import sys

import pytest

import action_fixtures as af
from ci_helpers import P, commit_all, needs_git, new_repo, outside_guard, parse_github_output, run_action, write_event

pytestmark = []


@pytest.fixture()
def clean(tmp_path):
    d = tmp_path / "ws" / "clean"
    af.make_clean(str(d))
    return d.parent


@pytest.fixture()
def bad(tmp_path):
    d = tmp_path / "ws" / "bad"
    af.make_bad(str(d))
    return d.parent


# ------------------------------------------------------------------ exit status contract: 0 / 1 / 2
def test_exit_0_when_nothing_reaches_the_threshold(clean, tmp_path, monkeypatch):
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, clean, path="clean")
    assert code == 0 and outputs["exit-code"] == "0" and outputs["findings-total"] == "0"
    assert outputs["highest-severity"] == "none" and "passed" in summary


def test_exit_1_when_findings_reach_the_threshold(bad, tmp_path, monkeypatch):
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, bad, path="bad")
    assert code == 1 and outputs["exit-code"] == "1"
    assert int(outputs["findings-critical"]) >= 1 and int(outputs["findings-high"]) >= 1
    assert outputs["highest-severity"] == "critical" and "FAILED" in summary


@pytest.mark.parametrize("fail_on,expected", [("none", 0), ("high", 1), ("medium", 1), ("low", 1)])
def test_fail_on_threshold(bad, tmp_path, monkeypatch, fail_on, expected):
    code, *_ = run_action(tmp_path, monkeypatch, bad, path="bad", fail_on=fail_on)
    assert code == expected


def test_threshold_is_respected_for_lower_severities(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "c.json").write_text(json.dumps({"saved_from": "/home/fx_zed_user/proj"}))  # one MEDIUM finding
    assert run_action(tmp_path, monkeypatch, ws, path="c.json", fail_on="high")[0] == 0
    assert run_action(tmp_path, monkeypatch, ws, path="c.json", fail_on="medium")[0] == 1
    assert run_action(tmp_path, monkeypatch, ws, path="c.json", fail_on="low")[0] == 1


def test_default_fail_on_is_medium_like_the_cli(tmp_path, monkeypatch):
    """Review decision 2026-10-24: the default was high, so the MEDIUM review findings (the fail-closed safety net:
    pickle-getattr-review, pickle-computed-callable, unresolved / incompletely analysed pickles) passed CI by default."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "c.json").write_text(json.dumps({"saved_from": "/home/fx_zed_user/proj"}))
    assert run_action(tmp_path, monkeypatch, ws, path="c.json")[0] == 1                  # MEDIUM >= medium (default)
    assert run_action(tmp_path, monkeypatch, ws, path="c.json", fail_on="high")[0] == 0  # opting out stays possible


@pytest.mark.parametrize("inputs", [
    {"fail_on": "critical"}, {"fail_on": "banana"}, {"changed_only": "maybe"}, {"deep": "2"}, {"fail_on_incomplete": "sometimes"},
    {"path": "does-not-exist"},
])
def test_exit_2_on_bad_input_with_an_error_annotation(clean, tmp_path, monkeypatch, inputs):
    inputs = {"path": "clean", **inputs}
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, clean, **inputs)
    assert code == 2 and outputs["exit-code"] == "2"
    assert any(l.startswith("::error ") for l in out.splitlines())


def test_exit_2_when_the_scanner_crashes_not_exit_1(clean, tmp_path, monkeypatch):
    from limnex_preflight.ci import action

    def boom(*a, **k):
        raise RuntimeError("simulated scanner bug with secret-looking text " + af.AWS)

    monkeypatch.setattr(action, "scan_paths", boom)
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, clean, path="clean")
    assert code == 2 and outputs["exit-code"] == "2"
    assert "scanner error" in out


def test_exit_2_for_a_missing_ignore_file_problem_is_not_reported_as_findings(clean, tmp_path, monkeypatch):
    (clean / "clean" / ".preflightignore").write_text("not-a-fingerprint\n")
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, clean, path="clean")
    assert code == 2 and "invalid entry" in out


# ------------------------------------------------------------------ fail-on-incomplete
def test_incomplete_scan_only_fails_when_asked(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    (ws / "m").mkdir(parents=True)
    (ws / "m" / "ok.json").write_text("{}")
    (ws / "m" / "blob.weights").write_bytes(bytes(range(256)) * 40)  # unsupported -> "Not scanned"
    (ws / "m" / "broken.zip").write_bytes(b"PK\x03\x04" + b"\x00" * 30)  # unreadable -> scan problem
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, ws, path="m")
    assert code == 0
    assert "Not scanned" in summary and "blob.weights" in summary
    assert any(l.startswith("::warning ") and "not fully scanned" in l for l in out.splitlines())
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, ws, path="m", fail_on_incomplete="true")
    assert code == 2 and outputs["exit-code"] == "2"
    assert any(l.startswith("::error ") and "not fully scanned" in l for l in out.splitlines())


def test_findings_take_precedence_over_incomplete(bad, tmp_path, monkeypatch):
    (bad / "bad" / "blob.weights").write_bytes(bytes(range(256)) * 40)
    code, *_ = run_action(tmp_path, monkeypatch, bad, path="bad", fail_on_incomplete="true")
    assert code == 1


# ------------------------------------------------------------------ outputs and SARIF
def test_outputs_summary_and_sarif(bad, tmp_path, monkeypatch):
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, bad, path="bad", sarif_file="reports/out.sarif")
    sarif = bad / "reports" / "out.sarif"
    assert sarif.exists() and outputs["sarif-path"] == str(sarif)
    assert int(outputs["files-scanned"]) == 3
    total = sum(int(outputs[f"findings-{s}"]) for s in ("critical", "high", "medium", "low", "info"))
    assert total == int(outputs["findings-total"])
    for s in ("critical", "high", "medium", "low", "info"):
        assert f"findings-{s}" in outputs
    assert "| critical | high | medium | low | info |" in summary and "aws-access-key-id" in summary


def test_no_sarif_unless_requested(clean, tmp_path, monkeypatch):
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, clean, path="clean")
    assert outputs["sarif-path"] == "" and not list(clean.rglob("*.sarif"))


def test_sarif_exists_even_when_nothing_was_scanned(clean, tmp_path, monkeypatch):
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, clean, path="clean", sarif_file="e.sarif")
    doc = json.loads((clean / "e.sarif").read_text(encoding="utf-8"))
    assert doc["runs"][0]["results"] == []


# ------------------------------------------------------------------ hostile paths and names
@pytest.mark.parametrize("dirname", ["dir with spaces", "-leading-dash", "--output=evil", "café 中文", "semi;colon $(touch pwned)", "quote'\"s"])
def test_unusual_path_inputs_work(tmp_path, monkeypatch, dirname):
    if sys.platform == "win32" and any(c in dirname for c in "\"$"):
        pytest.skip("not a valid Windows name")
    ws = tmp_path / "ws"
    d = ws / dirname
    af.make_bad(str(d))
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, ws, path=dirname)
    assert code == 1 and int(outputs["files-scanned"]) == 3
    assert not (ws / "pwned").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="newlines are not allowed in Windows file names")
def test_newline_in_the_path_input_and_in_file_names(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    d = ws / "line1\nline2"
    af.make_bad(str(d))
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, ws, path="line1\nline2", sarif_file="s\nx.sarif")
    assert code == 1 and int(outputs["files-scanned"]) == 3
    assert outputs["sarif-path"].endswith("s\nx.sarif") and os.path.exists(outputs["sarif-path"])
    assert set(outputs) >= {"exit-code", "findings-total", "sarif-path"}  # the newline did not corrupt $GITHUB_OUTPUT


@pytest.mark.skipif(sys.platform == "win32", reason="newlines are not allowed in Windows file names")
def test_hostile_file_name_cannot_forge_workflow_commands(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    d = ws / "m"
    d.mkdir(parents=True)
    evil = "x\n::error::forged annotation\n::add-mask::abc\n::set-output name=exit-code::0.json"
    (d / evil).write_text(json.dumps({"note": "key " + af.AWS}))
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, ws, path="m")
    assert code == 1
    live = outside_guard(out)  # lines GitHub would interpret
    assert not any("forged annotation" in l and not l.startswith("::error file=") for l in live)
    assert not any(l.startswith(("::add-mask", "::set-output")) for l in live)
    assert all(l.startswith(("::error ", "::warning ", "::notice ")) or not l.startswith("::") for l in live)
    assert outputs["exit-code"] == "1"
    assert "forged annotation" not in summary.replace("\\n", " ") or "\n::error" not in summary


def test_markdown_in_file_names_is_escaped_in_the_summary(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    (ws / "m").mkdir(parents=True)
    # Windows forbids | < > in file names; there the test uses the Markdown specials that are allowed
    name = "a`c[x](y)_#.json" if sys.platform == "win32" else "a|b`c<img src=x onerror=1>.json"
    (ws / "m" / name).write_text(json.dumps({"note": "key " + af.AWS}), encoding="utf-8")
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, ws, path="m")
    table = [l for l in summary.splitlines() if "`aws-access-key-id`" in l]
    assert table and "<img" not in table[0]
    assert all(l.count("|") == 7 for l in table), "an unescaped pipe in a file name would break the table"


# ------------------------------------------------------------------ the scanned repo cannot run code in the scanner
def _isolated_python_can_import_the_package():
    r = subprocess.run([sys.executable, "-I", "-c", "import limnex_preflight.ci"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return r.returncode == 0


@pytest.mark.skipif(not _isolated_python_can_import_the_package(), reason="package must be installed (pip install -e .) for python -I")
def test_a_stdlib_shadowing_file_in_the_scanned_repo_is_never_imported(tmp_path):
    ws = tmp_path / "ws"
    af.make_clean(str(ws / "m"))
    for name in ("json.py", "re.py", "limnex_preflight.py"):
        (ws / name).write_text("open('PWNED', 'w').write('x')\nraise SystemExit(99)\n")
    (ws / "limnex_preflight").mkdir()
    (ws / "limnex_preflight" / "__init__.py").write_text("open('PWNED2', 'w')\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith(P)}
    env.update({P + "PATH": "m", "GITHUB_WORKSPACE": str(ws)})
    r = subprocess.run([sys.executable, "-I", "-X", "utf8", "-m", "limnex_preflight.ci"], cwd=str(ws), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert r.returncode == 0, r.stderr.decode()[-500:]
    assert not (ws / "PWNED").exists() and not (ws / "PWNED2").exists()


def test_process_exit_codes_through_the_real_entry_point(tmp_path):
    if not _isolated_python_can_import_the_package():
        pytest.skip("package not installed")
    ws = tmp_path / "ws"
    af.make_clean(str(ws / "clean"))
    af.make_bad(str(ws / "bad"))

    def run(**inputs):
        env = {k: v for k, v in os.environ.items() if not k.startswith(P)}
        env.update({P + k.upper(): v for k, v in inputs.items()})
        env["GITHUB_WORKSPACE"] = str(ws)
        return subprocess.run([sys.executable, "-I", "-X", "utf8", "-m", "limnex_preflight.ci"], cwd=str(ws), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode

    assert run(path="clean") == 0
    assert run(path="bad") == 1
    assert run(path="bad", fail_on="none") == 0
    assert run(path="bad", fail_on="bogus") == 2
    assert run(path="nope") == 2


# ------------------------------------------------------------------ changed-only through the action
@needs_git
def test_changed_only_scans_exactly_the_changed_file(tmp_path, monkeypatch):
    repo = str(tmp_path / "chg")
    shas = af.make_changed_repo(repo)
    ev = {"GITHUB_EVENT_NAME": "push", "GITHUB_SHA": shas["c2"],
          "GITHUB_EVENT_PATH": write_event(tmp_path, {"before": shas["c1"], "after": shas["c2"]})}
    # c1..c2 changed ok.json only: the planted secret in legacy_bad.json is not in scope
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, repo, env_extra=ev, path=".", changed_only="true")
    assert code == 0 and outputs["files-scanned"] == "1" and outputs["findings-total"] == "0"
    assert "changed-only" in out and "1 under the scan path" in out
    # the same code with changed-only off finds it
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, repo, env_extra=ev, path=".", changed_only="false")
    assert code == 1 and int(outputs["files-scanned"]) == 3
    # c2..c3 changed legacy_bad.json: found, and the WHOLE file is scanned
    ev3 = {"GITHUB_EVENT_NAME": "push", "GITHUB_SHA": shas["c3"],
           "GITHUB_EVENT_PATH": write_event(tmp_path, {"before": shas["c2"], "after": shas["c3"]})}
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, repo, env_extra=ev3, path=".", changed_only="true")
    assert code == 1 and outputs["files-scanned"] == "1" and outputs["findings-high"] == "1"


@needs_git
def test_changed_only_with_no_changed_files_passes_and_still_writes_sarif(tmp_path, monkeypatch):
    repo = new_repo(tmp_path / "r")
    open(os.path.join(repo, "a.json"), "w").write("{}")
    c1 = commit_all(repo)
    c2 = commit_all(repo, "empty")
    ev = {"GITHUB_EVENT_NAME": "push", "GITHUB_SHA": c2, "GITHUB_EVENT_PATH": write_event(tmp_path, {"before": c1})}
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, repo, env_extra=ev, changed_only="true", sarif_file="s.sarif")
    assert code == 0 and outputs["files-scanned"] == "0" and "nothing to scan" in out
    assert json.loads(open(os.path.join(repo, "s.sarif"), encoding="utf-8").read())["runs"][0]["results"] == []


@needs_git
def test_changed_only_falls_back_to_everything_and_says_so(tmp_path, monkeypatch):
    repo = str(tmp_path / "chg")
    shas = af.make_changed_repo(repo)
    ev = {"GITHUB_EVENT_NAME": "push", "GITHUB_SHA": shas["c3"],
          "GITHUB_EVENT_PATH": write_event(tmp_path, {"before": "0" * 40})}
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, repo, env_extra=ev, changed_only="true")
    assert code == 1 and int(outputs["files-scanned"]) == 3
    assert "falling back to scanning EVERYTHING" in out and "first push" in out
    assert "falling back" in summary


@needs_git
def test_changed_file_under_a_path_with_an_ignore_file_is_judged_like_a_full_scan(tmp_path, monkeypatch):
    repo = new_repo(tmp_path / "r")
    models = os.path.join(repo, "models")
    os.makedirs(models)
    open(os.path.join(models, "keep.json"), "w").write("{}")
    c1 = commit_all(repo)
    bad = os.path.join(models, "new.json")
    open(bad, "w").write(json.dumps({"note": "key " + af.AWS}))
    c2 = commit_all(repo)
    ev = {"GITHUB_EVENT_NAME": "push", "GITHUB_SHA": c2, "GITHUB_EVENT_PATH": write_event(tmp_path, {"before": c1})}
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, repo, env_extra=ev, path="models", changed_only="true")
    assert code == 1
    fp = [l for l in out.splitlines() if "fingerprint:" in l][0].split("fingerprint:")[1].strip()
    open(os.path.join(models, ".preflightignore"), "w").write(fp + "  # accepted in test\n")
    c3 = commit_all(repo, "add ignore")
    ev = {"GITHUB_EVENT_NAME": "push", "GITHUB_SHA": c3, "GITHUB_EVENT_PATH": write_event(tmp_path, {"before": c1})}
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, repo, env_extra=ev, path="models", changed_only="true")
    assert code == 0, "the ignore file next to the scanned path must apply to the changed files too"
    code_full, *_ = run_action(tmp_path, monkeypatch, repo, env_extra=ev, path="models", changed_only="false")
    assert code_full == 0


@needs_git
def test_changed_archive_is_scanned_whole_not_just_the_diff(tmp_path, monkeypatch):
    import zipfile

    repo = new_repo(tmp_path / "r")
    z = os.path.join(repo, "bundle.zip")
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("old/config.json", json.dumps({"note": "key " + af.AWS}))  # planted in the UNCHANGED part
        f.writestr("notes.txt", "v1")
    c1 = commit_all(repo)
    with zipfile.ZipFile(z, "a") as f:
        f.writestr("added.txt", "only this member is new")
    c2 = commit_all(repo)
    ev = {"GITHUB_EVENT_NAME": "push", "GITHUB_SHA": c2, "GITHUB_EVENT_PATH": write_event(tmp_path, {"before": c1})}
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, repo, env_extra=ev, changed_only="true")
    assert code == 1 and outputs["files-scanned"] == "1" and "bundle.zip!/old/config.json" in out


@needs_git
def test_explicit_base_and_head_inputs(tmp_path, monkeypatch):
    repo = str(tmp_path / "chg")
    shas = af.make_changed_repo(repo)
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, repo, env_extra={"GITHUB_EVENT_NAME": "workflow_dispatch"},
                                      changed_only="true", base=shas["c1"], head=shas["c2"])
    assert code == 0 and outputs["files-scanned"] == "1"
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, repo, env_extra={"GITHUB_EVENT_NAME": "workflow_dispatch"},
                                      changed_only="true", base="--output=x")
    assert "falling back" in out and not os.path.exists(os.path.join(repo, "x"))


def test_report_only_mode_shows_warnings_not_errors(bad, tmp_path, monkeypatch):
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, bad, path="bad", fail_on="none")
    assert code == 0
    ann = [l for l in out.splitlines() if l.startswith("::")]
    assert not any(l.startswith("::error") for l in ann) and any(l.startswith("::warning ") for l in ann)
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, bad, path="bad", fail_on="high")
    assert any(l.startswith("::error file=") for l in out.splitlines())


def test_unreadable_torch_style_archive_fails_the_default_gate(tmp_path, monkeypatch):
    """Review 2026-11-11, finding 2: with default inputs a scan problem is only a warning, so an archive that PyTorch can
    load but the scanner cannot read passed. It is now a MEDIUM finding, which the default fail-on catches."""
    import io
    import pickle
    import struct
    import zipfile
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_STORED) as z:
        z.writestr("archive/data.pkl", pickle.dumps({"weights": [0.0]}, protocol=2))
        z.writestr("archive/version", b"3\n")
    data = bytearray(b.getvalue())
    cd = data.find(b"PK\x01\x02")
    struct.pack_into("<H", data, cd + 6, 99)  # "version needed to extract" 9.9: zipfile refuses, PyTorch does not
    ws = tmp_path / "ws"
    (ws / "m").mkdir(parents=True)
    (ws / "m" / "model.pt").write_bytes(bytes(data))
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, ws, path="m")
    assert code == 1 and outputs["exit-code"] == "1"


# ---- pre-release test 2026-11-19, finding F21: inputs that do nothing on their own say so --------------------------
def test_base_without_changed_only_is_reported_not_silently_ignored(tmp_path, monkeypatch):
    from ci_helpers import run_action
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "a.json").write_text("{}", encoding="utf-8")
    code, out, _, _ = run_action(tmp_path, monkeypatch, str(tmp_path / "w"), path=".", base="abc123")
    assert code == 0 and "::warning" in out and "changed-only" in out, out


def test_upload_sarif_without_a_sarif_file_is_reported(tmp_path, monkeypatch):
    from ci_helpers import run_action
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "a.json").write_text("{}", encoding="utf-8")
    code, out, _, _ = run_action(tmp_path, monkeypatch, str(tmp_path / "w"), path=".", upload_sarif="true")
    assert code == 0 and "::warning" in out and "sarif-file" in out, out


# ------------------------------------------------------------------ the SARIF report and links (pre-release test 2, N2)
def _link(target, link, is_dir=False):
    try:
        os.symlink(str(target), str(link), target_is_directory=is_dir)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks cannot be created here (Windows without Developer Mode)")


def test_sarif_is_not_written_through_a_link_committed_at_its_path(clean, tmp_path, monkeypatch):
    outside = tmp_path / "outside_target.txt"
    outside.write_text("keep me", encoding="utf-8")
    _link(outside, clean / "preflight.sarif")
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, clean, path="clean", sarif_file="preflight.sarif")
    assert code == 2 and "symbolic link" in out and outputs["sarif-path"] == ""
    assert outside.read_text(encoding="utf-8") == "keep me"


def test_sarif_is_not_written_through_a_linked_folder_on_its_path(clean, tmp_path, monkeypatch):
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    _link(outside, clean / "reports", is_dir=True)
    code, out, _, _ = run_action(tmp_path, monkeypatch, clean, path="clean", sarif_file="reports/x.sarif")
    assert code == 2 and "symbolic link" in out and not list(outside.iterdir())


def test_a_link_at_the_sarif_path_is_refused_before_scanning_simulated(clean, tmp_path, monkeypatch):
    # runs everywhere (no real link needed): a path the file system reports as a link is refused and nothing is written
    real_islink = os.path.islink
    monkeypatch.setattr(os.path, "islink", lambda p: str(p).endswith("preflight.sarif") or real_islink(p))
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, clean, path="clean", sarif_file="preflight.sarif")
    assert code == 2 and "symbolic link" in out and outputs["files-scanned"] == "0"
    assert not (clean / "preflight.sarif").exists()


def test_a_linked_folder_on_the_sarif_path_is_refused_simulated(clean, tmp_path, monkeypatch):
    real_islink = os.path.islink
    monkeypatch.setattr(os.path, "islink", lambda p: os.path.basename(str(p)) == "reports" or real_islink(p))
    code, out, _, _ = run_action(tmp_path, monkeypatch, clean, path="clean", sarif_file="reports/x.sarif")
    assert code == 2 and "'reports' is a symbolic link" in out


def test_sarif_outside_the_workspace_is_still_allowed(clean, tmp_path, monkeypatch):
    # e.g. ${{ runner.temp }}/preflight.sarif: the configured path is trusted, only links are refused
    target = tmp_path / "runner_temp" / "preflight.sarif"
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, clean, path="clean", sarif_file=str(target))
    assert code == 0 and target.exists() and outputs["sarif-path"] == str(target)


@pytest.mark.parametrize("value", ["yes", "1", "no", "0", "on", " true ", "true ", " false"])
def test_boolean_inputs_accept_only_true_or_false(clean, tmp_path, monkeypatch, value):
    # pre-release test 3, R13: " true " was true here but failed action.yml's `== 'true'`, so nothing was uploaded
    # pre-release test 2, N16: action.yml compares with 'true', so 'yes' meant true here and false in the upload step
    code, out, _, _ = run_action(tmp_path, monkeypatch, clean, path="clean", upload_sarif=value, sarif_file="x.sarif")
    assert code == 2 and "must be 'true' or 'false'" in out


@pytest.mark.parametrize("value", ["true", "TRUE", "False"])
def test_true_and_false_in_any_case_are_accepted(clean, tmp_path, monkeypatch, value):
    code, out, _, _ = run_action(tmp_path, monkeypatch, clean, path="clean", deep=value)
    assert code == 0, out



# ---- pre-release test 3, R8: a scan that read no file does not say "passed" ----------------------------------------------
def test_a_full_scan_that_checks_nothing_says_so_in_the_summary_and_the_log(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    (ws / "empty").mkdir(parents=True)
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, ws, path="empty")
    assert code == 0 and outputs["files-scanned"] == "0"
    assert summary.splitlines()[0].endswith("nothing was checked"), summary   # it read "passed"
    assert "::warning" in out and "nothing was checked" in out


# ---- retest of release candidate 11: files found but none read is not a green check ----------------------------------------
def test_a_full_scan_that_found_files_but_read_none_fails(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    (ws / "models").mkdir(parents=True)
    (ws / "models" / "weights.xyz").write_bytes(bytes(range(256)) * 64)     # an unsupported type, the only file
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, ws, path="models")
    assert code == 2 and outputs["files-scanned"] == "0" and outputs["files-not-scanned"] == "1"
    assert "::error" in out and "nothing was checked" in out and "(exit 2)" in out


@needs_git
def test_changed_only_touching_only_unreadable_files_still_passes(tmp_path, monkeypatch):
    repo = new_repo(tmp_path / "r")
    open(os.path.join(repo, "a.json"), "w").write("{}")
    c1 = commit_all(repo)
    with open(os.path.join(repo, "picture.xyz"), "wb") as f:
        f.write(bytes(range(256)) * 64)
    c2 = commit_all(repo, "a file no scanner reads")
    ev = {"GITHUB_EVENT_NAME": "push", "GITHUB_SHA": c2, "GITHUB_EVENT_PATH": write_event(tmp_path, {"before": c1})}
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, repo, env_extra=ev, changed_only="true")
    assert code == 0, out

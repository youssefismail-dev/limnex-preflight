"""Fixes from the fifth pre-release test (T1-T12).

T1: a NumPy string array inside a pickle or a joblib file was skipped with the weights, so a classifier whose class
labels held an e-mail address and a home path came back clean; the same strings in a .npy file were found."""
import pickle
import struct

import pytest

from limnex_preflight import cli
from limnex_preflight.detectors import ENGINE
from limnex_preflight.scanner import scan_paths

EMAIL, HOME = "jane.doe42@acme-labs.io", "owner:/home/janedoe42"       # fake values


def _rules(path):
    return {f.rule_id for f in scan_paths([str(path)]).findings}


@pytest.mark.parametrize("kind", ["U", "S", ">U"])
@pytest.mark.parametrize("how", ["pickle0", "pickle2", "pickle4", "pickle5", "joblib", "joblib-zlib", "joblib-p0"])
def test_string_arrays_inside_pickles_and_joblib_files_are_read(tmp_path, kind, how):
    np = pytest.importorskip("numpy")
    labels = np.array([EMAIL, HOME])
    if kind == "S":
        labels = labels.astype("S")
    elif kind == ">U":
        labels = labels.astype(labels.dtype.newbyteorder(">"))
    p = tmp_path / "model.pkl"
    if how.startswith("pickle"):
        p.write_bytes(pickle.dumps({"classes_": labels}, protocol=int(how[-1])))
    else:
        joblib = pytest.importorskip("joblib")
        joblib.dump({"classes_": labels}, str(p), compress=3 if how == "joblib-zlib" else 0,
                    protocol=0 if how == "joblib-p0" else None)
    assert {"email-address", "username-in-path"} <= _rules(p)


@pytest.mark.parametrize("how", ["pickle4", "pickle5", "joblib"])
def test_an_element_that_fills_its_width_does_not_run_into_the_next(tmp_path, how):
    """Elements are split by the dtype's width; a NUL-padding heuristic would glue "...acme-labs.io" to "abc"."""
    np = pytest.importorskip("numpy")
    labels = np.array([EMAIL, "abc"])           # the e-mail is the longest element: no padding after it
    p = tmp_path / "m.pkl"
    if how == "joblib":
        pytest.importorskip("joblib").dump({"y": labels}, str(p))
    else:
        p.write_bytes(pickle.dumps({"y": labels}, protocol=int(how[-1])))
    emails = [f for f in scan_paths([str(p)]).findings if f.rule_id == "email-address"]
    assert emails and all(f.masked.endswith(".io") for f in emails), [f.masked for f in emails]


def test_a_scikit_learn_classifier_with_personal_class_labels(tmp_path):
    np = pytest.importorskip("numpy")
    joblib = pytest.importorskip("joblib")
    linear_model = pytest.importorskip("sklearn.linear_model")
    clf = linear_model.LogisticRegression().fit(np.array([[0.0], [1.0], [0.1], [0.9]]), [EMAIL, HOME, EMAIL, HOME])
    for name, write in (("clf.joblib", lambda p: joblib.dump(clf, str(p), compress=3)),
                        ("clf.pkl", lambda p: p.write_bytes(pickle.dumps(clf, protocol=5)))):
        p = tmp_path / name
        write(p)
        assert {"email-address", "username-in-path"} <= _rules(p), name


def test_number_arrays_are_not_read_as_text(tmp_path):
    np = pytest.importorskip("numpy")
    p = tmp_path / "w.pkl"
    p.write_bytes(pickle.dumps({"w": np.arange(4096, dtype="float64"), "i": np.arange(64, dtype="int32")}, protocol=4))
    res = scan_paths([str(p)])
    assert not res.findings and not [n for n in res.notes if n.kind in ("error", "skipped", "degraded")]


# ---- T2: joblib's lz4 mode under a text name passed clean ---------------------------------------------------------------
LZ4_FRAME = b"\x04\x22\x4d\x18" + b"\x64\x40\xa7" + b"\x0b\x00\x00\x80" + b"(dp0\nVa\np1\n" + b"\x00\x00\x00\x00"


@pytest.mark.parametrize("name", ["lz4_p0.txt", "model.joblib", "weights", "notes.md"])
def test_an_lz4_frame_is_never_clean_or_text(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(LZ4_FRAME)
    res = scan_paths([str(p)])
    assert [s.kind for s in res.files] == ["lz4"]
    assert [f.rule_id for f in res.findings] == ["pickle-incomplete-analysis"]
    assert cli.main([str(p)]) == 1                     # MEDIUM: the default threshold stops


def test_joblibs_own_lz4_file_when_lz4_is_installed(tmp_path):
    joblib = pytest.importorskip("joblib")
    pytest.importorskip("lz4")
    p = tmp_path / "lz4_p0.txt"
    joblib.dump({"p": "/home/alice/x"}, str(p), compress=("lz4", 3), protocol=0)
    assert [f.rule_id for f in scan_paths([str(p)]).findings] == ["pickle-incomplete-analysis"]


# ---- T5: a pickle named .onnx or .safetensors got only a parse error ----------------------------------------------------
@pytest.mark.parametrize("name", ["model.onnx", "model.safetensors"])
@pytest.mark.parametrize("protocol", [0, 2, 4])
def test_a_pickle_under_a_format_name_without_a_signature_is_analysed(tmp_path, name, protocol):
    p = tmp_path / name
    p.write_bytes(pickle.dumps({"p": "/home/alice/x"}, protocol=protocol))
    res = scan_paths([str(p)])
    assert [s.kind for s in res.files] == ["pickle"] and {f.rule_id for f in res.findings} == {"username-in-path"}


def test_a_real_safetensors_header_starting_like_a_pickle_stays_safetensors(tmp_path):
    """A header length of 0x0280..0x05ff starts with the bytes 80 02..05, like a pickle: the header check decides."""
    header = b'{"__metadata__":{"p":"/home/alice/x"},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    header += b" " * (0x0280 - len(header))
    p = tmp_path / "model.safetensors"
    p.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 4)
    res = scan_paths([str(p)])
    assert [s.kind for s in res.files] == ["safetensors"] and {f.rule_id for f in res.findings} == {"username-in-path"}


# ---- T6: the report-overwrite guard ------------------------------------------------------------------------------------
def test_a_model_card_that_quotes_a_scan_is_not_overwritten(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "m").mkdir()
    card = "# My model\n\nScanned before upload:\n\nlimnex-preflight models/ --fail-on high\n"
    (tmp_path / "m" / "README.md").write_text(card, encoding="utf-8")
    (tmp_path / "m" / "empty.json").write_text("", encoding="utf-8")
    for dest in ("m/README.md", "m/empty.json"):
        assert cli.main(["--json", dest, "m"]) == 2, dest
    assert (tmp_path / "m" / "README.md").read_text(encoding="utf-8") == card
    assert (tmp_path / "m" / "empty.json").read_text(encoding="utf-8") == ""


def test_a_text_report_is_replaced_only_outside_the_scanned_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "a.json").write_text("{}", encoding="utf-8")
    for _ in range(2):
        assert cli.main(["m", "-o", "out.txt"]) == 0                   # outside: the second run replaces the first report
    assert cli.main(["m", "-o", "m/out.txt"]) == 0                     # inside: a new file is fine ...
    assert cli.main(["m", "-o", "m/out.txt"]) == 2                     # ... a text report there is not replaced
    for _ in range(2):
        assert cli.main(["m", "--json", "m/r.json"]) == 0              # a JSON report is, inside too


# ---- T7: a string-array .npy cut inside its data ------------------------------------------------------------------------
def test_a_string_array_npy_cut_inside_its_data_is_an_error(tmp_path):
    np = pytest.importorskip("numpy")
    p = tmp_path / "cut.npy"
    np.save(str(p), np.array(["jane.doe42@acme-labs.io", "x"]))
    p.write_bytes(p.read_bytes()[:-10])
    res = scan_paths([str(p)])
    assert any(n.kind == "error" and "shorter than its header" in n.message for n in res.notes)
    assert cli.main(["--fail-on", "none", str(p)]) == 3
    q = tmp_path / "cutnum.npy"                        # numbers are not read, so a cut there stays reduced coverage
    np.save(str(q), np.arange(8, dtype="float64"))
    q.write_bytes(q.read_bytes()[:-10])
    assert cli.main(["--fail-on", "none", str(q)]) == 0


# ---- T8: a PEM header alone is not a key --------------------------------------------------------------------------------
HEADER = "-----BEGIN RSA " + "PRIVATE KEY-----"
BODY = "FAKEfakeFAKEfake" * 4


@pytest.mark.parametrize("text", [
    f'if text.startswith("{HEADER}"):\n    kind = "rsa"\n',
    f"The line {HEADER} marks the start of a PEM key file.\n",
    f'<glob pattern="{HEADER}*"/>\n',
])
def test_a_key_header_without_a_body_is_not_a_finding(text):
    assert "private-key-pem" not in {h.rule.id for h in ENGINE.scan(text)}


@pytest.mark.parametrize("text", [
    f"{HEADER}\n{BODY}\n{BODY}\n",
    f"{HEADER}\r\n{BODY}\r\n",
    '{"key": "' + HEADER + r"\n" + BODY + r'\n"}',                   # escaped newlines in JSON
    '["' + HEADER + r'\n", "' + BODY + r'\n"]',                       # a notebook's list of lines
    f"{HEADER}\nProc-Type: 4,ENCRYPTED\nDEK-Info: AES-128-CBC,0123ABCD\n\n{BODY}\n",
])
def test_a_key_block_with_a_body_is_critical(text):
    hits = [h for h in ENGINE.scan(text) if h.rule.id == "private-key-pem"]
    assert hits and hits[0].severity.label == "critical"


def test_the_masked_key_says_body_redacted_once(tmp_path):
    p = tmp_path / "id_rsa.txt"
    p.write_text(f"{HEADER}\n{BODY}\n-----END RSA PRIVATE KEY-----\n", encoding="utf-8")
    from limnex_preflight.report import to_json
    import json
    finding = json.loads(to_json(scan_paths([str(p)]), "0", None))["findings"][0]
    assert finding["value_masked"].count("[body redacted]") == 1, finding["value_masked"]


# ---- T11: an annotation's file= keeps a backslash that is part of a Linux name ------------------------------------------
def test_an_annotation_path_keeps_a_backslash_on_linux(monkeypatch):
    from limnex_preflight.ci import commands
    monkeypatch.setattr(commands.os, "sep", "/")
    assert commands.workspace_relative(r"models/caf\xe9.json", "/ws") == r"models/caf\xe9.json"
    monkeypatch.setattr(commands.os, "sep", "\\")
    assert commands.workspace_relative(r"models\a.json", r"C:\ws") == "models/a.json"


# ---- T4: on a pull request, a problem in the pull request's own ignore file hid every finding ---------------------------
def test_a_pull_requests_own_invalid_ignore_file_warns_and_the_scan_still_reports(tmp_path, monkeypatch):
    from ci_helpers import HAVE_GIT, commit_all, new_repo, run_action, write_event
    if not HAVE_GIT:
        pytest.skip("git not installed")
    import action_fixtures as af
    repo = new_repo(tmp_path / "r")
    (tmp_path / "r" / "models").mkdir()
    (tmp_path / "r" / "models" / "ok.json").write_text("{}", encoding="utf-8")
    base = commit_all(repo, "base")
    af.git(repo, "checkout", "-q", "-b", "feature")
    (tmp_path / "r" / "models" / "conf.json").write_text('{"p": "/home/alice/x"}', encoding="utf-8")
    (tmp_path / "r" / "models" / ".preflightignore").write_text("username-in-path\n", encoding="utf-8")   # invalid
    head = commit_all(repo, "a finding and a broken ignore file")
    env = {"GITHUB_EVENT_NAME": "pull_request", "GITHUB_SHA": head,
           "GITHUB_EVENT_PATH": write_event(tmp_path, {"pull_request": {"base": {"sha": base}, "head": {"sha": head}}})}
    code, out, outputs, summary = run_action(tmp_path, monkeypatch, repo, env_extra=env, path="models")
    # it stopped with exit 2 BEFORE the scan (findings-total 0); now the scan runs and reports, and the run still exits 2
    assert code == 2 and outputs["findings-total"] == "1" and outputs["exit-code"] == "2", out
    assert "::error" in out and "not applied on pull requests" in out and "invalid entry" in out
    assert "everything was scanned and reported" in out and "username-in-path" in summary
    assert "username-in-path\n" not in out.split("not applied on pull requests", 1)[1].split("\n", 1)[0]
    push = {"GITHUB_EVENT_NAME": "push", "GITHUB_SHA": head,
            "GITHUB_EVENT_PATH": write_event(tmp_path, {"before": base, "after": head})}
    code, out, _, _ = run_action(tmp_path, monkeypatch, repo, env_extra=push, path="models")
    assert code == 2 and "invalid entry" in out          # on a push the file is applied, so it must be valid

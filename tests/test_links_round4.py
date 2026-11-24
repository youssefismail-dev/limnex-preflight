"""Links, pre-release test 4: S6 (a committed link into .git was followed), S7 (a hash-named blob reached through several
links was read under the first link's name in sort order), and R12 again (changed-only scanned a link out of the scanned
folder that a full scan does not follow, and gave a link another fingerprint than a full scan).

changed-only now lists the full scan's walk (nothing is read) and keeps what concerns a changed file, so both modes
decide every path the same way by construction. The decision tests run everywhere; real links only where they can be
created (Linux and macOS; Windows only in Developer Mode)."""
import json
import os

import pytest

from limnex_preflight import cli
from limnex_preflight.ci import action
from limnex_preflight.context import ScanOptions
from limnex_preflight.scanner import VCS_LINK, _link_name_rank, in_vcs_data, scan_paths, symlink_decision

from ci_helpers import HAVE_GIT, commit_all, new_repo, run_action, write_event


# ---- decisions (no real link needed) ---------------------------------------------------------------------------------
@pytest.mark.parametrize("target", ["/r/.git/config", "/r/sub/.hg/hgrc", "/r/.svn/entries", "/r/.git"])
def test_a_link_into_version_control_data_is_never_followed(target):
    assert in_vcs_data(os.path.abspath(target))
    for explicit, follow_all, confine in ((False, False, None), (False, True, None), (True, False, None),
                                          (True, True, os.path.abspath("/r"))):
        assert symlink_decision(os.path.abspath(target), True, explicit=explicit, scan_root=os.path.abspath("/r"),
                                follow_all=follow_all, confine_to=confine) == (False, VCS_LINK)


@pytest.mark.parametrize("target", ["/r/git/config", "/r/.github/workflows/a.yml", "/r/x.git/config", "/r/.gitignore"])
def test_names_that_only_look_like_version_control_data_are_not_refused(target):
    assert not in_vcs_data(os.path.abspath(target))


def test_the_link_name_that_says_most_about_a_blob_wins():
    ranked = sorted(["a_copy.txt", "notes.md", "weights", "model.onnx", "x.safetensors"], key=_link_name_rank)
    assert ranked[:2] == ["model.onnx", "x.safetensors"] and ranked[2] == "weights" and set(ranked[3:]) == {"a_copy.txt", "notes.md"}


def test_changed_only_keeps_the_full_scans_decisions_simulated(tmp_path, monkeypatch):
    """The filter alone, on a fixed walk: the changed path's own entry is kept whatever it says; a file scanned under a link
    name is kept when its content changed; a changed file below a folder the walk left out gets that folder's reason."""
    ws = tmp_path / "ws"
    (ws / "m" / ".venv").mkdir(parents=True)
    (ws / "blobs").mkdir()
    for rel in ("m/a.json", "m/b.json", "m/.venv/env.txt", "blobs/h1", "m/out.txt", "m/model.onnx"):
        (ws / rel).write_text("{}", encoding="utf-8")
    j = lambda *p: os.path.join(str(ws), *p)
    walk = [(j("m", ".venv"), "excluded", "excluded by --exclude .venv"),
            (j("m", "out.txt"), "skipped", "symbolic link points outside the scanned folder; not followed"),
            (j("m", "model.onnx"), None, None),            # the link a blob is scanned under
            (j("m", "a.json"), None, None),
            (j("m", "b.json"), None, None),
            (j("blobs", "h1"), "info", "the same file as the symbolic link m/model.onnx; scanned once, under that name")]
    monkeypatch.setattr(action, "iter_files", lambda *a, **k: iter(walk))
    real = os.path.realpath
    monkeypatch.setattr(action.os.path, "realpath",
                        lambda p: real(j("blobs", "h1")) if os.path.abspath(p) == j("m", "model.onnx") else real(p))
    changed = [j("m", "a.json"), j("m", "out.txt"), j("m", ".venv", "env.txt"), j("blobs", "h1")]
    entries, notes = action._changed_entries(changed, str(ws), ScanOptions())
    assert [(os.path.basename(p), level) for p, level, _ in entries] == [("out.txt", "skipped"), ("model.onnx", None),
                                                                         ("a.json", None), ("h1", "info")]
    assert [(n.kind, n.file.rsplit("/", 1)[-1], n.message) for n in notes] == [("excluded", "env.txt", "excluded by --exclude .venv")]


# ---- real links ------------------------------------------------------------------------------------------------------
def _link(target, link, is_dir=False):
    try:
        os.symlink(str(target), str(link), target_is_directory=is_dir)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks cannot be created here (Windows without Developer Mode)")


def _fake_git_config(repo):
    with open(os.path.join(repo, ".git", "config"), "a", encoding="utf-8") as fh:
        fh.write('[http]\n\textraheader = AUTHORIZATION: basic eC1hY2Nlc3MtdG9rZW46ZmFrZS1ub3QtYS10b2tlbg==\n'
                 '\tpath = /home/alice/fake-runner\n')


@pytest.mark.skipif(not HAVE_GIT, reason="git not installed")
@pytest.mark.parametrize("follow", [False, True])
def test_a_committed_link_into_git_is_never_followed(tmp_path, follow):
    repo = new_repo(tmp_path / "r")
    _fake_git_config(repo)
    os.makedirs(os.path.join(repo, "models"))
    _link(os.path.join("..", ".git", "config"), os.path.join(repo, "models", "x.txt"))
    _link(os.path.join(".git", "config"), os.path.join(repo, "y.txt"))
    for path in (repo, os.path.join(repo, "models")):
        res = scan_paths([path], ScanOptions(follow_symlinks=follow))
        assert not res.findings, path
        assert [n.kind for n in res.notes if n.message == VCS_LINK], path


@pytest.mark.skipif(not HAVE_GIT, reason="git not installed")
@pytest.mark.parametrize("changed_only", ["false", "true"])
def test_the_action_never_follows_a_link_into_git(tmp_path, monkeypatch, changed_only):
    repo = new_repo(tmp_path / "r")
    (tmp_path / "r" / "ok.json").write_text("{}", encoding="utf-8")
    base = commit_all(repo, "base")
    _fake_git_config(repo)
    os.makedirs(os.path.join(repo, "models"))
    _link(os.path.join("..", ".git", "config"), os.path.join(repo, "models", "x.txt"))
    head = commit_all(repo, "a link into .git")
    env = {"GITHUB_EVENT_NAME": "push", "GITHUB_SHA": head,
           "GITHUB_EVENT_PATH": write_event(tmp_path, {"before": base, "after": head})}
    code, out, outputs, _ = run_action(tmp_path, monkeypatch, repo, env_extra=env, path=".", changed_only=changed_only,
                                       fail_on="low")
    assert outputs["findings-total"] == "0" and VCS_LINK in out, out


def _cache(tmp_path):
    base = tmp_path / "cache"
    (base / "blobs").mkdir(parents=True)
    (base / "blobs" / "9c1d0e").write_bytes(b"\x08\x07\x12\x04test")
    snap = base / "snapshots" / "abc"
    snap.mkdir(parents=True)
    return base, snap


def test_a_blob_behind_a_text_named_and_a_model_named_link_is_read_as_the_model(tmp_path):
    base, snap = _cache(tmp_path)
    for name in ("a_copy.txt", "model.onnx", "z_notes.md"):
        _link(os.path.join("..", "..", "blobs", "9c1d0e"), snap / name)
    res = scan_paths([str(base)])
    assert [(s.path.rsplit("/", 1)[-1], s.kind) for s in res.files] == [("model.onnx", "onnx")]
    notes = {n.file.rsplit("/", 1)[-1]: n.message for n in res.notes if n.kind == "info"}
    assert set(notes) == {"a_copy.txt", "z_notes.md", "9c1d0e"}, notes          # the other links, and the blob itself
    assert all(m.endswith("snapshots/abc/model.onnx; scanned once, under that name") for m in notes.values()), notes


def _fingerprints(res):
    return sorted((f.rule_id, f.fingerprint) for f in res.findings)


@pytest.mark.skipif(not HAVE_GIT, reason="git not installed")
def test_changed_only_decides_each_link_as_a_full_scan_does(tmp_path, monkeypatch):
    """A link out of the scanned folder, a link to a walked file, a model name on a blob, a changed blob: the same
    entries, notes and fingerprints in both modes."""
    repo = new_repo(tmp_path / "r")
    r = tmp_path / "r"
    for rel, text in (("models/ok.json", "{}"), ("other/o.txt", "path=/home/alice/outside\n"),
                      ("models/conf.json", json.dumps({"p": "/home/alice/conf"})),
                      ("models/blobs/77aa", json.dumps({"p": "/home/alice/blob"}))):
        (r / rel).parent.mkdir(parents=True, exist_ok=True)
        (r / rel).write_text(text, encoding="utf-8")
    base = commit_all(repo, "base")
    _link(os.path.join("..", "other", "o.txt"), r / "models" / "outlink.txt")
    _link("conf.json", r / "models" / "a_link.txt")
    _link(os.path.join("blobs", "77aa"), r / "models" / "model.json")
    (r / "models" / "blobs" / "77aa").write_text(json.dumps({"p": "/home/alice/blob2"}), encoding="utf-8")
    head = commit_all(repo, "links")
    env = {"GITHUB_EVENT_NAME": "push", "GITHUB_SHA": head,
           "GITHUB_EVENT_PATH": write_event(tmp_path, {"before": base, "after": head})}
    full = run_action(tmp_path, monkeypatch, repo, env_extra=env, path="models", fail_on="none", sarif_file="full.sarif")
    part = run_action(tmp_path, monkeypatch, repo, env_extra=env, path="models", changed_only="true", fail_on="none",
                      sarif_file="part.sarif")
    assert "symbolic link points outside the scanned folder" in part[1], part[1]
    sar = {k: json.loads((r / k).read_text(encoding="utf-8")) for k in ("full.sarif", "part.sarif")}
    for doc in sar.values():     # the link out of the scanned folder is not followed in either mode
        assert not [x for x in doc["runs"][0]["results"]
                    if "outlink" in x["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]]
    prints = {k: sorted(x["partialFingerprints"]["limnexFingerprint/v1"] for x in v["runs"][0]["results"])
              for k, v in sar.items()}
    # the changed-only scan reads only changed content, so its fingerprints are a subset of the full scan's, and here the
    # full scan's every finding comes from a changed file or a link to one
    assert prints["part.sarif"] == prints["full.sarif"] and prints["full.sarif"], prints

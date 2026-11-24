"""Symbolic links (pre-release test 2026-11-19, finding F14: a Hugging Face cache snapshot, a tree of symlinks, scanned
nothing, and even a symlinked file named on the command line was refused).

The rules (scanner.symlink_decision):
  * a path named on the command line is followed;
  * inside a scanned folder, a link to a place inside that folder is noted, not followed: the place is scanned there;
  * a link to a place outside the scanned folder is followed only with --follow-symlinks;
  * in the Action nothing outside the workspace is ever followed, not even a named path: a pull request could turn the
    scanned folder into a link to the runner's own files, whose findings would then be printed in a public log;
  * a broken link, or one leading back into a folder already walked, is noted and not followed.
The first tests check the decision itself (they need no real link, so they run everywhere); the others create links,
which Windows allows only in Developer Mode or as administrator, so there they are skipped."""
import json
import os

import pytest

from limnex_preflight import cli
from limnex_preflight.context import ScanOptions
from limnex_preflight.scanner import scan_paths, symlink_decision

ROOT = os.path.abspath("/scan/root")
WS = os.path.abspath("/workspace")


def _d(target, *, explicit=False, follow_all=False, confine=None, exists=True):
    return symlink_decision(os.path.abspath(target), exists, explicit=explicit, scan_root=ROOT, follow_all=follow_all,
                            confine_to=confine)


# ---- the decision ------------------------------------------------------------------------------------------------------
def test_a_named_path_is_followed():
    assert _d("/anywhere/else/model.bin", explicit=True) == (True, None)


def test_a_link_inside_the_scanned_folder_is_scanned_there():
    follow, why = _d("/scan/root/models/model.bin")
    assert not follow and "inside the scanned folder" in why


def test_a_link_outside_needs_follow_symlinks():
    follow, why = _d("/home/user/.cache/huggingface/blobs/abc")
    assert not follow and "--follow-symlinks" in why
    assert _d("/home/user/.cache/huggingface/blobs/abc", follow_all=True) == (True, None)


def test_the_action_never_leaves_the_workspace():
    for explicit in (True, False):
        follow, why = _d("/home/runner/.ssh/id_rsa", explicit=explicit, follow_all=True, confine=WS)
        assert not follow and "outside the workspace" in why
    assert _d("/workspace/models/x.bin", explicit=True, confine=WS) == (True, None)


def test_a_broken_link_is_not_followed():
    follow, why = _d("/nowhere", explicit=True, exists=False)
    assert not follow and "broken" in why


def test_a_folder_whose_name_only_starts_like_the_root_is_outside():
    follow, why = _d("/scan/root-other/x.bin")
    assert not follow and "outside the scanned folder" in why


# ---- real links (skipped where they cannot be created) ------------------------------------------------------------------
def _link(target, link, is_dir=False):
    try:
        os.symlink(str(target), str(link), target_is_directory=is_dir)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks cannot be created here (Windows without Developer Mode)")


def _hf_cache(tmp_path):
    """models--org--name/{blobs/<hash>, snapshots/abc/config.json -> ../../blobs/<hash>}, as huggingface_hub lays it out."""
    base = tmp_path / "models--org--name"
    blob = base / "blobs" / "4f2e9a"
    blob.parent.mkdir(parents=True)
    blob.write_text(json.dumps({"output_dir": "/home/alice/train"}), encoding="utf-8")
    snap = base / "snapshots" / "abc"
    snap.mkdir(parents=True)
    _link(os.path.join("..", "..", "blobs", "4f2e9a"), snap / "config.json")
    return snap


def test_a_named_symlinked_file_is_scanned(tmp_path):
    snap = _hf_cache(tmp_path)
    res = scan_paths([str(snap / "config.json")])
    assert any(f.rule_id == "username-in-path" for f in res.findings), [n.message for n in res.notes]


def test_a_cache_snapshot_is_scanned_without_any_option(tmp_path):
    """A snapshot's links into its own repository's blobs folder are followed (user review before launch: it scanned
    nothing and passed); --follow-symlinks changes nothing there."""
    snap = _hf_cache(tmp_path)
    for options in (ScanOptions(), ScanOptions(follow_symlinks=True)):
        assert any(f.rule_id == "username-in-path" for f in scan_paths([str(snap)], options).findings)
    assert cli.main([str(snap), "--fail-on", "none"]) == 0


def test_a_loop_is_walked_once(tmp_path):
    d = tmp_path / "d"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "a.json").write_text("{}", encoding="utf-8")
    _link(d, d / "sub" / "back", is_dir=True)
    res = scan_paths([str(d)], ScanOptions(follow_symlinks=True))
    assert [s.path.rsplit("/", 1)[-1] for s in res.files] == ["a.json"]


def test_the_action_does_not_follow_a_scanned_path_out_of_the_workspace(tmp_path, monkeypatch):
    from ci_helpers import run_action
    secret = tmp_path / "runner_home"
    secret.mkdir()
    (secret / "creds.json").write_text(json.dumps({"path": "/home/runner/secret"}), encoding="utf-8")
    ws = tmp_path / "ws"
    ws.mkdir()
    _link(secret, ws / "models", is_dir=True)          # a pull request turned the scanned folder into a link
    code, out, _, _ = run_action(tmp_path, monkeypatch, str(ws), path="models", fail_on="low")
    assert "creds.json" not in out and "outside the workspace" in out, out



# ---- pre-release test 2 (2026-11-20): N5, N6, N7, N20 -------------------------------------------------------------------
def test_a_link_to_a_folder_scanned_anyway_does_not_make_the_scan_incomplete(tmp_path):
    d = tmp_path / "ckpt"
    (d / "checkpoint-100").mkdir(parents=True)
    (d / "checkpoint-100" / "config.json").write_text("{}", encoding="utf-8")
    _link("checkpoint-100", d / "latest", is_dir=True)
    res = scan_paths([str(d)])
    note = [n for n in res.notes if n.file.endswith("latest")]
    assert note and note[0].kind == "info" and "scanned there" in note[0].message, [(n.kind, n.message) for n in res.notes]
    assert cli.main([str(d), "--fail-on-incomplete"]) == 0          # N5: it used to exit 3


def test_a_link_into_an_excluded_folder_is_reported_as_excluded(tmp_path):
    root = tmp_path / "models"
    (root / ".venv" / "lib").mkdir(parents=True)
    (root / ".venv" / "lib" / "conf.txt").write_text("path=/home/alice/x\n", encoding="utf-8")
    _link(os.path.join("..", "models", ".venv", "lib", "conf.txt"), root / "model.txt")
    res = scan_paths([str(root)], ScanOptions(exclude=(".venv",)))
    note = [n for n in res.notes if n.file.endswith("model.txt")]
    assert note and note[0].kind == "excluded" and "left out by --exclude .venv" in note[0].message, note
    assert not any("scanned there" in n.message for n in res.notes)   # N6: it said "scanned there" though nothing was


def test_a_cache_repository_folder_scans_each_blob_under_its_link_name(tmp_path):
    base = tmp_path / "models--org--name"
    blob = base / "blobs" / "9c1d0e"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"\x08\x07\x12\x04test")            # the link's name, not the hash, tells the scanner the format
    snap = base / "snapshots" / "abc"
    snap.mkdir(parents=True)
    _link(os.path.join("..", "..", "blobs", "9c1d0e"), snap / "model.onnx")
    res = scan_paths([str(base)])
    scanned = {s.path.split("models--org--name/", 1)[-1]: s.kind for s in res.files}
    assert list(scanned) == ["snapshots/abc/model.onnx"] and scanned["snapshots/abc/model.onnx"] == "onnx", scanned
    assert any(n.kind == "info" and "same file as the symbolic link" in n.message for n in res.notes)
    assert not any("unsupported" in n.message for n in res.notes)    # N7: the blob was "unsupported file type"


def test_two_links_to_one_file_scan_it_once(tmp_path):
    d = tmp_path / "d"
    (d / "blobs").mkdir(parents=True)
    (d / "blobs" / "h1").write_text(json.dumps({"p": "/home/alice/x"}), encoding="utf-8")
    _link(os.path.join("blobs", "h1"), d / "a.json")
    _link(os.path.join("blobs", "h1"), d / "b.json")
    res = scan_paths([str(d)])
    assert [s.path.rsplit("/", 1)[-1] for s in res.files] == ["a.json"]
    assert len([f for f in res.findings if f.rule_id == "username-in-path"]) == 1


def test_a_folder_whose_links_all_point_outside_says_nothing_was_checked(tmp_path, capsys):
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "c.json").write_text(json.dumps({"p": "/home/alice/x"}), encoding="utf-8")
    (tmp_path / "snap").mkdir()
    _link(os.path.join("..", "outside", "c.json"), tmp_path / "snap" / "config.json")
    assert cli.main([str(tmp_path / "snap")]) == 3                    # files found, none read: not a pass (review)
    assert "nothing was checked" in capsys.readouterr().out          # N7: it read "scanned 0 file(s): no findings"


def test_a_broken_link_named_on_the_command_line_is_a_usage_error(tmp_path, capsys):
    _link(tmp_path / "gone.pkl", tmp_path / "dangling.pkl")
    assert cli.main([str(tmp_path / "dangling.pkl")]) == 2            # N20: it was exit 0
    assert "broken symbolic link" in capsys.readouterr().err


# ---- pre-release test 3, R1: a link's name must never decide how a file is read -----------------------------------------
def _zlib_pickle(path):
    import pickle
    import zlib
    path.write_bytes(zlib.compress(pickle.dumps({"p": "/home/alice/x"}, protocol=4), 3))   # how joblib compresses


def test_a_text_named_link_to_a_compressed_joblib_does_not_turn_off_the_pickle_analysis(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _zlib_pickle(repo / "model.joblib")
    _link("model.joblib", repo / "a_notes.txt")
    res = scan_paths([str(repo)])
    assert [(s.path.rsplit("/", 1)[-1], s.kind) for s in res.files] == [("model.joblib", "zlib")]
    assert any(f.rule_id == "username-in-path" for f in res.findings)          # it was "no findings", exit 0
    assert any(n.kind == "info" and n.file.endswith("a_notes.txt") and "own name" in n.message for n in res.notes)


def test_a_hash_named_blob_behind_a_text_named_link_is_still_analysed(tmp_path):
    repo = tmp_path / "repo"
    (repo / "blobs").mkdir(parents=True)
    _zlib_pickle(repo / "blobs" / "9f3a1c")
    _link(os.path.join("blobs", "9f3a1c"), repo / "a_notes.txt")
    res = scan_paths([str(repo)])
    assert any(f.rule_id == "username-in-path" for f in res.findings), [(s.path, s.kind) for s in res.files]


def test_a_named_link_is_read_by_the_file_s_own_name(tmp_path):
    _zlib_pickle(tmp_path / "model.joblib")
    _link("model.joblib", tmp_path / "a_notes.txt")
    res = scan_paths([str(tmp_path / "a_notes.txt")])
    assert [s.kind for s in res.files] == ["zlib"] and any(f.rule_id == "username-in-path" for f in res.findings)


def test_an_onnx_file_behind_a_text_named_link_is_read_as_onnx(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "model.onnx").write_bytes(b"\x08\x07\x12\x04test")
    _link("model.onnx", repo / "notes.txt")
    res = scan_paths([str(repo)])
    assert [(s.path.rsplit("/", 1)[-1], s.kind) for s in res.files] == [("model.onnx", "onnx")]


# ---- pre-release test 3, R4: a named pipe hung the scan --------------------------------------------------------------------
def test_a_named_pipe_is_listed_as_not_scanned_and_never_opened(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("no named pipes on this system")
    d = tmp_path / "d"
    d.mkdir()
    (d / "ok.json").write_text("{}", encoding="utf-8")
    os.mkfifo(str(d / "pipe"))
    for target in ([str(d)], [str(d / "pipe")]):
        res = scan_paths(target)                       # it used to wait forever for a writer
        assert any(n.kind == "skipped" and "not a regular file" in n.message for n in res.notes)

"""Fixes from the user review's retest of release candidate 8 (2026-11-21). All values here are fake.

1. A Hugging Face cache built by a newer huggingface_hub keeps Xet-backed files in a cache-wide store,
   <cache>/blobs/<2 hex>/<hash>; the repository's blobs/<etag> entry is a link to it. RC8 followed snapshot links only into
   the repository's own blobs/ folder, so most of the repository's files were skipped and the scan passed."""
import json
import os

import pytest

from limnex_preflight.scanner import symlink_decision

XET = "ab" + "3c" * 31                                   # 64 hex, the shared store's file name


def _layout(tmp_path):
    hub = tmp_path / "hub"
    repo = hub / "models--org--name"
    for d in ("blobs", "snapshots/abc", "refs"):
        (repo / d).mkdir(parents=True)
    shared = hub / "blobs" / XET[:2]
    shared.mkdir(parents=True)
    return hub, repo, shared


def _decide(target, scan_root):
    return symlink_decision(os.path.realpath(str(target)), True, explicit=False, scan_root=os.path.realpath(str(scan_root)),
                            follow_all=False, confine_to=None)


# ---- 1. the cache-wide blob store -----------------------------------------------------------------------------------------
def test_a_snapshot_follows_links_into_the_cache_wide_blob_store(tmp_path):
    hub, repo, shared = _layout(tmp_path)
    (shared / XET).write_text("{}", encoding="utf-8")
    snap = repo / "snapshots" / "abc"
    assert _decide(shared / XET, snap) == (True, None)
    assert _decide(shared / XET, repo) == (True, None)               # scanning the repository folder itself


def test_nothing_else_next_to_the_cache_is_followed(tmp_path):
    hub, repo, shared = _layout(tmp_path)
    for name in ("elsewhere/x.bin", "blobs-not/x.bin", "models--other--y/snapshots/def/x.bin"):
        (hub / name).parent.mkdir(parents=True, exist_ok=True)
        (hub / name).write_text("{}", encoding="utf-8")
        follow, why = _decide(hub / name, repo / "snapshots" / "abc")
        assert not follow and "outside the scanned folder" in why, name


def test_outside_a_cache_a_folder_named_blobs_gives_nothing(tmp_path):
    """The rule needs a cache repository (models--org--name with blobs/ and snapshots/) above the scanned folder."""
    (tmp_path / "blobs").mkdir()
    (tmp_path / "blobs" / "x.bin").write_text("{}", encoding="utf-8")
    (tmp_path / "project").mkdir()
    follow, _ = _decide(tmp_path / "blobs" / "x.bin", tmp_path / "project")
    assert not follow


def _link(target, link):
    try:
        os.symlink(str(target), str(link))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks cannot be created here (Windows without Developer Mode)")


def test_a_real_shared_blob_snapshot_is_read_and_its_finding_reported(tmp_path):
    """snapshots/abc/config.json -> ../../blobs/<etag> -> ../../blobs/ab/<hash>, as huggingface_hub lays it out."""
    from limnex_preflight import cli
    from limnex_preflight.scanner import scan_paths
    hub, repo, shared = _layout(tmp_path)
    token = "hf_" + "Qx7mV2pL9sT4wK8nR3jB6yF1dH5gZ0cAeU"           # fake, 34 characters after hf_
    (shared / XET).write_text(json.dumps({"token": token}), encoding="utf-8")
    _link(os.path.join("..", "..", "blobs", XET[:2], XET), repo / "blobs" / "9f8e7d")
    _link(os.path.join("..", "..", "blobs", "9f8e7d"), repo / "snapshots" / "abc" / "config.json")
    res = scan_paths([str(repo / "snapshots" / "abc")])
    assert [f.rule_id for f in res.findings] == ["huggingface-token"]
    assert cli.main([str(repo / "snapshots" / "abc")]) == 1


# ---- 2. a .keras weights member is not copied into memory ---------------------------------------------------------------
TOKEN = "hf_" + "Lm4Rt8Wq2Zx6Cv0Bn3Hj7Kp1Df5Gs9Ay2Eu"        # fake


def _h5(path, mb=0):
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    with h5py.File(str(path), "w") as h:
        h.attrs["token"] = TOKEN
        if mb:
            h.create_dataset("layers/dense/vars/0", data=np.zeros(mb * 1024 * 1024 // 4, dtype="float32"))


def _keras(tmp_path, compression, mb=0):
    import zipfile
    h5 = tmp_path / "w.h5"
    _h5(h5, mb)
    k = tmp_path / ("model_%d.keras" % compression)
    with zipfile.ZipFile(str(k), "w", compression=compression) as z:
        z.writestr("config.json", json.dumps({"class_name": "Sequential"}))
        z.write(str(h5), "model.weights.h5")
    return k


def _headers(monkeypatch):
    from limnex_preflight.formats import hdf5_fmt
    seen = []
    real = hdf5_fmt._child_command

    def spy(header):
        seen.append(dict(header))
        return real(header)
    monkeypatch.setattr(hdf5_fmt, "_child_command", spy)
    return seen


def test_a_stored_weights_member_is_read_in_place(tmp_path, monkeypatch):
    import zipfile
    from limnex_preflight.scanner import scan_paths
    seen = _headers(monkeypatch)
    k = _keras(tmp_path, zipfile.ZIP_STORED)
    res = scan_paths([str(k)])
    assert [f.rule_id for f in res.findings] == ["huggingface-token"]
    assert len(seen) == 1 and seen[0]["path"] == os.path.abspath(str(k)) and seen[0]["offset"] > 0
    with open(str(k), "rb") as f:                       # the window is exactly the member's bytes
        f.seek(seen[0]["offset"])
        assert f.read(seen[0]["length"]) == (tmp_path / "w.h5").read_bytes()


def test_a_compressed_weights_member_is_piped_and_gives_the_same_findings(tmp_path, monkeypatch):
    import zipfile
    from limnex_preflight.scanner import scan_paths
    seen = _headers(monkeypatch)
    k = _keras(tmp_path, zipfile.ZIP_DEFLATED)
    res = scan_paths([str(k)])
    assert [f.rule_id for f in res.findings] == ["huggingface-token"]
    assert seen[0]["path"] is None and "offset" not in seen[0]


def test_a_member_of_a_nested_archive_has_no_window(tmp_path):
    import io
    import zipfile
    from limnex_preflight.formats.archive import stored_window
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.h5", b"x" * 100)
    with zipfile.ZipFile(buf) as z:
        assert stored_window(buf, z.infolist()[0]) is None          # not a file on disk: piped as before


def test_the_window_reads_and_seeks_like_a_file(tmp_path):
    from limnex_preflight.formats.hdf5_child import Window
    p = tmp_path / "f.bin"
    p.write_bytes(b"HEAD" + bytes(range(100)) + b"TAIL")
    with open(str(p), "rb") as f:
        w = Window(f, 4, 100)
        assert w.read(3) == bytes([0, 1, 2]) and w.tell() == 3
        assert w.seek(-2, 2) == 98 and w.read() == bytes([98, 99]) and w.read(5) == b""
        assert w.seek(10) == 10 and w.read(2) == bytes([10, 11])
        with pytest.raises(ValueError):
            w.seek(-1)


@pytest.mark.parametrize("compression", ["stored", "deflated"])
def test_memory_does_not_grow_with_the_weights(tmp_path, compression):
    """48 MB of weights: the scanner's own allocations stay far below the member's size (RC8: about 2x the size)."""
    import tracemalloc
    import zipfile
    from limnex_preflight.scanner import scan_paths
    k = _keras(tmp_path, zipfile.ZIP_STORED if compression == "stored" else zipfile.ZIP_DEFLATED, mb=48)
    tracemalloc.start()
    try:
        res = scan_paths([str(k)])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert [f.rule_id for f in res.findings] == ["huggingface-token"]
    assert peak < 16 * 1024 * 1024, peak


# ---- 3. a Windows Trainer run folder --------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    r"logging_dir: C:\Users\x\out\runs\Aug09_14-23-11_MARIA-LAPTOP",          # as Python stores it (a pickle, a log)
    r'{"logging_dir": "C:\Users\x\out\runs\Aug09_14-23-11_MARIA-LAPTOP"}',  # as JSON writes it
    "logging_dir: out/runs/Aug09_14-23-11_MARIA-LAPTOP",                     # unchanged
])
def test_the_machine_name_in_a_run_folder_is_found_with_either_slash(text):
    from limnex_preflight.detectors import ENGINE
    hits = [h for h in ENGINE.scan(text) if h.rule.id == "hostname"]
    assert [h.value for h in hits] == ["MARIA-LAPTOP"], text


def test_a_windows_run_folder_on_localhost_is_still_not_a_finding():
    from limnex_preflight.detectors import ENGINE
    assert not [h for h in ENGINE.scan(r"out\runs\Aug09_14-23-11_localhost") if h.rule.id == "hostname"]


# ---- 4. the per-rule limit is said, not silent ----------------------------------------------------------------------------
def _csv(tmp_path, rows):
    p = tmp_path / "customers.csv"
    p.write_text("id,email\n" + "".join(f"{i},client{i}.name@mailbox-{i % 7}.net\n" for i in range(rows)), encoding="utf-8")
    return p


def test_more_matches_than_the_limit_are_said_in_reduced_coverage(tmp_path):
    from limnex_preflight.detectors import MAX_HITS_PER_RULE
    from limnex_preflight.scanner import scan_paths
    res = scan_paths([str(_csv(tmp_path, 400))])
    assert sum(f.rule_id == "email-address" for f in res.findings) == MAX_HITS_PER_RULE
    notes = [n for n in res.notes if n.kind == "degraded" and "email-address" in n.message]
    assert len(notes) == 1 and f"only the first {MAX_HITS_PER_RULE}" in notes[0].message


@pytest.mark.parametrize("rows", [10, 200])
def test_up_to_the_limit_there_is_no_note(tmp_path, rows):
    from limnex_preflight.scanner import scan_paths
    res = scan_paths([str(_csv(tmp_path, rows))])
    assert sum(f.rule_id == "email-address" for f in res.findings) == rows
    assert not [n for n in res.notes if "matches of rule" in n.message]


# ---- 5. one read per file through links that lead out of the folder (2026-11-22) ------------------------------------------
def _shared_cache(tmp_path, revisions=("abc",)):
    hub, repo, shared = _layout(tmp_path)
    token = "hf_" + "Wd3Kq8Zp1Lm6Tx4Rv9Bn2Cy7Hs5Fg0JeXa"         # fake, 34 characters after hf_
    (shared / XET).write_text(json.dumps({"token": token}), encoding="utf-8")
    _link(os.path.join("..", "..", "blobs", XET[:2], XET), repo / "blobs" / "9f8e7d")
    for rev in revisions:
        (repo / "snapshots" / rev).mkdir(parents=True, exist_ok=True)
        _link(os.path.join("..", "..", "blobs", "9f8e7d"), repo / "snapshots" / rev / "config.json")
    return repo


@pytest.mark.parametrize("revisions", [("abc",), ("abc", "def")])
def test_scanning_the_whole_cache_repository_reads_each_file_once(tmp_path, revisions):
    """blobs/<etag> and every snapshots/<rev>/config.json lead to one file in <cache>/blobs/: one read, one finding, under
    the snapshot's name, never the hash."""
    from limnex_preflight.scanner import scan_paths
    repo = _shared_cache(tmp_path, revisions)
    res = scan_paths([str(repo)])
    assert [f.rule_id for f in res.findings] == ["huggingface-token"]
    assert res.findings[0].file.replace("\\", "/").endswith("snapshots/abc/config.json")
    assert any("scanned once" in n.message and n.file.replace("\\", "/").endswith("blobs/9f8e7d") for n in res.notes)


def test_two_followed_links_to_one_outside_file_read_it_once(tmp_path):
    """The same rule for --follow-symlinks (it was planned for 0.2: two links out of the folder to one file read it twice)."""
    from limnex_preflight.context import ScanOptions
    from limnex_preflight.scanner import scan_paths
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "settings.json").write_text(json.dumps({"token": "hf_" + "Pq2Ws7Ex4Rc9Tv1Yb6Un3Im8Ok5Lj0HgZt"}),
                                           encoding="utf-8")
    proj = tmp_path / "proj"
    proj.mkdir()
    _link(outside / "settings.json", proj / "a.json")
    _link(outside / "settings.json", proj / "b.json")
    res = scan_paths([str(proj)], ScanOptions(follow_symlinks=True))
    assert [f.rule_id for f in res.findings] == ["huggingface-token"]


def test_a_cache_repository_own_blob_link_is_the_last_choice_of_name(tmp_path):
    from limnex_preflight.scanner import _out_link_rank
    hub, repo, shared = _layout(tmp_path)
    blob = repo / "blobs" / "9f8e7d"
    snap = repo / "snapshots" / "abc" / "config.json"
    assert _out_link_rank(str(snap)) < _out_link_rank(str(blob))
    other = tmp_path / "proj" / "blobs"                                 # a folder named blobs outside a cache: no rule
    other.mkdir(parents=True)
    assert _out_link_rank(str(other / "x.json")) == _out_link_rank(str(snap))


# ---- 6. Hugging Face's own cache metadata is left out on purpose, and said (retest of release candidate 11) -------------
def test_cache_metadata_is_left_out_and_listed(tmp_path):
    from limnex_preflight.scanner import HF_METADATA, scan_paths
    hub, repo, shared = _layout(tmp_path)
    (shared / (XET + ".refs")).write_text("models--org--name/blobs/" + "e7" * 32 + "\n", encoding="utf-8")
    (repo / "trees").mkdir()
    (repo / "trees" / ("c0ffee" * 6 + "abcd.json")).write_text(
        '{"files": ["runs/Aug09_14-23-11_MARIA-LAPTOP/events.out"]}', encoding="utf-8")
    res = scan_paths([str(hub)])
    assert not res.findings, [(f.rule_id, f.file) for f in res.findings]
    left = [n for n in res.notes if n.kind == "excluded" and n.message == HF_METADATA]
    assert len(left) == 2


def test_a_trees_folder_outside_a_cache_is_still_scanned(tmp_path):
    from limnex_preflight.scanner import scan_paths
    (tmp_path / "proj" / "trees").mkdir(parents=True)
    (tmp_path / "proj" / "trees" / "a.json").write_text('{"logging_dir": "out/runs/Aug09_14-23-11_MARIA-LAPTOP"}',
                                                       encoding="utf-8")
    (tmp_path / "proj" / ("ab" * 32 + ".refs")).write_text("x", encoding="utf-8")    # not in a blobs/<2 hex>/ folder
    res = scan_paths([str(tmp_path / "proj")])
    assert [f.rule_id for f in res.findings] == ["hostname"]

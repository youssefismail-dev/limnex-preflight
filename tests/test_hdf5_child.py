"""The HDF5 reader's child process (pre-release test 2026-11-19, finding F1: a damaged .h5 file made h5py hang forever).

h5py runs in a child that reports at least every half second; a child that stops reporting is stopped and the file is
scanned as raw strings, with the reduced coverage reported. These tests use stand-in children (plain Python one-liners),
so the behaviour is checked without needing a file that hangs h5py on this platform.
"""
import json
import sys
import time

import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("numpy")

from limnex_preflight.formats import hdf5_fmt  # noqa: E402
from limnex_preflight.scanner import scan_paths  # noqa: E402

HOME = "/home/alice/models/run1"          # a home path: the username-in-path rule fires on it


def _h5(tmp_path):
    p = tmp_path / "m.h5"
    with h5py.File(p, "w") as h:
        h.attrs["note"] = HOME
    return p


def _messages(res):
    return [n.message for n in res.notes]


def _stand_in(monkeypatch, code):
    monkeypatch.setattr(hdf5_fmt, "_child_command", lambda header: [sys.executable, "-c", code])


def test_a_silent_child_is_stopped_and_the_file_falls_back_to_raw_strings(tmp_path, monkeypatch):
    p = _h5(tmp_path)
    _stand_in(monkeypatch, "import time; time.sleep(120)")          # never reports: the hang
    monkeypatch.setenv(hdf5_fmt.STALL_ENV, "1")
    t = time.time()
    res = scan_paths([str(p)])
    assert time.time() - t < 30
    assert any("made no progress for 1 s" in m and "raw strings" in m for m in _messages(res)), _messages(res)
    assert any(f.rule_id == "username-in-path" for f in res.findings)  # still found, by the raw-string scan


def test_a_slow_child_that_keeps_reporting_is_not_stopped(tmp_path, monkeypatch):
    p = _h5(tmp_path)
    beat = json.dumps({"items": []})
    item = json.dumps({"items": [["value", HOME, 'hdf5 attribute "note" of /', "note"]]})
    done = json.dumps({"done": True, "skips": {}})
    code = (f"import time, sys\nfor _ in range(6):\n    print({beat!r}, flush=True); time.sleep(0.4)\n"
            f"print({item!r}, flush=True); print({done!r}, flush=True)")
    _stand_in(monkeypatch, code)                                     # 2.4 s in all, a report every 0.4 s
    monkeypatch.setenv(hdf5_fmt.STALL_ENV, "1")
    res = scan_paths([str(p)])
    assert not res.notes, _messages(res)                              # no fallback, no reduced coverage
    assert any(f.rule_id == "username-in-path" for f in res.findings)


def test_a_crashing_child_falls_back_like_an_h5py_error(tmp_path, monkeypatch):
    p = _h5(tmp_path)
    _stand_in(monkeypatch, "import sys; sys.exit(3)")
    res = scan_paths([str(p)])
    assert any("could not read the file (ChildProcessError)" in m for m in _messages(res)), _messages(res)
    assert any(f.rule_id == "username-in-path" for f in res.findings)


def test_the_real_child_finds_what_the_old_in_process_reader_found(tmp_path):
    p = _h5(tmp_path)
    res = scan_paths([str(p)])
    assert not res.notes, _messages(res)
    assert [f.rule_id for f in res.findings] == ["username-in-path"]


def test_an_archive_member_goes_through_the_pipe(tmp_path):
    import zipfile
    z = tmp_path / "bundle.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(_h5(tmp_path), "inner/model.h5")
    res = scan_paths([str(z)])
    assert any(f.rule_id == "username-in-path" for f in res.findings), [(f.rule_id, f.location) for f in res.findings]
    assert not any("hdf5" in m for m in _messages(res)), _messages(res)


def test_the_working_folder_cannot_shadow_the_childs_imports(tmp_path, monkeypatch):
    (tmp_path / "h5py.py").write_text("raise SystemExit('shadowed')\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    p = tmp_path / "m.h5"
    with h5py.File(p, "w") as h:
        h.attrs["note"] = HOME
    res = scan_paths([str(p)])
    assert not res.notes, _messages(res)                              # the real h5py was used
    assert any(f.rule_id == "username-in-path" for f in res.findings)



# ---- pre-release test 2 (2026-11-20): N18 ---------------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [("inf", 10.0), ("-inf", 10.0), ("nan", 10.0), ("abc", 10.0), ("-5", 1.0),
                                           ("0", 1.0), ("2.5", 2.5), ("1e9", 3600.0)])
def test_the_stall_setting_is_always_a_finite_number_of_seconds(monkeypatch, raw, expected):
    from limnex_preflight.formats import hdf5_fmt
    monkeypatch.setenv(hdf5_fmt.STALL_ENV, raw)
    assert hdf5_fmt._stall_s() == expected

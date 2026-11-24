"""A short run of evaluation/fuzz_formats.py on every CI run: damaged benign files never crash, hang, exhaust memory, or
come back clean when truncated. The full campaign (300 files per format) is in evaluation/results/fuzz.md."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))

import fuzz_formats  # noqa: E402


def test_damaged_benign_files_never_crash_hang_or_pass_as_clean(tmp_path):
    rep = fuzz_formats.run(cases_per_format=8, seed=7, timeout=60.0, max_mb=256.0, workdir=str(tmp_path))
    assert rep["cases"] >= 8 * 10, "the seed formats were not built"
    assert not rep["failures"], rep["failures"][:10]



def test_text_seeds_are_the_same_bytes_on_every_os(tmp_path):
    """Pre-release test 2, N8: the text seeds were written with the OS's line ends (CRLF on Windows), which shifted the
    fuzzer's random stream, so the same seed gave different cases on Windows and Linux."""
    import fixtures as fx
    for fmt in fx.FORMATS:
        if fmt.ext not in (".ipynb", ".json", ".yaml", ".txt"):
            continue
        p = tmp_path / ("seed" + fmt.ext)
        fmt.writer(str(p), fx.clean_items())
        assert b"\r" not in p.read_bytes(), fmt.name



def test_an_onnx_file_cut_between_two_top_level_fields_is_a_known_limit_not_a_pass():
    """A protobuf file cut exactly where a top-level field ends is a complete, valid file: no reader can see the cut
    (found by the fuzzer, 2026-11-20). Only that exact cut is excused; a cut inside a field must still be reported."""
    import fuzz_formats as fz
    data = bytes([0x08, 0x07]) + bytes([0x12, 0x03]) + b"abc" + bytes([0x72, 0x02]) + b"xy"   # fields 1, 2, 14
    assert fz.protobuf_field_ends(data) == [2, 7, 11]
    rec = {"findings": 0, "notes": [], "peak_mb": 1.0}
    on_boundary = {"format": "onnx", "mutation": "truncate", "bytes": 7, "unparsed_from": None, "complete_at": [2, 7, 11]}
    inside = dict(on_boundary, bytes=5)
    assert fz.judge(on_boundary, rec, 256)[0].startswith("known-limit: cut between two top-level protobuf fields")
    assert fz.judge(inside, rec, 256) == ["clean-cut: truncated file reported as clean and complete"]


def test_seeds_do_not_depend_on_the_clock_or_the_system(tmp_path, monkeypatch):
    """Pre-release test 3, R6: zip entries carried the current local time and the writing OS (0 on Windows, 3 elsewhere),
    so the zip-based seeds changed every two seconds and between systems, and with one shared random stream every later
    format's damaged files moved with them. Now: fixed entry metadata, one stream per format."""
    import time
    import fuzz_formats as fz
    first = fz.seed_digests(fz.build_seeds(str(tmp_path / "a"), 5))
    fixed = time.struct_time((2001, 2, 3, 4, 5, 6, 5, 34, 0))
    monkeypatch.setattr(time, "localtime", lambda *a: fixed)
    monkeypatch.setattr(sys, "platform", "linux" if sys.platform == "win32" else "win32")
    second = fz.seed_digests(fz.build_seeds(str(tmp_path / "b"), 5))
    assert first == second, [f for f in first if first[f] != second.get(f)]


def test_normalize_zip_changes_only_the_time_and_the_system(tmp_path):
    import io
    import zipfile
    import fuzz_formats as fz

    def make(system, when):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            info = zipfile.ZipInfo("a.txt", date_time=when)
            info.create_system = system
            z.writestr(info, "hello")
        return buf.getvalue()

    a, b = make(0, (2001, 2, 3, 4, 5, 6)), make(3, (2026, 11, 20, 12, 0, 0))
    assert a != b and fz.normalize_zip(a) == fz.normalize_zip(b)
    assert zipfile.ZipFile(io.BytesIO(fz.normalize_zip(a))).read("a.txt") == b"hello"
    assert fz.normalize_zip(b"not a zip") == b"not a zip"

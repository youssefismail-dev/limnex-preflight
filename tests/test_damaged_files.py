"""Damaged files found by evaluation/fuzz_formats.py: each one was reported clean, or demanded gigabytes, before its fix.

Rule: a damaged file is reported as incomplete (an error or degraded note, or a finding), never as clean. Every file built
here is benign; only its framing is broken.
"""
import io
import json
import struct
import tracemalloc
import zipfile
import zlib

from limnex_preflight.context import ScanOptions
from limnex_preflight.scanner import scan_paths


def scan(p):
    return scan_paths([str(p)], ScanOptions())


def incomplete(res):
    return bool(res.findings) or any(n.kind in ("error", "degraded") for n in res.notes)


def _zip(members, stored=True):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED) as z:
        for name, data in members:
            z.writestr(name, data)
    return b.getvalue()


def test_a_gguf_with_a_huge_dimension_count_is_truncated_not_20_gigabytes(tmp_path):
    body = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 1, 0)  # version 3, one tensor, no metadata
    body += struct.pack("<Q", 6) + b"weight" + struct.pack("<I", 2_500_000_000)  # tensor name, then a damaged ndims
    p = tmp_path / "model.gguf"
    p.write_bytes(body + b"\x00" * 64)
    tracemalloc.start()
    try:
        res = scan(p)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 32 * 1024 * 1024, f"{peak / 2 ** 20:.0f} MB"
    assert any(n.kind == "degraded" and "gguf" in n.message for n in res.notes)


def test_a_cut_zip_that_holds_a_stored_zip_is_not_read_as_the_inner_one(tmp_path):
    inner = _zip([("deep/config.json", json.dumps({"a": 1}))])
    outer = _zip([("inner.zip", inner), ("README.txt", "bundle\n" * 20)])
    cut = outer[: outer.index(b"README.txt") + 40]  # inside the second member: the inner zip's end record is the last one
    p = tmp_path / "bundle.zip"
    p.write_bytes(cut)
    res = scan(p)
    assert incomplete(res)
    assert any("joined together" in n.message for n in res.notes)


def test_two_zips_glued_together_say_that_the_first_was_not_read(tmp_path):
    p = tmp_path / "two.zip"
    p.write_bytes(_zip([("first.txt", "one\n")]) + _zip([("second.txt", "two\n")]))
    res = scan(p)
    assert any(n.kind == "degraded" and "before the first member" in n.message for n in res.notes)


def test_an_intact_zip_gets_no_layout_note(tmp_path):
    p = tmp_path / "ok.zip"
    p.write_bytes(_zip([("a.txt", "hello\n")], stored=False))
    assert not any(n.kind in ("error", "degraded") for n in scan(p).notes)


def test_a_cut_zlib_stream_is_incomplete(tmp_path):
    import pickle

    data = zlib.compress(pickle.dumps({"layer": list(range(400))}, protocol=4))
    p = tmp_path / "model.joblib"
    p.write_bytes(data[: len(data) // 3])
    res = scan(p)
    # content is lost, so it is an error like a cut gzip stream (pre-release test 4, S5; it was "reduced coverage")
    assert any(n.kind == "error" and "cut short" in n.message for n in res.notes)


def test_data_after_a_zlib_stream_is_reported(tmp_path):
    import pickle

    p = tmp_path / "model.joblib"
    p.write_bytes(zlib.compress(pickle.dumps({"a": 1}, protocol=4)) + b"extra bytes after the stream")
    assert any(n.kind == "degraded" and "after the end of the compressed stream" in n.message for n in scan(p).notes)


def _npy_object_header():
    header = "{'descr': '|O', 'fortran_order': False, 'shape': (1,), }"
    header += " " * (118 - len(header) - 1) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode("latin-1")


def test_an_npy_cut_inside_its_header_padding_is_truncated(tmp_path):
    p = tmp_path / "a.npy"
    p.write_bytes(_npy_object_header()[:74])  # the dict is complete, only the padding is cut
    assert any(n.kind == "error" and "header cut short" in n.message for n in scan(p).notes)   # test 4, S5


def test_an_npy_object_array_without_its_pickle_is_truncated(tmp_path):
    p = tmp_path / "a.npy"
    p.write_bytes(_npy_object_header())
    assert any(n.kind == "degraded" and "without its pickle" in n.message for n in scan(p).notes)


def test_a_catboost_file_shorter_than_its_core_is_truncated(tmp_path):
    core = b"\x00" * 64 + json.dumps({"loss_function": "Logloss", "train_dir": "catboost_info"}).encode() + b"\x00"
    full = b"CBM1" + struct.pack("<I", len(core)) + core + b"\x00" * 32
    p = tmp_path / "model.cbm"
    for cut in (5, 8 + len(core) // 2):  # inside the size field, and inside the core
        p.write_bytes(full[:cut])
        assert any(n.kind == "degraded" and "catboost" in n.message for n in scan(p).notes), cut
    p.write_bytes(full)
    assert not any(n.kind in ("error", "degraded") for n in scan(p).notes)


def test_a_container_name_with_other_content_is_damaged_or_misnamed(tmp_path):
    p = tmp_path / "weights.zip"
    p.write_bytes(b"P")  # a zip cut to its first byte
    res = scan(p)
    assert any(n.kind == "degraded" and "misnamed" in n.message for n in res.notes)
    q = tmp_path / "notes.txt"
    q.write_text("plain text\n")
    assert not any(n.kind == "degraded" for n in scan(q).notes)


# ---- review 2026-11-11, finding 2: a PyTorch-style archive the scanner cannot read must not pass the default gate.
# PyTorch reads zips with its own reader, which accepts some header damage that Python's zipfile rejects (checked on
# torch 2.13 with a checkpoint of two zero tensors: a changed CRC, a "version needed" of 9.9, a local name that differs
# from the central one). Every archive built here is benign: a plain dict, then one header field changed.
def _torch_like(pkl_name="archive/data.pkl"):
    import pickle
    return _zip([(pkl_name, pickle.dumps({"weights": [0.0, 0.0]}, protocol=2)), ("archive/version", b"3\n")])


def _central_entry(data, suffix):
    off = data.find(b"PK\x01\x02")
    while off >= 0:
        nlen, elen, clen = struct.unpack_from("<HHH", data, off + 28)
        if data[off + 46: off + 46 + nlen].endswith(suffix):
            return off
        off = data.find(b"PK\x01\x02", off + 46 + nlen + elen + clen)
    raise AssertionError("member not found")


def _patched(data, off, fmt, value):
    b = bytearray(data)
    struct.pack_into(fmt, b, off, value)
    return bytes(b)


def _rules(res):
    return {f.rule_id for f in res.findings}


def test_torch_style_archive_with_an_unreadable_pickle_member_is_a_medium_finding(tmp_path):
    data = _torch_like()
    cd = _central_entry(data, b"data.pkl")
    p = tmp_path / "model.pt"
    p.write_bytes(_patched(data, cd + 16, "<I", 0xDEADBEEF))  # CRC changed
    res = scan(p)
    hits = [f for f in res.findings if f.rule_id == "unscannable-model-archive"]
    assert hits and all(int(f.severity) == 2 for f in hits)  # MEDIUM: the default gate fails
    assert any(n.kind == "error" for n in res.notes)          # the scan problem is still listed


def test_torch_style_archive_the_zip_reader_cannot_open_is_a_medium_finding(tmp_path):
    data = _torch_like()
    cd = _central_entry(data, b"data.pkl")
    p = tmp_path / "model.pt"
    p.write_bytes(_patched(data, cd + 6, "<H", 99))  # "version needed to extract" 9.9
    assert "unscannable-model-archive" in _rules(scan(p))


def test_other_unreadable_archives_stay_scan_problems_only(tmp_path):
    data = _zip([("docs/readme.txt", b"hello there, nothing here"), ("docs/notes.txt", b"more plain text")])
    cd = _central_entry(data, b"readme.txt")
    p = tmp_path / "bundle.zip"
    p.write_bytes(_patched(data, cd + 16, "<I", 0xDEADBEEF))
    res = scan(p)
    assert "unscannable-model-archive" not in _rules(res)
    assert any(n.kind == "error" for n in res.notes)


def test_a_readable_torch_style_archive_has_no_such_finding(tmp_path):
    p = tmp_path / "model.pt"
    p.write_bytes(_torch_like())
    assert "unscannable-model-archive" not in _rules(scan(p))

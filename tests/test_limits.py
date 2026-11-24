"""Documented blind spots. These tests pin the CURRENT behaviour so README claims stay honest."""
import base64
import json
import struct

import pytest

from limnex_preflight.context import ScanOptions
from limnex_preflight.scanner import scan_paths

AWS = "AKIA" + "IOSFODNN7EXAMPLE"


def scan(p, **kw):
    return scan_paths([str(p)], ScanOptions(**kw))


def test_base64_wrapped_secret_is_not_found(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"blob": base64.b64encode(AWS.encode()).decode()}))
    assert scan(p).findings == []


def test_secret_split_across_fields_is_not_found(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"a": AWS[:10], "b": AWS[10:]}))
    assert scan(p).findings == []


def test_utf16_secret_in_unknown_binary_is_not_found_even_with_deep(tmp_path):
    p = tmp_path / "weights.dat"
    p.write_bytes(b"\x00\xff" * 64 + AWS.encode("utf-16-le") + b"\xff\x00" * 64)
    assert scan(p, deep=True).findings == []


def test_secret_in_raw_tensor_bytes_needs_deep(tmp_path):
    header = json.dumps({"w": {"dtype": "U8", "shape": [20], "data_offsets": [0, 20]}}).encode()
    p = tmp_path / "t.safetensors"
    p.write_bytes(struct.pack("<Q", len(header)) + header + AWS.encode())
    assert scan(p).findings == []
    assert [f.rule_id for f in scan(p, deep=True).findings] == ["aws-access-key-id"]


def test_image_outputs_in_notebooks_are_skipped_by_design(tmp_path):
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [{"cell_type": "code", "metadata": {}, "source": "x", "execution_count": 1,
          "outputs": [{"output_type": "display_data", "metadata": {}, "data": {"image/png": base64.b64encode(AWS.encode()).decode() * 10}}]}]}
    p = tmp_path / "n.ipynb"
    p.write_text(json.dumps(nb))
    assert scan(p).findings == []


def test_unknown_module_in_pickle_is_not_flagged(tmp_path):
    # A deny-list cannot know what an arbitrary installed package does when called.
    p = tmp_path / "x.pkl"
    p.write_bytes(b"\x80\x02csome_unknown_pkg.helpers\nrun_payload\n)R.")
    assert scan(p).findings == []


def test_a_small_text_file_does_not_allocate_the_whole_size_cap(tmp_path):
    """Found by evaluation/fuzz_formats.py: f.read(cap + 1) allocated the 256 MB text cap up front for every text file."""
    import tracemalloc

    p = tmp_path / "config.json"
    p.write_text(json.dumps({"layers": [{"name": f"dense_{i}"} for i in range(50)]}))
    tracemalloc.start()
    try:
        scan(p)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 32 * 1024 * 1024, f"peak allocation {peak / 2 ** 20:.0f} MB for a {p.stat().st_size}-byte file"


def test_read_limited_keeps_its_contract():
    import io

    from limnex_preflight.formats.binary import read_limited

    data = bytes(range(256)) * 9000  # 2.3 MB: more than one read chunk
    assert read_limited(io.BytesIO(data), 10) == data[:10]
    assert read_limited(io.BytesIO(data), len(data)) == data
    assert read_limited(io.BytesIO(data), len(data) + 5) == data
    assert read_limited(io.BytesIO(data), 2 * 1024 * 1024) == data[:2 * 1024 * 1024]
    assert read_limited(io.BytesIO(b""), 10) == b""


def test_a_damaged_pickle_cannot_demand_gigabytes(tmp_path):
    """Found by evaluation/fuzz_formats.py: a length field rewritten to 2**32 made pickletools' f.read(n) allocate 4 GB.

    The pickle below is benign and only 14 bytes long: PROTO 4, then BINBYTES8 claiming 2**32 bytes that are not there.
    """
    import tracemalloc

    p = tmp_path / "damaged.pkl"
    p.write_bytes(b"\x80\x04" + b"\x8e" + (2 ** 32).to_bytes(8, "little") + b"abc")
    tracemalloc.start()
    try:
        res = scan(p)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 32 * 1024 * 1024, f"peak allocation {peak / 2 ** 20:.0f} MB for a 14-byte file"
    assert any(f.rule_id == "pickle-incomplete-analysis" for f in res.findings), "a damaged pickle is incomplete, never clean"


def test_carving_a_small_binary_does_not_allocate_the_carve_window(tmp_path):
    import tracemalloc

    p = tmp_path / "model.cbm"
    p.write_bytes(b"CBM1" + b"\x00" * 64 + b'{"loss_function":"Logloss","train_dir":"catboost_info"}' + b"\x00" * 900)
    tracemalloc.start()
    try:
        scan(p)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 32 * 1024 * 1024, f"peak allocation {peak / 2 ** 20:.0f} MB for a {p.stat().st_size}-byte file"

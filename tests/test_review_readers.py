"""Review 2026-11-11, second round (the other readers). Every file built here is benign."""
import json
import struct

import pytest

from limnex_preflight.context import ScanContext, ScanOptions
from limnex_preflight.scanner import scan_paths


def scan(p):
    return scan_paths([str(p)], ScanOptions())


def rules(res):
    return {f.rule_id for f in res.findings}


# ---- B: ONNX external data that points outside the model's folder ------------------------------------------------
def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _ld(field, payload):
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def _onnx_with_external(location):
    strstr = _ld(1, b"location") + _ld(2, location.encode())
    tensor = _ld(8, b"w") + _ld(13, strstr)
    graph = _ld(5, tensor) + _ld(2, b"g")
    return b"\x08\x07" + _ld(7, graph)   # ir_version 7, then the graph


@pytest.mark.parametrize("location", ["../../outside.bin", "/etc/data.bin", r"C:\data\w.bin", "sub/../../x.bin",
                                      "~/w.bin", "file:///tmp/w.bin"])
def test_onnx_external_data_outside_the_folder_is_flagged(tmp_path, location):
    p = tmp_path / "m.onnx"
    p.write_bytes(_onnx_with_external(location))
    assert "onnx-external-data-path" in rules(scan(p))


@pytest.mark.parametrize("location", ["weights.bin", "sub/weights.bin", "./w.bin", "model.onnx_data"])
def test_onnx_external_data_inside_the_folder_is_quiet(tmp_path, location):
    p = tmp_path / "m.onnx"
    p.write_bytes(_onnx_with_external(location))
    assert "onnx-external-data-path" not in rules(scan(p))


# ---- E: YAML type tags that some Python YAML loaders turn into objects --------------------------------------------
def test_yaml_python_object_tags_are_flagged(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("model:\n  layers: 3\n  extra: !!python/object:collections.OrderedDict {}\n")
    res = scan(p)
    hits = [f for f in res.findings if f.rule_id == "unsafe-yaml-python-tag"]
    assert hits and hits[0].line == 3


def test_plain_yaml_and_harmless_python_tags_are_quiet(tmp_path):
    p = tmp_path / "config.yml"
    p.write_text("model:\n  shape: !!python/tuple [1, 2]\n  name: resnet\n")
    assert "unsafe-yaml-python-tag" not in rules(scan(p))


# ---- F: bytes after the last safetensors tensor -------------------------------------------------------------------
def _safetensors(extra=b""):
    header = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    return struct.pack("<Q", len(header)) + header + b"\x00" * 8 + extra


def test_text_after_the_last_tensor_is_scanned(tmp_path):
    p = tmp_path / "m.safetensors"
    p.write_bytes(_safetensors(b"saved by fx_zed_user from /home/fx_zed_user/models/run1"))
    assert "username-in-path" in rules(scan(p))


def test_a_normal_safetensors_file_is_clean(tmp_path):
    p = tmp_path / "m.safetensors"
    p.write_bytes(_safetensors())
    res = scan(p)
    assert not res.findings and not res.notes


# ---- A: HDF5 parts that could not be read are reported, never silent ----------------------------------------------
h5py = pytest.importorskip("h5py")
np = pytest.importorskip("numpy")


def _notes(res):
    return [n.message for n in res.notes]


def test_a_large_string_dataset_is_reported_as_not_scanned(tmp_path):
    p = tmp_path / "m.h5"
    with h5py.File(p, "w") as h:
        h.create_dataset("vocab", data=np.array([b"tok%d" % i for i in range(10_001)]))
    res = scan(p)
    assert any("string dataset" in m and "not scanned" in m for m in _notes(res)), _notes(res)


def test_stopping_at_the_object_limit_is_reported(tmp_path, monkeypatch):
    from limnex_preflight.formats import hdf5_fmt
    monkeypatch.setattr(hdf5_fmt, "MAX_OBJECTS", 3)
    p = tmp_path / "m.h5"
    with h5py.File(p, "w") as h:
        for i in range(6):
            h.create_group(f"g{i}")
    assert any("objects" in m for m in _notes(scan(p))), _notes(scan(p))


def test_an_unreadable_attribute_is_reported():
    from limnex_preflight.formats import hdf5_fmt

    class Attrs(dict):
        def __getitem__(self, k):
            raise OSError("cannot read")

    ctx = ScanContext("m.h5")
    skips = hdf5_fmt._Skips()
    hdf5_fmt._attrs(ctx, Attrs({"secret_attr": 1}), "/", skips)
    skips.report(ctx)
    assert any("attribute" in n.message for n in ctx.notes)


def test_a_normal_h5_file_has_no_notes(tmp_path):
    p = tmp_path / "m.h5"
    with h5py.File(p, "w") as h:
        h.attrs["model_config"] = json.dumps({"class_name": "Sequential", "config": {"layers": []}})
        h.create_dataset("w", data=np.zeros(4))
        h.create_dataset("names", data=np.array([b"a", b"b"]))
    res = scan(p)
    assert not res.notes, _notes(res)

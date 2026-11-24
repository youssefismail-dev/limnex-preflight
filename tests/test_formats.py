import os

import pytest

import fixtures as fx
from limnex_preflight.context import fingerprint


def key_of(display):
    top, _, rest = display.partition("!/")
    return os.path.basename(top) + ("!/" + rest if rest else "")


def found(planted, findings):
    return any(
        f.rule_id == planted.rule and f.fingerprint == fingerprint(planted.rule, key_of(f.file), planted.value)
        for f in findings
    )


def test_all_formats_built(built):
    formats, _ = built
    names = {f.format for f in formats}
    # core formats must always be present; others need optional libs
    assert {"pickle", "torch-zip", "torch-legacy", "safetensors", "gguf", "ipynb", "json", "yaml", "txt", "cfg", "catboost", "zip-nested"} <= names


def test_every_planted_value_is_found_in_every_format(built, scan):
    formats, _ = built
    missing = []
    for fixture in formats:
        if fixture.kind != "planted":
            continue
        res = scan(fixture.path)
        for p in fx.PLANTED:
            if not found(p, res.findings):
                missing.append((fixture.format, p.label))
    assert not missing, missing


def test_clean_twins_have_no_findings(built, scan):
    formats, _ = built
    noisy = {}
    for fixture in formats:
        if fixture.kind == "clean":
            res = scan(fixture.path)
            if res.findings:
                noisy[fixture.format] = [(f.rule_id, f.location, f.masked) for f in res.findings]
    assert not noisy, noisy


def test_no_raw_planted_value_ever_appears_in_a_report(built, scan):
    from limnex_preflight import __version__
    from limnex_preflight.report import to_human, to_json, to_sarif

    formats, _ = built
    paths = [f.path for f in formats if f.kind == "planted"]
    res = scan(*paths)
    blob = to_human(res, __version__, verbose=True) + to_json(res, __version__, None) + to_sarif(res, __version__)
    for p in fx.PLANTED:
        needle = p.value if p.rule != "private-key-pem" else "FAKEfakeFAKEfake"
        if len(needle) > 12:
            assert needle not in blob, f"{p.label} leaked into a report"
    assert "Xk9-Passw0rd-4471" not in blob and "fx_alice_ml" not in blob


def test_json_walker_reports_json_path(tmp_path, scan):
    import json

    p = tmp_path / "c.json"
    p.write_text(json.dumps({"a": {"b": [{"note": "/home/fx_zed_user/proj"}]}}))
    f = scan(p).findings[0]
    assert f.location == "$.a.b[0].note"


def test_notebook_locations_name_cell_and_output(tmp_path, scan):
    import json

    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
        {"cell_type": "code", "metadata": {}, "execution_count": 1, "source": ["x = 1\n"], "outputs": [
            {"output_type": "stream", "name": "stderr", "text": ["warn\n", "loading /home/fx_zed_user/model.bin\n"]}]}]}
    p = tmp_path / "n.ipynb"
    p.write_text(json.dumps(nb))
    f = scan(p).findings[0]
    assert f.location.startswith("cell 1 (code) output 1 (stream stderr)") and f.location.endswith("line 2")


def test_zip_member_chain_in_file_name(built, scan):
    formats, _ = built
    z = next(f for f in formats if f.format == "zip-nested" and f.kind == "planted")
    res = scan(z.path)
    assert any("!/inner.zip!/deep/config.json" in f.file for f in res.findings)


def test_safetensors_tensor_data_is_not_read_without_deep(tmp_path, scan):
    import json
    import struct

    secret = "hf" + "_" + "FAKEfakeFAKEfakeFAKEfakeFAKE" + "012345"
    header = json.dumps({"w": {"dtype": "U8", "shape": [len(secret)], "data_offsets": [0, len(secret)]}}).encode()
    p = tmp_path / "t.safetensors"
    p.write_bytes(struct.pack("<Q", len(header)) + header + secret.encode())
    assert scan(p).findings == []
    deep = scan(p, deep=True)
    assert [f.rule_id for f in deep.findings] == ["huggingface-token"]


def test_unsupported_files_are_reported_not_silently_ignored(tmp_path, scan):
    p = tmp_path / "blob.weights"
    p.write_bytes(bytes(range(256)) * 20)
    res = scan(p)
    assert any(n.kind == "skipped" and "unsupported" in n.message for n in res.notes)


def _incomplete(res):
    from limnex_preflight.cli import INCOMPLETE_KINDS
    return [n for n in res.notes if n.kind in INCOMPLETE_KINDS]


def _benign_files():
    import io as _io
    import json as _json
    import struct as _struct
    np = pytest.importorskip("numpy")
    arr = np.arange(20000, dtype=np.float32)
    b = _io.BytesIO()
    np.save(b, arr)
    hdr = _json.dumps({"w": {"dtype": "F32", "shape": [20000], "data_offsets": [0, 80000]},
                       "__metadata__": {"format": "pt"}}).encode()
    key = b"general.name"
    gguf = (b"GGUF" + _struct.pack("<I", 3) + _struct.pack("<Q", 0) + _struct.pack("<Q", 1) + _struct.pack("<Q", len(key))
            + key + _struct.pack("<I", 8) + _struct.pack("<Q", 6) + b"benign")
    # (file name, intact bytes, where to cut: inside the payload / metadata, never in padding)
    return [("arr.npy", b.getvalue(), 128 + 40000), ("w.safetensors", _struct.pack("<Q", len(hdr)) + hdr + arr.tobytes(),
            8 + len(hdr) + 40000), ("m.gguf", gguf, 45)]


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_a_truncated_file_is_reported_incomplete_never_clean(tmp_path, scan, idx):
    """Review finding 2026-10-24: npy and safetensors cut inside their data passed as clean; GGUF said 'info'."""
    name, data, cut = _benign_files()[idx]
    whole = tmp_path / name
    whole.write_bytes(data)
    assert not _incomplete(scan(whole)), "an intact file must not be reported incomplete"
    cut_file = tmp_path / ("cut_" + name)
    cut_file.write_bytes(data[:cut])
    assert _incomplete(scan(cut_file)), f"a truncated {name} was reported as clean"


# --- found by dogfooding on the author's own projects (2026-10-29) ---------------------------
def test_non_ascii_utf8_text_with_an_unlisted_extension_is_scanned(tmp_path, scan):
    """The text sniff judged bytes, so UTF-8 continuation bytes counted as unprintable: Arabic, Cyrillic or CJK
    text (and even many em dashes) made a file 'unsupported' while its ASCII twin was scanned."""
    body = ("ملاحظات التدريب: النموذج يعمل جيداً\n" * 50) + "data in /home/fx_zed_user/proj\n"
    p = tmp_path / "notes.rst"
    p.write_text(body, encoding="utf-8")
    res = scan(p)
    assert [f.rule_id for f in res.findings] == ["username-in-path"]
    assert not any("unsupported" in n.message for n in res.notes)


def test_code_files_are_text_but_data_tables_stay_with_the_sniff(tmp_path, scan):
    """Model repositories publish custom model code and run scripts; they must not depend on the size-limited sniff.
    Data tables do on purpose (big URL datasets made a scan 8x slower and buried the model findings)."""
    from limnex_preflight.scanner import detect_kind

    for name in ("modeling_custom.py", "run.sh"):
        assert detect_kind(b"x = 1\n", name, 6) == "text", name
    assert detect_kind(b"a,b\n1,2\n", "train.csv", 8) == "unknown"

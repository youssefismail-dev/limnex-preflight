import io
import json
import zipfile

from limnex_preflight.context import ScanOptions
from limnex_preflight.scanner import scan_paths


def test_nested_archive_depth_limit(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("c.json", json.dumps({"p": "/home/fx_deep_user/x"}))
    data = inner.getvalue()
    for _ in range(5):
        outer = io.BytesIO()
        with zipfile.ZipFile(outer, "w") as z:
            z.writestr("n.zip", data)
        data = outer.getvalue()
    p = tmp_path / "deep.zip"
    p.write_bytes(data)
    shallow = scan_paths([str(p)], ScanOptions(max_depth=2))
    assert shallow.findings == [] and any("nesting" in n.message for n in shallow.notes)
    full = scan_paths([str(p)], ScanOptions(max_depth=10))
    assert [f.rule_id for f in full.findings] == ["username-in-path"]


def test_encrypted_and_corrupt_members_are_reported(tmp_path):
    p = tmp_path / "bad.zip"
    p.write_bytes(b"PK\x03\x04" + b"\x00" * 40)
    res = scan_paths([str(p)])
    assert res.errors


def test_zip_member_names_and_comments_are_scanned(tmp_path):
    p = tmp_path / "n.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("home/fx_name_user/notes.bin", b"\x00\x01")
        z.comment = b"built by fx.person@gmail.com"
    res = scan_paths([str(p)])
    assert {f.rule_id for f in res.findings} >= {"email-address"}


def test_decompression_bomb_is_not_expanded(tmp_path):
    import gzip

    p = tmp_path / "bomb.json.gz"
    with gzip.open(p, "wb") as f:
        f.write(b"0" * (3 * 1024 * 1024))
    res = scan_paths([str(p)], ScanOptions(max_decompressed_bytes=1024 * 1024))
    assert any("decompresses to more than" in n.message for n in res.notes)


def test_tar_and_tgz_members_are_scanned(tmp_path):
    import tarfile

    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"p": "/home/fx_tar_user/x"}))
    for name, mode in (("m.tar", "w"), ("m.tar.gz", "w:gz")):
        with tarfile.open(tmp_path / name, mode) as tf:
            tf.add(cfg, arcname="pkg/config.json")
        res = scan_paths([str(tmp_path / name)])
        assert [f.rule_id for f in res.findings] == ["username-in-path"], name
        assert "!/pkg/config.json" in res.findings[0].file


def test_bz2_and_xz_compressed_pickles_are_scanned(tmp_path):
    import bz2
    import lzma
    import pickle

    payload = pickle.dumps({"p": "/home/fx_zip_user/x"}, protocol=4)
    (tmp_path / "m.pkl.bz2").write_bytes(bz2.compress(payload))
    (tmp_path / "m.pkl.xz").write_bytes(lzma.compress(payload))
    for name in ("m.pkl.bz2", "m.pkl.xz"):
        res = scan_paths([str(tmp_path / name)])
        assert [f.rule_id for f in res.findings] == ["username-in-path"], name

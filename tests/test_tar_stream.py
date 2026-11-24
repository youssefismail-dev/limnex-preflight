"""Pre-release test 2026-11-19, finding F8: a compressed tar (model.tar.gz, .tgz, .tar.bz2, .tar.xz) was decompressed whole
into memory and dropped whole when it passed the per-stream limit (512 MB), so a typical SageMaker model.tar.gz was not
scanned at all. A compressed tar is now read as a stream, member by member; only a member that is itself over the limit
is left out (and reported). Small limits stand in for 512 MB here."""
import io
import os
import tarfile

import pytest

from limnex_preflight.context import MB, ScanOptions
from limnex_preflight.scanner import scan_paths

HOME = "/home/alice/sagemaker/model"


def _member(tf, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def _archive(tmp_path, mode, ext, big=0):
    p = tmp_path / f"model{ext}"
    with tarfile.open(p, mode) as tf:
        for i in range(3):
            _member(tf, f"model/weights_{i}.bin", os.urandom(400 * 1024))     # 1.2 MB in all: over the 1 MB limit
        if big:
            _member(tf, "model/huge.bin", os.urandom(big))
        _member(tf, "model/code/config.json", ('{"output_dir": "%s"}' % HOME).encode())
    return p


@pytest.mark.parametrize("mode,ext", [("w:gz", ".tar.gz"), ("w:gz", ".tgz"), ("w:bz2", ".tar.bz2"), ("w:xz", ".tar.xz")])
def test_a_compressed_tar_over_the_limit_is_scanned_member_by_member(tmp_path, mode, ext):
    res = scan_paths([str(_archive(tmp_path, mode, ext))], ScanOptions(max_decompressed_bytes=1 * MB))
    assert any(f.rule_id == "username-in-path" for f in res.findings), [n.message for n in res.notes]
    assert not any("decompresses to more than" in n.message for n in res.notes), [n.message for n in res.notes]


def test_only_a_member_over_the_limit_is_left_out(tmp_path):
    res = scan_paths([str(_archive(tmp_path, "w:gz", ".tar.gz", big=2 * MB))], ScanOptions(max_decompressed_bytes=1 * MB))
    assert any(f.rule_id == "username-in-path" for f in res.findings)
    skipped = [n for n in res.notes if "huge.bin" in n.file]
    assert skipped and skipped[0].kind == "skipped" and "more than 1 MB" in skipped[0].message, [(n.file, n.message) for n in res.notes]


def test_a_cut_compressed_tar_is_not_clean(tmp_path):
    p = _archive(tmp_path, "w:gz", ".tar.gz")
    p.write_bytes(p.read_bytes()[: p.stat().st_size // 2])
    res = scan_paths([str(p)], ScanOptions(max_decompressed_bytes=1 * MB))
    assert any(n.kind in ("error", "degraded") for n in res.notes), [n.message for n in res.notes]


def test_a_plain_gzip_file_is_still_scanned(tmp_path):
    import gzip
    p = tmp_path / "config.json.gz"
    p.write_bytes(gzip.compress(('{"output_dir": "%s"}' % HOME).encode()))
    res = scan_paths([str(p)])
    assert any(f.rule_id == "username-in-path" for f in res.findings)

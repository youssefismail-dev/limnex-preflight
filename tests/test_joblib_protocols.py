"""joblib files written with pickle protocol 0 or 1 (pre-release test 4, S1).

Protocols 0 and 1 have no marker byte, and they store joblib's NumpyArrayWrapper through copyreg._reconstructor instead
of NEWOBJ. Three things let such a file pass unanalysed: the zlib and lzma probes wanted the protocol 2+ marker, so a
compressed joblib under a text name (or behind a link with a joblib name) was read as text; the array bytes after the
wrapper were read as instructions, so the walk broke and the content was "not scanned" even as model.joblib; and joblib's
lzma format was parsed as a raw pickle. joblib.load reads all of these."""
import bz2
import gzip
import lzma
import os
import zlib

import pytest

from limnex_preflight import cli
from limnex_preflight.scanner import scan_paths

# joblib.dump({"path": "/home/alice/x", "w": numpy.array([1.0, 2.0])}, f, protocol=0), byte for byte (joblib 1.5.3): the
# array's 16 raw bytes follow the wrapper's BUILD ("b"), after a padding-length byte (15) and 15 bytes of padding.
P0_JOBLIB = (
    b"(dp0\nVpath\np1\nV/home/alice/x\np2\nsVw\np3\nccopy_reg\n_reconstructor\np4\n(cjoblib.numpy_pickle\n"
    b"NumpyArrayWrapper\np5\nc__builtin__\nobject\np6\nNtp7\nRp8\n(dp9\nVsubclass\np10\ncnumpy\nndarray\np11\nsVshape\n"
    b"p12\n(I2\ntp13\nsVorder\np14\nVC\np15\nsVdtype\np16\ncnumpy\ndtype\np17\n(Vf8\np18\nI00\nI01\ntp19\nRp20\n(I3\n"
    b"V<\np21\nNNNI-1\nI-1\nI0\ntp22\nbsVallow_mmap\np23\nI01\nsVnumpy_array_alignment_bytes\np24\nI16\nsb\x0f"
    + b"\xff" * 15 + b"\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@s."
)
COMPRESS = {
    "zlib": lambda b: zlib.compress(b, 3),
    "gzip": gzip.compress,
    "bz2": bz2.compress,
    "xz": lzma.compress,
    "lzma": lambda b: lzma.compress(b, format=lzma.FORMAT_ALONE),
}
INCOMPLETE = ("error", "skipped", "degraded")


def _rules(res):
    return sorted({f.rule_id for f in res.findings})


@pytest.mark.parametrize("name", ["model.joblib", "notes.txt", "README.md", "config.json", "weights"])
@pytest.mark.parametrize("comp", [None, *COMPRESS])
def test_a_protocol_0_joblib_with_an_array_is_analysed_under_any_name(tmp_path, name, comp):
    p = tmp_path / name
    p.write_bytes(P0_JOBLIB if comp is None else COMPRESS[comp](P0_JOBLIB))
    res = scan_paths([str(p)])
    assert _rules(res) == ["username-in-path"]
    assert not [n for n in res.notes if n.kind in INCOMPLETE]


@pytest.mark.parametrize("comp", ["zlib", "lzma"])
def test_joblibs_own_wrappers_are_recognised_under_an_onnx_name_too(tmp_path, comp):
    """ONNX is judged by its name; a zlib or lzma stream is checked first, so a joblib named model.onnx is not misread."""
    p = tmp_path / "model.onnx"
    p.write_bytes(COMPRESS[comp](P0_JOBLIB))
    assert _rules(scan_paths([str(p)])) == ["username-in-path"]


def test_content_that_is_no_pickle_inside_a_compressed_joblib_is_medium_not_a_note(tmp_path):
    """Content joblib would hand to its unpickler but that does not parse is reported for review (it was "not scanned",
    exit 0 by default)."""
    p = tmp_path / "model.joblib"
    p.write_bytes(gzip.compress(bytes(range(1, 256)) * 40))
    res = scan_paths([str(p)])
    assert "pickle-incomplete-analysis" in _rules(res)
    assert cli.main([str(p)]) == 1


def test_text_inside_joblibs_zlib_wrapper_still_gets_every_text_rule(tmp_path):
    p = tmp_path / "config.json"
    p.write_bytes(zlib.compress(b'{"log_dir": "/home/alice/x"}'))
    assert "username-in-path" in _rules(scan_paths([str(p)]))


def test_a_cut_zlib_stream_is_an_error_like_a_cut_gzip_stream(tmp_path):
    data = zlib.compress(P0_JOBLIB * 20)
    for name, blob in (("cut_zlib.joblib", data[: len(data) // 2]),
                       ("cut_gzip.joblib", gzip.compress(P0_JOBLIB * 20)[:200])):
        p = tmp_path / name
        p.write_bytes(blob)
        assert any(n.kind == "error" for n in scan_paths([str(p)]).notes), name
        assert cli.main(["--fail-on", "none", str(p)]) == 3, name


def _link(target, link):
    try:
        os.symlink(str(target), str(link))
    except (OSError, NotImplementedError):
        pytest.skip("symlinks cannot be created here (Windows without Developer Mode)")


@pytest.mark.parametrize("comp", [None, "zlib", "lzma", "gzip"])
def test_a_joblib_name_on_a_link_to_a_text_named_protocol_0_joblib(tmp_path, comp):
    """The third and fourth test's shape: model.joblib -> data/notes.txt."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "notes.txt").write_bytes(P0_JOBLIB if comp is None else COMPRESS[comp](P0_JOBLIB))
    _link(data / "notes.txt", tmp_path / "model.joblib")
    for args in ([str(tmp_path)], [str(tmp_path / "model.joblib")]):
        res = scan_paths(args)
        assert _rules(res) == ["username-in-path"], args


@pytest.mark.parametrize("protocol", [0, 1])
@pytest.mark.parametrize("comp", [0, *COMPRESS])
@pytest.mark.parametrize("name", ["model.joblib", "notes.txt", "weights"])
def test_joblib_itself_protocol_0_and_1(tmp_path, protocol, comp, name):
    """The same with files written by joblib (several arrays: numbers, strings, booleans and an object array)."""
    joblib = pytest.importorskip("joblib")
    np = pytest.importorskip("numpy")
    obj = {"path": "/home/alice/x", "w": np.arange(6, dtype="float64"), "s": np.array(["a", "bb"]),
           "b": np.array([True, False]), "o": np.array([{"k": 1}], dtype=object)}
    p = tmp_path / name
    joblib.dump(obj, str(p), compress=comp if comp == 0 else (comp, 3), protocol=protocol)
    assert joblib.load(str(p))["path"] == "/home/alice/x"
    res = scan_paths([str(p)])
    assert _rules(res) == ["username-in-path"]
    assert not [n for n in res.notes if n.kind in INCOMPLETE]

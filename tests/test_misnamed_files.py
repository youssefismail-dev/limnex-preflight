"""A file's name must not turn its analysis off (pre-release test 3, R1).

A compressed joblib is a zlib stream around a pickle. It used to be recognised only under a pickle extension, so the same
file under a text name ("notes.txt", or a link with that name) was read as text and the pickle inside was never analysed.
Compressed pickles are now recognised by their content, under any name; a link's name decides nothing when the file has
a name of its own (see tests/test_symlinks.py for the real links)."""
import bz2
import os
import pickle
import types
import zlib

import pytest

from limnex_preflight import scanner
from limnex_preflight.scanner import detection_name, scan_paths

PAYLOAD = {"p": "/home/alice/x"}       # a planted fake user path: one username-in-path finding


def _kinds_and_rules(res):
    return [s.kind for s in res.files], sorted({f.rule_id for f in res.findings})


@pytest.mark.parametrize("name", ["notes.txt", "README.md", "config.json", "weights", "model.data"])
def test_a_zlib_compressed_pickle_is_analysed_under_any_name(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(zlib.compress(pickle.dumps(PAYLOAD, protocol=4), 3))
    kinds, rules = _kinds_and_rules(scan_paths([str(p)]))
    assert kinds == ["zlib"] and "username-in-path" in rules


@pytest.mark.parametrize("name", ["notes.md", "weights", "model.data"])
def test_a_bz2_compressed_pickle_is_analysed_under_any_name(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(bz2.compress(pickle.dumps(PAYLOAD, protocol=4)))
    kinds, rules = _kinds_and_rules(scan_paths([str(p)]))
    assert kinds == ["bz2"] and "username-in-path" in rules


@pytest.mark.parametrize("text", ["x^2 + y^2 = r^2\n", "x\x01 starts like a zlib header but inflates to no pickle\n"])
def test_text_that_starts_like_a_compressed_stream_stays_text(tmp_path, text):
    p = tmp_path / "notes.txt"
    p.write_bytes(text.encode("latin-1"))
    kinds, _ = _kinds_and_rules(scan_paths([str(p)]))
    assert kinds == ["text"]


def test_a_link_name_is_used_only_when_the_file_has_no_extension(monkeypatch):
    """The rule itself, without a real link (Windows cannot create one without Developer Mode)."""
    links = {os.path.abspath("/r/a_notes.txt"): os.path.abspath("/r/model.joblib"),
             os.path.abspath("/r/snap/model.onnx"): os.path.abspath("/r/blobs/9f3a1c")}
    monkeypatch.setattr(scanner.os.path, "islink", lambda p: os.path.abspath(p) in links)
    monkeypatch.setattr(scanner.os.path, "realpath", lambda p: links.get(os.path.abspath(p), os.path.abspath(p)))
    assert detection_name("/r/a_notes.txt") == "model.joblib"          # the file's own name wins
    assert detection_name("/r/snap/model.onnx") == "model.onnx"        # a hash-named blob: the link's name is the only hint
    assert detection_name("/r/plain.json") == "plain.json"


def test_a_file_that_is_not_regular_is_never_opened(tmp_path, monkeypatch):
    """A named pipe waits for a writer when opened (pre-release test 3, R4); simulated here, real in test_symlinks.py."""
    p = tmp_path / "pipe"
    p.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(scanner, "stat", types.SimpleNamespace(S_ISREG=lambda mode: False))
    opened = []
    monkeypatch.setattr(scanner, "open", lambda *a, **k: opened.append(a) or (_ for _ in ()).throw(AssertionError("opened")),
                        raising=False)
    res = scan_paths([str(p)])
    assert not opened
    assert [s.kind for s in res.files] == ["unsupported"]
    assert any(n.kind == "skipped" and "not a regular file" in n.message for n in res.notes)



@pytest.mark.parametrize("name", ["model.joblib", "notes.txt", "weights"])
def test_an_lzma_compressed_pickle_is_analysed_under_any_name(tmp_path, name):
    """joblib's compress="lzma" format has no fixed signature; it was read as a broken pickle under .joblib and as text
    under a text name."""
    import lzma
    p = tmp_path / name
    p.write_bytes(lzma.compress(pickle.dumps(PAYLOAD, protocol=4), format=lzma.FORMAT_ALONE))
    kinds, rules = _kinds_and_rules(scan_paths([str(p)]))
    assert kinds == ["xz"] and "username-in-path" in rules


def test_text_starting_with_a_bracket_is_not_taken_for_lzma(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("]] closing brackets, then text\n", encoding="utf-8")
    kinds, _ = _kinds_and_rules(scan_paths([str(p)]))
    assert kinds == ["text"]

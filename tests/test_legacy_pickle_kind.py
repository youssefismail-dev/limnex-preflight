"""Pre-release test 2026-11-19, finding F3: pickles in protocol 0 or 1 have no marker byte, so with an extension shared by
other formats (.bin .p .model .state .jl) they were sniffed as text or reported unsupported, never analysed as pickles.
Now a file with such an extension that is, from its first byte to its end, one complete protocol-0/1 pickle is a pickle.
Ordinary pickles made by Python's own pickler only."""
import pickle
from collections import OrderedDict

import pytest

from limnex_preflight.scanner import scan_paths

HOME = "/home/alice/projects/model"           # a home path: the username-in-path rule fires on it
AMBIGUOUS = (".bin", ".p", ".model", ".state", ".jl")


def _rules(res):
    return [f.rule_id for f in res.findings]


@pytest.mark.parametrize("protocol", [0, 1])
@pytest.mark.parametrize("ext", AMBIGUOUS)
def test_a_protocol_0_or_1_pickle_with_an_ambiguous_extension_is_analysed(tmp_path, ext, protocol):
    p = tmp_path / f"state{ext}"
    p.write_bytes(pickle.dumps(OrderedDict([("out_dir", HOME), ("epochs", 3)]), protocol=protocol))
    res = scan_paths([str(p)])
    assert "username-in-path" in _rules(res), (ext, protocol, _rules(res), [n.message for n in res.notes])
    assert not any("not scanned" in n.message.lower() or "unsupported" in n.message.lower() for n in res.notes)


def test_a_pickle_followed_by_other_bytes_is_not_taken_for_a_whole_pickle(tmp_path):
    p = tmp_path / "weights.bin"
    p.write_bytes(pickle.dumps({"a": 1}, protocol=0) + b"\x00\x01\x02 trailing binary data of another format" * 20)
    res = scan_paths([str(p)])
    assert "username-in-path" not in _rules(res)


@pytest.mark.parametrize("text", [b"hello world\nthis is a plain note\n", b".\n", b"(not a pickle at all"])
def test_text_with_an_ambiguous_extension_is_not_a_pickle(tmp_path, text):
    p = tmp_path / "notes.p"
    p.write_bytes(text + b"see " + HOME.encode() + b"\n")
    res = scan_paths([str(p)])
    assert "username-in-path" in _rules(res), "a text file must still be scanned as text"


def test_another_binary_format_with_an_ambiguous_extension_stays_unknown(tmp_path):
    p = tmp_path / "model.bin"
    p.write_bytes(bytes(range(256)) * 64)
    res = scan_paths([str(p)])
    assert not res.findings


# The loader does not look at the name: pickle.load and torch.load read "model.data" or "weights" as readily as "x.pkl".
# So a whole protocol-0/1 pickle is analysed whatever it is called (follow-up to F3: the first fix covered only the five
# extensions other formats share).
OTHER_NAMES = ("model.data", "weights", "checkpoint.v2", "notes.txt", "config.json", "run.log")


@pytest.mark.parametrize("protocol", [0, 1])
@pytest.mark.parametrize("fname", OTHER_NAMES)
def test_a_protocol_0_or_1_pickle_under_any_name_is_analysed(tmp_path, fname, protocol):
    p = tmp_path / fname
    p.write_bytes(pickle.dumps(OrderedDict([("out_dir", HOME), ("epochs", 3)]), protocol=protocol))
    res = scan_paths([str(p)])
    assert "username-in-path" in _rules(res), (fname, protocol, _rules(res), [n.message for n in res.notes])
    assert [f.kind for f in res.files] == ["pickle"], (fname, protocol, [f.kind for f in res.files])


@pytest.mark.parametrize("fname,text", [
    ("urls.csv", b"url,label\nhttp://example.com/a,0\n"),
    ("data", b"Version 1\nThis is a plain note\n"),
    ("notes.txt", b"STOP.\n"),
    ("cfg.json", b'{"a": 1}\n'),
    ("x.log", b"(.\n"),
])
def test_text_under_other_names_stays_text(tmp_path, fname, text):
    p = tmp_path / fname
    p.write_bytes(text + b"see " + HOME.encode() + b"\n")
    res = scan_paths([str(p)])
    assert "username-in-path" in _rules(res), "a text file must still be scanned as text"
    assert [f.kind for f in res.files] != ["pickle"]

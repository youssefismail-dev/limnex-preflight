import io
import os
import pickle
import sys

import pytest

import fixtures as fx
from limnex_preflight.formats.pickle_scan import classify_global


def rules(res):
    return {f.rule_id for f in res.findings}


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return p


@pytest.mark.parametrize("mod,name,rule", [
    ("os", "system", "unsafe-pickle-exec"),
    ("posix", "system", "unsafe-pickle-exec"),
    ("nt", "system", "unsafe-pickle-exec"),
    ("subprocess", "Popen", "unsafe-pickle-exec"),
    ("builtins", "eval", "unsafe-pickle-exec"),
    ("__builtin__", "exec", "unsafe-pickle-exec"),
    ("builtins", "__import__", "unsafe-pickle-exec"),
    ("numpy.os", "system", "unsafe-pickle-exec"),
    ("builtins", "setattr", "unsafe-pickle-io"),
    ("socket", "socket", "unsafe-pickle-io"),
    ("shutil", "rmtree", "unsafe-pickle-io"),
    ("torch", "load", "unsafe-pickle-nested"),
    ("pickle", "loads", "unsafe-pickle-nested"),
])
def test_classification(mod, name, rule):
    assert classify_global(mod, name)[0] == rule


@pytest.mark.parametrize("mod,name", [
    ("collections", "OrderedDict"), ("torch._utils", "_rebuild_tensor_v2"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "_reconstruct"), ("_codecs", "encode"), ("builtins", "set"), ("builtins", "slice"),
    ("os.path", "join"), ("_io", "BytesIO"), ("copyreg", "_reconstructor"), ("sklearn.linear_model._logistic", "LogisticRegression"),
    ("pathlib", "PosixPath"), ("functools", "partial"), ("builtins", "getattr"),
])
def test_benign_globals_not_flagged(mod, name):
    assert classify_global(mod, name) is None


def test_every_unsafe_variant_is_flagged_and_benign_ones_are_not(built, scan):
    _, unsafe = built
    for case in unsafe:
        res = scan(case.path)
        got = rules(res)
        if case.malicious:
            assert set(case.expect_rules) <= got, f"{case.id} {case.description}: {got}"
        else:
            assert not any(r.startswith(("unsafe-pickle", "pickle-")) for r in got), f"{case.id}: {got}"


def test_unsafe_findings_are_a_separate_category_and_severity(built, scan):
    _, unsafe = built
    case = next(c for c in unsafe if c.id == "U04")
    f = scan(case.path).findings[0]
    assert f.category == "unsafe-code" and f.severity.name == "CRITICAL"
    assert f.masked == "os.system"


def test_memo_indirection_is_resolved(built, scan):
    _, unsafe = built
    case = next(c for c in unsafe if c.id == "U05")
    res = scan(case.path)
    assert any(f.masked == "builtins.exec" for f in res.findings)


def test_scanning_never_executes_anything(tmp_path, scan):
    marker = tmp_path / "pwned"
    side = tmp_path / "sidefx"
    side.mkdir()
    (side / "limnex_sidefx_module.py").write_text(f"open({str(marker)!r}, 'w').write('imported')\n")
    sys.path.insert(0, str(side))
    try:
        payload = (b"cos\nsystem\n(S'touch " + str(marker).encode() + b"'\ntR."
                   + b"\x80\x02climnex_sidefx_module\nthing\n.")
        p = write(tmp_path, "evil.pkl", payload)
        res = scan(p)
        assert "unsafe-pickle-exec" in rules(res)
        assert not marker.exists(), "scanner executed or imported pickle content"
        assert "limnex_sidefx_module" not in sys.modules
    finally:
        sys.path.remove(str(side))


def test_truncated_pickle_reports_incomplete_not_clean(tmp_path, scan):
    data = fx.benign_pickle_bytes()[:-5]
    res = scan(write(tmp_path, "cut.pkl", data))
    assert "pickle-incomplete-analysis" in rules(res)


def test_trailing_garbage_is_reported(tmp_path, scan):
    res = scan(write(tmp_path, "t.pkl", fx.benign_pickle_bytes() + b"JUNKJUNKJUNK"))
    assert "pickle-trailing-data" in rules(res)


def test_joblib_array_payload_is_skipped_correctly(tmp_path, scan):
    joblib = pytest.importorskip("joblib")
    np = pytest.importorskip("numpy")
    obj = {"a": np.arange(1000, dtype="float64"), "path": "/home/fx_zed_user/x", "b": np.arange(7, dtype="int16")}
    p = tmp_path / "m.joblib"
    joblib.dump(obj, str(p))
    res = scan(p)
    assert "pickle-incomplete-analysis" not in rules(res)
    assert "username-in-path" in rules(res)


def test_sklearn_model_is_clean(built, scan):
    _, unsafe = built
    case = next((c for c in unsafe if c.id == "B02"), None)
    if case is None:
        pytest.skip("scikit-learn not installed")
    assert scan(case.path).findings == []


def test_pickle_hidden_inside_bytes_is_followed(tmp_path, scan):
    inner = fx.pickle_proto4_system()
    outer = pickle.dumps({"blob": inner, "note": "x"}, protocol=4)
    res = scan(write(tmp_path, "nested.pkl", outer))
    hits = [f for f in res.findings if f.rule_id == "unsafe-pickle-exec"]
    assert hits and "!/bytes@" in hits[0].file


def test_raw_bin_without_pickle_magic_is_not_parsed_as_pickle(tmp_path, scan):
    p = write(tmp_path, "weights.bin", bytes(range(256)) * 40)
    res = scan(p)
    assert not any(f.rule_id.startswith("pickle-") for f in res.findings)
    assert any("unsupported" in n.message for n in res.notes)


def test_bin_with_pickle_magic_is_scanned(tmp_path, scan):
    p = write(tmp_path, "training_args.bin", pickle.dumps({"output_dir": "/home/fx_zed_user/out"}, protocol=4))
    assert {f.rule_id for f in scan(p).findings} == {"username-in-path"}


def test_getattr_is_judged_by_attribute_not_by_name(built, scan):
    _, unsafe = built
    by_id = {c.id: c for c in unsafe}
    assert "unsafe-pickle-io" in rules(scan(by_id["U06"].path))   # getattr(x, "__init__")
    assert not scan(by_id["B03"].path).findings                    # getattr(ultralytics Detect, "forward") is routine


def test_loader_on_analysable_bytes_is_not_flagged_but_its_payload_is_scanned(built, scan, tmp_path):
    _, unsafe = built
    by_id = {c.id: c for c in unsafe}
    assert not rules(scan(by_id["B04"].path))                      # tensor-by-value: payload is a benign torch zip
    assert "unsafe-pickle-nested" in rules(scan(by_id["U17"].path))  # loader on opaque bytes


def test_malicious_payload_inside_loader_bytes_is_found(tmp_path, scan):
    import io, zipfile, struct
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("archive/data.pkl", fx.pickle_proto4_system())
    blob = inner.getvalue()
    data = (b"\x80\x04" + fx._su("torch.storage") + fx._su("_load_from_bytes") + b"\x93\x94B" + struct.pack("<I", len(blob)) + blob
            + b"\x85\x94R\x94.")
    res = scan(write(tmp_path, "t.pkl", data))
    hit = [f for f in res.findings if f.rule_id == "unsafe-pickle-exec"]
    assert hit and "!/bytes@" in hit[0].file


def _gt(attr, **kw):
    return fx._getattr_call(attr, **kw)


@pytest.mark.parametrize("attr", ["os", "_os", "sys", "_subprocess", "builtins", "system", "__globals__", "popen"] + fx.GETATTR_NEW_DANGEROUS)
def test_getattr_walks_to_dangerous_names_are_high(tmp_path, scan, attr):
    res = scan(write(tmp_path, "g.pkl", _gt(attr)))
    hits = [f for f in res.findings if f.rule_id == "unsafe-pickle-io"]
    assert hits and hits[0].severity.name == "HIGH"


@pytest.mark.parametrize("opcode", ["REDUCE", "INST", "OBJ"])
def test_every_call_opcode_is_checked(tmp_path, scan, opcode):
    assert "unsafe-pickle-io" in rules(scan(write(tmp_path, "g.pkl", _gt("execvp", opcode=opcode))))
    assert "pickle-getattr-review" in rules(scan(write(tmp_path, "h.pkl", _gt("frobnicate", opcode=opcode))))
    assert not scan(write(tmp_path, "i.pkl", _gt("forward", opcode=opcode))).findings


def test_allow_list_is_tiny_documented_and_data_derived():
    from limnex_preflight.formats.pickle_scan import DANGEROUS_ATTRS, GETATTR_ALLOWED

    assert GETATTR_ALLOWED == {"forward"}
    assert not (GETATTR_ALLOWED & DANGEROUS_ATTRS)


@pytest.mark.parametrize("attr", ["keys", "weight", "load_state_dict", "frobnicate", "my_custom_hook"])
def test_unknown_getattr_names_are_reviewed_never_silent(tmp_path, scan, attr):
    res = scan(write(tmp_path, "g.pkl", _gt(attr)))
    assert [f.rule_id for f in res.findings] == ["pickle-getattr-review"]
    assert res.findings[0].severity.name == "MEDIUM" and res.findings[0].category == "unsafe-code"


def test_computed_attribute_name_is_high(tmp_path, scan):
    res = scan(write(tmp_path, "g.pkl", fx.pickle_getattr_computed_name()))
    assert [f.rule_id for f in res.findings] == ["unsafe-pickle-io"] and res.findings[0].severity.name == "HIGH"


def test_getattr_handed_to_partial_is_reported(tmp_path, scan):
    res = scan(write(tmp_path, "g.pkl", fx.pickle_getattr_referenced_only()))
    assert "pickle-getattr-review" in rules(res)


def test_yolo_style_checkpoint_has_no_high_finding(built, scan):
    _, unsafe = built
    case = next(c for c in unsafe if c.id == "B03")
    assert scan(case.path).findings == []


def test_loader_through_inst_is_flagged(tmp_path, scan):
    assert "unsafe-pickle-nested" in rules(scan(write(tmp_path, "l.pkl", fx.pickle_loader_via_inst())))


def test_genuine_yolo_shaped_getattr_target_does_not_change_severity(tmp_path, scan):
    # whatever the target class is, only the NAME decides
    res = scan(write(tmp_path, "g.pkl", fx._getattr_call("forward", b"cos\nsystem\n")))
    assert "unsafe-pickle-exec" in rules(res)  # importing os.system is HIGH/CRITICAL on its own


@pytest.mark.parametrize("mod,name,rule", [
    # logging.config RCE gadget family (review finding 2026-10-24: previously unflagged)
    ("logging.config", "listen", "unsafe-pickle-exec"),
    ("logging.config", "fileConfig", "unsafe-pickle-exec"),
    ("logging.config", "dictConfig", "unsafe-pickle-exec"),
    # torch native-code loaders (load / compile a shared object -> machine code)
    ("torch.ops", "load_library", "unsafe-pickle-exec"),
    ("torch.utils.cpp_extension", "load", "unsafe-pickle-exec"),
    ("torch.utils.cpp_extension", "load_inline", "unsafe-pickle-exec"),
    # a nested loader variant alongside torch.jit.load
    ("torch.jit", "_load_for_lite_interpreter", "unsafe-pickle-nested"),
])
def test_deny_list_covers_known_gadgets_found_in_review(mod, name, rule):
    assert classify_global(mod, name) is not None
    assert classify_global(mod, name)[0] == rule


@pytest.mark.parametrize("mod,name", [
    ("copyreg", "_reconstructor"), ("copyreg", "__newobj__"), ("functools", "partial"),
    ("builtins", "bytearray"), ("builtins", "memoryview"), ("warnings", "catch_warnings"),
    ("collections", "OrderedDict"), ("pandas.core.frame", "DataFrame"),
])
def test_benign_machinery_stays_quiet(mod, name):
    assert classify_global(mod, name) is None


@pytest.mark.parametrize("mod,name,rule", [
    # the same callables under their C-module name (review finding 2026-10-24)
    ("_operator", "methodcaller", "unsafe-pickle-io"),
    ("_operator", "attrgetter", "unsafe-pickle-io"),
    # protocol 4 dotted names: the method reached through a class is judged, not only the module
    ("pathlib", "Path.unlink", "unsafe-pickle-io"),
    ("pathlib", "Path.write_text", "unsafe-pickle-io"),
    ("pathlib", "Path.chmod", "unsafe-pickle-io"),
])
def test_aliases_and_dotted_names_are_judged(mod, name, rule):
    assert classify_global(mod, name) is not None and classify_global(mod, name)[0] == rule


@pytest.mark.parametrize("mod,name", [("pathlib", "Path"), ("pathlib", "PurePosixPath"), ("collections", "OrderedDict"),
                                      ("ultralytics.nn.modules.head", "Detect.forward")])
def test_dotted_check_leaves_benign_names_quiet(mod, name):
    assert classify_global(mod, name) is None


class _Computed:
    """Benign: rebuilds as functools.partial(int, base=2)("101") == 5. Its callable is the RESULT of a call."""

    def __reduce__(self):
        import functools
        return functools.partial(int, base=2), ("101",)


def test_a_call_on_a_computed_callable_is_reported_for_review(tmp_path, scan):
    p = write(tmp_path, "computed.pkl", pickle.dumps(_Computed(), protocol=4))
    res = scan(p)
    assert "pickle-computed-callable" in rules(res)
    plain = write(tmp_path, "plain.pkl", pickle.dumps({"weights": [1.0, 2.0], "name": "model"}, protocol=4))
    assert "pickle-computed-callable" not in rules(scan(plain))


# ---- review 2026-11-11, finding 1: the bytes the walker skips as joblib array data are read as instructions by a plain
# pickle loader. Built from ordinary objects by Python's own pickler; the "hidden" part is a harmless method reference
# that the scanner reports for review when it sees it (getattr(int, 'bit_length')).
class _DTypeStandIn:
    def __init__(self, spec):
        self.spec = spec

    def __reduce__(self):
        return _DTypeStandIn, (self.spec,)


class NumpyArrayWrapper:  # only the class NAME matters to the scanner
    pass


def _wrapper_followed_by(block: bytes) -> bytes:
    w = NumpyArrayWrapper()
    w.dtype, w.shape, w.order, w.allow_mmap, w.subclass = _DTypeStandIn("u1"), (len(block),), "C", False, None
    outer = pickle.dumps(w, protocol=2)
    assert outer.endswith(b"b.")  # BUILD, STOP
    return outer[:-1] + block + b"."


def test_content_in_a_skipped_array_block_is_still_analysed(tmp_path, scan):
    marker = pickle.dumps(int.bit_length, protocol=2)
    assert "pickle-getattr-review" in rules(scan(write(tmp_path, "alone.pkl", marker)))
    blob = _wrapper_followed_by(b"0" + marker[2:-1])  # POP, then the marker without its PROTO and STOP
    assert pickle.loads(blob) == int.bit_length         # a plain loader reads the block as instructions
    res = scan(write(tmp_path, "m.joblib", blob))
    hits = [f for f in res.findings if f.rule_id == "pickle-getattr-review"]
    assert hits and "plain pickle loader" in hits[0].location


def test_ordinary_array_bytes_in_the_block_give_no_finding(tmp_path, scan):
    blob = _wrapper_followed_by(bytes(range(256)) * 4)
    res = scan(write(tmp_path, "m.joblib", blob))
    assert not any(r.startswith(("unsafe-pickle", "pickle-")) for r in rules(res)), rules(res)


def test_real_joblib_file_gets_no_extra_findings_from_the_second_reading(tmp_path, scan):
    joblib = pytest.importorskip("joblib")
    np = pytest.importorskip("numpy")
    obj = {"a": np.arange(1000, dtype="float64"), "b": np.random.default_rng(0).random(5000), "c": np.arange(7, dtype="int16")}
    p = tmp_path / "m.joblib"
    joblib.dump(obj, str(p))
    assert not any(r.startswith(("unsafe-pickle", "pickle-")) for r in rules(scan(p)))


# ---- review 2026-11-11, finding 4: a second pickle in the old text protocol (no 0x80 first byte) was only "trailing data"
def test_a_second_pickle_in_protocol_0_is_analysed_not_only_carved(tmp_path, scan):
    first = pickle.dumps({"weights": [0.0, 1.0]}, protocol=2)
    second = pickle.dumps(int.bit_length, protocol=0)   # harmless; reported for review when analysed
    assert second[:1] != b"\x80"
    res = scan(write(tmp_path, "two.pkl", first + second))
    assert "pickle-getattr-review" in rules(res)
    assert "pickle-trailing-data" not in rules(res)


def test_junk_after_a_pickle_is_still_trailing_data_and_nothing_more(tmp_path, scan):
    res = scan(write(tmp_path, "t.pkl", fx.benign_pickle_bytes() + bytes(range(256)) * 4))
    got = rules(res)
    assert "pickle-trailing-data" in got
    assert not got & {"pickle-unresolved-global", "pickle-computed-callable", "pickle-incomplete-analysis"}

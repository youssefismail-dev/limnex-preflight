"""0.1.1: getattr on fastai's own methods is quiet only as exact (module, class, name) pairs on an object made from that
class (CHANGELOG 0.1.1; the list: evaluation/fastai_pairs_from_source.py). Every pickle here is written opcode by opcode and only names
things; fastai is not installed and nothing is ever loaded."""
import pytest

import fixtures as fx
from limnex_preflight.formats.pickle_scan import FASTAI_GETATTR_ALLOWED


def rules(res):
    return {f.rule_id for f in res.findings}


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def glob(module, cls):
    return b"c" + module.encode() + b"\n" + cls.encode() + b"\n"


def made_by(how, module="fastai.data.core", cls="TfmdLists"):
    """Bytes that leave one object of module.cls on the stack, made the way real pickles make it."""
    g = glob(module, cls)
    return {
        "NEWOBJ": g + b")\x81",
        "NEWOBJ_EX": g + b")}\x92",
        "OBJ": b"(" + g + b"o",
        "INST": b"(i" + module.encode() + b"\n" + cls.encode() + b"\n",
        "copyreg._reconstructor": b"ccopyreg\n_reconstructor\n" + g + b"cbuiltins\nobject\nN\x87R",
        "the class itself": g,
    }[how]


def name_bytes(attr):
    return b"X" + len(attr.encode()).to_bytes(4, "little") + attr.encode()


CLASS_ITSELF = {("fastai.vision.core", "PILImage"), ("fastai.torch_core", "TensorImage")}


@pytest.mark.parametrize("module,cls,attr", sorted(FASTAI_GETATTR_ALLOWED))
def test_every_allowed_pair_is_quiet(tmp_path, scan, module, cls, attr):
    how = "the class itself" if (module, cls) in CLASS_ITSELF else "NEWOBJ"
    res = scan(write(tmp_path, "export.pkl", fx._getattr_call(attr, made_by(how, module, cls))))
    assert not res.findings


@pytest.mark.parametrize("how", ["NEWOBJ", "NEWOBJ_EX", "OBJ", "INST", "copyreg._reconstructor", "the class itself"])
@pytest.mark.parametrize("call", ["REDUCE", "INST", "OBJ"])
def test_each_way_of_naming_the_class_is_quiet(tmp_path, scan, how, call):
    target = made_by(how)
    res = scan(write(tmp_path, "export.pkl", fx._getattr_call("dataloaders", target, opcode=call)))
    assert not res.findings


def test_the_object_reached_through_the_memo_is_quiet(tmp_path, scan):
    """Real exports make the object first, remember it (BINPUT), and take its method later (BINGET), before its state."""
    data = (b"\x80\x02" + made_by("NEWOBJ") + b"q\x00" + b"0" + b"cbuiltins\ngetattr\n" + b"h\x00"
            + name_bytes("dataloaders") + b"\x86R.")
    assert not scan(write(tmp_path, "export.pkl", data)).findings


def test_protocol_4_stack_global_class_is_quiet(tmp_path, scan):
    target = b"\x8c\x12fastai.vision.core\x8c\x08PILImage\x93"
    assert not scan(write(tmp_path, "export.pkl", fx._getattr_call("create", target))).findings


@pytest.mark.parametrize("module", ["__main__", "fastai_x.data.core", "fastaix.data.core", "mylib.data.core",
                                    "fastai.data", "fastai.data.core.extra"])
def test_the_same_pair_outside_fastai_or_elsewhere_in_it_is_still_reviewed(tmp_path, scan, module):
    res = scan(write(tmp_path, "export.pkl", fx._getattr_call("dataloaders", made_by("NEWOBJ", module=module))))
    assert rules(res) == {"pickle-getattr-review"}


@pytest.mark.parametrize("cls,attr", [("TfmdLists", "frobnicate"), ("Datasets", "before_call"), ("TfmdLists", "create")])
def test_a_name_that_is_not_on_its_class_list_is_still_reviewed(tmp_path, scan, cls, attr):
    res = scan(write(tmp_path, "export.pkl", fx._getattr_call(attr, made_by("NEWOBJ", cls=cls))))
    assert rules(res) == {"pickle-getattr-review"}


@pytest.mark.parametrize("attr", ["system", "save", "load", "open", "__globals__", "os", "_subprocess"])
def test_dangerous_names_stay_high_even_on_a_fastai_class(tmp_path, scan, attr):
    res = scan(write(tmp_path, "export.pkl", fx._getattr_call(attr, made_by("NEWOBJ"))))
    hits = [f for f in res.findings if f.rule_id == "unsafe-pickle-io"]
    assert hits and hits[0].severity.name == "HIGH"


def test_calling_the_allowed_method_is_still_flagged(tmp_path, scan):
    data = b"\x80\x02cbuiltins\ngetattr\n" + made_by("NEWOBJ") + name_bytes("dataloaders") + b"\x86R)R."
    assert rules(scan(write(tmp_path, "export.pkl", data))) == {"pickle-computed-callable"}


@pytest.mark.parametrize("target", [
    glob("fastai.data.core", "TfmdLists") + b")R",        # made by a plain call of the class: not a naming opcode
    glob("fastai.data.core", "some_factory") + b")R",     # whatever a function returned
    b"h\x07",                                           # an unknown memo entry
])
def test_an_object_of_unknown_class_is_still_reviewed(tmp_path, scan, target):
    res = scan(write(tmp_path, "export.pkl", fx._getattr_call("dataloaders", target)))
    assert rules(res) == {"pickle-getattr-review"}


RECLASS_TO = glob("somepkg", "Builder")
SLOT_RECLASS = b"N}" + name_bytes("__class__") + RECLASS_TO + b"s\x86"          # (None, {'__class__': Builder})
MAIN_RECLASS = b"}" + name_bytes("__class__") + RECLASS_TO + b"s"               # {'__class__': Builder}
SLOT_UNSEEN = b"N" + glob("somepkg", "make_slots") + b")R\x86"                    # (None, <what a call returned>)
# tester of 0.1.1, L1: a state the scan cannot see at all may be a (state, slotstate) pair, so it counts as well
WHOLE_UNSEEN = glob("somepkg", "make_state") + b")R"                               # <what a call returned>
MAIN_UNSEEN = glob("somepkg", "make_main") + b")RN\x86"                            # (<what a call returned>, None)
GLOBAL_STATE = glob("somepkg", "STATE")                                            # a module attribute
MEMO_STATE = b"h\x07"                                                              # an unknown memo entry


@pytest.mark.parametrize("state", [SLOT_RECLASS, MAIN_RECLASS, SLOT_UNSEEN, WHOLE_UNSEEN, MAIN_UNSEEN, GLOBAL_STATE,
                                   MEMO_STATE],
                         ids=["slot-state", "main-state", "unseen", "whole-unseen", "main-unseen", "global", "memo"])
@pytest.mark.parametrize("how", ["NEWOBJ", "copyreg._reconstructor"])
def test_an_object_whose_state_may_change_its_class_is_reviewed(tmp_path, scan, how, state):
    """Review of 0.1.1, finding 3: NEWOBJ Flip, then BUILD with slot state {'__class__': X}, made getattr quiet on an
    object that is really an X. Its owner is now unknown, so the getattr is reviewed as in 0.1.0."""
    target = made_by(how, "fastai.vision.augment", "Flip") + state + b"b"
    res = scan(write(tmp_path, "export.pkl", fx._getattr_call("before_call", target)))
    assert "pickle-getattr-review" in rules(res)


@pytest.mark.parametrize("state", [b"}" + name_bytes("p") + b"K\x01s",                                 # {'p': 1}
                                   b"N}" + name_bytes("p") + b"K\x01s\x86",                            # (None, {'p': 1})
                                   b"}" + name_bytes("p") + b"K\x01sN\x86"])                           # ({'p': 1}, None)
def test_an_ordinary_state_keeps_the_pair_quiet(tmp_path, scan, state):
    target = made_by("NEWOBJ", "fastai.vision.augment", "Flip") + state + b"b"
    assert not scan(write(tmp_path, "export.pkl", fx._getattr_call("before_call", target))).findings


def test_the_list_names_only_fastai_and_no_dangerous_name():
    from limnex_preflight.formats.pickle_scan import DANGEROUS_ATTRS, getattr_name_is_high
    for module, cls, attr in FASTAI_GETATTR_ALLOWED:
        assert module == "fastai" or module.startswith("fastai."), module
        assert attr not in DANGEROUS_ATTRS and not (attr.startswith("__") and attr.endswith("__")), attr
        assert not getattr_name_is_high(attr), attr


# -- pairs from fastai's source, by method and inheritance (evaluation/fastai_pairs_from_source.py) -----------------------
# The families the independent review of 1e5b385 found still failing (finding 2), made the way real exports make them:
# Hook and ProgressCallback objects by NEWOBJ (a bound method kept on them); PILMask, TensorPoint and FilteredBase taken
# as the class itself (ImageBlock/MaskBlock/PointBlock keep `cls.create`; protocol 2 writes getattr(class, name)).
REVIEW_FAMILIES = [
    ("fastai.callback.hook", "Hook", "hook_fn", "NEWOBJ"),                       # every U-Net (DynamicUnet's hooks)
    ("fastai.vision.core", "PILMask", "create", "the class itself"),             # every segmentation DataBlock
    ("fastai.collab", "TabularCollab", "dataloaders", "NEWOBJ"),                 # every collaborative-filtering learner
    ("fastai.vision.core", "TensorPoint", "create", "the class itself"),         # keypoint learners
    ("fastai.callback.progress", "ProgressCallback", "_write_stats", "NEWOBJ"),  # an export made during fit
    ("fastai.data.core", "FilteredBase", "dataloaders", "the class itself"),     # first check set (section 9)
]


@pytest.mark.parametrize("module,cls,attr,how", REVIEW_FAMILIES, ids=[f"{c}.{a}" for _, c, a, _ in REVIEW_FAMILIES])
def test_the_families_the_review_found_are_quiet(tmp_path, scan, module, cls, attr, how):
    res = scan(write(tmp_path, "export.pkl", fx._getattr_call(attr, made_by(how, module, cls))))
    assert not res.findings


@pytest.mark.parametrize("module,cls,attr", [
    ("fastai.callback.hook", "Hook", "hook_func"),        # data the Hook keeps (the user's function), not its method
    ("fastai.callback.hook", "Hook", "stored"),
    ("fastai.vision.core", "PILMask", "_open_args"),       # a class attribute, not a method
    ("fastai.collab", "TabularCollab", "procs"),
    ("fastai.tabular.core", "Tabular", "x_names"),         # a method made a property by fastcore's properties(...)
    ("__main__", "Hook", "hook_fn"),                       # a user's own class of the same name
    ("fastai.vision.core", "PILMask", "hook_fn"),          # a real method name, on a class that does not have it
])
def test_names_that_are_not_fastai_methods_of_that_class_are_still_reviewed(tmp_path, scan, module, cls, attr):
    res = scan(write(tmp_path, "export.pkl", fx._getattr_call(attr, made_by("NEWOBJ", module, cls))))
    assert rules(res) == {"pickle-getattr-review"}


def test_the_hand_read_pairs_and_the_reviews_pairs_are_all_on_the_list():
    """Section 10 point 5: the rule reproduced all 11 hand-read pairs and the review's six; they must stay listed."""
    from limnex_preflight.formats.fastai_pairs import FASTAI_SOURCE_PAIRS
    from limnex_preflight.formats.pickle_scan import _FASTAI_STUDY_PAIRS
    assert _FASTAI_STUDY_PAIRS <= FASTAI_SOURCE_PAIRS
    assert {(m, c, a) for m, c, a, _ in REVIEW_FAMILIES} <= FASTAI_SOURCE_PAIRS


# tester of 0.1.1, H1: from Python 3.13, pickle names the class of a function taken from a class through the function's
# own module; fastcore's @patch keeps the module it was written in. So TensorImage.lighting, patched in
# fastai.vision.augment, is written getattr(fastai.vision.augment.TensorImage, 'lighting') (rule 3 of the generator).
PY313_NAMES = [
    ("fastai.vision.augment", "TensorImage", "lighting"),   # seen by the tester in two fastai 2.8 vision exports
    ("fastai.vision.augment", "TensorImage", "hsv"),
    ("fastai.callback.schedule", "Learner", "fit_one_cycle"),
    ("fastai.callback.data", "Datasets", "weighted_dataloaders"),
]


@pytest.mark.parametrize("module,cls,attr", PY313_NAMES, ids=[f"{m}.{c}.{a}" for m, c, a in PY313_NAMES])
def test_a_patched_method_named_through_its_own_module_is_quiet(tmp_path, scan, module, cls, attr):
    assert (module, cls, attr) in FASTAI_GETATTR_ALLOWED
    res = scan(write(tmp_path, "export.pkl", fx._getattr_call(attr, made_by("the class itself", module, cls))))
    assert not res.findings


def test_the_second_name_is_only_for_the_class_the_method_was_patched_onto():
    """A function's qualified name holds the class it was patched onto, never a subclass, and only patched methods
    get a second module: plain methods and every 0.1.1 pair keep exactly the names they had."""
    second = {(m, c, a) for m, c, a in FASTAI_GETATTR_ALLOWED
              if (c, a) in {(c2, a2) for _, c2, a2 in PY313_NAMES} and m in {m2 for m2, _, _ in PY313_NAMES}}
    assert second == set(PY313_NAMES)
    assert ("fastai.vision.augment", "TensorMask", "lighting") not in FASTAI_GETATTR_ALLOWED
    assert ("fastai.vision.augment", "Flip", "before_call") in FASTAI_GETATTR_ALLOWED      # defined there: one name


def test_old_exports_keep_before_call_although_fastai_2_8_stopped_storing_it():
    """fastai 2.7 kept `self.before_call` bound on every random transform; 2.8.12 only keeps one given by the user. The
    list is the union of both versions, so 2.7 exports (most of the Hub) stay quiet."""
    assert ("fastai.vision.augment", "Zoom", "before_call") in FASTAI_GETATTR_ALLOWED
    assert ("fastai.vision.augment", "RandTransform", "before_call") in FASTAI_GETATTR_ALLOWED


# -- fastai 2.8: cloudpickle's bound methods (amendment 3, tester of 0.1.1, B2) -------------------------------------------
# fastai 2.8 exports with cloudpickle, which writes a bound method as _builtin_type('MethodType')(<function>, <object>):
# a call on what a call returned. These are the bindings the 2.8 study saw, written the three ways the function appears.
CP = "cloudpickle.cloudpickle"
BINDINGS = [
    (("fastai.data.core", "FilteredBase", "dataloaders"), ("fastai.data.core", "TfmdLists")),
    (("fastai.data.core", "FilteredBase", "dataloaders"), ("fastai.data.core", "Datasets")),
    (("fastai.data.core", "FilteredBase", "dataloaders"), ("fastai.tabular.core", "TabularPandas")),
    (("fastai.vision.augment", "Resize", "before_call"), ("fastai.vision.augment", "Resize")),
    (("fastai.vision.augment", "SpaceTfm", "before_call"), ("fastai.vision.augment", "Brightness")),
    (("fastai.vision.augment", "AffineCoordTfm", "before_call"), ("fastai.vision.augment", "Flip")),
    (("fastai.vision.augment", "AffineCoordTfm", "before_call"), ("fastai.vision.augment", "Dihedral")),
    (("fastai.vision.augment", "AffineCoordTfm", "before_call"), ("fastai.vision.augment", "Rotate")),
    (("fastai.vision.augment", "RandomResizedCropGPU", "before_call"), ("fastai.vision.augment", "RandomResizedCropGPU")),
    (("fastai.text.models.awdlstm", "WeightDropout", "_do_nothing"), ("fastai.text.models.awdlstm", "WeightDropout")),
    (("fastai.callback.hook", "Hook", "hook_fn"), ("fastai.callback.hook", "Hook")),
    (("fastai.callback.progress", "ProgressCallback", "_write_stats"), ("fastai.callback.progress", "ProgressCallback")),
]


def builtin_type(name, module=CP):
    return glob(module, "_builtin_type") + name_bytes(name) + b"\x85R"


def function(how, module, cls, attr):
    """The function, named through its class: a dotted GLOBAL, a STACK_GLOBAL (protocol 4+), or getattr(C, 'm')."""
    return {
        "dotted GLOBAL": glob(module, f"{cls}.{attr}"),
        "STACK_GLOBAL": name_bytes(module) + name_bytes(f"{cls}.{attr}") + b"\x93",
        "getattr": b"c__builtin__\ngetattr\n" + glob(module, cls) + name_bytes(attr) + b"\x86R",
    }[how]


def bound(fn, obj, maker=None):
    return b"\x80\x04" + (maker if maker is not None else builtin_type("MethodType")) + fn + obj + b"\x86R"


FORMS = ["dotted GLOBAL", "STACK_GLOBAL", "getattr"]


@pytest.mark.parametrize("how", FORMS)
@pytest.mark.parametrize("func,owner", BINDINGS, ids=[f"{f[1]}.{f[2]}-on-{o[1]}" for f, o in BINDINGS])
def test_cloudpickle_binding_fastais_method_to_its_object_is_quiet(tmp_path, scan, func, owner, how):
    assert func in FASTAI_GETATTR_ALLOWED and (owner[0], owner[1], func[2]) in FASTAI_GETATTR_ALLOWED
    data = bound(function(how, *func), made_by("NEWOBJ", *owner)) + b"."
    assert not scan(write(tmp_path, "export.pkl", data)).findings


def test_bindings_sharing_one_methodtype_through_the_memo_are_quiet(tmp_path, scan):
    mt = builtin_type("MethodType")
    one = mt + b"q\x00" + function("dotted GLOBAL", "fastai.data.core", "FilteredBase", "dataloaders") \
        + made_by("NEWOBJ", "fastai.data.core", "TfmdLists") + b"\x86R"
    two = b"h\x00" + function("dotted GLOBAL", "fastai.data.core", "FilteredBase", "dataloaders") \
        + made_by("NEWOBJ", "fastai.data.core", "Datasets") + b"\x86R"
    assert not scan(write(tmp_path, "export.pkl", b"\x80\x04" + one + two + b"\x86.")).findings


GOOD_FN = function("dotted GLOBAL", "fastai.data.core", "FilteredBase", "dataloaders")
GOOD_OBJ = made_by("NEWOBJ", "fastai.data.core", "TfmdLists")
STILL_REVIEWED = {
    "another builtin type": bound(GOOD_FN, GOOD_OBJ, builtin_type("FunctionType")),
    "a computed type name": bound(GOOD_FN, GOOD_OBJ, glob(CP, "_builtin_type") + glob("somepkg", "pick") + b")R\x85R"),
    "a look-alike module": bound(GOOD_FN, GOOD_OBJ, builtin_type("MethodType", "cloudpickle.cloudpickle_fast")),
    "an outside look-alike": bound(GOOD_FN, GOOD_OBJ, builtin_type("MethodType", "somepkg")),
    "object class not listed for the name": bound(GOOD_FN, made_by("NEWOBJ", "fastai.vision.augment", "Flip")),
    "a user's class": bound(GOOD_FN, made_by("NEWOBJ", "__main__", "TfmdLists")),
    "function not listed": bound(function("dotted GLOBAL", "fastai.data.core", "TfmdLists", "frobnicate"), GOOD_OBJ),
    "function outside fastai": bound(function("dotted GLOBAL", "spacy.language", "Language", "pipe"),
                                     made_by("NEWOBJ", "spacy.lang.en", "English")),
    "function stored as code": bound(glob(CP, "_make_function") + builtin_type("CodeType") + b"\x85R", GOOD_OBJ),
    "a deeper dotted name": bound(glob("fastai.data.core", "FilteredBase.dataloaders.__func__"), GOOD_OBJ),
    "a HIGH name": bound(function("dotted GLOBAL", "fastai.data.core", "TfmdLists", "__reduce_ex__"), GOOD_OBJ),
    "object of unknown class": bound(GOOD_FN, glob("fastai.data.core", "some_factory") + b")R"),
    "a re-classed object": bound(GOOD_FN, GOOD_OBJ + SLOT_RECLASS + b"b"),
    "three arguments": b"\x80\x04" + builtin_type("MethodType") + GOOD_FN + GOOD_OBJ + b"N\x87R",
}


@pytest.mark.parametrize("case", sorted(STILL_REVIEWED))
def test_every_other_methodtype_call_is_still_reported(tmp_path, scan, case):
    res = scan(write(tmp_path, "export.pkl", STILL_REVIEWED[case] + b"."))
    assert "pickle-computed-callable" in rules(res)


def test_calling_the_bound_method_is_still_reported(tmp_path, scan):
    data = bound(GOOD_FN, GOOD_OBJ) + b")R."
    res = scan(write(tmp_path, "export.pkl", data))
    assert [f.rule_id for f in res.findings] == ["pickle-computed-callable"]       # the call; the binding is quiet


def test_fastais_create_stored_as_bytecode_is_still_reported(tmp_path, scan):
    """2.8 image exports: ImageBlock keeps cls.create, a classmethod cloudpickle cannot name, so it stores its code.
    A static scan cannot tell fastai's bytecode from anyone else's, so both calls stay reported."""
    code = builtin_type("CodeType") + b")R"                                   # CodeType(...): a computed call
    data = bound(glob(CP, "_make_function") + b"(" + code + b"tR", glob("fastai.vision.core", "PILImage")) + b"."
    res = scan(write(tmp_path, "export.pkl", data))
    assert [f.rule_id for f in res.findings].count("pickle-computed-callable") == 2

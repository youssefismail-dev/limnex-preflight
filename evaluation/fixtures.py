"""Builds the evaluation / test fixtures: small model files of every supported format.

Every planted value is FAKE and is assembled from fragments at run time, so that no secret-shaped
literal exists in this repository. Values contain words like FAKE/fake on purpose, except the
generic high-entropy sample, which is a random-looking string that is not a credential for anything.

Nothing here ever *loads* a model file; files are only written.
"""

from __future__ import annotations

import base64
import io
import json
import os
import pickle
import re
import struct
import sys
import zipfile
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------------------
# Planted values
# --------------------------------------------------------------------------------------



_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


def _zput(z: zipfile.ZipFile, name: str, data) -> None:
    """ZipFile.writestr with a fixed time, system and mode. writestr(name, ...) stamps the current local time and the OS
    that wrote the entry, so the same fixture differed from run to run and between Windows and Linux (pre-release test 3,
    R6: the zip-based fuzz seeds changed every two seconds)."""
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIME)
    info.compress_type = z.compression
    info.create_system = 3
    info.external_attr = 0o600 << 16
    z.writestr(info, data)

@dataclass(frozen=True)
class Planted:
    label: str  # short id, used as the neutral field name "meta_<label>"
    kind: str  # "secret" | "personal"
    text: str  # the text planted into files (may include context such as "KAGGLE_KEY=")
    value: str  # the exact value the scanner fingerprints
    rule: str  # rule id expected to fire


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


_F28 = "FAKEfakeFAKEfakeFAKEfakeFAKE"  # 28 chars
_F16 = "FAKEfakeFAKEfake"  # 16 chars
_HEX32A = "0123456789abcdef0123456789abcdef"
_HEX32B = "fedcba9876543210fedcba9876543210"


def _planted() -> List[Planted]:
    p: List[Planted] = []

    def add(label, kind, rule, value, text=None):
        p.append(Planted(label, kind, text if text is not None else value, value, rule))

    # --- secrets ---
    add("aws_key", "secret", "aws-access-key-id", "AKIA" + "IOSFODNN7" + "EXAMPLE")  # documented AWS example
    secret = "wJalrXUtnFEMI/K7MDENG/" + "bPxRfiCYEXAMPLEKEY"  # documented AWS example
    add("aws_secret", "secret", "aws-secret-access-key", secret, "aws_secret_access_key=" + secret)
    add("gcp_key", "secret", "gcp-api-key", "AIza" + _F28 + "0123456")
    add("gcp_oauth", "secret", "gcp-oauth-client-secret", "GOCSPX" + "-" + _F28)
    azk = ("FAKE" * 21) + "fa" + "=="
    add("azure_key", "secret", "azure-storage-account-key", azk, "DefaultEndpointsProtocol=https;AccountName=fixture;AccountKey=" + azk)
    sas = "FAKEfakeFAKEfakeFAKEfakeFAKEfake0123456789%3D"
    add("azure_sas", "secret", "azure-sas-token", sas, "https://fixture.blob.core.windows.net/c?sv=2022-11-02&ss=b&srt=co&sp=rwdlac&sig=" + sas)
    add("azure_ad", "secret", "azure-ad-client-secret", "abc" + "8Q~" + _F28 + "012")
    add("github", "secret", "github-token", "ghp" + "_" + _F28 + "01234567")
    add("github_pat", "secret", "github-fine-grained-pat", "github_pat" + "_" + _F16 + "012345" + "_" + (_F28 * 2) + "012")
    add("gitlab", "secret", "gitlab-token", "glpat" + "-" + _F16 + "0123")
    add("hf", "secret", "huggingface-token", "hf" + "_" + _F28 + "012345")
    add("openai", "secret", "openai-api-key", "sk" + "-proj-" + _F28 + _F16 + "0123")
    add("anthropic", "secret", "anthropic-api-key", "sk-ant" + "-api03-" + _F16 * 5)
    add("kaggle_env", "secret", "kaggle-api-key", _HEX32A, "KAGGLE_KEY=" + _HEX32A)
    add("slack", "secret", "slack-token", "xoxb" + "-123456789012-1234567890123-" + _F16 + "fake0123")
    add("stripe", "secret", "stripe-secret-key", "sk_live" + "_" + _F28 + "0123")
    add("twilio", "secret", "twilio-api-key", "SK" + _HEX32B)
    pem_body = "\n".join([_F28 * 2 + "FAKEFAKE"] * 3)
    pem = "-----BEGIN RSA " + "PRIVATE KEY-----\n" + pem_body + "\n-----END RSA " + "PRIVATE KEY-----"
    header = "-----BEGIN RSA " + "PRIVATE KEY-----"
    add("pem", "secret", "private-key-pem", header + re.sub(r"\\n|\\r|\s", "", pem[len(header):len(header) + 400])[:128], pem)
    jwt = _b64url(b'{"alg":"HS256","typ":"JWT"}') + "." + _b64url(b'{"sub":"fake-limnex-fixture"}') + "." + "FAKEsignatureFAKEsignature"
    add("jwt", "secret", "jwt", jwt)
    add("url_pw", "secret", "password-in-url", "Fakepass-0123", "postgres://fixture_user:" + "Fakepass-0123" + "@db-host.fixture.test:5432/ml")
    add("generic", "secret", "generic-credential", "Xk9-Passw0rd-4471", "db_password=Xk9-Passw0rd-4471")
    add("entropy", "secret", "high-entropy-string", "q8Zr3KmX7vLp2NwYt5HcJd9BfGs4AeU6oIiR1PzV")
    # --- personal data ---
    add("win_user", "personal", "username-in-path", "fx_alice_ml", "C:\\Users\\fx_alice_ml\\Documents\\train.py")
    add("linux_user", "personal", "username-in-path", "fx_bob_research", "/home/fx_bob_research/experiments/run1/config.yaml")
    add("mac_user", "personal", "username-in-path", "fx.carol.nair", "/Users/fx.carol.nair/Library/data/set.csv")
    add("kaggle_owner", "personal", "username-in-path", "fx_dave_kaggle", "/kaggle/input/datasets/fx_dave_kaggle/xray-images/img_001.png")
    drive = "/content/drive/MyDrive/fx_thesis_project/ckpt_final.pt"
    add("colab_drive", "personal", "platform-path", drive)
    add("email", "personal", "email-address", "fx.erin.person@gmail.com")
    add("host", "personal", "hostname", "gpu-node-07.corp.internal")
    add("ip_private", "personal", "ip-address", "10.23.45.67")
    add("ip_public", "personal", "ip-address", "8.8.4.4", "remote_host=8.8.4.4")
    return p


PLANTED: List[Planted] = _planted()
BY_LABEL: Dict[str, Planted] = {p.label: p for p in PLANTED}

# Harmless-looking but "interesting" content for the clean twins: must produce NO findings.
CLEAN_VALUES = [
    "Linear(in_features=768, out_features=3072)", "encoder.layer.11.attention.self.query.weight", "<s>", "</s>", "[CLS]",
    "contact: user@example.com", "/home/ubuntu/work/model", "C:\\Users\\Public\\Documents", "http://localhost:8080/v1",
    "sha256:" + "ab" * 32, "d41d8cd98f00b204e9800998ecf8427e", "123e4567-e89b-12d3-a456-426614174000",
    "version 1.2.3.4", "bert-base-uncased", "transformers==4.30.2", "git@github.com:org/repo.git", "torch.float32",
    "pad_token: <pad>", "max_position_embeddings: 512", "Apache License, Version 2.0", "token_type_ids",
    "api_key: YOUR_API_KEY_HERE", "password = <your-password>",
    # shapes that produced false alarms on real public models (see evaluation/results/hf_review.md), with made-up names
    "https://huggingface.co/production/uploads/5f0e1d2c3b4a59687c8d9e0f/Qm7kLp2xRt-VwZ",
    "wav2vec2-xls-r-300m-xx-250/checkpoint-16800/pytorch_model", "1/attention/self/Sqrt_2_output_0_CUDAExecutionProvider",
    "Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/lib (3.10.0.2)", "client = Client(api_key=os.environ[\"TOKEN\"])",
    "Resolving host.example... 185.199.110.133, 185.199.111.133", "someuser/TinyLM2-135M-Instruct-GGUF",
    "Build info:\n    Author: Some Dev <some.dev@corp-tools.io>\n    Summary: Release 1.2.3. commit_hash:0123abcd",
    "httpwwwexamplecomvideopartnersemailindexhtmlurlvideotech20080805",
]

# Unsafe pickle payloads. They are harmless strings and are never executed.
CMD = "echo limnex-fixture"


def _op(name: bytes) -> bytes:
    return name


def pickle_proto0_system() -> bytes:
    return b"cos\nsystem\n(S'" + CMD.encode() + b"'\ntR."


def pickle_proto2_subprocess() -> bytes:
    return b"\x80\x02csubprocess\nPopen\nX\x13\x00\x00\x00echo limnex-fixture\x85R."


def pickle_proto0_eval() -> bytes:
    return b"c__builtin__\neval\n(S'1+1'\ntR."


def _su(text: str) -> bytes:
    b = text.encode()
    return b"\x8c" + bytes([len(b)]) + b + b"\x94"  # SHORT_BINUNICODE + MEMOIZE


def pickle_proto4_system() -> bytes:
    """PROTO 4, then os.system(<cmd>) built with STACK_GLOBAL + REDUCE (what pickle.dumps emits)."""
    return b"\x80\x04" + _su("os") + _su("system") + b"\x93\x94" + _su(CMD) + b"\x85\x94R\x94."


def pickle_memo_indirect_exec() -> bytes:
    """exec via STACK_GLOBAL whose operands come from the memo, with unrelated strings in between."""
    return (b"\x80\x04"
            b"\x8c\x08builtins\x94"          # 0: 'builtins'
            b"\x8c\x04exec\x94"              # 1: 'exec'
            b"\x8c\x0bjunk-string\x94"       # 2: decoy pushed last
            b"0"                             # POP decoy
            b"h\x00h\x01\x93\x94"            # BINGET 0, BINGET 1, STACK_GLOBAL
            b"\x8c\x0bprint('hi')\x94\x85\x94R\x94.")


def pickle_getattr() -> bytes:
    """getattr(collections.OrderedDict, '__init__'): the first step of an attribute-walking gadget chain."""
    return b"\x80\x02cbuiltins\ngetattr\nccollections\nOrderedDict\nX\x08\x00\x00\x00__init__\x86R."


def _getattr_call(attr: str, target: bytes = b"ccollections\nOrderedDict\n", opcode: str = "REDUCE") -> bytes:
    """A pickle that calls builtins.getattr(<target>, <attr>) through REDUCE, INST or OBJ."""
    name = b"X" + len(attr.encode()).to_bytes(4, "little") + attr.encode()
    if opcode == "REDUCE":
        return b"\x80\x02cbuiltins\ngetattr\n" + target + name + b"\x86R."
    if opcode == "INST":
        return b"\x80\x02(" + target + name + b"ibuiltins\ngetattr\n."
    if opcode == "OBJ":
        return b"\x80\x02(cbuiltins\ngetattr\n" + target + name + b"o."
    raise ValueError(opcode)


def pickle_getattr_benign() -> bytes:
    """getattr(ultralytics.nn.modules.head.Detect, 'forward'): how pickle stores an unbound method, as YOLO-style
    detection checkpoints commonly do."""
    return _getattr_call("forward", b"cultralytics.nn.modules.head\nDetect\n").replace(b"cbuiltins\ngetattr", b"c__builtin__\ngetattr")


def pickle_getattr_computed_name() -> bytes:
    """getattr(OrderedDict, str('abc')): the attribute name is computed, not a constant."""
    return (b"\x80\x02cbuiltins\ngetattr\nccollections\nOrderedDict\ncbuiltins\nstr\nX\x03\x00\x00\x00abc\x85R\x86R.")


def pickle_getattr_referenced_only() -> bytes:
    """functools.partial(getattr): getattr is handed over, never called directly."""
    return b"\x80\x02cfunctools\npartial\ncbuiltins\ngetattr\n\x85R."


def pickle_loader_via_inst() -> bytes:
    """torch.load(<opaque bytes>) reached through INST instead of REDUCE."""
    return b"\x80\x02(C\x08NOTAPICKitorch\nload\n."


# Attribute names a name-list could easily forget; every one must be HIGH. (Newly covered after review.)
GETATTR_NEW_DANGEROUS = [
    "execvp", "execlp", "execle", "execvpe", "spawnlp", "spawnvp", "spawnve", "spawnvpe", "posix_spawn", "posix_spawnp",
    "run_path", "run_module", "exec_module", "write_text", "write_bytes", "write",
]


def pickle_torch_load_from_bytes_unanalysable() -> bytes:
    """torch.storage._load_from_bytes(<bytes that are neither a zip nor a pickle>)"""
    return b"\x80\x02ctorch.storage\n_load_from_bytes\nC\x08NOTAPICK\x85R."


def benign_pickle_bytes(planted: Optional[Dict[str, str]] = None) -> bytes:
    obj = {"weights_note": "layer.0.weight", "config": {"hidden": 8, "act": "gelu"}, "items": list(range(5))}
    if planted:
        obj.update(planted)
    return pickle.dumps(obj, protocol=4)


def two_pickles_second_evil() -> bytes:
    return benign_pickle_bytes() + pickle_proto2_subprocess()


def corrupt_then_evil() -> bytes:
    """A benign pickle head, then a bad opcode byte, then a payload: the parser cannot continue."""
    return benign_pickle_bytes()[:-1] + b"\xff\xfe" + pickle_proto0_system()


# --------------------------------------------------------------------------------------
# Format writers. Each writes ONE file and returns nothing. ``items`` maps meta_<label> -> text.
# --------------------------------------------------------------------------------------
Items = Dict[str, str]


def _items(labels: Optional[List[str]] = None) -> Items:
    chosen = PLANTED if labels is None else [BY_LABEL[l] for l in labels]
    return {"meta_" + p.label: p.text for p in chosen}


def clean_items() -> Items:
    return {f"meta_clean_{i:02d}": v for i, v in enumerate(CLEAN_VALUES)}


def w_pickle(path: str, items: Items) -> None:
    with open(path, "wb") as f:
        pickle.dump({"state": "model.layer.0", "dims": (3, 4), **items}, f, protocol=4)


def w_pickle_proto2(path: str, items: Items) -> None:
    with open(path, "wb") as f:
        pickle.dump({"state": "model.layer.0", **items}, f, protocol=2)


def w_torch_zip(path: str, items: Items) -> None:
    """Handwritten PyTorch zip layout (archive/data.pkl, version, byteorder, data/0 storage)."""
    data = pickle.dumps({"state_dict": {"layer.weight": "storage-ref"}, "args": items}, protocol=2)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        _zput(z, "archive/data.pkl", data)
        _zput(z, "archive/byteorder", "little")
        _zput(z, "archive/version", "3\n")
        _zput(z, "archive/data/0", os.urandom(64))


def w_torch_legacy(path: str, items: Items) -> None:
    """Legacy (pre-1.6) torch.save layout: magic, protocol, sys_info, main object, storage keys, raw storages."""
    with open(path, "wb") as f:
        pickle.dump(0x1950A86A20F9469CFC6C, f, protocol=2)
        pickle.dump(1001, f, protocol=2)
        pickle.dump({"protocol_version": 1001, "little_endian": True, "type_sizes": {"short": 2, "int": 4, "long": 8}}, f, protocol=2)
        pickle.dump({"state_dict": {"w": "0"}, "args": items}, f, protocol=2)
        pickle.dump(["0"], f, protocol=2)
        f.write(struct.pack("<q", 4) + os.urandom(16))


def w_joblib(path: str, items: Items, compress=0) -> None:
    import joblib
    import numpy as np

    obj = {"coef": np.arange(12, dtype="float64").reshape(3, 4), "idx": np.arange(5, dtype="int32"), "info": items}
    joblib.dump(obj, path, compress=compress)


def w_joblib_zlib(path: str, items: Items) -> None:
    w_joblib(path, items, compress=3)


def w_npy_object(path: str, items: Items) -> None:
    import numpy as np

    arr = np.empty(1, dtype=object)
    arr[0] = items
    np.save(path, arr, allow_pickle=True)


def w_npy_strings(path: str, items: Items) -> None:
    import numpy as np

    np.save(path, np.array(list(items.values()), dtype="U"))


def w_npz(path: str, items: Items) -> None:
    import numpy as np

    arr = np.empty(1, dtype=object)
    arr[0] = items
    np.savez(path, weights=np.arange(6, dtype="float32"), notes=np.array(list(items.values()), dtype="U"), objs=arr)


def w_safetensors(path: str, items: Items) -> None:
    header = {"__metadata__": dict(items, format="pt"), "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h + b"\x00" * 8)


def w_onnx(path: str, items: Items) -> None:
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["x"], ["y"], name="identity0", doc_string=items.get("meta_email", "node doc"))
    init = helper.make_tensor("weights_0", TensorProto.FLOAT, [1], [1.0])
    graph = helper.make_graph([node], "g", [x], [y], initializer=[init], doc_string=items.get("meta_host", "graph doc"))
    model = helper.make_model(graph, producer_name="fixture", opset_imports=[helper.make_opsetid("", 13)])
    model.doc_string = items.get("meta_aws_key", "model doc")
    for k, v in items.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.save(model, path)


def w_onnx_initializer_names(path: str, items: Items) -> None:
    """Secrets hidden in initializer (tensor) names, node names and string attributes."""
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    inits = [helper.make_tensor(v, TensorProto.FLOAT, [1], [1.0]) for v in list(items.values())[:3]]
    node = helper.make_node("Identity", ["x"], ["y"], name=list(items.values())[3 % len(items)], note=list(items.values())[4 % len(items)])
    graph = helper.make_graph([node], "g", [x], [y], initializer=inits)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)


def w_h5(path: str, items: Items) -> None:
    import h5py

    with h5py.File(path, "w") as h:
        h.attrs["keras_version"] = "2.15.0"
        h.attrs["backend"] = "tensorflow"
        h.attrs["model_config"] = json.dumps({"class_name": "Sequential", "config": {"name": "seq", "layers": [
            {"class_name": "Dense", "config": {"name": "dense_1", "units": 4, **items}}]}})
        g = h.create_group("model_weights/dense_1")
        g.attrs["weight_names"] = [b"dense_1/kernel:0"]
        g.create_dataset("dense_1/kernel:0", data=[[1.0, 2.0]])
        h.create_dataset("notes", data=list(items.values()), dtype=h5py.string_dtype())


def w_keras(path: str, items: Items) -> None:
    import h5py

    buf = io.BytesIO()
    with h5py.File(buf, "w") as h:
        h.create_group("layers/dense").create_dataset("vars/0", data=[1.0, 2.0])
    with zipfile.ZipFile(path, "w") as z:
        _zput(z, "config.json", json.dumps({"module": "keras", "class_name": "Sequential", "config": {"name": "seq", **items}}))
        _zput(z, "metadata.json", json.dumps({"keras_version": "3.0.0", "date_saved": "2024-01-01@00:00:00"}))
        _zput(z, "model.weights.h5", buf.getvalue())


def w_catboost(path: str, items: Items) -> None:
    """A CBM1-framed blob with the kind of JSON training options real files embed (train_dir etc.).

    Like a real file, the declared core size covers the part that holds the training options; extra bytes follow it.
    """
    opts = json.dumps({"train_dir": "catboost_info", "loss_function": "Logloss", **items})
    core = os.urandom(64) + b"\x00\x00" + opts.encode() + b"\x00"
    with open(path, "wb") as f:
        f.write(b"CBM1" + struct.pack("<I", len(core)) + core + os.urandom(32))


def _gstr(s: str, v1: bool = False) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def w_gguf(path: str, items: Items) -> None:
    kvs = [("general.architecture", "llama"), ("general.name", "fixture-model")] + [("general." + k.replace("meta_", ""), v) for k, v in items.items()]
    kvs.append(("tokenizer.chat_template", "{% for m in messages %}{{ m.content }}{% endfor %}"))
    out = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", len(kvs))
    for k, v in kvs:
        out += _gstr(k) + struct.pack("<I", 8) + _gstr(v)
    out += _gstr("token_embd.weight") + struct.pack("<I", 1) + struct.pack("<Q", 4) + struct.pack("<I", 0) + struct.pack("<Q", 0)
    out += b"\x00" * (-len(out) % 32) + b"\x00" * 16
    with open(path, "wb") as f:
        f.write(out)


def _nb_lines(lines):
    """nbformat multiline string: every element ends with a newline except the last."""
    return [l + "\n" for l in lines[:-1]] + lines[-1:]


def w_ipynb(path: str, items: Items) -> None:
    vals = list(items.items())
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "colab": {"name": "fixture.ipynb"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Fixture notebook\n", (vals[0][1] if vals else "")]},
            {"cell_type": "code", "execution_count": 1, "metadata": {}, "outputs": [
                {"output_type": "stream", "name": "stdout", "text": _nb_lines([v for _, v in vals[1:6]])},
                {"output_type": "error", "ename": "ValueError", "evalue": "boom", "traceback": ["\u001b[0;31mTraceback\u001b[0m", *[v for _, v in vals[6:9]]]},  # nbformat: no line terminators
                {"output_type": "display_data", "data": {"text/plain": _nb_lines([v for _, v in vals[9:12]]), "image/png": "iVBORw0KGgo="}, "metadata": {}},
            ], "source": ["import os\n"] + [f"x{i} = {json.dumps(v)}\n" for i, (_, v) in enumerate(vals[12:])]},
        ],
    }
    with open(path, "w", encoding="utf-8", newline="\n") as f:   # same bytes on every OS (pre-release test 2, N8)
        json.dump(nb, f, indent=1)  # nbformat writes notebooks with indent=1


def w_json(path: str, items: Items) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:   # same bytes on every OS (pre-release test 2, N8)
        json.dump({"model_type": "fixture", **items}, f, indent=1)  # one key per line, like most real configs


def w_yaml(path: str, items: Items) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:   # same bytes on every OS (pre-release test 2, N8)
        f.write("model_type: fixture\n")
        for k, v in items.items():
            f.write(f"{k}: {json.dumps(v)}\n")


def w_txt(path: str, items: Items) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:   # same bytes on every OS (pre-release test 2, N8)
        for k, v in items.items():
            f.write(f"{k}: {v}\n")


def w_cfg(path: str, items: Items) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:   # same bytes on every OS (pre-release test 2, N8)
        f.write("[model]\ntype = fixture\n")
        for k, v in items.items():
            f.write(f"{k} = {v.replace(chr(10), ' ')}\n")


def w_zip_nested(path: str, items: Items) -> None:
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        _zput(z, "deep/config.json", json.dumps(items))
    with zipfile.ZipFile(path, "w") as z:
        _zput(z, "inner.zip", inner.getvalue())
        _zput(z, "README.txt", "bundle\n")


@dataclass
class Format:
    name: str
    ext: str
    writer: Callable[[str, Items], None]
    needs: Tuple[str, ...] = ()
    pickle_based: bool = False
    # planted labels this format is not expected to preserve verbatim (documented limits)
    skip_labels: Tuple[str, ...] = ()


FORMATS: List[Format] = [
    Format("pickle", ".pkl", w_pickle, pickle_based=True),
    Format("pickle-proto2", ".pickle", w_pickle_proto2, pickle_based=True),
    Format("torch-zip", ".pt", w_torch_zip, pickle_based=True),
    Format("torch-legacy", ".pth", w_torch_legacy, pickle_based=True),
    Format("joblib", ".joblib", w_joblib, ("joblib", "numpy"), pickle_based=True),
    Format("joblib-zlib", ".joblib", w_joblib_zlib, ("joblib", "numpy"), pickle_based=True),
    Format("npy-object", ".npy", w_npy_object, ("numpy",), pickle_based=True),
    Format("npy-strings", ".npy", w_npy_strings, ("numpy",)),
    Format("npz", ".npz", w_npz, ("numpy",), pickle_based=True),
    Format("safetensors", ".safetensors", w_safetensors),
    Format("onnx", ".onnx", w_onnx, ("onnx",)),
    Format("keras-h5", ".h5", w_h5, ("h5py",)),
    Format("keras-v3", ".keras", w_keras, ("h5py",)),
    Format("catboost", ".cbm", w_catboost),
    Format("gguf", ".gguf", w_gguf),
    Format("ipynb", ".ipynb", w_ipynb),
    Format("json", ".json", w_json),
    Format("yaml", ".yaml", w_yaml),
    Format("txt", ".txt", w_txt),
    Format("cfg", ".cfg", w_cfg),
    Format("zip-nested", ".zip", w_zip_nested),
]


def have(mods: Tuple[str, ...]) -> bool:
    import importlib.util

    return all(importlib.util.find_spec(m) is not None for m in mods)


@dataclass
class Fixture:
    format: str
    kind: str  # "planted" | "clean"
    path: str
    labels: List[str] = field(default_factory=list)


def build_format_fixtures(outdir: str) -> List[Fixture]:
    """One planted file and one clean twin per supported format (skipping formats whose writer lib is missing)."""
    out: List[Fixture] = []
    os.makedirs(outdir, exist_ok=True)
    for fmt in FORMATS:
        if not have(fmt.needs):
            continue
        planted_path = os.path.join(outdir, f"planted_{fmt.name}{fmt.ext}")
        clean_path = os.path.join(outdir, f"clean_{fmt.name}{fmt.ext}")
        fmt.writer(planted_path, _items())
        fmt.writer(clean_path, clean_items())
        out.append(Fixture(fmt.name, "planted", planted_path, [p.label for p in PLANTED]))
        out.append(Fixture(fmt.name, "clean", clean_path))
    return out


# --------------------------------------------------------------------------------------
# Unsafe-pickle fixtures (for scanner and tool comparison)
# --------------------------------------------------------------------------------------
@dataclass
class UnsafeCase:
    id: str
    description: str
    path: str
    malicious: bool
    expect_rules: Tuple[str, ...] = ()
    group: str = ""


def _w(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


def build_unsafe_fixtures(outdir: str) -> List[UnsafeCase]:
    os.makedirs(outdir, exist_ok=True)
    cases: List[UnsafeCase] = []

    def add(id_, desc, fname, data=None, writer=None, malicious=True, rules=("unsafe-pickle-exec",), group=""):
        path = os.path.join(outdir, fname)
        if writer:
            writer(path)
        else:
            _w(path, data)
        cases.append(UnsafeCase(id_, desc, path, malicious, tuple(rules) if malicious else (), group))

    add("U01", "os.system, protocol 0 GLOBAL", "u01_proto0_system.pkl", pickle_proto0_system())
    add("U02", "subprocess.Popen, protocol 2 GLOBAL", "u02_proto2_popen.pkl", pickle_proto2_subprocess())
    add("U03", "builtins.eval, protocol 0 (__builtin__ alias)", "u03_eval.pkl", pickle_proto0_eval())
    add("U04", "os.system, protocol 4 STACK_GLOBAL", "u04_proto4_system.pkl", pickle_proto4_system())
    add("U05", "builtins.exec, STACK_GLOBAL operands via memo with decoy string", "u05_memo_exec.pkl", pickle_memo_indirect_exec())
    add("U06", "builtins.getattr(OrderedDict, '__init__') (attribute-walking gadget start)", "u06_getattr.pkl", pickle_getattr(), rules=("unsafe-pickle-io",))
    add("U07", "benign pickle then a SECOND pickle with subprocess.Popen", "u07_second_pickle.pkl", two_pickles_second_evil())
    add("U08", "benign head, corrupt opcode, payload after", "u08_corrupt_then_payload.pkl", corrupt_then_evil(),
        rules=("pickle-incomplete-analysis", "unsafe-pickle-exec"))

    def torch_evil(path):
        with zipfile.ZipFile(path, "w") as z:
            _zput(z, "archive/data.pkl", pickle_proto4_system())
            _zput(z, "archive/version", "3\n")
            _zput(z, "archive/data/0", os.urandom(32))

    add("U09", "os.system inside PyTorch zip data.pkl", "u09_torch_zip.pt", writer=torch_evil)

    def torch_legacy_evil(path):
        with open(path, "wb") as f:
            pickle.dump(0x1950A86A20F9469CFC6C, f, protocol=2)
            pickle.dump(1001, f, protocol=2)
            pickle.dump({"protocol_version": 1001}, f, protocol=2)
            f.write(pickle_proto4_system())
            pickle.dump(["0"], f, protocol=2)
            f.write(struct.pack("<q", 2) + b"\x00\x00")

    add("U10", "os.system as the main object of a legacy torch.save file", "u10_torch_legacy.pth", writer=torch_legacy_evil)

    if have(("joblib", "numpy")):
        def joblib_evil(path):
            import joblib
            import numpy as np

            class Boom:
                def __reduce__(self):
                    import os as _os
                    return (_os.system, (CMD,))

            joblib.dump({"a": np.arange(100, dtype="float64"), "b": np.arange(50, dtype="int32"), "x": Boom()}, path)

        add("U11", "os.system AFTER numpy arrays in a joblib file (arrays sit inline between opcodes)", "u11_joblib.joblib", writer=joblib_evil)

        def joblib_evil_zlib(path):
            import joblib
            import numpy as np

            class Boom:
                def __reduce__(self):
                    import os as _os
                    return (_os.system, (CMD,))

            joblib.dump({"a": np.arange(100, dtype="float64"), "x": Boom()}, path, compress=3)

        add("U12", "same, zlib-compressed joblib", "u12_joblib_zlib.joblib", writer=joblib_evil_zlib)

        def npy_evil(path):
            import numpy as np

            class Boom:
                def __reduce__(self):
                    import os as _os
                    return (_os.system, (CMD,))

            a = np.empty(1, dtype=object)
            a[0] = Boom()
            np.save(path, a, allow_pickle=True)

        add("U13", "os.system inside a NumPy object array (.npy)", "u13_npy.npy", writer=npy_evil)

        def npz_evil(path):
            import numpy as np

            class Boom:
                def __reduce__(self):
                    import os as _os
                    return (_os.system, (CMD,))

            a = np.empty(1, dtype=object)
            a[0] = Boom()
            np.savez(path, ok=np.arange(3), evil=a)

        add("U14", "os.system inside an .npz member", "u14_npz.npz", writer=npz_evil)

    if have(("h5py",)):
        def keras_lambda_h5(path):
            import h5py

            code = base64.b64encode(b"fixture-bytecode-not-real").decode()
            cfg = {"class_name": "Sequential", "config": {"name": "seq", "layers": [
                {"class_name": "Lambda", "config": {"name": "lambda_1", "function": [code, None, None], "function_type": "lambda"}}]}}
            with h5py.File(path, "w") as h:
                h.attrs["model_config"] = json.dumps(cfg)

        add("U18", "Keras 2 HDF5 model with a Lambda layer (serialised code)", "u18_keras_lambda.h5", writer=keras_lambda_h5,
            rules=("unsafe-keras-lambda",))

        def keras3_lambda(path):
            code = base64.b64encode(b"fixture-bytecode-not-real").decode()
            cfg = {"module": "keras", "class_name": "Sequential", "config": {"layers": [
                {"module": "keras.layers", "class_name": "Lambda", "config": {"name": "lambda", "function": {
                    "class_name": "__lambda__", "config": {"code": code, "defaults": None, "closure": None}}}}]}}
            with zipfile.ZipFile(path, "w") as z:
                _zput(z, "config.json", json.dumps(cfg))
                _zput(z, "metadata.json", json.dumps({"keras_version": "3.0.0"}))

        add("U19", "Keras 3 .keras archive with a Lambda layer", "u19_keras3_lambda.keras", writer=keras3_lambda,
            rules=("unsafe-keras-lambda",))

    def nested_zip_evil(path):
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as z:
            _zput(z, "pkg/model.pkl", pickle_proto4_system())
        with zipfile.ZipFile(path, "w") as z:
            _zput(z, "bundle/inner.zip", inner.getvalue())

    add("U15", "os.system inside a zip inside a zip", "u15_nested.zip", writer=nested_zip_evil)

    def gz_evil(path):
        import gzip

        with gzip.open(path, "wb") as f:
            f.write(pickle_proto4_system())

    add("U16", "os.system in a gzip-compressed pickle", "u16_gz.pkl.gz", writer=gz_evil)

    add("U17", "torch.storage._load_from_bytes on bytes that cannot be analysed", "u17_unanalysable_loader.pkl",
        pickle_torch_load_from_bytes_unanalysable(), rules=("unsafe-pickle-nested",))
    add("B01", "benign pickle (dict of strings and ints)", "b01_benign.pkl", benign_pickle_bytes(), malicious=False)
    def yolo_zip(path):
        with zipfile.ZipFile(path, "w") as z:
            _zput(z, "yolov8n/data.pkl", pickle_getattr_benign())
            _zput(z, "yolov8n/version", "3\n")
            _zput(z, "yolov8n/data/0", os.urandom(32))

    add("B03", "YOLO-style checkpoint: getattr(ultralytics Detect, 'forward') must NOT be HIGH", "b03_yolo_style.pt",
        writer=yolo_zip, malicious=False)

    for k, attr in enumerate(GETATTR_NEW_DANGEROUS):
        add(f"U{20 + k}", f"getattr(OrderedDict, {attr!r})", f"u{20 + k}_getattr_{attr}.pkl", _getattr_call(attr),
            rules=("unsafe-pickle-io",), group="getattr with a dangerous attribute name (16 files)")
    n = 20 + len(GETATTR_NEW_DANGEROUS)
    add(f"U{n}", "getattr(OrderedDict, 'frobnicate'): unknown name -> review finding, not silence", f"u{n}_getattr_unknown.pkl",
        _getattr_call("frobnicate"), rules=("pickle-getattr-review",))
    add(f"U{n + 1}", "getattr(OrderedDict, '__init__') through INST", f"u{n + 1}_getattr_inst.pkl",
        _getattr_call("__init__", opcode="INST"), rules=("unsafe-pickle-io",))
    add(f"U{n + 2}", "getattr(OrderedDict, 'execvp') through OBJ", f"u{n + 2}_getattr_obj.pkl",
        _getattr_call("execvp", opcode="OBJ"), rules=("unsafe-pickle-io",))
    add(f"U{n + 3}", "getattr with a computed attribute name", f"u{n + 3}_getattr_computed.pkl",
        pickle_getattr_computed_name(), rules=("unsafe-pickle-io",))
    add(f"U{n + 4}", "getattr passed to functools.partial, never called directly", f"u{n + 4}_getattr_partial.pkl",
        pickle_getattr_referenced_only(), rules=("pickle-getattr-review",))
    add(f"U{n + 5}", "torch.load on opaque bytes through INST", f"u{n + 5}_loader_inst.pkl",
        pickle_loader_via_inst(), rules=("unsafe-pickle-nested",))
    def torch_tensor_by_value(path):
        """pickle.dumps(<tensor>)-style payload: torch.storage._load_from_bytes(<torch zip bytes>)"""
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as z:
            _zput(z, "archive/data.pkl", benign_pickle_bytes())
            _zput(z, "archive/version", "3\n")
        blob = inner.getvalue()
        with open(path, "wb") as f:
            f.write(b"\x80\x04" + _su("torch.storage") + _su("_load_from_bytes") + b"\x93\x94" + b"B" + struct.pack("<I", len(blob)) + blob
                    + b"\x85\x94R\x94.")

    add("B04", "torch tensor pickled by value (torch.storage._load_from_bytes with an analysable torch zip)",
        "b04_tensor_by_value.pkl", writer=torch_tensor_by_value, malicious=False)

    if have(("joblib", "numpy", "sklearn")):
        def sk_benign(path):
            import joblib
            from sklearn.linear_model import LogisticRegression
            import numpy as np

            m = LogisticRegression().fit(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1]))
            joblib.dump(m, path)

        add("B02", "benign scikit-learn LogisticRegression via joblib", "b02_sklearn.joblib", writer=sk_benign, malicious=False)
    return cases


def build_all(outdir: str) -> Tuple[List[Fixture], List[UnsafeCase]]:
    f = build_format_fixtures(os.path.join(outdir, "formats"))
    u = build_unsafe_fixtures(os.path.join(outdir, "unsafe"))
    return f, u


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "fixtures_out"
    fx, un = build_all(target)
    print(f"{len(fx)} format fixtures, {len(un)} unsafe-pickle fixtures written to {target}")

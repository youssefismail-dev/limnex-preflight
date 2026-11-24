"""PyTorch's ONNX exporter labels graph nodes with the metadata key ``pkg.torch.onnx.original_node_name``.

Found by the held-out round of 2026-11-05: the host-name key rule matched any key that ENDS in ``node_name``, so every such
label (a graph node name, not a machine) became a LOW ``hostname-field`` finding (167 of the 177 LOW findings on 60
ordinary repositories came from two exported ONNX files). A key that really is a host name must still be reported.
"""

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402


def _model(path, entries):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    for k, v in entries:
        e = x.metadata_props.add()
        e.key, e.value = k, v
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"], name="n0")], "g", [x], [y])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), str(path))


def test_torch_exporter_node_labels_are_not_host_names(tmp_path, scan):
    p = tmp_path / "exported.onnx"
    _model(p, [("pkg.torch.onnx.original_node_name", "p_layer1_weight"), ("pkg.torch.onnx.name_scopes", "layer1")])
    assert [f.rule_id for f in scan(p).findings] == []


def test_a_real_host_name_key_in_onnx_metadata_is_still_reported(tmp_path, scan):
    p = tmp_path / "host.onnx"
    _model(p, [("node_name", "gpu-box-17.corp"), ("training.host_name", "ml-train-03")])
    assert "hostname-field" in {f.rule_id for f in scan(p).findings}

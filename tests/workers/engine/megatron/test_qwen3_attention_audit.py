import importlib.util
from pathlib import Path

import torch


def _load_attention_audit_module():
    module_path = (
        Path(__file__).resolve().parents[5]
        / "megatron-bridge/src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/attention_audit.py"
    )
    spec = importlib.util.spec_from_file_location("qwen3_attention_audit_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_attention_execution_audit_records_response_mask_state():
    audit = _load_attention_audit_module()
    input_ids = torch.tensor([[10, 11, 12, 13]])
    rows = [{"row": 0, "input_ids_sha256": "sample", "positions": [2, 3]}]
    attention_mask = torch.tensor(
        [[[[False, True, False, True], [False, False, True, True], [False, False, False, True], [False, False, False, False]]]]
    )

    audit.configure_attention_audit(rows, input_ids, padded_length=4)
    audit.capture_attention_execution(
        1,
        core_attention=object(),
        path="core_attention",
        packed_seq_params=None,
        attention_bias=None,
        inference_context=None,
        attention_mask=attention_mask,
    )

    execution = audit.attention_execution_audit()
    assert len(execution) == 1
    assert execution[0]["core_attention_type"] == "object"
    assert execution[0]["path"] == "core_attention"
    assert execution[0]["has_packed_seq_params"] is False
    assert execution[0]["response_mask"] == [
        {"present": True, "shape": [1, 1, 4, 4], "dtype": "torch.bool", "query_masked_count": 1, "query_unmasked_count": 3, "self_masked": False},
        {"present": True, "shape": [1, 1, 4, 4], "dtype": "torch.bool", "query_masked_count": 0, "query_unmasked_count": 4, "self_masked": False},
    ]
    assert audit.clear_attention_audit() == []


def test_all_valid_2d_mask_identifies_only_unpadded_boolean_masks():
    audit = _load_attention_audit_module()

    assert audit.is_all_valid_2d_mask(torch.ones((2, 4), dtype=torch.bool))
    assert not audit.is_all_valid_2d_mask(torch.tensor([[True, False]], dtype=torch.bool))
    assert not audit.is_all_valid_2d_mask(torch.ones((1, 1, 4, 4), dtype=torch.bool))
    assert not audit.is_all_valid_2d_mask(torch.ones((1, 4), dtype=torch.int64))
    assert not audit.is_all_valid_2d_mask(None)


def test_qwen_valid_mask_is_converted_to_the_te_mask_convention():
    audit = _load_attention_audit_module()
    valid_mask = torch.tensor([[True, True, False], [True, False, False]], dtype=torch.bool)

    converted = audit.qwen_valid_mask_to_te_mask(valid_mask)

    assert torch.equal(
        converted,
        torch.tensor(
            [[[[False, False, True]]], [[[False, True, True]]]],
            dtype=torch.bool,
        ),
    )
    assert audit.qwen_valid_mask_to_te_mask(torch.ones((1, 3), dtype=torch.bool)) is None
    non_boolean = torch.ones((1, 3), dtype=torch.int64)
    assert audit.qwen_valid_mask_to_te_mask(non_boolean) is non_boolean

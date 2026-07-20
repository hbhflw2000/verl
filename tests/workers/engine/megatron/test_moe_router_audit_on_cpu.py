# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import torch

from verl.workers.engine.megatron.transformer_impl import (
    _append_moe_dispatcher_metadata,
    _append_moe_replay_route_metadata,
    _install_moe_gate_logits_audit,
    _install_moe_mlp_stage_audit,
)


class _MethodGateRouter:
    def __init__(self):
        self.weight = torch.tensor([[1.0, -1.0], [0.5, 2.0]])

    def gating(self, input_tensor):
        return torch.nn.functional.linear(input_tensor, self.weight)


class _ModuleGateRouter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gating = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.gating.weight.copy_(torch.tensor([[1.0, -1.0], [0.5, 2.0]]))


class _AuditMlp:
    def route(self, hidden_states):
        return hidden_states + 1, hidden_states.bool()

    def preprocess(self, probs, indices):
        return probs + 1, indices

    def dispatch(self, probs):
        return probs + 1

    def routed_experts_compute(self, dispatched):
        return dispatched + 1

    def combine(self, expert_output):
        return expert_output + 1

    def postprocess(self, combined):
        return combined + 1


class _ReplayState:
    def __init__(self, topk):
        self.recorded_topk_idx = topk
        self.target_topk_idx = topk


class _MetadataMlp:
    def __init__(self, topk):
        self.router = type("Router", (), {"router_replay": _ReplayState(topk)})()
        self.token_dispatcher = type(
            "Dispatcher",
            (),
            {
                "routing_map": torch.tensor([[True, False], [False, True]]),
                "tokens_per_expert": torch.tensor([1, 1]),
                "input_splits": torch.tensor([1, 1]),
                "output_splits": torch.tensor([1, 1]),
                "output_splits_tp": torch.tensor([2]),
                "reversed_local_input_permutation_mapping": torch.tensor([0, 1]),
            },
        )()


def test_moe_gate_audit_captures_and_restores_method_gate():
    router = _MethodGateRouter()
    original_gate = router.gating
    destination = {"value": None}
    cleanup = _install_moe_gate_logits_audit(router, destination)
    assert cleanup is not None

    input_tensor = torch.tensor([[2.0, 3.0]])
    expected = original_gate(input_tensor)
    assert torch.equal(router.gating(input_tensor), expected)
    assert torch.equal(destination["value"], expected)

    cleanup()
    assert router.gating.__func__ is original_gate.__func__


def test_moe_gate_audit_captures_module_gate():
    router = _ModuleGateRouter()
    destination = {"value": None}
    cleanup = _install_moe_gate_logits_audit(router, destination)
    assert cleanup is not None

    input_tensor = torch.tensor([[2.0, 3.0]])
    expected = router.gating(input_tensor)
    assert torch.equal(destination["value"], expected)

    cleanup()


def test_moe_mlp_stage_audit_fingerprints_and_restores_methods():
    mlp = _AuditMlp()
    original_route = mlp.route
    destination = []
    cleanup = _install_moe_mlp_stage_audit(mlp, 1, destination)
    assert cleanup is not None

    probs, indices = mlp.route(torch.tensor([[1.0, 2.0]]))
    probs, indices = mlp.preprocess(probs, indices)
    dispatched = mlp.dispatch(probs)
    expert_output = mlp.routed_experts_compute(dispatched)
    combined = mlp.combine(expert_output)
    assert torch.equal(mlp.postprocess(combined), torch.tensor([[7.0, 8.0]]))

    stages = {entry["stage"] for entry in destination}
    assert {"route_input:0", "route_output:0", "dispatch_output:0", "postprocess_output:0"} <= stages
    assert all(entry["stats"]["numel"] == 2 for entry in destination)
    cleanup()
    assert mlp.route.__func__ is original_route.__func__


def test_moe_replay_metadata_audit_compares_target_map_and_dispatcher_state():
    topk = torch.tensor([[0], [1]])
    mlp = _MetadataMlp(topk)
    destination = []
    probs = torch.tensor([[0.7, 0.0], [0.0, 0.8]])
    routing_map = torch.tensor([[True, False], [False, True]])
    _append_moe_replay_route_metadata(mlp, 1, (probs, routing_map), destination)
    _append_moe_dispatcher_metadata(mlp, 1, (torch.ones(2, 1),), destination)

    route, dispatcher = destination
    assert route["route_vs_target_map_mismatch_count"] == 0
    assert route["recorded_topk"]["sha256"] == route["target_topk"]["sha256"]
    assert dispatcher["input_splits"]["values"] == [1, 1]

    mlp.router.router_replay.target_topk_idx = torch.tensor([[1], [1]])
    destination = []
    _append_moe_replay_route_metadata(mlp, 1, (probs, routing_map), destination)
    assert destination[0]["route_vs_target_map_mismatch_count"] == 2

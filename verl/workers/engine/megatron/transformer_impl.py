# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import hashlib
import json
import logging
import os
import socket
from functools import partial
from typing import Any, Callable, ContextManager, Iterator, Optional

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core import tensor_parallel
from megatron.core.package_info import __version__
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.models.mcore import get_mcore_weight_converter
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.debug.logprob_audit import response_score_positions
from verl.utils.device import get_device_id, get_device_name
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction, apply_router_replay_patch
from verl.utils.megatron.router_replay_utils import (
    RouterReplayHelper,
    get_thd_sequence_parallel_padding_mask,
    merge_router_topk_indices,
    pp_gather,
    reorder_and_merge_vpp_layers,
    set_router_replay_data,
)
from verl.utils.megatron.tensor_parallel import (
    vocab_parallel_entropy,
    vocab_parallel_log_probs_from_logits,
    vocab_parallel_sum_pi_squared,
)
from verl.utils.megatron_peft_utils import (
    add_base_layer_suffix,
    build_peft_config_for_vllm,
    gather_ep_lora_adapter_weights_for_vllm,
    pack_3d_moe_lora_adapter_weights_for_vllm,
)
from verl.utils.megatron_utils import (
    check_mtp_config,
    get_megatron_module_device,
    get_megatron_mtp_loss,
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
    patch_engine_mtp,
    register_megatron_training_hooks,
    unwrap_model,
)
from verl.utils.model import extract_multi_modal_inputs, load_mcore_dist_weights
from verl.utils.seqlen_balancing import restore_dynamic_batch
from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import postprocess_batch_func, prepare_micro_batches
from .utils import set_random_seed

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_MEGATRON_VOCAB_DEBUG_COUNTS = {}
_MEGATRON_DECODER_AUDIT_COUNTS = {}


def _vocab_parallel_debug_enabled() -> bool:
    return bool(os.getenv("VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL", "").strip())


def _component_audit_enabled() -> bool:
    return os.getenv("VERL_OMNI_MEGATRON_LOGPROB_COMPONENT_AUDIT", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _decoder_component_audit_enabled() -> bool:
    """Emit small per-layer activation summaries for the fixed-sequence parity probe."""
    return os.getenv("VERL_OMNI_MEGATRON_DECODER_COMPONENT_AUDIT", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _moe_router_audit_enabled() -> bool:
    return os.getenv("VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _moe_router_audit_layers() -> set[int]:
    raw = os.getenv("VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT_LAYERS", "1,4,6,16,32,48")
    return {int(value) for value in raw.split(",") if value.strip().isdigit()}


def _moe_mlp_stage_audit_enabled() -> bool:
    return os.getenv("VERL_OMNI_MEGATRON_MOE_MLP_STAGE_AUDIT", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _moe_mlp_stage_audit_layers() -> set[int]:
    raw = os.getenv("VERL_OMNI_MEGATRON_MOE_MLP_STAGE_AUDIT_LAYERS", "1")
    return {int(value) for value in raw.split(",") if value.strip().isdigit()}


def _moe_replay_metadata_audit_enabled() -> bool:
    return os.getenv("VERL_OMNI_MEGATRON_MOE_REPLAY_METADATA_AUDIT", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _router_replay_audit_action() -> str | None:
    """Return the active replay phase while an audited forward is still live."""
    actions = {
        action.value
        for router in RouterReplay.router_instances
        if (action := getattr(router, "router_replay_action", None)) is not None
    }
    return ",".join(sorted(actions)) if actions else None


def _debug_jsonable(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _debug_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_debug_jsonable(v) for v in value]
    try:
        return int(value)
    except Exception:
        try:
            return float(value)
        except Exception:
            return str(value)


def _debug_tensor_2d_slice(value: torch.Tensor, rows: int, tokens: int, tail: bool = False):
    if value is None:
        return None
    try:
        tensor = value.detach()
        if tensor.dim() == 0:
            return _debug_jsonable(tensor.cpu().item())
        if tensor.dim() == 1:
            tensor_slice = tensor[-tokens:] if tail else tensor[:tokens]
            return _debug_jsonable(tensor_slice.cpu().tolist())
        tensor_slice = tensor[:rows, -tokens:] if tail else tensor[:rows, :tokens]
        return _debug_jsonable(tensor_slice.cpu().tolist())
    except Exception as exc:
        return {"error": repr(exc), "shape": _debug_jsonable(list(getattr(value, "shape", [])))}


def _token_ids_sha256(token_ids: torch.Tensor) -> str:
    values = token_ids.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _vector_row_stats(rows: torch.Tensor) -> list[dict[str, float | list[float]]]:
    rows = rows.detach().float()
    if rows.ndim != 2:
        return []
    head_width = min(8, rows.shape[-1])
    return [
        {
            "sum": float(row.sum().cpu().item()),
            "square_sum": float((row * row).sum().cpu().item()),
            "head": row[:head_width].cpu().tolist(),
        }
        for row in rows
    ]


def _tensor_fingerprint(value: torch.Tensor | None) -> dict | None:
    """Record cheap, layout-sensitive anchors for static cross-backend checks."""
    if value is None or value.ndim != 2:
        return None
    tensor = value.detach().float()
    height, width = tensor.shape
    anchors = sorted(
        {
            (0, 0),
            (0, min(1, width - 1)),
            (min(1, height - 1), 0),
            (height // 2, width // 2),
            (height - 1, width - 1),
        }
    )
    return {
        "shape": [int(height), int(width)],
        "sum": float(tensor.sum().cpu().item()),
        "square_sum": float((tensor * tensor).sum().cpu().item()),
        "anchors": [
            {"index": [row, column], "value": float(tensor[row, column].cpu().item())}
            for row, column in anchors
        ],
    }


def _vocab_parallel_weight_row_stats(
    weight: torch.Tensor | None,
    local_target: torch.Tensor,
    owned: torch.Tensor,
    index: torch.Tensor,
) -> list[dict[str, float | list[float]]] | None:
    if weight is None or weight.ndim != 2 or weight.shape[0] <= int(local_target.max().item()):
        return None
    owned_rows = owned[index]
    local_rows = weight.detach().float()[local_target[index]]
    local_rows = local_rows * owned_rows.unsqueeze(-1)
    row_sum = local_rows.sum(dim=-1)
    row_square_sum = (local_rows * local_rows).sum(dim=-1)
    head_width = min(8, local_rows.shape[-1])
    row_head = local_rows[:, :head_width].contiguous()
    if torch.distributed.is_initialized():
        tp_group = mpu.get_tensor_model_parallel_group()
        torch.distributed.all_reduce(row_sum, group=tp_group)
        torch.distributed.all_reduce(row_square_sum, group=tp_group)
        torch.distributed.all_reduce(row_head, group=tp_group)
    return [
        {
            "sum": float(row_sum[row].cpu().item()),
            "square_sum": float(row_square_sum[row].cpu().item()),
            "head": row_head[row].cpu().tolist(),
        }
        for row in range(index.numel())
    ]


def _vocab_parallel_target_linear(
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    hidden: torch.Tensor | None,
    local_target: torch.Tensor,
    owned: torch.Tensor,
    index: torch.Tensor,
) -> list[float] | None:
    """Recompute selected vocab logits from the captured LM-head input."""
    if (
        weight is None
        or hidden is None
        or weight.ndim != 2
        or hidden.ndim != 2
        or weight.shape[0] <= int(local_target.max().item())
        or weight.shape[1] != hidden.shape[-1]
    ):
        return None
    owned_rows = owned[index]
    local_values = (hidden.detach().float()[index] * weight.detach().float()[local_target[index]]).sum(dim=-1)
    if bias is not None and bias.ndim == 1 and bias.shape[0] > int(local_target.max().item()):
        local_values = local_values + bias.detach().float()[local_target[index]]
    local_values = local_values.masked_fill(~owned_rows, 0.0)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(local_values, group=mpu.get_tensor_model_parallel_group())
    return [float(value.cpu().item()) for value in local_values]


def _align_pre_lm_hidden(hidden: torch.Tensor | None, logits: torch.Tensor) -> torch.Tensor | None:
    if hidden is None or hidden.ndim != 3:
        return None
    if hidden.shape[:2] == logits.shape[:2]:
        return hidden
    if hidden.shape[0] == logits.shape[1] and hidden.shape[1] == logits.shape[0]:
        return hidden.transpose(0, 1).contiguous()
    return None


def _gather_sequence_parallel_hidden(
    hidden: torch.Tensor | None, sequence_parallel: bool
) -> torch.Tensor | None:
    """Match LinearForLastLayer's gather before comparing hidden to gathered logits."""
    if hidden is None or hidden.ndim != 3 or not sequence_parallel:
        return hidden
    return tensor_parallel.gather_from_sequence_parallel_region(
        hidden, tensor_parallel_output_grad=False
    )


def _get_output_layer(model) -> torch.nn.Module | None:
    try:
        module = unwrap_model(model)
        if isinstance(module, list):
            module = module[0]
        return module.thinker.language_model.output_layer
    except (AttributeError, IndexError, TypeError):
        return None


def _get_input_embedding(model) -> torch.nn.Module | None:
    try:
        module = unwrap_model(model)
        if isinstance(module, list):
            module = module[0]
        return module.thinker.language_model.embedding.word_embeddings
    except (AttributeError, IndexError, TypeError):
        return None


def _get_final_layernorm(model) -> torch.nn.Module | None:
    try:
        module = unwrap_model(model)
        if isinstance(module, list):
            module = module[0]
        return module.thinker.language_model.decoder.final_layernorm
    except (AttributeError, IndexError, TypeError):
        return None


def _first_tensor_argument(args, kwargs, preferred_names: tuple[str, ...]) -> torch.Tensor | None:
    """Get a module's activation without accidentally recording an optional weight."""
    if args and isinstance(args[0], torch.Tensor):
        return args[0]
    for name in preferred_names:
        value = (kwargs or {}).get(name)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _first_tensor_output(value) -> torch.Tensor | None:
    """Return the activation tensor from common Megatron module return shapes."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor_output(item)
            if tensor is not None:
                return tensor
    return None


def _tensor_leaves(value) -> list[torch.Tensor]:
    """Flatten tensor-bearing method inputs/outputs without retaining activations."""
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _tensor_leaves(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in _tensor_leaves(item)]
    return []


def _tensor_stage_stats(value: torch.Tensor) -> dict:
    """Return a bounded numeric fingerprint suitable for cross-forward comparison."""
    flat = value.detach().reshape(-1).float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "sum": float(flat.sum().item()),
        "square_sum": float(torch.square(flat).sum().item()),
        "abs_max": float(flat.abs().max().item()) if flat.numel() else 0.0,
        "head": [float(item) for item in flat[:8].tolist()],
    }


def _tensor_exact_audit(value: torch.Tensor | None, *, include_values: bool = False) -> dict | None:
    """Emit a small exact fingerprint for a debug-only cross-forward comparison."""
    if not isinstance(value, torch.Tensor):
        try:
            value = torch.as_tensor(value)
        except (RuntimeError, TypeError, ValueError):
            return None
    tensor = value.detach().contiguous()
    raw = tensor.view(torch.uint8).cpu().numpy()
    result = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
    }
    if tensor.numel():
        numeric = tensor.float()
        result.update(
            {
                "sum": float(numeric.sum().item()),
                "square_sum": float(torch.square(numeric).sum().item()),
                "nonzero": int(torch.count_nonzero(tensor).item()),
            }
        )
    if include_values and tensor.numel() <= 256:
        result["values"] = _debug_jsonable(tensor.cpu().tolist())
    return result


def _target_routing_map(target_topk: torch.Tensor | None, routing_map: torch.Tensor | None) -> torch.Tensor | None:
    if (
        not isinstance(target_topk, torch.Tensor)
        or not isinstance(routing_map, torch.Tensor)
        or target_topk.ndim != 2
        or routing_map.ndim != 2
        or target_topk.shape[0] != routing_map.shape[0]
    ):
        return None
    if target_topk.numel() and (target_topk.min() < 0 or target_topk.max() >= routing_map.shape[1]):
        return None
    return torch.zeros_like(routing_map, dtype=torch.bool).scatter_(1, target_topk.to(torch.long), True)


def _append_moe_replay_route_metadata(mlp, layer_number: int, output, destination: list[dict]) -> None:
    if not isinstance(output, (tuple, list)) or len(output) < 2:
        return
    probs, routing_map = output[:2]
    if not isinstance(probs, torch.Tensor) or not isinstance(routing_map, torch.Tensor):
        return
    replay = getattr(getattr(mlp, "router", None), "router_replay", None)
    recorded_topk = getattr(replay, "recorded_topk_idx", None)
    target_topk = getattr(replay, "target_topk_idx", None)
    target_map = _target_routing_map(target_topk, routing_map)
    per_expert = routing_map.sum(dim=0)
    entry = {
        "layer": layer_number,
        "phase": "route",
        "input": None,
        "routing_probs": _tensor_exact_audit(probs),
        "routing_map": _tensor_exact_audit(routing_map, include_values=routing_map.numel() <= 256),
        "per_expert": _tensor_exact_audit(per_expert, include_values=True),
        "recorded_topk": _tensor_exact_audit(recorded_topk),
        "target_topk": _tensor_exact_audit(target_topk),
        "target_map": _tensor_exact_audit(target_map, include_values=target_map is not None and target_map.numel() <= 256),
    }
    if target_map is not None:
        entry["route_vs_target_map_mismatch_count"] = int(torch.count_nonzero(routing_map != target_map).item())
    destination.append(entry)


def _append_moe_dispatcher_metadata(mlp, layer_number: int, route_input, destination: list[dict]) -> None:
    dispatcher = getattr(mlp, "token_dispatcher", None)
    if dispatcher is None:
        return
    entry = {
        "layer": layer_number,
        "phase": "dispatcher_preprocess",
        "input": _tensor_exact_audit(_first_tensor_output(route_input)),
        "routing_map": _tensor_exact_audit(getattr(dispatcher, "routing_map", None)),
        "tokens_per_expert": _tensor_exact_audit(getattr(dispatcher, "tokens_per_expert", None), include_values=True),
        "input_splits": _tensor_exact_audit(getattr(dispatcher, "input_splits", None), include_values=True),
        "output_splits": _tensor_exact_audit(getattr(dispatcher, "output_splits", None), include_values=True),
        "output_splits_tp": _tensor_exact_audit(getattr(dispatcher, "output_splits_tp", None), include_values=True),
        "local_permutation": _tensor_exact_audit(
            getattr(dispatcher, "reversed_local_input_permutation_mapping", None)
        ),
    }
    destination.append(entry)


def _install_moe_mlp_stage_audit(
    mlp, layer_number: int, destination: list[dict], metadata_destination: list[dict] | None = None
) -> Callable[[], None] | None:
    """Fingerprint MCore MoE stages while preserving its regular method dispatch."""
    method_names = ("route", "preprocess", "dispatch", "routed_experts_compute", "combine", "postprocess")
    originals = {}
    recorded: set[tuple[str, int]] = set()

    def capture(stage: str, value) -> None:
        for index, tensor in enumerate(_tensor_leaves(value)):
            key = (stage, index)
            if key in recorded:
                continue
            recorded.add(key)
            destination.append(
                {
                    "layer": layer_number,
                    "stage": f"{stage}:{index}",
                    "stats": _tensor_stage_stats(tensor),
                }
            )

    for method_name in method_names:
        original = getattr(mlp, method_name, None)
        if not callable(original):
            continue
        originals[method_name] = original

        def audited_method(*args, _original=original, _method_name=method_name, **kwargs):
            capture(f"{_method_name}_input", (args, kwargs))
            output = _original(*args, **kwargs)
            capture(f"{_method_name}_output", output)
            if metadata_destination is not None:
                if _method_name == "route":
                    _append_moe_replay_route_metadata(mlp, layer_number, output, metadata_destination)
                    if metadata_destination:
                        metadata_destination[-1]["input"] = _tensor_exact_audit(_first_tensor_output(args))
                elif _method_name == "preprocess":
                    _append_moe_dispatcher_metadata(mlp, layer_number, args, metadata_destination)
            return output

        try:
            setattr(mlp, method_name, audited_method)
        except (AttributeError, TypeError):
            for name, saved in originals.items():
                setattr(mlp, name, saved)
            return None

    if not originals:
        return None

    def restore_methods():
        for method_name, original in originals.items():
            setattr(mlp, method_name, original)

    return restore_methods


def _install_moe_gate_logits_audit(router, destination: dict[str, torch.Tensor | None]) -> Callable[[], None] | None:
    """Capture gate logits for both module- and method-based MCore routers."""
    gate = getattr(router, "gating", None)
    if isinstance(gate, torch.nn.Module):
        handle = gate.register_forward_hook(
            lambda _module, _args, output: destination.__setitem__("value", _first_tensor_output(output))
        )
        return handle.remove
    if not callable(gate):
        return None

    original_gate = gate

    def audited_gate(*args, **kwargs):
        output = original_gate(*args, **kwargs)
        destination["value"] = _first_tensor_output(output)
        return output

    try:
        setattr(router, "gating", audited_gate)
    except (AttributeError, TypeError):
        return None

    def restore_gate():
        setattr(router, "gating", original_gate)

    return restore_gate


def _get_decoder_layers(model) -> list[torch.nn.Module]:
    try:
        module = unwrap_model(model)
        if isinstance(module, list):
            module = module[0]
        return list(module.thinker.language_model.decoder.layers)
    except (AttributeError, IndexError, TypeError):
        return []


def _audit_sequence_rows(input_ids: torch.Tensor, response_lengths: torch.Tensor) -> tuple[list[dict], int]:
    """Build response-token coordinates without depending on padded BSHD tensors."""
    rows = []
    max_length = 0
    for row_idx in range(int(input_ids.shape[0])):
        token_ids = input_ids[row_idx].detach()
        valid_length = int(token_ids.shape[0])
        max_length = max(max_length, valid_length)
        response_length = int(response_lengths[row_idx].item())
        positions = list(response_score_positions(valid_length, response_length))
        if positions:
            rows.append(
                {
                    "row": row_idx,
                    "input_ids_sha256": _token_ids_sha256(token_ids),
                    "input_len": valid_length,
                    "response_len": response_length,
                    "positions": positions,
                }
            )
    tp_world = mpu.get_tensor_model_parallel_world_size()
    padded_length = max_length + (-max_length % tp_world)
    return rows, padded_length


def _local_sequence_axis(tensor: torch.Tensor, batch_size: int, padded_length: int) -> tuple[int, int] | None:
    """Infer BSH/SBH layout and the global offset of this SP shard."""
    if tensor.ndim != 3:
        return None
    if tensor.shape[0] == batch_size:
        axis, local_length = 1, int(tensor.shape[1])
    elif tensor.shape[1] == batch_size:
        axis, local_length = 0, int(tensor.shape[0])
    else:
        return None

    tp_world = mpu.get_tensor_model_parallel_world_size()
    if local_length == padded_length:
        return axis, 0
    if local_length * tp_world == padded_length:
        return axis, mpu.get_tensor_model_parallel_rank() * local_length
    return None


def _decoder_component_stats(
    value, audit_rows: list[dict], input_ids: torch.Tensor, batch_size: int, padded_length: int
) -> list[dict]:
    tensor = _first_tensor_output(value)
    if tensor is None:
        return []
    layout = _local_sequence_axis(tensor, batch_size, padded_length)
    if layout is None:
        return []
    sequence_axis, offset = layout
    local_length = tensor.shape[sequence_axis]
    records = []
    for audit_row in audit_rows:
        row_idx = audit_row["row"]
        for response_index, model_position in enumerate(audit_row["positions"]):
            local_position = model_position - offset
            if not 0 <= local_position < local_length:
                continue
            vector = (
                tensor[row_idx, local_position]
                if sequence_axis == 1
                else tensor[local_position, row_idx]
            )
            stats = _vector_row_stats(vector.unsqueeze(0))
            if stats:
                records.append(
                    {
                        "input_ids_sha256": audit_row["input_ids_sha256"],
                        "input_len": audit_row["input_len"],
                        "response_len": audit_row["response_len"],
                        "response_index": response_index,
                        "model_position": model_position,
                        "input_token_id": int(input_ids[row_idx, model_position].item()),
                        "stats": stats[0],
                    }
                )
    return records


def _moe_router_audit_rows(
    value,
    router_input: torch.Tensor | None,
    router_logits: torch.Tensor | None,
    audit_rows: list[dict],
    input_ids: torch.Tensor,
    batch_size: int,
    padded_length: int,
) -> list[dict]:
    """Capture the complete router boundary at response-token coordinates."""
    if not isinstance(value, (tuple, list)) or len(value) < 2 or router_input is None:
        return []
    probs, routing_map = value[:2]
    if not isinstance(probs, torch.Tensor) or not isinstance(routing_map, torch.Tensor):
        return []
    if probs.ndim != 2 or routing_map.ndim != 2 or probs.shape != routing_map.shape:
        return []
    layout = _local_sequence_axis(router_input, batch_size, padded_length)
    if layout is None:
        return []
    sequence_axis, offset = layout
    local_length = router_input.shape[sequence_axis]
    records = []
    for audit_row in audit_rows:
        row_idx = audit_row["row"]
        for response_index, model_position in enumerate(audit_row["positions"]):
            local_position = model_position - offset
            if not 0 <= local_position < local_length:
                continue
            flat_index = local_position * batch_size + row_idx
            if not 0 <= flat_index < routing_map.shape[0]:
                continue
            expert_ids = torch.nonzero(routing_map[flat_index], as_tuple=False).flatten()
            input_vector = (
                router_input[row_idx, local_position]
                if sequence_axis == 1
                else router_input[local_position, row_idx]
            )
            logit_vector = None
            logit_stats = None
            logit_top_ids = []
            logit_top_values = []
            if router_logits is not None:
                logits_layout = _local_sequence_axis(router_logits, batch_size, padded_length)
                if logits_layout == layout:
                    logits_axis, logits_offset = logits_layout
                    logits_position = model_position - logits_offset
                    if 0 <= logits_position < router_logits.shape[logits_axis]:
                        logit_vector = (
                            router_logits[row_idx, logits_position]
                            if logits_axis == 1
                            else router_logits[logits_position, row_idx]
                        )
            if logit_vector is not None:
                logit_stats = _vector_row_stats(logit_vector.unsqueeze(0))[0]
                top_values, top_ids = torch.topk(logit_vector.detach().float(), min(16, logit_vector.numel()))
                logit_top_ids = [int(item) for item in top_ids.cpu().tolist()]
                logit_top_values = [float(item) for item in top_values.cpu().tolist()]
            raw_topk_size = int(expert_ids.numel())
            raw_topk_margin = None
            if logit_vector is not None and 0 < raw_topk_size < logit_vector.numel():
                boundary_values = torch.topk(
                    logit_vector.detach().float(), raw_topk_size + 1
                ).values
                raw_topk_margin = float(boundary_values[raw_topk_size - 1] - boundary_values[raw_topk_size])
            records.append(
                {
                    "input_ids_sha256": audit_row["input_ids_sha256"],
                    "input_len": audit_row["input_len"],
                    "response_len": audit_row["response_len"],
                    "response_index": response_index,
                    "model_position": model_position,
                    "input_token_id": int(input_ids[row_idx, model_position].item()),
                    "expert_ids": [int(value) for value in expert_ids.detach().cpu().tolist()],
                    "expert_probs": [
                        float(value)
                        for value in probs[flat_index, expert_ids].detach().float().cpu().tolist()
                    ],
                    "router_input_stats": _vector_row_stats(input_vector.unsqueeze(0))[0],
                    "router_logits_stats": logit_stats,
                    "router_logit_top_expert_ids": logit_top_ids,
                    "router_logit_top_values": logit_top_values,
                    # This is the global raw-logit top-k boundary. Group-limited
                    # routing may apply additional constraints afterward.
                    "router_raw_topk_size": raw_topk_size,
                    "router_raw_topk_margin": raw_topk_margin,
                }
            )
    return records


def _write_decoder_component_audit(
    audit: dict[tuple[int, str], list[dict]],
    input_ids: torch.Tensor,
    response_lengths: torch.Tensor,
    model,
    attention_stage_audit: list[dict] | None = None,
    attention_execution_audit: list[dict] | None = None,
    moe_router_audit: dict[int, list[dict]] | None = None,
    moe_router_weight_audit: list[dict] | None = None,
    moe_mlp_stage_audit: list[dict] | None = None,
    moe_replay_metadata_audit: list[dict] | None = None,
):
    base_path = os.getenv("VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL", "").strip()
    if not base_path or (
        not audit
        and not attention_stage_audit
        and not attention_execution_audit
        and not moe_router_audit
        and not moe_mlp_stage_audit
        and not moe_replay_metadata_audit
    ):
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.getenv("RANK", "0"))
    key = (rank, "decoder_component")
    limit = int(os.getenv("VERL_OMNI_MEGATRON_DECODER_COMPONENT_AUDIT_LIMIT", "1"))
    count = _MEGATRON_DECODER_AUDIT_COUNTS.get(key, 0)
    if count >= limit:
        return
    _MEGATRON_DECODER_AUDIT_COUNTS[key] = count + 1
    try:
        audit_rows, padded_length = _audit_sequence_rows(input_ids, response_lengths)
        if not audit_rows:
            return
        module = unwrap_model(model)
        if isinstance(module, list):
            module = module[0]
        config = module.config
        records = []
        for (layer, component), values in sorted(audit.items()):
            for entry in values:
                records.append({"layer": layer, "component": component, **entry})

        def _group_ranks(group):
            try:
                return torch.distributed.get_process_group_ranks(group)
            except (AttributeError, RuntimeError):
                return []

        payload = {
            "event": "megatron_decoder_component_audit",
            "rank": rank,
            "count": count,
            "router_replay_action": _router_replay_audit_action(),
            "tp_rank": mpu.get_tensor_model_parallel_rank(),
            "pp_rank": mpu.get_pipeline_model_parallel_rank(),
            "ep_rank": mpu.get_expert_model_parallel_rank(),
            "topology": {
                "tp": mpu.get_tensor_model_parallel_world_size(),
                "pp": mpu.get_pipeline_model_parallel_world_size(),
                "ep": mpu.get_expert_model_parallel_world_size(),
                "sequence_parallel": bool(getattr(config, "sequence_parallel", False)),
                "moe_dispatcher": getattr(config, "moe_token_dispatcher_type", None),
                "moe_router_score": getattr(config, "moe_router_score_function", None),
                "moe_router_pre_softmax": getattr(config, "moe_router_pre_softmax", None),
                "moe_router_expert_bias": getattr(config, "moe_router_enable_expert_bias", None),
                "attention_backend": str(getattr(config, "attention_backend", None)),
                "rotary_interleaved": getattr(config, "rotary_interleaved", None),
                "apply_rotary_pos_emb_in_fp32": getattr(config, "apply_rotary_pos_emb_in_fp32", None),
                "global_rank": rank,
                "hostname": socket.gethostname(),
                "tp_group_ranks": _group_ranks(mpu.get_tensor_model_parallel_group()),
                "pp_group_ranks": _group_ranks(mpu.get_pipeline_model_parallel_group()),
                "ep_group_ranks": _group_ranks(mpu.get_expert_model_parallel_group()),
                "dp_group_ranks": _group_ranks(mpu.get_data_parallel_group()),
            },
            "decoder_component_audit": records,
            "attention_stage_audit": attention_stage_audit or [],
            "attention_execution_audit": attention_execution_audit or [],
            "moe_router_audit": [
                {"layer": layer, **entry}
                for layer, entries in sorted((moe_router_audit or {}).items())
                for entry in entries
            ],
            "moe_router_weight_audit": moe_router_weight_audit or [],
            "moe_mlp_stage_audit": moe_mlp_stage_audit or [],
            "moe_replay_metadata_audit": moe_replay_metadata_audit or [],
        }
        path = f"{base_path}.decoder.rank{rank}.jsonl"
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_debug_jsonable(payload), ensure_ascii=True) + "\n")
    except Exception as exc:
        logger.warning("Failed to write Megatron decoder component audit: %r", exc)


def _write_vocab_parallel_debug(
    logits: torch.Tensor,
    label: torch.Tensor,
    log_probs: torch.Tensor | None = None,
    audit_input_ids: torch.Tensor | None = None,
    audit_attention_mask: torch.Tensor | None = None,
    audit_response_lengths: torch.Tensor | None = None,
    audit_input_weight: torch.Tensor | None = None,
    audit_output_weight: torch.Tensor | None = None,
    audit_output_bias: torch.Tensor | None = None,
    audit_pre_lm_hidden: torch.Tensor | None = None,
    audit_pre_final_norm_hidden: torch.Tensor | None = None,
    audit_final_norm_weight: torch.Tensor | None = None,
    audit_sequence_parallel: bool = False,
    audit_temperature: torch.Tensor | None = None,
):
    base_path = os.getenv("VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL", "").strip()
    if not base_path:
        return
    try:
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else int(os.getenv("RANK", "0"))
        key = (rank, "vocab_parallel")
        limit = int(os.getenv("VERL_OMNI_MEGATRON_VOCAB_DEBUG_LIMIT", "2"))
        count = _MEGATRON_VOCAB_DEBUG_COUNTS.get(key, 0)
        if count >= limit:
            return
        _MEGATRON_VOCAB_DEBUG_COUNTS[key] = count + 1

        rows = int(os.getenv("VERL_OMNI_MEGATRON_BSHD_DEBUG_ROWS", "2"))
        tokens = int(os.getenv("VERL_OMNI_MEGATRON_BSHD_DEBUG_TOKENS", "16"))
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_world = mpu.get_tensor_model_parallel_world_size()
        vocab_per_partition = int(logits.shape[-1])
        vocab_start = tp_rank * vocab_per_partition
        vocab_end = vocab_start + vocab_per_partition

        with torch.no_grad():
            logits_f = logits.detach().float()
            label_l = label.detach().long()
            owned = (label_l >= vocab_start) & (label_l < vocab_end)
            local_target = (label_l - vocab_start).clamp(min=0, max=vocab_per_partition - 1)
            target_logits = logits_f.gather(dim=-1, index=local_target.unsqueeze(-1)).squeeze(-1)
            target_logits = target_logits.masked_fill(~owned, 0.0)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(target_logits, group=mpu.get_tensor_model_parallel_group())

            logits_max = logits_f.max(dim=-1).values
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
            shifted_exp_sum = (logits_f - logits_max.unsqueeze(-1)).exp().sum(dim=-1)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(shifted_exp_sum, group=mpu.get_tensor_model_parallel_group())
            logsumexp = logits_max + shifted_exp_sum.log()
            manual_log_probs = target_logits - logsumexp

            local_top_values, local_top_indices = logits_f.max(dim=-1)
            local_top_token_ids = local_top_indices + vocab_start

            payload = {
                "event": "vocab_parallel_debug",
                "rank": rank,
                "local_rank": os.getenv("LOCAL_RANK"),
                "count": count,
                "tp_rank": tp_rank,
                "tp_world": tp_world,
                "vocab_start": vocab_start,
                "vocab_end": vocab_end,
                "logits_shape": _debug_jsonable(list(logits.shape)),
                "label_shape": _debug_jsonable(list(label.shape)),
                "label_head": _debug_tensor_2d_slice(label_l, rows, tokens),
                "label_tail": _debug_tensor_2d_slice(label_l, rows, tokens, tail=True),
                "owned_head": _debug_tensor_2d_slice(owned, rows, tokens),
                "owned_tail": _debug_tensor_2d_slice(owned, rows, tokens, tail=True),
                "target_logit_head": _debug_tensor_2d_slice(target_logits, rows, tokens),
                "target_logit_tail": _debug_tensor_2d_slice(target_logits, rows, tokens, tail=True),
                "logsumexp_head": _debug_tensor_2d_slice(logsumexp, rows, tokens),
                "logsumexp_tail": _debug_tensor_2d_slice(logsumexp, rows, tokens, tail=True),
                "manual_log_probs_head": _debug_tensor_2d_slice(manual_log_probs, rows, tokens),
                "manual_log_probs_tail": _debug_tensor_2d_slice(manual_log_probs, rows, tokens, tail=True),
                "local_top_token_head": _debug_tensor_2d_slice(local_top_token_ids, rows, tokens),
                "local_top_token_tail": _debug_tensor_2d_slice(local_top_token_ids, rows, tokens, tail=True),
                "local_top_logit_head": _debug_tensor_2d_slice(local_top_values, rows, tokens),
                "local_top_logit_tail": _debug_tensor_2d_slice(local_top_values, rows, tokens, tail=True),
            }
            if audit_output_weight is not None:
                payload["audit_output_weight_shape"] = _debug_jsonable(list(audit_output_weight.shape))
            if audit_input_weight is not None:
                payload["audit_input_weight_shape"] = _debug_jsonable(list(audit_input_weight.shape))
            if audit_output_bias is not None:
                payload["audit_output_bias_shape"] = _debug_jsonable(list(audit_output_bias.shape))
            if audit_pre_lm_hidden is not None:
                payload["audit_pre_lm_hidden_shape"] = _debug_jsonable(list(audit_pre_lm_hidden.shape))
            payload["audit_component_capture"] = {
                "pre_lm_hidden": audit_pre_lm_hidden is not None,
                "pre_final_norm_hidden": audit_pre_final_norm_hidden is not None,
                "final_norm_weight": audit_final_norm_weight is not None,
            }
            if audit_input_ids is not None and audit_attention_mask is not None and audit_response_lengths is not None:
                audit_rows = []
                for row_idx in range(min(int(logits.shape[0]), int(audit_input_ids.shape[0]))):
                    valid_input = audit_input_ids[row_idx][audit_attention_mask[row_idx].bool()]
                    valid_length = int(valid_input.numel())
                    response_length = int(audit_response_lengths[row_idx].item())
                    positions = list(response_score_positions(valid_length, response_length))
                    if not positions:
                        continue
                    index = torch.tensor(positions, device=logits.device, dtype=torch.long)
                    audit_rows.append(
                        {
                            "input_ids_sha256": _token_ids_sha256(valid_input),
                            "input_len": valid_length,
                            "response_len": response_length,
                            "score_positions": positions,
                            "label_token_ids": label_l[row_idx, index].detach().cpu().tolist(),
                            "target_logits": target_logits[row_idx, index].detach().cpu().tolist(),
                            "logsumexp": logsumexp[row_idx, index].detach().cpu().tolist(),
                            "manual_log_probs": manual_log_probs[row_idx, index].detach().cpu().tolist(),
                            "ce_log_probs": (
                                log_probs[row_idx, index].detach().float().cpu().tolist()
                                if log_probs is not None
                                else None
                            ),
                        }
                    )
                    audit_row = audit_rows[-1]
                    lm_head_weight = _vocab_parallel_weight_row_stats(
                        audit_output_weight, local_target[row_idx], owned[row_idx], index
                    )
                    if lm_head_weight is not None:
                        audit_row["lm_head_weight"] = lm_head_weight
                    input_vocab_per_partition = (
                        int(audit_input_weight.shape[0])
                        if audit_input_weight is not None and audit_input_weight.ndim == 2
                        else vocab_per_partition
                    )
                    input_vocab_start = tp_rank * input_vocab_per_partition
                    input_vocab_end = input_vocab_start + input_vocab_per_partition
                    input_token_ids = audit_input_ids[row_idx, index].detach().long()
                    input_owned = (input_token_ids >= input_vocab_start) & (input_token_ids < input_vocab_end)
                    input_local = (input_token_ids - input_vocab_start).clamp(
                        min=0, max=input_vocab_per_partition - 1
                    )
                    input_embedding = _vocab_parallel_weight_row_stats(
                        audit_input_weight,
                        input_local,
                        input_owned,
                        torch.arange(index.numel(), device=logits.device),
                    )
                    if input_embedding is not None:
                        audit_row["input_token_ids"] = input_token_ids.cpu().tolist()
                        audit_row["input_embedding"] = input_embedding
                    pre_final_norm_hidden = _align_pre_lm_hidden(
                        _gather_sequence_parallel_hidden(audit_pre_final_norm_hidden, audit_sequence_parallel), logits
                    )
                    if pre_final_norm_hidden is not None:
                        audit_row["pre_final_norm_hidden"] = _vector_row_stats(
                            pre_final_norm_hidden[row_idx, index]
                        )
                    if audit_final_norm_weight is not None and audit_final_norm_weight.ndim == 1:
                        audit_row["final_norm_weight"] = _vector_row_stats(
                            audit_final_norm_weight.detach().float().unsqueeze(0)
                        )[0]
                    hidden = _align_pre_lm_hidden(
                        _gather_sequence_parallel_hidden(audit_pre_lm_hidden, audit_sequence_parallel), logits
                    )
                    if hidden is not None:
                        audit_row["pre_lm_hidden"] = _vector_row_stats(hidden[row_idx, index])
                        manual_target_logits = _vocab_parallel_target_linear(
                            audit_output_weight,
                            audit_output_bias,
                            hidden[row_idx],
                            local_target[row_idx],
                            owned[row_idx],
                            index,
                        )
                        if manual_target_logits is not None:
                            if audit_temperature is not None:
                                manual_target_logits = [
                                    value / float(audit_temperature[row_idx, position].item())
                                    for value, position in zip(manual_target_logits, positions)
                                ]
                            audit_row["manual_lm_head_target_logits"] = manual_target_logits
                payload["response_score_audit"] = audit_rows
            if log_probs is not None:
                diff = log_probs.detach().float() - manual_log_probs
                payload.update(
                    {
                        "ce_log_probs_head": _debug_tensor_2d_slice(log_probs.detach().float(), rows, tokens),
                        "ce_log_probs_tail": _debug_tensor_2d_slice(log_probs.detach().float(), rows, tokens, tail=True),
                        "ce_minus_manual_head": _debug_tensor_2d_slice(diff, rows, tokens),
                        "ce_minus_manual_tail": _debug_tensor_2d_slice(diff, rows, tokens, tail=True),
                        "ce_minus_manual_abs_max": float(diff.abs().max().detach().cpu().item()),
                    }
                )

        path = f"{base_path}.vocab.rank{rank}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_debug_jsonable(payload), ensure_ascii=True) + "\n")
    except Exception as exc:
        print(f"[MegatronVocabDebug] failed to write vocab debug record: {exc}", flush=True)


class MegatronEngine(BaseEngine):
    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        assert self.engine_config.use_mbridge, "use_mbridge must be True"
        self._init_device_mesh()

        set_random_seed(seed=self.engine_config.seed)

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_grad = self.engine_config.grad_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload

        self.mode = None

        self.layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        self.weight_converter = None

        # QAT configuration
        self._qat_config = getattr(self.engine_config, "qat", None)
        self._qat_enabled = self._qat_config is not None and getattr(self._qat_config, "enable", False)
        if self._qat_enabled:
            if self.engine_config.vanilla_mbridge:
                raise ValueError(
                    "QAT requires non-vanilla Megatron bridge. "
                    "Please set 'use_mbridge=True' and 'vanilla_mbridge=False'."
                )
            logger.info(f"QAT enabled in MegatronEngine: mode={self._qat_config.mode}")

        # Router replay configuration for MoE models
        self.enable_routing_replay = self.engine_config.router_replay.mode != "disabled"
        logger.info(f"enable_routing_replay in MegatronEngine: {self.enable_routing_replay}")
        if self.enable_routing_replay:
            apply_router_replay_patch()
            self.mini_layer_topk_idx_list = []
        # Apply checkpoint patch for MoE models
        from verl.utils.device import is_cuda_available, is_npu_available

        if is_npu_available and __version__ >= "0.16.0":
            from verl.models.mcore.patch import apply_mtp_inference_patch

            apply_mtp_inference_patch()

        if is_cuda_available:
            from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward

            apply_patch_megatron_recomputation_backward()

    def _init_device_mesh(self):
        # TODO: set different parallelism for actor, critic, ref
        if mpu.is_initialized():
            return

        extra_args = dict()

        if self.engine_config.dynamic_context_parallel:
            assert "dynamic_context_parallel" in inspect.signature(mpu.initialize_model_parallel).parameters, (
                "dynamic_context_parallel is not supported in your megatron version, "
                + "please update your megatron version to the latest version"
            )
            assert self.engine_config.max_seqlen_per_dp_cp_rank is not None, (
                "max_seqlen_per_dp_cp_rank is required when dynamic_context_parallel is enabled"
            )
            extra_args["dynamic_context_parallel"] = self.engine_config.dynamic_context_parallel

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.engine_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.engine_config.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=self.engine_config.virtual_pipeline_model_parallel_size,
            use_sharp=False,
            context_parallel_size=self.engine_config.context_parallel_size,
            expert_model_parallel_size=self.engine_config.expert_model_parallel_size,
            expert_tensor_parallel_size=self.engine_config.expert_tensor_parallel_size,
            nccl_communicator_config_path=None,
            **extra_args,
        )

    def _build_tf_config(self):
        from verl.utils.megatron_utils import mapping_string_to_attn_backend
        from verl.utils.torch_dtypes import PrecisionType

        check_mtp_config(self.model_config, self.engine_config)

        self.param_dtype = PrecisionType.to_dtype(self.engine_config.dtype)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        override_transformer_config = mapping_string_to_attn_backend({**self.engine_config.override_transformer_config})
        if self.engine_config.dynamic_context_parallel:
            override_transformer_config["max_seqlen_per_dp_cp_rank"] = self.engine_config.max_seqlen_per_dp_cp_rank
            # note(baiyan): we must set the transformer_config.dynamic_context_parallel to False
            # because of the bad coupling design in Megatron-LM
            # https://github.com/xiaoyao0115/Megatron-LM/blob/88733ab6614e3e91b9d095172f41e7d8b5d8e9d4/megatron/core/pipeline_parallel/dynamic_cp_schedule.py#L552-L553
            # but it does not affect the functionality of dynamic CP, so we can use it to avoid the coupling.
            override_transformer_config["dynamic_context_parallel"] = False
            override_transformer_config["context_parallel_size"] = mpu.get_data_parallel_world_size()
        self.provider = None
        self.vanilla_bridge = self.engine_config.vanilla_mbridge

        if self.vanilla_bridge:
            from verl.models.mcore.mbridge import AutoBridge

            bridge = AutoBridge.from_config(self.model_config.hf_config, dtype=self.param_dtype)
            bridge.set_extra_args(**override_transformer_config)
            tf_config = bridge.config
            tf_config.fp16 = self.param_dtype == torch.float16
            tf_config.bf16 = self.param_dtype == torch.bfloat16
        else:
            from verl.models.mcore.bridge import AutoBridge

            # Use Megatron-Bridge to convert HF config to Megatron config
            bridge = AutoBridge.from_hf_pretrained(
                self.model_config.local_path, trust_remote_code=self.model_config.trust_remote_code
            )
            # Get Megatron provider and configure it
            provider = bridge.to_megatron_provider(load_weights=False)

            # Match verl implementation (need variable_seq_lengths)
            from megatron.core.transformer.enums import AttnBackend

            provider_overrides = {
                "tensor_model_parallel_size": self.engine_config.tensor_model_parallel_size,
                "pipeline_model_parallel_size": self.engine_config.pipeline_model_parallel_size,
                "expert_model_parallel_size": self.engine_config.expert_model_parallel_size,
                "expert_tensor_parallel_size": self.engine_config.expert_tensor_parallel_size,
                "virtual_pipeline_model_parallel_size": self.engine_config.virtual_pipeline_model_parallel_size,
                "context_parallel_size": self.engine_config.context_parallel_size,
                "sequence_parallel": self.engine_config.sequence_parallel,
                "variable_seq_lengths": True,
                "attention_backend": AttnBackend.flash,
                "moe_token_dispatcher_type": "alltoall",
                "moe_router_load_balancing_type": "none",
            }
            for key, value in override_transformer_config.items():
                provider_overrides[key] = value
            if self.enable_routing_replay:
                provider_overrides["enable_routing_replay"] = True

            if self._qat_enabled:
                from megatron.bridge.models.gpt_provider import modelopt_transformer_layer_spec

                provider.transformer_layer_spec = modelopt_transformer_layer_spec

            provider.apply_overrides_and_finalize(
                dtype=self.param_dtype,
                overrides=provider_overrides,
            )
            self.provider = provider
            tf_config = None  # Will be set after model creation
        self.bridge = bridge

        if not self.bridge:
            self.weight_converter = get_mcore_weight_converter(self.model_config.hf_config, self.dtype)

        # Set enable_routing_replay directly on tf_config instead of passing through
        # override_transformer_config, because dataclass subclasses like MLATransformerConfig
        # generate their own __init__ and don't inherit the patched TransformerConfig.__init__
        # that accepts this kwarg.
        if self.enable_routing_replay and tf_config is not None:
            tf_config.enable_routing_replay = True

        if torch.distributed.get_rank() == 0:
            if tf_config is not None:
                print(f"TF config: {tf_config}")
        self.tf_config = tf_config

        from verl.workers.config.megatron_peft import get_peft_cls

        self.peft_cls = get_peft_cls(
            model_config=self.model_config, bridge=self.bridge, provider=self.provider, dtype=self.param_dtype
        )

    def _build_megatron_module(self):
        from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
        from verl.utils.model import print_model_size

        self.is_value_model = self.model_config.model_type == "value_model"
        if self.engine_config.forward_only:
            wrap_with_ddp = False
        else:
            wrap_with_ddp = True

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=self.is_value_model,
            wrap_with_ddp=wrap_with_ddp,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            use_megatron_fsdp=self.engine_config.use_megatron_fsdp,
        )
        if self.is_value_model:
            self.model_config.hf_config.tie_word_embeddings = False

        module, updated_tf_config = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.model_config.hf_config,
            bridge=self.bridge,
            provider=self.provider,
            override_model_config=self.engine_config.override_mcore_model_config,
            override_ddp_config=self.engine_config.override_ddp_config,
            peft_cls=self.peft_cls,
            peft_config=self.model_config.get("lora", None),
        )
        self.tf_config = updated_tf_config
        print(f"module: {len(module)}")

        if self.engine_config.use_dist_checkpointing:
            load_mcore_dist_weights(
                module, self.engine_config.dist_checkpointing_path, is_value_model=self.is_value_model
            )
        else:
            if self.vanilla_bridge:
                self.bridge.load_weights(module, self.model_config.local_path)
            else:
                allowed_mismatched_params = []
                if self.is_value_model:
                    allowed_mismatched_params = ["output_layer.weight"]
                self.bridge.load_hf_weights(
                    module, self.model_config.local_path, allowed_mismatched_params=allowed_mismatched_params
                )

        if torch.distributed.get_rank() == 0:
            print_model_size(module[0])

        if self.enable_routing_replay:
            print(f"routing replay layers: {len(RouterReplay.router_instances)}")

        return module

    def _maybe_enable_fused_kernels(self):
        if not self.engine_config.use_fused_kernels:
            return

        if self.is_value_model or self.model_config.mtp.enable:
            logger.warning_once(
                "Fused kernels are not supported for value models or when MTP is enabled in Megatron engine; disabling."
            )
            self.engine_config.use_fused_kernels = False
            return

        from verl.models.mcore.model_forward_fused import patch_fused_forward

        for model in self.module:
            patch_fused_forward(model)

    def _build_optimizer(self):
        from verl.utils.megatron.optimizer import get_megatron_optimizer, init_megatron_optim_config

        optim_config_megatron = init_megatron_optim_config(
            self.optimizer_config,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            fp16=self.param_dtype == torch.float16,
        )
        optimizer = get_megatron_optimizer(model=self.module, config=optim_config_megatron)
        register_megatron_training_hooks(self.module, optimizer)
        return optimizer

    def _build_lr_scheduler(self):
        from verl.utils.megatron.optimizer import get_megatron_optimizer_param_scheduler

        optimizer_scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer, config=self.optimizer_config
        )
        return optimizer_scheduler

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self):
        return (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0
        )

    def initialize(self):
        self._build_tf_config()

        self.module = self._build_megatron_module()

        if self._qat_enabled and not self.engine_config.forward_only:
            from verl.utils.modelopt import apply_qat_to_modules

            self.module = apply_qat_to_modules(self.module, self._qat_config)

        self._maybe_enable_fused_kernels()

        if self.model_config.mtp.enable:
            patch_engine_mtp(self.module, self.model_config)
        elif (
            self.engine_config.forward_only
            and self.engine_config.override_transformer_config.get("mtp_num_layers") == 0
        ):
            from verl.models.mcore.mtp_patch import patch_postprocess

            for model in self.module:
                patch_postprocess(model)

        # For forward_only, we don't need optimizer, lr_scheduler, checkpoint_mananager
        if self.engine_config.forward_only:
            self.optimizer = None
            self.lr_scheduler = None
            self.to(device="cpu", model=self._is_offload_param, optimizer=False, grad=False)
            log_gpu_memory_usage("After offload model during init (forward_only)", logger=logger)
            return

        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

        full_reshardable = self.engine_config.dist_ckpt_optim_fully_reshardable
        mem_eff = self.engine_config.distrib_optim_fully_reshardable_mem_efficient

        tmp_config = OmegaConf.create(
            {
                "model": {"path": self.model_config.local_path},
                "megatron": {
                    "dist_ckpt_optim_fully_reshardable": full_reshardable,
                    "distrib_optim_fully_reshardable_mem_efficient": mem_eff,
                },
            }
        )

        role = "actor" if not self.is_value_model else "critic"

        self.checkpoint_mananager = MegatronCheckpointManager(
            config=tmp_config,
            checkpoint_config=self.checkpoint_config,
            model_config=self.model_config.hf_config,
            transformer_config=self.tf_config,
            role=role,
            model=self.module,
            arch=self.model_config.architectures[0],
            hf_config=self.model_config.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=self.model_config.share_embeddings_and_output_weights,
            processing_class=self.model_config.get_processor(),
            optimizer=self.optimizer,
            optimizer_scheduler=self.lr_scheduler,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.optimizer_config.use_checkpoint_opt_param_scheduler,
            bridge=self.bridge,
            provider=self.provider,
            peft_cls=self.peft_cls,
            use_dist_checkpointing=self.engine_config.use_dist_checkpointing,
            use_megatron_fsdp=self.engine_config.use_megatron_fsdp,
        )

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def train_mode(self, **kwargs):
        """
        Context manager entry for switching the engine and model into training mode.

        Usage:
            with engine.train_mode():
                # runs in training mode
        """
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """
        Context manager entry for switching the engine and model into evaluation mode.

        Usage:
            with engine.eval_mode():
                # runs in evaluation mode
        """
        return EngineEvalModeCtx(self, **kwargs)

    def optimizer_zero_grad(self):
        """
        Zero out gradients of all parameters before starting a new backward pass.
        """
        self.optimizer.zero_grad()
        # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
        for chunk in self.module:
            # if use distributed optimizer, zero grad buffer will be handled by optimizer
            chunk.zero_grad_buffer()

    def optimizer_step(self):
        """
        Perform an optimization step to update model parameters based on accumulated gradients.

        Returns:
            grad_norm (float): The norm of the gradients before clipping or update.
        """
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()

        if update_successful:
            # allgather already execute in optimizer.step in new megatron
            pass
        else:
            raise NotImplementedError("Megatron optimizer step failed. This should not happen")

        return grad_norm

    def lr_scheduler_step(self):
        """
        Advance the learning rate scheduler by one step.

        Returns:
            current_lr (float or list[float]): Updated learning rate(s).
        """
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        self.lr_scheduler.step(1)
        return get_megatron_last_lr(self.optimizer)

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.
        Note that this function executes irrespective of offload config. It serves as manual control

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        device_name = get_device_name()

        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_megatron_model_to_gpu(self.module, load_grad=grad)
            if optimizer and self.optimizer is not None:
                load_megatron_optimizer(self.optimizer)
        elif device == "cpu":
            if model:
                offload_megatron_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_megatron_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def get_data_parallel_rank(self):
        if self.engine_config.dynamic_context_parallel:
            # in order to let every dp-cp group has full data to split, we set dp=1
            return 0
        return mpu.get_data_parallel_rank()

    def get_data_parallel_size(self):
        if self.engine_config.dynamic_context_parallel:
            # in order to let every dp-cp group has full data to split, we set dp=1
            return 1
        return mpu.get_data_parallel_world_size()

    def get_data_parallel_group(self):
        return mpu.get_data_parallel_group()

    def get_model_parallel_group(self):
        return mpu.get_model_parallel_group()

    def get_context_parallel_group(self):
        return mpu.get_context_parallel_group()

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Save model, optimizer, and scheduler states to a checkpoint.

        Args:
            local_path: Local filesystem path to save checkpoint.
            hdfs_path: Optional HDFS path to copy checkpoint.
            global_step: Integer training step number for naming.
            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.
        """
        origin_module_device = get_megatron_module_device(self.module)
        if self._is_offload_param or origin_module_device == "cpu":
            load_megatron_model_to_gpu(self.module, load_grad=True)
        self.checkpoint_mananager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: bool = True, **kwargs
    ) -> None:
        """
        Load model, optimizer, and scheduler states from a checkpoint.

        Args:
            local_path: Local filesystem path of the checkpoint.
            hdfs_path: Optional HDFS path where checkpoint is stored.
            del_local_after_load: Whether to delete local copy after loading.
        """
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.optimizer)

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        tu.assign_non_tensor(data, sp_size=self.engine_config.context_parallel_size)

        # compute num_tokens in global batch for loss normalization
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        if vpp_size is not None and vpp_size > 1:
            num_batches_divided_by = self.tf_config.microbatch_group_size_per_vp_stage
        else:
            num_batches_divided_by = None

        micro_batches, indices = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            num_batches_divided_by=num_batches_divided_by,
            same_micro_num_in_dp=True,
            min_num_micro_batch=None,
        )

        if num_batches_divided_by is not None:
            assert len(micro_batches) % num_batches_divided_by == 0, (
                f"micro_batches {micro_batches} must be divisible by num_batches_divided_by "
                f"{num_batches_divided_by} for megatron backend"
            )

        # compute input shapes for pp stages
        n_micro_batch = len(micro_batches)

        for micro_batch in micro_batches:
            tu.assign_non_tensor(micro_batch, num_micro_batch=n_micro_batch)

        forward_backward_func = get_forward_backward_func()

        postprocess_micro_batch_func = partial(
            self.postprocess_micro_batch_func,
            forward_only=forward_only,
            loss_function=loss_function,
        )

        tu.assign_non_tensor(data, num_micro_batch=n_micro_batch)

        forward_step = partial(
            self.forward_step,
            logits_processor_func=loss_function,
            postprocess_micro_batch_func=postprocess_micro_batch_func,
        )

        enable_routing_replay = tu.get_non_tensor_data(data, key="enable_routing_replay", default=False)

        if enable_routing_replay:
            # Set to REPLAY mode: for R3 mode or actor update phase in R2 mode
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            has_replay_routes = "routed_experts" in data.keys()
            if forward_only and self.engine_config.router_replay.mode == "R2" and not has_replay_routes:
                # In R2 mode, forward_only calls (e.g., compute_log_probs) need to record routing information
                RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.module,
            num_microbatches=n_micro_batch,
            seq_length=1,  # the communication shape is obtained via p2p comm
            micro_batch_size=1,  # the communication shape is obtained via p2p comm
            forward_only=forward_only,
        )

        if self.model_config.mtp.enable and mpu.is_pipeline_last_stage(ignore_virtual=True):
            # All CP ranks must participate in the all_reduce inside get_megatron_mtp_loss,
            # because save_loss_to_tracker uses avg_group=DP+CP group.
            # Only collect metrics on the src rank afterward.
            metrics = get_megatron_mtp_loss(n_micro_batch)
            if self.is_mp_src_rank_with_outputs():
                if "metrics" not in losses_reduced[0]:
                    losses_reduced[0]["metrics"] = {}
                losses_reduced[0]["metrics"].update(metrics)

        if RouterReplayHelper.is_r2_record_action(self.tf_config):
            if self.tf_config.virtual_pipeline_model_parallel_size is not None:
                # config = self.actor_module[0].module.module.config
                vp_size = len(self.module)
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                bs = n_micro_batch
                topk_idx_td = reorder_and_merge_vpp_layers(
                    self.mini_layer_topk_idx_list, bs, vp_size, microbatch_group_size_per_vp_stage
                )
            else:
                tensors = [tensor for nt in self.mini_layer_topk_idx_list for tensor in nt.unbind()]
                topk_idx_td = torch.nested.as_nested_tensor(tensors, layout=torch.jagged)
            self.mini_layer_topk_idx_list = []

            layers_topk_idx = pp_gather(topk_idx_td.to(torch.uint8), self.tf_config)
            use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=True)
            if use_dynamic_bsz and indices is not None:
                layers_topk_idx = restore_dynamic_batch(layers_topk_idx, indices)

        output = {}
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            output = postprocess_batch_func(output_lst=losses_reduced, indices=indices, data=data)
            if RouterReplayHelper.is_r2_record_action(self.tf_config):
                output["model_output"]["routed_experts"] = layers_topk_idx
        if enable_routing_replay:
            RouterReplay.clear_global_indices()
            RouterReplay.clear_global_router_replay_action()
        return output

    def get_per_tensor_param(self, base_sync_done=False, **kwargs):
        peft_config = None
        non_merge_lora_sync = self.peft_cls is not None and not self.model_config.lora.get("merge", False)
        adapter_only = base_sync_done and non_merge_lora_sync
        if non_merge_lora_sync:
            peft_config = build_peft_config_for_vllm(self.model_config.lora)
        # when lora adapter only, we only load adapter weights when base sync is done, otherwise load all weights
        load_megatron_model_to_gpu(self.module, load_grad=False, load_frozen_params=not adapter_only)
        if self.vanilla_bridge:
            per_tensor_param = self.bridge.export_weights(self.module)
        elif adapter_only:
            per_tensor_param = self.bridge.export_adapter_weights(self.module)
            per_tensor_param = gather_ep_lora_adapter_weights_for_vllm(per_tensor_param)
            per_tensor_param = pack_3d_moe_lora_adapter_weights_for_vllm(
                per_tensor_param, model_type=self.model_config.hf_config.model_type
            )
        else:
            per_tensor_param = (
                self.bridge.export_hf_weights(self.module, merge_adapter_weights=False)
                if non_merge_lora_sync
                else self.bridge.export_hf_weights(self.module)
            )
            if non_merge_lora_sync:
                per_tensor_param = add_base_layer_suffix(
                    per_tensor_param, model_type=self.model_config.hf_config.model_type
                )

        # QAT: process weights through QATWeightExporter for quantized weight sync to vLLM
        if self._qat_enabled:
            from verl.utils.modelopt import export_qat_weights

            per_tensor_param = export_qat_weights(per_tensor_param, self.module, self._qat_config.mode, self.bridge)

        return per_tensor_param, peft_config

    def disable_adapter(self) -> ContextManager:
        return self.peft_cls.disable_adapter(self.module)

    def forward_step(self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def postprocess_micro_batch_func(self, output, data: TensorDict, forward_only: bool, loss_function):
        raise NotImplementedError("postprocess_micro_batch_func must be implemented in subclass")


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: MegatronEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, MegatronEngine)
        super().__enter__()
        # mcore module is a list of model chunk in each vpp stage
        for module in self.engine.module:
            module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, MegatronEngine)
        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: MegatronEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, MegatronEngine)
        super().__enter__()
        # mcore module is a list of model chunk in each vpp stage
        for module in self.engine.module:
            module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, MegatronEngine)
        self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend="megatron")
class MegatronEngineWithLMHead(MegatronEngine):
    def prepare_model_inputs(self, batch: TensorDict):
        input_ids = batch["input_ids"]
        loss_mask = batch["loss_mask"].to(bool)
        multi_modal_inputs = extract_multi_modal_inputs(batch.get("multi_modal_inputs", []))

        routed_experts = batch.get("routed_experts", None)

        return {
            "input_ids": input_ids,
            "attention_mask": batch.get("attention_mask", None),
            "loss_mask": loss_mask,
            "multi_modal_inputs": multi_modal_inputs,
            "routed_experts": routed_experts,
        }

    def prepare_model_outputs(self, output: dict, data: TensorDict):
        return output

    def forward_step(
        self, batch_iter: Iterator[TensorDict], model, logits_processor_func, postprocess_micro_batch_func
    ):
        batch: TensorDict = next(batch_iter)

        if self.engine_config.dynamic_context_parallel:
            # split the batch and give the sub-batches to each dp-cp group
            from verl.utils.megatron_utils import dynamic_cp_split_batch

            batch = dynamic_cp_split_batch(
                batch=batch,
                engine_config=self.engine_config,
                dp_size=mpu.get_data_parallel_world_size(),
                dp_rank=mpu.get_data_parallel_rank(),
            )

        batch = batch.to(get_device_id())
        use_fused_kernels = tu.get_non_tensor_data(batch, key="use_fused_kernels", default=False)
        calculate_entropy = tu.get_non_tensor_data(batch, key="calculate_entropy", default=False)
        calculate_sum_pi_squared = tu.get_non_tensor_data(batch, key="calculate_sum_pi_squared", default=False)
        distillation_use_topk = tu.get_non_tensor_data(batch, key="distillation_use_topk", default=False)

        if calculate_sum_pi_squared and use_fused_kernels:
            raise NotImplementedError(
                "calculate_sum_pi_squared=True is not supported with use_fused_kernels=True: "
                "fused kernels do not materialize the full logits tensor needed for Σπ²."
            )
        pad_mode = tu.get_non_tensor_data(batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        temperature = batch["temperature"]
        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]
        local_cp_size = tu.get_non_tensor_data(data=batch, key="local_cp_size", default=None)
        loss_mask = model_inputs["loss_mask"]

        unwrapped_model = unwrap_model(model)
        if hasattr(unwrapped_model, "vp_stage"):
            vp_rank = unwrapped_model.vp_stage
        else:
            vp_rank = 0

        if RouterReplayHelper.is_replay_backward_action(self.tf_config, vp_rank):
            router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
            for router in router_instance_list:
                router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

        if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
            layers_topk_idx = model_inputs["routed_experts"]
            set_router_replay_data(layers_topk_idx, attention_mask, self.tf_config, vp_rank)

        if pad_mode == DatasetPadMode.NO_PADDING:
            label = input_ids.clone()
        else:
            raise NotImplementedError(f"Pad mode {pad_mode} is not supported for megatron engine")

        if use_fused_kernels:
            if not self.engine_config.use_remove_padding:
                logger.warning_once(
                    "Fused kernels require `use_remove_padding=True` for Megatron engine. Falling back to non-fused."
                )
                use_fused_kernels = False
            elif isinstance(temperature, torch.Tensor):
                if temperature.numel() != 1:
                    logger.warning_once(
                        "Fused kernels do not support per-sample temperature. Falling back to non-fused."
                    )
                    use_fused_kernels = False
                else:
                    temperature_value = float(temperature.item())
            else:
                temperature_value = float(temperature)

        if use_fused_kernels:
            from verl.models.mcore import get_mcore_forward_fused_model_engine_fn

            fused_forward_fn = get_mcore_forward_fused_model_engine_fn(self.model_config.hf_config)
            output = fused_forward_fn(
                model=model,
                input_ids=input_ids,
                labels=label,
                multi_modal_inputs=multi_modal_inputs,
                temperature=temperature_value,
                calculate_entropy=calculate_entropy,
                pad_token_id=self.model_config.tokenizer.pad_token_id,
            )
        else:
            if not isinstance(temperature, torch.Tensor):
                temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

            temperature = temperature.to(torch.float32)
            assert temperature.shape[0] == input_ids.shape[0]
            temperature = verl_F.expand_as_nested(temperature, input_ids)  # (bsz, j1)
            from verl.models.mcore import get_mcore_engine_forward_fn

            forward_fn = get_mcore_engine_forward_fn(self.model_config.hf_config)
            data_format = "thd" if self.engine_config.use_remove_padding else "bshd"
            component_audit = data_format == "bshd" and _component_audit_enabled()
            decoder_component_audit = data_format == "bshd" and _decoder_component_audit_enabled()
            audit_capture: dict[str, torch.Tensor] = {}
            audit_output_layer = _get_output_layer(model) if component_audit else None
            audit_input_embedding = _get_input_embedding(model) if component_audit else None
            audit_final_layernorm = _get_final_layernorm(model) if component_audit else None
            audit_hooks = []
            audit_cleanups: list[Callable[[], None]] = []
            decoder_audit: dict[tuple[int, str], list[dict]] = {}
            decoder_audit_rows: list[dict] = []
            decoder_audit_padded_length = 0
            decoder_audit_response_lengths = None
            attention_stage_audit: list[dict] = []
            attention_execution_audit: list[dict] = []
            moe_router_audit: dict[int, list[dict]] = {}
            moe_router_weight_audit: list[dict] = []
            moe_mlp_stage_audit: list[dict] = []
            moe_replay_metadata_audit: list[dict] = []
            clear_attention_audit = None
            get_attention_execution_audit = None
            if decoder_component_audit:
                decoder_audit_response_lengths = loss_mask.sum(dim=-1)
                decoder_audit_rows, decoder_audit_padded_length = _audit_sequence_rows(
                    input_ids, decoder_audit_response_lengths
                )
                from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention_audit import (
                    attention_execution_audit as _attention_execution_audit,
                    clear_attention_audit as _clear_attention_audit,
                    configure_attention_audit,
                )

                configure_attention_audit(decoder_audit_rows, input_ids, decoder_audit_padded_length)
                clear_attention_audit = _clear_attention_audit
                get_attention_execution_audit = _attention_execution_audit

                def _capture_decoder_component(layer_number: int, component: str, *, pre_hook: bool = False):
                    def _capture(value):
                        key = (layer_number, component)
                        if key in decoder_audit:
                            return
                        stats = _decoder_component_stats(
                            value,
                            decoder_audit_rows,
                            input_ids,
                            int(input_ids.shape[0]),
                            decoder_audit_padded_length,
                        )
                        if stats:
                            decoder_audit[key] = stats

                    if pre_hook:
                        def _pre_hook(_module, args, kwargs):
                            _capture(_first_tensor_argument(args, kwargs, ("hidden_states", "input_")))

                        return _pre_hook

                    def _forward_hook(_module, _args, output):
                        _capture(output)

                    return _forward_hook

                def _capture_moe_router(layer_number: int, router):
                    router_input = {"value": None}
                    router_logits = {"value": None}

                    def _pre_hook(_module, args, kwargs):
                        router_input["value"] = _first_tensor_argument(
                            args, kwargs, ("input", "hidden_states")
                        )

                    def _forward_hook(_module, _args, output):
                        if layer_number in moe_router_audit:
                            return
                        rows = _moe_router_audit_rows(
                            output,
                            router_input["value"],
                            router_logits["value"],
                            decoder_audit_rows,
                            input_ids,
                            int(input_ids.shape[0]),
                            decoder_audit_padded_length,
                        )
                        if rows:
                            moe_router_audit[layer_number] = rows

                    return _pre_hook, _forward_hook, router_logits

                for decoder_layer in _get_decoder_layers(model):
                    layer_number = int(getattr(decoder_layer, "layer_number", -1))
                    if layer_number < 1:
                        continue
                    audit_hooks.append(
                        decoder_layer.register_forward_pre_hook(
                            _capture_decoder_component(layer_number, "layer_input", pre_hook=True), with_kwargs=True
                        )
                    )
                    audit_hooks.append(
                        decoder_layer.register_forward_hook(_capture_decoder_component(layer_number, "layer"))
                    )
                    self_attention = getattr(decoder_layer, "self_attention", None)
                    if self_attention is not None:
                        audit_hooks.append(
                            self_attention.register_forward_hook(
                                _capture_decoder_component(layer_number, "self_attention")
                            )
                        )
                    pre_mlp_layernorm = getattr(decoder_layer, "pre_mlp_layernorm", None)
                    if pre_mlp_layernorm is not None:
                        audit_hooks.append(
                            pre_mlp_layernorm.register_forward_pre_hook(
                                _capture_decoder_component(
                                    layer_number, "post_attention_residual", pre_hook=True
                                ),
                                with_kwargs=True,
                            )
                        )
                        audit_hooks.append(
                            pre_mlp_layernorm.register_forward_hook(
                                _capture_decoder_component(layer_number, "post_attention_norm")
                            )
                        )
                    mlp = getattr(decoder_layer, "mlp", None)
                    if mlp is not None:
                        audit_hooks.append(
                            mlp.register_forward_hook(_capture_decoder_component(layer_number, "mlp"))
                        )
                        if (
                            _moe_mlp_stage_audit_enabled()
                            and layer_number in _moe_mlp_stage_audit_layers()
                        ):
                            mlp_stage_cleanup = _install_moe_mlp_stage_audit(
                                mlp,
                                layer_number,
                                moe_mlp_stage_audit,
                                moe_replay_metadata_audit if _moe_replay_metadata_audit_enabled() else None,
                            )
                            if mlp_stage_cleanup is not None:
                                audit_cleanups.append(mlp_stage_cleanup)
                        router = getattr(mlp, "router", None)
                        if _moe_router_audit_enabled() and layer_number in _moe_router_audit_layers() and router is not None:
                            router_pre_hook, router_forward_hook, router_logits = _capture_moe_router(layer_number, router)
                            audit_hooks.append(router.register_forward_pre_hook(router_pre_hook, with_kwargs=True))
                            audit_hooks.append(router.register_forward_hook(router_forward_hook))
                            gate = getattr(router, "gating", None)
                            if gate is not None:
                                fingerprint = _tensor_fingerprint(
                                    getattr(gate, "weight", getattr(router, "weight", None))
                                )
                                if fingerprint is not None:
                                    moe_router_weight_audit.append({"layer": layer_number, **fingerprint})
                                gate_cleanup = _install_moe_gate_logits_audit(router, router_logits)
                                if gate_cleanup is not None:
                                    audit_cleanups.append(gate_cleanup)
            if audit_output_layer is not None:
                def _capture_pre_lm_hidden(_module, args, kwargs=None):
                    # Qwen3-VL's GPT postprocess calls LinearForLastLayer with
                    # input_ as a keyword; do not mistake its optional weight
                    # override for the hidden activation.
                    hidden = _first_tensor_argument(args, kwargs, ("input_", "hidden_states"))
                    if hidden is not None:
                        audit_capture["pre_lm_hidden"] = hidden.detach()

                audit_hooks.append(
                    audit_output_layer.register_forward_pre_hook(_capture_pre_lm_hidden, with_kwargs=True)
                )
            if audit_final_layernorm is not None:
                def _capture_pre_final_norm_hidden(_module, args, kwargs=None):
                    hidden = _first_tensor_argument(args, kwargs, ("hidden_states", "input_"))
                    if hidden is not None:
                        audit_capture["pre_final_norm_hidden"] = hidden.detach()

                audit_hooks.append(
                    audit_final_layernorm.register_forward_pre_hook(
                        _capture_pre_final_norm_hidden, with_kwargs=True
                    )
                )

            def logits_processor(
                logits,
                label,
                temperature,
                audit_input_ids=None,
                audit_attention_mask=None,
                audit_response_lengths=None,
            ):
                assert logits.shape[:2] == label.shape[:2]
                # avoid non-positive temperature such as padding
                temperature[temperature <= 0] = 1e-8
                assert torch.all(temperature > 0).item(), f"temperature tensor must be positive. Got {temperature}"
                logits.div_(temperature.unsqueeze(dim=-1).to(logits.dtype))
                ret = {}
                # sum_pi_squared is non-destructive — must run before vocab_parallel_entropy.
                if calculate_sum_pi_squared:
                    ret["sum_pi_squared"] = vocab_parallel_sum_pi_squared(logits)
                if calculate_entropy:
                    logits_bak = logits.clone()
                    # # disable the hint until the fused_kernel is optimized for triton>=3.3
                    # if torch.distributed.get_rank() == 0:
                    #     logger.warning_once(
                    #         "For memory-efficient computation, enable fused kernels via "
                    #         "`actor_rollout_ref.model.use_fused_kernels=True`. "
                    #         "The current `clone()` operation ensures correctness but increases memory usage."
                    #     )
                    entropy = vocab_parallel_entropy(logits)
                    ret["entropy"] = entropy
                else:
                    logits_bak = logits

                # logits_processor_func return tensors with shape (1, total_nnz/cp_size)
                if distillation_use_topk:
                    ret.update(logits_processor_func(student_logits=logits_bak, data=batch, data_format=data_format))
                vocab_debug_logits = logits_bak.detach().clone() if _vocab_parallel_debug_enabled() else None
                log_probs = vocab_parallel_log_probs_from_logits(logits_bak, label)
                if vocab_debug_logits is not None:
                    _write_vocab_parallel_debug(
                        vocab_debug_logits,
                        label,
                        log_probs,
                        audit_input_ids=audit_input_ids,
                        audit_attention_mask=audit_attention_mask,
                        audit_response_lengths=audit_response_lengths,
                        audit_input_weight=(
                            audit_input_embedding.weight if audit_input_embedding is not None else None
                        ),
                        audit_output_weight=(
                            audit_output_layer.weight if audit_output_layer is not None else None
                        ),
                        audit_output_bias=(
                            getattr(audit_output_layer, "bias", None) if audit_output_layer is not None else None
                        ),
                        audit_pre_lm_hidden=audit_capture.get("pre_lm_hidden"),
                        audit_pre_final_norm_hidden=audit_capture.get("pre_final_norm_hidden"),
                        audit_final_norm_weight=(
                            getattr(audit_final_layernorm, "weight", None)
                            if audit_final_layernorm is not None
                            else None
                        ),
                        audit_sequence_parallel=bool(
                            getattr(audit_output_layer, "sequence_parallel", False)
                        ),
                        audit_temperature=temperature,
                    )
                ret["log_probs"] = log_probs
                return ret

            response_attention_mask = None
            if attention_mask is not None and not loss_mask.is_nested:
                response_attention_mask = attention_mask[:, -loss_mask.shape[-1] :]
            logits_processor_args = {
                "label": label,
                "temperature": temperature,
                "loss_mask": loss_mask,
                "response_attention_mask": response_attention_mask,
            }
            if component_audit:
                # The final response tokens are contiguous in the unpadded
                # sequence. Keep only their counts so the audit can emit the
                # exact causal score positions without carrying broad tensors.
                logits_processor_args["audit_input_ids"] = input_ids
                logits_processor_args["audit_attention_mask"] = attention_mask
                logits_processor_args["audit_response_lengths"] = loss_mask.sum(dim=-1)

            if RouterReplayHelper.is_r2_record_action(self.tf_config, vp_rank):
                RouterReplay.set_global_record_padding_mask(get_thd_sequence_parallel_padding_mask(input_ids))
            else:
                RouterReplay.clear_global_record_padding_mask()

            try:
                output = forward_fn(
                    model,
                    input_ids,
                    multi_modal_inputs,
                    logits_processor=logits_processor,
                    logits_processor_args=logits_processor_args,
                    vision_model=hasattr(self.model_config.hf_config, "vision_config"),
                    pad_token_id=self.model_config.tokenizer.pad_token_id,
                    data_format=data_format,
                    mtp_enable_train=self.model_config.mtp.enable and self.model_config.mtp.enable_train,
                    local_cp_size=local_cp_size,
                )
            finally:
                for audit_hook in audit_hooks:
                    audit_hook.remove()
                for audit_cleanup in reversed(audit_cleanups):
                    audit_cleanup()
                if clear_attention_audit is not None:
                    if get_attention_execution_audit is not None:
                        attention_execution_audit = get_attention_execution_audit()
                    attention_stage_audit = clear_attention_audit()
            if decoder_component_audit and decoder_audit_response_lengths is not None:
                _write_decoder_component_audit(
                    decoder_audit,
                    input_ids,
                    decoder_audit_response_lengths,
                    model,
                    attention_stage_audit=attention_stage_audit,
                    attention_execution_audit=attention_execution_audit,
                    moe_router_audit=moe_router_audit,
                    moe_router_weight_audit=moe_router_weight_audit,
                    moe_mlp_stage_audit=moe_mlp_stage_audit,
                    moe_replay_metadata_audit=moe_replay_metadata_audit,
                )

        # Router replay: record routing decisions for R2 mode
        if RouterReplayHelper.is_r2_record_action(self.tf_config, vp_rank):
            merge_router_topk_indices(attention_mask, input_ids, self.mini_layer_topk_idx_list, self.tf_config, vp_rank)

        # Router replay: switch to backward replay mode for next backward pass
        if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
            router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
            for router in router_instance_list:
                router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

        return output, partial(postprocess_micro_batch_func, data=batch, local_cp_size=local_cp_size)

    def postprocess_micro_batch_func(
        self, output, data: TensorDict, forward_only: bool, loss_function, local_cp_size=None
    ):
        # For memory efficiency
        # We move calculation of entropy to compute_log_probs, forward_only == True
        device = data["input_ids"].device
        model_output = self.prepare_model_outputs(output, data)

        if loss_function is not None:
            # TODO(baiyan): How to support hybrid context parallel with dp_group,
            # now the dp_group is not used, so just leave it as is, but what if we need to use it?
            loss, metrics = loss_function(model_output=model_output, data=data, dp_group=self.get_data_parallel_group())
            # scale loss by num_micro_batch because megatron will scale loss
            # by n_micro_batch inside pp schedule
            scaled_loss = loss * data["num_micro_batch"]
        else:
            assert forward_only, "forward_only must be True when loss_function is None"
            loss = torch.tensor(1.0, device=device)
            scaled_loss = loss
            metrics = {}
        if local_cp_size is not None:
            # aggregate model_output by DP-CP groups
            from verl.utils.megatron_utils import dynamic_cp_merge_output

            model_output = dynamic_cp_merge_output(
                model_output,
                dp_size=mpu.get_data_parallel_world_size(),
                dp_rank=mpu.get_data_parallel_rank(),
                local_cp_size=local_cp_size,
            )

        output = {
            "model_output": model_output,
            "loss": loss.detach().item(),
            "metrics": metrics,
        }

        # return loss and stats
        return scaled_loss, output


@EngineRegistry.register(model_type="value_model", backend="megatron")
class MegatronEngineWithValueHead(MegatronEngineWithLMHead):
    # for value head
    def forward_step(self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func):
        batch: TensorDict = next(batch_iter)
        batch = batch.to(get_device_id())
        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]

        from verl.models.mcore import get_mcore_engine_forward_fn

        forward_fn = get_mcore_engine_forward_fn(self.model_config.hf_config)

        output = forward_fn(
            model,
            input_ids,
            multi_modal_inputs,
            value_model=True,
            vision_model=hasattr(self.model_config.hf_config, "vision_config"),
            pad_token_id=self.model_config.tokenizer.pad_token_id,
            data_format="thd" if self.engine_config.use_remove_padding else "bshd",
        )

        return output, partial(postprocess_micro_batch_func, data=batch)

    def prepare_model_outputs(self, output: dict | torch.Tensor, data: TensorDict):
        return {"values": output}

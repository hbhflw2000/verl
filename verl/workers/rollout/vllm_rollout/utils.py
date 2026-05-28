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
import ctypes
import hashlib
from contextlib import contextmanager
import json
import logging
import os
import platform
import signal
import threading
import time
from types import MethodType
from typing import Any, Literal, Optional, get_args

import torch
from vllm.outputs import RequestOutput

from verl.utils.device import is_npu_available
from verl.utils.vllm import TensorLoRARequest, VLLMHijack
from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights

try:
    from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

    from verl.utils.vllm_omni import OmniTensorLoRARequest, VLLMOmniHijack

    _VLLM_OMNI_AVAILABLE = True
except (ImportError, RuntimeError):  # optional stack; ImportError if missing, RuntimeError e.g. diffusers/transformers
    CustomPipelineWorkerExtension = None  # type: ignore[assignment]
    OmniTensorLoRARequest = None  # type: ignore[assignment]
    VLLMOmniHijack = None  # type: ignore[assignment]
    _VLLM_OMNI_AVAILABLE = False

# Use object as fallback base so the class definition is always valid even when
# vllm_omni is not installed (None is not a valid base class).
_OmniWorkerBase = CustomPipelineWorkerExtension if _VLLM_OMNI_AVAILABLE else object

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# magic numbers that ensure we are using the same LoRA adapter during the rollout and training process
VLLM_LORA_INT_ID = 123
VLLM_LORA_NAME = "123"
VLLM_LORA_PATH = "simon_lora_path"

VLLM_ASCEND_REQUIRED_ENV_VARS = {"VLLM_ALL2ALL_BACKEND": "flashinfer_all2allv", "VLLM_ASCEND_ENABLE_NZ": "0"}


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "0").lower() in {"1", "true", "yes", "on"}


def _tensor_debug_metadata(tensor: Any) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        return {"type": type(tensor).__name__}

    metadata: dict[str, Any] = {
        "type": type(tensor).__name__,
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "nbytes": int(tensor.nbytes),
        "stride": tuple(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "is_contiguous": bool(tensor.is_contiguous()),
        "requires_grad": bool(tensor.requires_grad),
    }
    try:
        metadata["data_ptr"] = int(tensor.data_ptr())
    except Exception as exc:
        metadata["data_ptr_error"] = repr(exc)
    try:
        metadata["storage_data_ptr"] = int(tensor.untyped_storage().data_ptr())
    except Exception as exc:
        metadata["storage_data_ptr_error"] = repr(exc)
    return metadata


def _value_debug_patterns() -> list[str]:
    raw = os.getenv("VERL_OMNI_WEIGHT_VALUE_DEBUG_NAMES")
    if raw:
        return [item.strip() for item in raw.split(",") if item.strip()]

    target_name = os.getenv("VERL_OMNI_WEIGHT_SYNC_TARGET_NAME", "thinker.audio_tower.conv2d1.bias")
    default_patterns = [
        target_name,
        "audio_tower.conv2d1.bias",
        "embed_tokens.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.qkv_proj.weight",
        "layers.0.mlp.experts.0.gate_proj.weight",
        "layers.0.mlp.experts.0.up_proj.weight",
        "layers.0.mlp.experts.0.down_proj.weight",
        "layers.0.mlp.experts.w13_weight",
        "layers.0.mlp.experts.w2_weight",
    ]
    return list(dict.fromkeys(pattern for pattern in default_patterns if pattern))


def _matches_value_debug_name(name: str) -> bool:
    return any(pattern and pattern in name for pattern in _value_debug_patterns())


def _tensor_value_debug(tensor: Any) -> dict[str, Any]:
    payload = _tensor_debug_metadata(tensor)
    if not isinstance(tensor, torch.Tensor):
        return payload

    try:
        sample_count = max(0, int(os.getenv("VERL_OMNI_WEIGHT_VALUE_DEBUG_SAMPLE", "16")))
        max_full_stats_numel = max(0, int(os.getenv("VERL_OMNI_WEIGHT_VALUE_DEBUG_MAX_FULL_NUMEL", "1048576")))
        flat = tensor.detach().view(-1)
        sample = flat[: min(sample_count, flat.numel())].detach().float().cpu().contiguous()
        payload["sample_values"] = sample.tolist()
        payload["sample_sha256"] = hashlib.sha256(sample.numpy().tobytes()).hexdigest()
        if flat.numel() <= max_full_stats_numel:
            stats_tensor = flat.detach().float()
            payload["full_stats"] = {
                "sum": float(stats_tensor.sum().item()),
                "mean": float(stats_tensor.mean().item()) if stats_tensor.numel() else 0.0,
                "std": float(stats_tensor.std(unbiased=False).item()) if stats_tensor.numel() else 0.0,
                "min": float(stats_tensor.min().item()) if stats_tensor.numel() else 0.0,
                "max": float(stats_tensor.max().item()) if stats_tensor.numel() else 0.0,
            }
        else:
            payload["full_stats_skipped_numel"] = int(flat.numel())
    except Exception as exc:
        payload["value_debug_error"] = repr(exc)
    return payload


def _canonical_weight_name(name: str | None) -> str | None:
    if not name:
        return None
    value = str(name)
    changed = True
    while changed:
        changed = False
        for prefix in ("thinker.", "language_model."):
            if value.startswith(prefix):
                value = value[len(prefix) :]
                changed = True
        if value.startswith("model."):
            value = value[len("model.") :]
            changed = True
    return value


def _loader_shard_id(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if "shard_id" in kwargs:
        return kwargs["shard_id"]
    if args:
        # For stacked linear loaders this is typically q/k/v or 0/1. For MoE
        # loaders the first positional arg may be weight_name, but then shard_id
        # should be present in kwargs.
        return args[-1]
    return None


def _loader_expert_id(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int | None:
    expert_id = kwargs.get("expert_id")
    if expert_id is None:
        return None
    try:
        return int(expert_id)
    except Exception:
        return None


def _source_name_from_loader(param_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    for arg in args:
        if isinstance(arg, str) and arg.endswith(".weight"):
            return _canonical_weight_name(arg)

    canonical = _canonical_weight_name(param_name) or param_name
    shard_id = _loader_shard_id(args, kwargs)
    shard = str(shard_id).strip("\"'")
    if "qkv_proj.weight" in canonical:
        shard_to_name = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}
        source = shard_to_name.get(shard)
        if source:
            return canonical.replace("qkv_proj.weight", f"{source}.weight")
    if "gate_up_proj.weight" in canonical:
        if shard in {"0", "gate", "gate_proj", "w1"}:
            return canonical.replace("gate_up_proj.weight", "gate_proj.weight")
        if shard in {"1", "up", "up_proj", "w3"}:
            return canonical.replace("gate_up_proj.weight", "up_proj.weight")
    if "experts.w13_weight" in canonical:
        expert_id = _loader_expert_id(args, kwargs)
        expert = 0 if expert_id is None else expert_id
        if shard in {"w1", "0", "gate", "gate_proj"}:
            return canonical.replace("experts.w13_weight", f"experts.{expert}.gate_proj.weight")
        if shard in {"w3", "1", "up", "up_proj"}:
            return canonical.replace("experts.w13_weight", f"experts.{expert}.up_proj.weight")
    if "experts.w2_weight" in canonical:
        expert_id = _loader_expert_id(args, kwargs)
        expert = 0 if expert_id is None else expert_id
        return canonical.replace("experts.w2_weight", f"experts.{expert}.down_proj.weight")
    return None


def _slice_tuple_to_str(slices: tuple[Any, ...]) -> list[str]:
    parts = []
    for item in slices:
        if isinstance(item, slice):
            parts.append(f"{item.start}:{item.stop}:{item.step}")
        else:
            parts.append(str(item))
    return parts


def _distributed_rank_world() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "rank_basis": "single_process",
        "rank": 0,
        "world_size": 1,
        "global_rank": 0,
        "global_world_size": 1,
        "tp_rank": None,
        "tp_world_size": None,
        "tp_group_available": False,
    }
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return metadata
    try:
        metadata["global_rank"] = int(torch.distributed.get_rank())
        metadata["global_world_size"] = int(torch.distributed.get_world_size())
        metadata["rank"] = metadata["global_rank"]
        metadata["world_size"] = metadata["global_world_size"]
        metadata["rank_basis"] = "global"
    except Exception as exc:
        metadata["global_rank_error"] = repr(exc)

    try:
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        metadata["tp_rank"] = int(tp_group.rank_in_group)
        metadata["tp_world_size"] = int(tp_group.world_size)
        metadata["tp_group_available"] = True
        metadata["rank"] = metadata["tp_rank"]
        metadata["world_size"] = metadata["tp_world_size"]
        metadata["rank_basis"] = "tp_group"
    except Exception as exc:
        metadata["tp_group_error"] = repr(exc)
    return metadata


def _local_row_slice(total_rows: int, world_size: int, rank: int) -> slice:
    if world_size > 1 and total_rows % world_size == 0:
        rows = total_rows // world_size
        return slice(rank * rows, (rank + 1) * rows)
    return slice(0, total_rows)


def _expected_loaded_param_slice(
    param_name: str,
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], torch.Tensor | None]:
    canonical = _canonical_weight_name(param_name) or param_name
    shard_id = _loader_shard_id(args, kwargs)
    shard = str(shard_id).strip("\"'")
    expert_id = _loader_expert_id(args, kwargs)
    data = param.data
    loaded_shape = tuple(loaded_weight.shape)
    rank_metadata = _distributed_rank_world()
    rank = int(rank_metadata["rank"])
    world_size = int(rank_metadata["world_size"])
    payload: dict[str, Any] = {
        "param_name": param_name,
        "canonical_param_name": canonical,
        "source_name": _source_name_from_loader(param_name, args, kwargs),
        "shard_id": shard_id,
        "expert_id": expert_id,
        "rank": rank,
        "world_size": world_size,
        "rank_basis": rank_metadata.get("rank_basis"),
        "global_rank": rank_metadata.get("global_rank"),
        "global_world_size": rank_metadata.get("global_world_size"),
        "tp_rank": rank_metadata.get("tp_rank"),
        "tp_world_size": rank_metadata.get("tp_world_size"),
        "tp_group_available": rank_metadata.get("tp_group_available"),
        "tp_group_error": rank_metadata.get("tp_group_error"),
        "global_rank_error": rank_metadata.get("global_rank_error"),
        "param_shape": tuple(data.shape),
        "loaded_weight_shape": loaded_shape,
    }

    try:
        slices: tuple[Any, ...] | None = None
        loaded_slices: tuple[Any, ...] | None = None
        if "qkv_proj.weight" in canonical and data.ndim >= 2 and loaded_weight.ndim >= 2:
            source_rows = int(loaded_weight.shape[0])
            local_source = _local_row_slice(source_rows, world_size, rank)
            rows = int((local_source.stop or source_rows) - (local_source.start or 0))
            if shard == "q":
                start = 0
            elif shard == "k":
                start = int(data.shape[0]) - 2 * rows
            elif shard == "v":
                start = int(data.shape[0]) - rows
            else:
                start = None
            if start is not None:
                slices = (slice(start, start + rows), *[slice(None)] * (data.ndim - 1))
                loaded_slices = (local_source, *[slice(None)] * (loaded_weight.ndim - 1))
        elif "gate_up_proj.weight" in canonical and data.ndim >= 2 and loaded_weight.ndim >= 2:
            source_rows = int(loaded_weight.shape[0])
            local_source = _local_row_slice(source_rows, world_size, rank)
            rows = int((local_source.stop or source_rows) - (local_source.start or 0))
            if shard in {"0", "gate", "gate_proj", "w1"}:
                start = 0
            elif shard in {"1", "up", "up_proj", "w3"}:
                start = rows
            else:
                start = None
            if start is not None:
                slices = (slice(start, start + rows), *[slice(None)] * (data.ndim - 1))
                loaded_slices = (local_source, *[slice(None)] * (loaded_weight.ndim - 1))
        elif "experts.w13_weight" in canonical and expert_id is not None and data.ndim == 3 and loaded_weight.ndim == 2:
            local_source_rows = _local_row_slice(int(loaded_weight.shape[0]), world_size, rank)
            source_rows = int((local_source_rows.stop or loaded_weight.shape[0]) - (local_source_rows.start or 0))
            if data.shape[1] >= 2 * source_rows and data.shape[2] == loaded_weight.shape[1]:
                rows = source_rows
                if shard in {"w1", "0", "gate", "gate_proj"}:
                    start = 0
                elif shard in {"w3", "1", "up", "up_proj"}:
                    start = rows
                else:
                    start = None
                if start is not None:
                    slices = (expert_id, slice(start, start + rows), slice(None))
                    loaded_slices = (local_source_rows, slice(None))
            else:
                local_source_cols = _local_row_slice(int(loaded_weight.shape[1]), world_size, rank)
                source_cols = int((local_source_cols.stop or loaded_weight.shape[1]) - (local_source_cols.start or 0))
                if data.shape[2] >= 2 * source_cols and data.shape[1] == loaded_weight.shape[0]:
                    cols = source_cols
                else:
                    cols = None
                if cols is not None:
                    if shard in {"w1", "0", "gate", "gate_proj"}:
                        start = 0
                    elif shard in {"w3", "1", "up", "up_proj"}:
                        start = cols
                    else:
                        start = None
                    if start is not None:
                        slices = (expert_id, slice(None), slice(start, start + cols))
                        loaded_slices = (slice(None), local_source_cols)
        elif "experts.w2_weight" in canonical and expert_id is not None and data.ndim == 3 and loaded_weight.ndim == 2:
            slices = (expert_id, slice(None), slice(None))
            if tuple(data[slices].shape) == loaded_shape:
                loaded_slices = (slice(None), slice(None))
            elif data[slices].shape[0] == loaded_weight.shape[0]:
                local_source_cols = _local_row_slice(int(loaded_weight.shape[1]), world_size, rank)
                loaded_slices = (slice(None), local_source_cols)
            else:
                local_source_rows = _local_row_slice(int(loaded_weight.shape[0]), world_size, rank)
                loaded_slices = (local_source_rows, slice(None))

        if slices is None:
            payload["status"] = "unsupported"
            return payload, None

        sliced = data[slices]
        payload["status"] = "ok"
        payload["slices"] = _slice_tuple_to_str(slices)
        payload["slice_shape"] = tuple(sliced.shape)
        payload["shape_matches_loaded_weight"] = tuple(sliced.shape) == loaded_shape
        if loaded_slices is not None:
            loaded_sliced = loaded_weight[loaded_slices]
            payload["loaded_weight_slices"] = _slice_tuple_to_str(loaded_slices)
            payload["loaded_weight_slice_shape"] = tuple(loaded_sliced.shape)
            payload["slice_matches_loaded_weight_slice"] = tuple(sliced.shape) == tuple(loaded_sliced.shape)
            payload["loaded_weight_slice_value"] = _tensor_value_debug(loaded_sliced)
        return payload, sliced
    except Exception as exc:
        payload["status"] = "failed"
        payload["exception"] = repr(exc)
        return payload, None


def _packed_slice_value_debug(
    param_name: str,
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    payload, tensor = _expected_loaded_param_slice(param_name, param, loaded_weight, args, kwargs)
    if tensor is not None:
        payload["value"] = _tensor_value_debug(tensor)
    return payload


def _cumem_debug_metadata(tensor: Any) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor):
        return {"type": type(tensor).__name__}
    try:
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        data_ptr = int(tensor.data_ptr())
        storage_ptr = int(tensor.untyped_storage().data_ptr())
        matches: list[dict[str, Any]] = []
        for ptr, data in allocator.pointer_to_data.items():
            size = int(data.handle[1])
            begin = int(ptr)
            end = begin + size
            if begin <= data_ptr < end or begin <= storage_ptr < end:
                matches.append(
                    {
                        "ptr": begin,
                        "size_bytes": size,
                        "tag": data.tag,
                        "data_ptr_offset": data_ptr - begin,
                        "storage_ptr_offset": storage_ptr - begin,
                        "has_cpu_backup": data.cpu_backup_tensor is not None,
                        "backup_nbytes": (
                            int(data.cpu_backup_tensor.numel() * data.cpu_backup_tensor.element_size())
                            if data.cpu_backup_tensor is not None
                            else 0
                        ),
                    }
                )
        return {
            "current_tag": allocator.current_tag,
            "tracked_allocations": len(allocator.pointer_to_data),
            "matched": bool(matches),
            "matches": matches,
        }
    except Exception as exc:
        return {"error": repr(exc)}


def _weight_debug_metadata(weights: list[tuple[str, torch.Tensor]], limit: int | None = None) -> list[dict[str, Any]]:
    items = weights if limit is None else weights[:limit]
    return [{"name": name, "tensor": _tensor_debug_metadata(tensor)} for name, tensor in items]


def _selected_weight_value_debug(weights: list[tuple[str, torch.Tensor]], limit: int = 32) -> list[dict[str, Any]]:
    selected = []
    for name, tensor in weights:
        if not _matches_value_debug_name(name):
            continue
        selected.append({"name": name, "tensor": _tensor_debug_metadata(tensor), "value": _tensor_value_debug(tensor)})
        if len(selected) >= limit:
            break
    return selected


def _selected_model_param_value_debug(model: Any, limit: int = 32) -> list[dict[str, Any]]:
    selected = []
    try:
        named_params = model.named_parameters(remove_duplicate=False)
    except TypeError:
        named_params = model.named_parameters()
    except Exception:
        logger.exception("Failed to enumerate model parameters for vLLM-Omni value debug")
        return selected

    for name, param in named_params:
        if not _matches_value_debug_name(name):
            continue
        selected.append(
            {
                "name": name,
                "param": _tensor_debug_metadata(param),
                "param_data": _tensor_debug_metadata(param.data),
                "value": _tensor_value_debug(param.data),
                "cumem": _cumem_debug_metadata(param.data),
            }
        )
        if len(selected) >= limit:
            break
    return selected


def _dump_weight_sync_debug(event: str, payload: dict[str, Any]) -> str | None:
    dump_dir = os.getenv("VERL_OMNI_WEIGHT_SYNC_DUMP_DIR")
    if not dump_dir:
        return None

    try:
        os.makedirs(dump_dir, exist_ok=True)
        filename = (
            f"verl_omni_weight_sync_{event}_pid{os.getpid()}_"
            f"tid{threading.get_ident()}_{int(time.time() * 1000)}.json"
        )
        path = os.path.join(dump_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return path
    except Exception:
        logger.exception("Failed to dump vLLM-Omni weight sync debug payload for event=%s", event)
        return None


def _sync_tensor_device(tensor: Any) -> None:
    if not isinstance(tensor, torch.Tensor):
        return
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def _candidate_debug_param_names(target_name: str) -> list[str]:
    candidates = [target_name]
    if target_name.startswith("thinker."):
        stripped = target_name[len("thinker.") :]
        candidates.append(stripped)
        candidates.append(f"language_model.{stripped}")
    if target_name.startswith("thinker.model."):
        candidates.append(f"language_model.model.{target_name[len('thinker.model.'):]}")
    if target_name.startswith("thinker.lm_head."):
        candidates.append(f"language_model.lm_head.{target_name[len('thinker.lm_head.'):]}")
    return list(dict.fromkeys(candidates))


def _find_debug_param(model: Any, target_name: str) -> tuple[str | None, torch.nn.Parameter | None]:
    try:
        named_params = dict(model.named_parameters(remove_duplicate=False))
    except TypeError:
        named_params = dict(model.named_parameters())
    except Exception:
        logger.exception("Failed to enumerate model parameters for vLLM-Omni local copy probe")
        return None, None

    candidates = _candidate_debug_param_names(target_name)
    for candidate in candidates:
        param = named_params.get(candidate)
        if param is not None:
            return candidate, param

    suffixes = [candidate.split(".", 1)[-1] for candidate in candidates if "." in candidate]
    for name, param in named_params.items():
        if any(name.endswith(suffix) for suffix in suffixes):
            return name, param
    return None, None


def _probe_model_param_local_copy(model: Any, target_name: str, event: str) -> dict[str, Any]:
    resolved_name, param = _find_debug_param(model, target_name)
    payload: dict[str, Any] = {
        "event": event,
        "target_name": target_name,
        "resolved_name": resolved_name,
        "pid": os.getpid(),
        "thread_id": threading.get_ident(),
    }
    if param is None:
        payload["status"] = "not_found"
        dump_path = _dump_weight_sync_debug(event, payload)
        logger.warning(
            "vLLM-Omni local copy probe target not found: target=%s dump=%s",
            target_name,
            dump_path,
        )
        return payload

    original = None
    scratch = None
    try:
        payload["param_before"] = _tensor_debug_metadata(param.data)
        payload["cumem_before"] = _cumem_debug_metadata(param.data)
        _sync_tensor_device(param.data)
        original = param.data.detach().clone()
        scratch = torch.zeros_like(param.data)
        payload["scratch"] = _tensor_debug_metadata(scratch)

        param.data.copy_(scratch)
        _sync_tensor_device(param.data)
        payload["param_after_zero_copy"] = _tensor_debug_metadata(param.data)
        payload["cumem_after_zero_copy"] = _cumem_debug_metadata(param.data)

        param.data.copy_(original)
        _sync_tensor_device(param.data)
        payload["param_after_restore"] = _tensor_debug_metadata(param.data)
        payload["cumem_after_restore"] = _cumem_debug_metadata(param.data)
        payload["status"] = "ok"
        dump_path = _dump_weight_sync_debug(event, payload)
        logger.info(
            "vLLM-Omni local copy probe passed: event=%s target=%s resolved=%s dump=%s",
            event,
            target_name,
            resolved_name,
            dump_path,
        )
        return payload
    except Exception as exc:
        payload["status"] = "failed"
        payload["exception"] = repr(exc)
        payload["param_on_failure"] = _tensor_debug_metadata(param.data)
        payload["cumem_on_failure"] = _cumem_debug_metadata(param.data)
        dump_path = _dump_weight_sync_debug(event, payload)
        logger.exception(
            "vLLM-Omni local copy probe failed: event=%s target=%s resolved=%s dump=%s",
            event,
            target_name,
            resolved_name,
            dump_path,
        )
        raise
    finally:
        del scratch
        del original


@contextmanager
def _debug_weight_loader_failures(model: Any, weights: list[tuple[str, torch.Tensor]]):
    value_debug_enabled = _env_enabled("VERL_OMNI_WEIGHT_VALUE_DEBUG")
    if not _env_enabled("VERL_OMNI_WEIGHT_SYNC_DEBUG") and not value_debug_enabled:
        yield
        return

    try:
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    except Exception:
        logger.exception("Failed to import vLLM default_weight_loader for weight sync debug")
        yield
        return

    restore: list[tuple[torch.nn.Parameter, bool, Any]] = []
    seen_params: set[int] = set()

    def make_debug_loader(param_name: str, original_loader: Any):
        def debug_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor, *args, **kwargs):
            should_dump_values = value_debug_enabled and _matches_value_debug_name(param_name)
            value_payload: dict[str, Any] | None = None
            if should_dump_values:
                value_payload = {
                    "event": "weight_loader_value",
                    "param_name": param_name,
                    "canonical_param_name": _canonical_weight_name(param_name),
                    "inferred_source_name": _source_name_from_loader(param_name, args, kwargs),
                    "loader": repr(original_loader),
                    "loader_args": [repr(arg) for arg in args],
                    "loader_kwargs": {key: repr(value) for key, value in kwargs.items()},
                    "param_before": _tensor_value_debug(param.data),
                    "loaded_weight": _tensor_value_debug(loaded_weight),
                    "incoming_count": len(weights),
                    "incoming_selected": _selected_weight_value_debug(weights),
                }
                value_payload["expected_loaded_slice_before"] = _packed_slice_value_debug(
                    param_name, param, loaded_weight, args, kwargs
                )
            try:
                result = original_loader(param, loaded_weight, *args, **kwargs)
                if value_payload is not None:
                    value_payload["param_after"] = _tensor_value_debug(param.data)
                    value_payload["expected_loaded_slice_after"] = _packed_slice_value_debug(
                        param_name, param, loaded_weight, args, kwargs
                    )
                    value_payload["cumem_after"] = _cumem_debug_metadata(param.data)
                    dump_path = _dump_weight_sync_debug("weight_loader_value", value_payload)
                    logger.info(
                        "vLLM-Omni weight loader value debug: param=%s dump=%s",
                        param_name,
                        dump_path,
                    )
                return result
            except Exception as exc:
                payload = {
                    "event": "weight_loader_failure",
                    "exception": repr(exc),
                    "param_name": param_name,
                    "loader": repr(original_loader),
                    "loader_args": [repr(arg) for arg in args],
                    "loader_kwargs": {key: repr(value) for key, value in kwargs.items()},
                    "param": _tensor_debug_metadata(param),
                    "param_data": _tensor_debug_metadata(param.data),
                    "loaded_weight": _tensor_debug_metadata(loaded_weight),
                    "loaded_weight_value": _tensor_value_debug(loaded_weight) if value_debug_enabled else None,
                    "incoming_count": len(weights),
                    "incoming_sample": _weight_debug_metadata(weights, limit=16),
                    "incoming_all": _weight_debug_metadata(weights),
                    "incoming_selected": _selected_weight_value_debug(weights) if value_debug_enabled else [],
                }
                dump_path = _dump_weight_sync_debug("weight_loader_failure", payload)
                logger.exception(
                    "vLLM-Omni weight loader failed: param=%s dump=%s param=%s loaded_weight=%s",
                    param_name,
                    dump_path,
                    payload["param_data"],
                    payload["loaded_weight"],
                )
                raise

        return debug_weight_loader

    try:
        for param_name, param in model.named_parameters(remove_duplicate=False):
            param_id = id(param)
            if param_id in seen_params:
                continue
            seen_params.add(param_id)

            had_loader = hasattr(param, "weight_loader")
            original_loader = getattr(param, "weight_loader", default_weight_loader)
            try:
                setattr(param, "weight_loader", make_debug_loader(param_name, original_loader))
            except Exception:
                logger.exception("Failed to install vLLM-Omni weight sync debug hook for param=%s", param_name)
                continue
            restore.append((param, had_loader, original_loader))
        yield
    finally:
        for param, had_loader, original_loader in restore:
            try:
                if had_loader:
                    setattr(param, "weight_loader", original_loader)
                else:
                    delattr(param, "weight_loader")
            except Exception:
                logger.exception("Failed to restore vLLM-Omni weight loader debug hook")


def set_death_signal():
    """Kill the current process when the parent process exits."""
    if platform.system() != "Linux":
        return
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(1, signal.SIGKILL)
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGKILL)


def get_device_uuid(device_id: int) -> str:
    from vllm.platforms import current_platform

    # Convert torch.npu.current_device to its corresponding ASCEND_RT_VISIBLE_DEVICES.
    if is_npu_available:
        if os.getenv("ASCEND_RT_VISIBLE_DEVICES") is not None:
            npu_visible_devices = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
            assert device_id < len(npu_visible_devices), f"device_id {device_id} must less than {npu_visible_devices}"
            return "NPU-" + npu_visible_devices[device_id]
        else:
            return f"NPU-{device_id}"
    else:
        return current_platform.get_device_uuid(device_id)


def get_vllm_max_lora_rank(lora_rank: int):
    """
    For vLLM, automatically adjusts the `max_lora_rank` to the nearest allowed value.
    The allowed values are retrieved from vLLM's MaxLoRARanks type definition.
    """
    assert lora_rank > 0, f"lora_rank must be greater than 0, get {lora_rank}"

    try:
        from vllm.config.lora import MaxLoRARanks
    except Exception:
        # FIXME: migrate vllm version https://github.com/vllm-project/vllm/blob/main/vllm/config/lora.py#L25
        MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]

    vllm_max_lora_ranks = sorted(get_args(MaxLoRARanks))
    if lora_rank > vllm_max_lora_ranks[-1]:
        raise ValueError(f"lora_rank must be less than or equal to {vllm_max_lora_ranks[-1]}, but got {lora_rank}")

    for rank in vllm_max_lora_ranks:
        if lora_rank <= rank:
            return rank


# https://github.com/vllm-project/vllm/issues/13175
def monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    2. Online FP8 quantization
    """

    def __new__(cls, **kwargs):
        set_death_signal()

        # 1. patch for Lora
        VLLMHijack.hijack()
        # 2. patch online fp8 quant
        if os.environ.get("VERL_VLLM_FP8_QUANT_ENABLED", "0") == "1":
            apply_vllm_fp8_patches()
        # 3. patch QAT (compressed-tensors NVFP4) for dynamic weight loading
        vllm_config = kwargs.get("vllm_config")
        quant_config = getattr(vllm_config, "quant_config", None) if vllm_config else None
        _is_qat_model = getattr(quant_config, "quant_format", None) == "nvfp4-pack-quantized"
        _is_modelopt_qat = type(quant_config).__name__ == "ModelOptNvFp4Config"
        if _is_qat_model:
            from verl.utils.qat import apply_qat_patches

            apply_qat_patches()
            logger.info("Applied QAT (compressed-tensors) patches in vLLM worker subprocess")
        elif _is_modelopt_qat:
            from verl.utils.modelopt import apply_modelopt_nvfp4_patches

            apply_modelopt_nvfp4_patches()
            logger.info("Applied ModelOpt NVFP4 patches in vLLM worker subprocess")

        # TODO: For ascend NPU, when the corresponding vllm-ascend version is upgraded to v0.13.0,
        # please remove the VLLM_ASCEND_REQUIRED_ENV_VARS variable replacement action.
        # This is only a fix for vllm version < v0.13.0.
        if is_npu_available:
            for k in VLLM_ASCEND_REQUIRED_ENV_VARS:
                if k not in os.environ:
                    os.environ[k] = VLLM_ASCEND_REQUIRED_ENV_VARS[k]

        instance = super().__new__(cls)
        instance._is_qat_model = _is_qat_model
        instance._is_modelopt_qat = _is_modelopt_qat
        return instance

    def monkey_patch_model(self, vocab_size: int):
        # patch compute_logits to avoid sampling OOV token
        monkey_patch_compute_logits(self.model_runner.model, vocab_size)
        # patch weight loader to support MoE model
        patch_vllm_moe_model_weight_loader(self.model_runner.model)

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        """Update the weights of the rollout model."""
        from vllm.platforms import current_platform

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        if current_platform.device_type == "npu" and self.device is None:
            self.device = torch.device(f"npu:{self.local_rank}")

        # In async mode, make sure the old lora is removed before adding the new one
        if peft_config and base_sync_done:
            self.remove_lora(VLLM_LORA_INT_ID)

        use_standard_weight_load = not (peft_config and base_sync_done) and not is_fp8_model(
            self.model_runner.vllm_config
        )

        if self._is_qat_model:
            # QAT (compressed-tensors): Prepare for weight loading BEFORE receiving any buckets
            from verl.utils.qat import prepare_qat_for_load_weights

            prepare_qat_for_load_weights(self.model_runner.model, device=self.device)
            logger.info("QAT: prepare_qat_for_load_weights completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import prepare_modelopt_for_weight_reload

            prepare_modelopt_for_weight_reload(self.model_runner.model, device=self.device)
            logger.info("ModelOpt: prepare_modelopt_for_weight_reload completed")
        elif use_standard_weight_load:
            # Re-apply here because async IPC weight sync can happen long after init and lose MoE weight_loader attrs.
            patch_vllm_moe_model_weight_loader(self.model_runner.model)

        assert self.device is not None
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        receiver.receive_weights(
            on_bucket_received=lambda weights: self._update_weights(
                weights, peft_config=peft_config, base_sync_done=base_sync_done
            )
        )

        if self._is_qat_model:
            # QAT (compressed-tensors): call process_weights_after_loading AFTER all buckets are received
            from verl.utils.qat import manual_process_weights_after_loading

            manual_process_weights_after_loading(self.model_runner.model)
            logger.info("QAT: process_weights_after_loading completed")
        elif self._is_modelopt_qat:
            from verl.utils.modelopt.vllm_modelopt_patch import modelopt_process_weights_after_loading

            modelopt_process_weights_after_loading(self.model_runner.model)
            logger.info("ModelOpt QAT: process_weights_after_loading completed")
        elif use_standard_weight_load:
            # Some post-load transforms are non-idempotent; run once after all buckets.
            from vllm.model_executor.model_loader.utils import process_weights_after_loading

            model = self.model_runner.model
            model_config = self.model_runner.vllm_config.model_config
            process_weights_after_loading(model, model_config, self.device)

    def _update_weights(self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool):
        if peft_config and base_sync_done:
            weights = dict(weights)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=weights,
            )
            self.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            # Add the FP8 related logic here as sharding manager has been deprecated.
            # Check if FP8 quantization is enabled and apply appropriate weight loading
            if is_fp8_model(self.model_runner.vllm_config):
                logger.info(f"FP8 model detected (async): {self.model_runner.vllm_config.quant_config}")
                # Convert bf16 weights to fp8 format before loading
                loaded_params = load_quanted_weights(weights, self.model_runner)
                logger.info(f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}")
            else:
                logger.info("Loading standard weights (non-FP8, async)")
                self.model_runner.model.load_weights(weights)

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for communication."""
        if not hasattr(self, "device_uuid") or not self.device_uuid:
            self.device_uuid = get_device_uuid(self.device.index)
        return f"ipc:///tmp/rl-colocate-zmq-{self.device_uuid}.sock"


class vLLMOmniColocateWorkerExtension(_OmniWorkerBase):
    """
    The class for vLLM-Omni's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.

    Feature support:
    1. LoRA
    """

    def __new__(cls, **kwargs):
        assert _VLLM_OMNI_AVAILABLE, "vLLM-Omni is required to use vLLMOmniColocateWorkerExtension"
        set_death_signal()

        # 1. patch for Lora
        VLLMOmniHijack.hijack()

        return super().__new__(cls)

    def update_weights_from_ipc(self, peft_config: dict = None, base_sync_done=False, use_shm: bool = False):
        """Update the weights of the rollout model."""

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import BucketedWeightReceiver

        # In async mode, make sure the old lora is removed before adding the new one
        if peft_config and base_sync_done:
            self.remove_lora(VLLM_LORA_INT_ID)

        assert self.device is not None
        receiver = BucketedWeightReceiver(
            zmq_handle=self._get_zmq_handle(),
            device=self.device,
            use_shm=use_shm,
        )
        receiver.receive_weights(
            on_bucket_received=lambda weights: self._update_weights(
                weights, peft_config=peft_config, base_sync_done=base_sync_done
            )
        )

    def probe_weight_update_local_copy(self, target_name: str | None = None, event: str = "pre_ipc_local_copy"):
        """Probe whether a restored vLLM-Omni parameter can be written before IPC load."""

        if not _env_enabled("VERL_OMNI_PRE_IPC_LOCAL_COPY_DEBUG"):
            return {"event": event, "status": "disabled"}

        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None)
        if model is None:
            payload = {"event": event, "status": "missing_model", "target_name": target_name}
            dump_path = _dump_weight_sync_debug(event, payload)
            logger.warning("vLLM-Omni local copy probe missing model: dump=%s", dump_path)
            return payload

        target_name = target_name or os.getenv(
            "VERL_OMNI_WEIGHT_SYNC_TARGET_NAME", "thinker.audio_tower.conv2d1.bias"
        )
        return _probe_model_param_local_copy(model, target_name, event)

    def _update_weights(self, weights: list[tuple[str, torch.Tensor]], peft_config: dict, base_sync_done: bool):
        if peft_config and base_sync_done:
            weights = dict(weights)
            lora_request = OmniTensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=weights,
            )
            self.add_lora(lora_request)
            logger.info(f"vLLM-Omni load weights, loaded_params: {len(weights)}")
        else:
            logger.info("Loading standard weights (async)")
            input_names = [name for name, _ in weights]
            if os.getenv("VERL_OMNI_SKIP_WEIGHT_UPDATE", "0").lower() in {"1", "true", "yes"}:
                logger.warning(
                    "Skipping vLLM-Omni standard weight update due to VERL_OMNI_SKIP_WEIGHT_UPDATE; "
                    "num_tensors=%d sample=%s",
                    len(input_names),
                    input_names[:8],
                )
                return

            model_runner = getattr(self, "model_runner", None)
            model = getattr(model_runner, "model", None)
            if model is not None and hasattr(model, "load_weights"):
                try:
                    target_name = os.getenv(
                        "VERL_OMNI_WEIGHT_SYNC_TARGET_NAME", "thinker.audio_tower.conv2d1.bias"
                    )
                    if _env_enabled("VERL_OMNI_PRE_LOAD_LOCAL_COPY_DEBUG") and any(
                        name == target_name for name, _ in weights
                    ):
                        _probe_model_param_local_copy(model, target_name, "pre_load_local_copy")
                    if _env_enabled("VERL_OMNI_WEIGHT_VALUE_DEBUG"):
                        incoming_selected = _selected_weight_value_debug(weights)
                        if incoming_selected:
                            dump_path = _dump_weight_sync_debug(
                                "pre_load_value_debug",
                                {
                                    "event": "pre_load_value_debug",
                                    "incoming_count": len(weights),
                                    "incoming_names_sample": input_names[:32],
                                    "incoming_selected": incoming_selected,
                                    "model_selected_before": _selected_model_param_value_debug(model),
                                },
                            )
                            logger.info("vLLM-Omni pre-load value debug dump=%s", dump_path)
                    with _debug_weight_loader_failures(model, weights):
                        loaded_params = model.load_weights(weights)
                except Exception as exc:
                    payload = {
                        "event": "load_weights_exception",
                        "exception": repr(exc),
                        "incoming_count": len(weights),
                        "incoming_sample": _weight_debug_metadata(weights, limit=32),
                        "incoming_all": _weight_debug_metadata(weights),
                        "incoming_selected": (
                            _selected_weight_value_debug(weights)
                            if _env_enabled("VERL_OMNI_WEIGHT_VALUE_DEBUG")
                            else []
                        ),
                    }
                    dump_path = _dump_weight_sync_debug("load_weights_exception", payload)
                    logger.exception(
                        "vLLM-Omni load_weights failed: dump=%s incoming=%d sample=%s",
                        dump_path,
                        len(weights),
                        input_names[:16],
                    )
                    raise
                logger.info(f"vLLM-Omni load standard weights, loaded_params: {len(loaded_params)}")
                if _env_enabled("VERL_OMNI_WEIGHT_VALUE_DEBUG"):
                    loaded_params_list = sorted(loaded_params)
                    selected_loaded = [name for name in loaded_params_list if _matches_value_debug_name(name)]
                    dump_path = _dump_weight_sync_debug(
                        "post_load_value_debug",
                        {
                            "event": "post_load_value_debug",
                            "incoming_count": len(weights),
                            "loaded_count": len(loaded_params),
                            "incoming_names_sample": input_names[:32],
                            "loaded_names_sample": loaded_params_list[:32],
                            "selected_loaded": selected_loaded[:32],
                            "model_selected_after": _selected_model_param_value_debug(model),
                        },
                    )
                    logger.info("vLLM-Omni post-load value debug dump=%s", dump_path)
                if _env_enabled("VERL_OMNI_WEIGHT_SYNC_DEBUG"):
                    key_substrs = (
                        "embed_tokens.weight",
                        "lm_head.weight",
                        "mlp.gate.weight",
                        "mlp.experts.0.gate_proj.weight",
                        "mlp.experts.0.up_proj.weight",
                        "mlp.experts.0.down_proj.weight",
                    )
                    key_input_names = [name for name in input_names if any(key in name for key in key_substrs)]
                    key_loaded_names = [name for name in loaded_params if any(key in name for key in key_substrs)]
                    logger.info(
                        "vLLM-Omni weight sync debug: incoming=%d loaded=%d sample_in=%s sample_loaded=%s "
                        "key_in=%s key_loaded=%s",
                        len(input_names),
                        len(loaded_params),
                        input_names[:8],
                        sorted(loaded_params)[:8],
                        key_input_names[:12],
                        sorted(key_loaded_names)[:12],
                    )
            elif hasattr(self, "reload_weights"):
                self.reload_weights(weights_iterator=weights)
                logger.info(f"vLLM-Omni reloaded standard weights, num_tensors: {len(weights)}")
            else:
                raise AttributeError("vLLM-Omni worker has neither model.load_weights nor reload_weights")

    def _get_zmq_handle(self) -> str:
        """Get ZMQ handle for communication."""
        if not hasattr(self, "device_uuid") or not self.device_uuid:
            self.device_uuid = get_device_uuid(self.device.index)
        return f"ipc:///tmp/rl-colocate-zmq-{self.device_uuid}.sock"


class SuppressSignalInThread:
    def __enter__(self):
        self.original_signal = signal.signal

        def no_op_signal(sig, action):
            if threading.current_thread() is not threading.main_thread():
                print(f"Ignored signal {sig} in thread {threading.current_thread().name}")
                return
            return self.original_signal(sig, action)

        signal.signal = no_op_signal
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.signal = self.original_signal


def build_cli_args_from_config(config: dict[str, Any]) -> list[str]:
    """
    Convert a config dictionary to CLI arguments for vLLM server.

    Handles different value types appropriately:
    - None: skipped
    - bool True: adds '--key'
    - bool False: skipped
    - list: expands to '--key item1 item2 ...'
    - empty list: skipped (vLLM uses nargs="+" which requires at least one value)
    - dict: JSON serialized
    - other: string converted

    Args:
        config: Dictionary of configuration key-value pairs

    Returns:
        List of CLI argument strings
    """
    cli_args = []
    for k, v in config.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                cli_args.append(f"--{k}")
        elif isinstance(v, list):
            if not v:
                # Skip empty lists - vLLM uses nargs="+" which requires at least one value
                continue
            # Lists need to be expanded as multiple separate arguments
            # e.g., --cuda-graph-sizes 1 2 4 8 becomes ['--cuda-graph-sizes', '1', '2', '4', '8']
            cli_args.append(f"--{k}")
            cli_args.extend([str(item) for item in v])
        else:
            cli_args.append(f"--{k}")
            # Use json.dumps for dict to ensure valid JSON format
            cli_args.append(json.dumps(v) if isinstance(v, dict) else str(v))
    return cli_args


def extract_prompt_logprobs(output: RequestOutput, num_prompt_logprobs: Optional[int], result_dict: dict[str, list]):
    """Extract prompt log probabilities from generation output."""
    if num_prompt_logprobs is None:
        return

    prompt_logprobs_ls, prompt_ids_ls = [], []
    # NOTE: logprob of first prompt token is None.
    for logprobs_dict in output.prompt_logprobs[1:]:
        if num_prompt_logprobs == 0:
            token_id_str = list(logprobs_dict.keys())[0]
            logprob = logprobs_dict[token_id_str].logprob
            prompt_logprobs_ls.append([logprob])
            prompt_ids_ls.append([int(token_id_str)])
        else:
            prompt_ids = [None] * num_prompt_logprobs
            prompt_logprobs = [None] * num_prompt_logprobs
            # We get either top-k logprobs or top-k plus the sampled logprob (if sampled token is not in top-k)
            assert len(logprobs_dict) in [num_prompt_logprobs, num_prompt_logprobs + 1], len(logprobs_dict)
            for token_id_str, token_logprob in logprobs_dict.items():
                rank = token_logprob.rank
                if rank > num_prompt_logprobs:
                    continue  # the sampled token is not in the top-k
                logprob = token_logprob.logprob
                prompt_ids[rank - 1] = int(token_id_str)
                prompt_logprobs[rank - 1] = logprob
            prompt_logprobs_ls.append(prompt_logprobs)
            prompt_ids_ls.append(prompt_ids)

    # NOTE: pad a dummy prompt logprob for last prompt token.
    prompt_logprobs_ls.append([0.0] * max(num_prompt_logprobs, 1))
    prompt_ids_ls.append([0] * max(num_prompt_logprobs, 1))

    result_dict["prompt_ids"] = prompt_ids_ls
    result_dict["prompt_logprobs"] = prompt_logprobs_ls

# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
Bucketed weight transfer via ZMQ + IPC (or shared memory fallback).

Not recommended depending on vllm for this file.
"""

import gc
import hashlib
import json
import logging
import os
from multiprocessing import shared_memory
import threading
import time
from typing import Callable, TypedDict

import torch
import zmq
from torch.multiprocessing.reductions import reduce_tensor

from verl.utils.device import get_device_id, get_device_name, get_torch_device

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class TensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "0").lower() in {"1", "true", "yes", "on"}


def _tensor_debug_metadata(tensor):
    if not isinstance(tensor, torch.Tensor):
        return {"type": type(tensor).__name__}

    metadata = {
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
    return [
        target_name,
        "embed_tokens.weight",
        "layers.0.input_layernorm.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.mlp.experts.0.gate_proj.weight",
        "layers.0.mlp.experts.0.up_proj.weight",
        "layers.0.mlp.experts.0.down_proj.weight",
    ]


def _matches_value_debug_name(name: str) -> bool:
    return any(pattern and pattern in name for pattern in _value_debug_patterns())


def _tensor_value_debug(tensor):
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


def _dump_weight_transfer_debug(event: str, payload: dict) -> str | None:
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
        logger.exception("Failed to dump bucketed weight transfer debug payload for event=%s", event)
        return None


# copy from https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_utils.py
def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer


def create_shared_memory(size: int, name: str):
    """Create shared memory for weight transfer. If already exists, attach to it."""
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
        assert shm.size >= size, f"Stale shm segment '{name}': expected {size} bytes, got {shm.size}"
    return shm


def rebuild_shared_memory(name: str, size: int, dtype=torch.uint8):
    """Rebuild tensor from shared memory."""
    shm = shared_memory.SharedMemory(name=name)
    tensor = torch.frombuffer(shm.buf[:size], dtype=dtype)

    return tensor, shm


class BucketedWeightSender:
    """
    Send model weights via bucketed IPC transfer over ZMQ.

    Packs weight tensors into a fixed-size communication buffer and sends them
    in buckets to the receiver. Supports CUDA IPC and shared memory fallback.

    Args:
        zmq_handle: ZMQ IPC socket path (e.g., "ipc:///tmp/rl-colocate-zmq-<uuid>.sock")
        bucket_size_mb: Communication buffer size in MB
        use_shm: Use shared memory instead of CUDA IPC (for NPU compatibility)
    """

    def __init__(
        self,
        zmq_handle: str,
        bucket_size_mb: int = 512,
        use_shm: bool = False,
    ):
        self.zmq_handle = zmq_handle
        self.bucket_size_mb = bucket_size_mb
        self.bucket_size = int(bucket_size_mb) << 20
        self.use_shm = use_shm

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    def _dump_sender_bucket_debug(
        self,
        bucket_index: int,
        bucket_meta: dict[str, TensorMetadata],
        source_tensors: list[dict],
        is_last: bool,
    ) -> None:
        if not _env_enabled("VERL_OMNI_WEIGHT_SYNC_DEBUG"):
            return

        target_name = os.getenv("VERL_OMNI_WEIGHT_SYNC_TARGET_NAME", "thinker.audio_tower.conv2d1.bias")
        value_debug_enabled = _env_enabled("VERL_OMNI_WEIGHT_VALUE_DEBUG")
        should_dump = (
            bucket_index == 0
            or target_name in bucket_meta
            or (value_debug_enabled and any(_matches_value_debug_name(name) for name in bucket_meta))
        )
        if not should_dump:
            return

        payload = {
            "event": "sender_bucket",
            "bucket_index": bucket_index,
            "is_last": is_last,
            "zmq_handle": self.zmq_handle,
            "bucket_size_mb": self.bucket_size_mb,
            "use_shm": self.use_shm,
            "target_name": target_name,
            "target_in_bucket": target_name in bucket_meta,
            "bucket_names_sample": list(bucket_meta.keys())[:32],
            "bucket_count": len(bucket_meta),
            "bucket_meta": {
                name: {
                    "name": meta["name"],
                    "shape": tuple(meta["shape"]),
                    "dtype": str(meta["dtype"]),
                    "offset": int(meta["offset"]),
                }
                for name, meta in bucket_meta.items()
            },
            "source_tensors": source_tensors,
            "buffer": _tensor_debug_metadata(self.buffer),
        }
        dump_path = _dump_weight_transfer_debug("sender_bucket", payload)
        logger.info(
            "BucketedWeightSender debug dump: bucket=%d count=%d target_in_bucket=%s dump=%s",
            bucket_index,
            len(bucket_meta),
            target_name in bucket_meta,
            dump_path,
        )

    async def async_send_weights(self, weights):
        """
        Send weights to the receiver. Accepts a sync generator or async iterator.

        Args:
            weights: Generator or async iterator yielding (name, tensor) pairs
        """
        from verl.workers.rollout.utils import ensure_async_iterator

        try:
            self._init_socket()
            self._init_buffer()

            # send bucket weights
            offset = 0
            bucket_meta: dict[str, TensorMetadata] = {}
            bucket_source_tensors: list[dict] = []
            bucket_index = 0
            target_name = os.getenv("VERL_OMNI_WEIGHT_SYNC_TARGET_NAME", "thinker.audio_tower.conv2d1.bias")
            # dtype = PrecisionType.to_dtype(self.config.dtype)
            async for name, weight in ensure_async_iterator(weights):
                # model parameters are in fp32 full precision
                # (vermouth1992) we should not force cast weight here because some parameters
                # (such as moe gate) have to keep fp32 precision. If a weight is bf16 in the rollout side,
                # the rollout should automatically cast on demand. However, this would incur a higher weight
                # transfer volume.
                # weight = weight.to(dtype, non_blocking=True)

                # fill the tensor bucket
                if offset + weight.nbytes > self.bucket_size:
                    get_torch_device().synchronize()
                    self._dump_sender_bucket_debug(
                        bucket_index,
                        bucket_meta,
                        bucket_source_tensors,
                        is_last=False,
                    )
                    self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                    self.socket.recv()
                    bucket_meta = {}
                    bucket_source_tensors = []
                    offset = 0
                    bucket_index += 1

                # TODO: slice embedding layer weight into chunks
                assert offset + weight.nbytes <= self.bucket_size, (
                    f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket."
                    f"Please increase rollout.update_weights_bucket_megabytes({self.bucket_size_mb} MB)."
                )
                bucket_meta[name] = {
                    "name": name,
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "offset": offset,
                }
                if _env_enabled("VERL_OMNI_WEIGHT_SYNC_DEBUG") and (
                    bucket_index == 0
                    or name == target_name
                    or len(bucket_source_tensors) < 8
                    or (_env_enabled("VERL_OMNI_WEIGHT_VALUE_DEBUG") and _matches_value_debug_name(name))
                ):
                    source_tensor_payload = {"name": name, "tensor": _tensor_debug_metadata(weight)}
                    if _env_enabled("VERL_OMNI_WEIGHT_VALUE_DEBUG") and _matches_value_debug_name(name):
                        source_tensor_payload["value"] = _tensor_value_debug(weight)
                    bucket_source_tensors.append(source_tensor_payload)
                self.buffer[offset : offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
                offset += weight.nbytes

            # send the last bucket
            get_torch_device().synchronize()
            self._dump_sender_bucket_debug(
                bucket_index,
                bucket_meta,
                bucket_source_tensors,
                is_last=True,
            )
            self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": True})
            self.socket.recv()
        finally:
            self._cleanup()

    def _init_socket(self):
        """Initialize ZMQ REQ socket and bind."""
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.bind(self.zmq_handle)

    def _init_buffer(self):
        """build communication buffer"""
        buffer, shm = None, None
        if not self.use_shm:
            buffer = torch.empty(self.bucket_size, dtype=torch.uint8, device=f"{get_device_name()}:{get_device_id()}")
            handle = reduce_tensor(buffer)
            self.socket.send_pyobj(handle)
        else:
            import uuid

            # Create unique name for shared memory
            shm_name = f"verl_weights_{uuid.uuid4().hex}"
            shm = create_shared_memory(self.bucket_size, shm_name)
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)

            comm_metadata = {"name": shm_name, "size": self.bucket_size}
            self.socket.send_pyobj(comm_metadata)

        self.socket.recv()
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self):
        """clean up"""
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            del self.shm
            self.shm = None
        gc.collect()
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()


class BucketedWeightReceiver:
    """
    Receive model weights via bucketed IPC transfer over ZMQ.

    Receives weight tensors from BucketedWeightSender and passes each
    bucket to a callback for processing (e.g., loading into the model).

    Args:
        zmq_handle: ZMQ IPC socket path (must match sender)
        device: Target device for received tensors
        use_shm: Use shared memory instead of CUDA IPC
    """

    def __init__(
        self,
        zmq_handle: str,
        device: torch.device,
        use_shm: bool = False,
    ):
        self.zmq_handle = zmq_handle
        self.device = device
        self.use_shm = use_shm

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    def receive_weights(self, on_bucket_received: callable):
        """
        Receive weights from sender and process each bucket via callback.

        Args:
            on_bucket_received: Callback function(weights: list[(name, tensor)]) called per bucket.
        """
        try:
            self._init_socket()
            self._init_buffer()

            # receive bucket and update weights
            while True:
                metadata = self.socket.recv_pyobj()
                weights, tensor = [], None
                for name, meta in metadata["bucket_meta"].items():
                    shape, dtype, offset = meta["shape"], meta["dtype"], meta["offset"]
                    size = dtype.itemsize * shape.numel()
                    tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)
                    if self.use_shm:
                        tensor = tensor.to(self.device)
                    weights.append((name, tensor))
                on_bucket_received(weights)
                get_torch_device().synchronize()
                self.socket.send(b"")
                del weights, tensor
                if metadata["is_last"]:
                    break
        finally:
            self._cleanup()

    def _init_socket(self):
        """Initialize ZMQ REP socket and connect."""
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.connect(self.zmq_handle)

    def _init_buffer(self):
        """Receive and rebuild communication buffer from sender."""
        comm_metadata = self.socket.recv_pyobj()
        buffer, shm = None, None
        if not self.use_shm:
            handle = comm_metadata
            buffer = rebuild_ipc(handle, self.device.index)
            assert buffer.dtype == torch.uint8
        else:
            shm_name = comm_metadata["name"]
            shm_size = comm_metadata["size"]
            buffer, shm = rebuild_shared_memory(shm_name, shm_size, dtype=torch.uint8)
        self.socket.send(b"")
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self):
        """clean up"""
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        # Synchronize before releasing the buffer to ensure all async ops
        # referencing it (e.g. clone, .to()) have completed.
        get_torch_device().synchronize()
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            del self.shm
            self.shm = None
        gc.collect()
        get_torch_device().ipc_collect()
        get_torch_device().empty_cache()

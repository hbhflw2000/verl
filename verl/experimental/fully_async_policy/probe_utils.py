# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Small dependency-light helpers for fully-async diagnostic probes."""

from typing import Any

import torch


def router_replay_tensor_metrics(value: Any) -> dict[str, float]:
    """Return a compact, order-sensitive fingerprint for recorded MoE routes."""
    if not isinstance(value, torch.Tensor):
        return {"present": 0.0}

    flat = (value.values() if value.is_nested else value).detach().reshape(-1).to(torch.int64)
    if flat.numel() == 0:
        return {"present": 1.0, "numel": 0.0, "sum": 0.0, "square_sum": 0.0, "sample_hash": 0.0}

    # A bounded strided sample keeps the probe inexpensive while making a
    # simple route-order mix-up visible in the JSONL summary.
    stride = max(flat.numel() // 64, 1)
    sample = flat[::stride][:64]
    weights = torch.arange(1, sample.numel() + 1, device=sample.device, dtype=torch.int64)
    return {
        "present": 1.0,
        "numel": float(flat.numel()),
        "sum": float(flat.sum().item()),
        "square_sum": float((flat * flat).sum().item()),
        "sample_hash": float((sample * weights).sum().item()),
    }

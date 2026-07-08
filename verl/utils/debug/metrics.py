# Copyright 2025 Individual Contributor: TomQunChaoA
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

import logging
import math
import os

import torch

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        print(
            f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}"
        )
        return torch.ones_like(tensor1)
    # transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)

    # calculate diff
    diff_mask = tensor1 != tensor2

    valid_diff_mask = diff_mask & (mask == 1)

    diff_counts = valid_diff_mask.sum(dim=1)

    return diff_counts


def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # implemention of https://arxiv.org/pdf/2506.13585
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        return float("nan")
    pair_mask = mask.bool() & torch.isfinite(tensor1) & torch.isfinite(tensor2)
    mt1 = torch.masked_select(tensor1.detach().float(), pair_mask)
    mt2 = torch.masked_select(tensor2.detach().float(), pair_mask)
    if mt1.numel() < 2 or mt2.numel() < 2:
        return float("nan")
    if torch.std(mt1) == 0 or torch.std(mt2) == 0:
        return float("nan")
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()


def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)


def _masked_float_values(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tensor.shape != mask.shape:
        return torch.empty(0, dtype=torch.float32, device=tensor.device)
    values = torch.masked_select(tensor.detach().float(), mask.bool())
    return values[torch.isfinite(values)]


def _masked_pair_values(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor1.shape != tensor2.shape or tensor1.shape != mask.shape:
        empty = torch.empty(0, dtype=torch.float32, device=tensor1.device)
        return empty, empty
    pair_mask = mask.bool() & torch.isfinite(tensor1) & torch.isfinite(tensor2)
    return (
        torch.masked_select(tensor1.detach().float(), pair_mask),
        torch.masked_select(tensor2.detach().float(), pair_mask),
    )


def _masked_corrcoef(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> float:
    if tensor1.shape != tensor2.shape or tensor1.shape != mask.shape:
        return float("nan")
    values1, values2 = _masked_pair_values(tensor1, tensor2, mask)
    if values1.numel() < 2 or values2.numel() < 2:
        return float("nan")
    if torch.std(values1) == 0 or torch.std(values2) == 0:
        return float("nan")
    return torch.corrcoef(torch.stack([values1, values2], dim=0))[0][1].detach().item()


def _masked_mean_abs_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> float:
    if tensor1.shape != tensor2.shape or tensor1.shape != mask.shape:
        return float("nan")
    values = _masked_float_values(torch.abs(tensor1.detach().float() - tensor2.detach().float()), mask)
    if values.numel() == 0:
        return float("nan")
    return values.mean().detach().item()


def _rollout_corr_shift_metrics(
    actor_old_log_probs: torch.Tensor,
    rollout_old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, float]:
    if (
        actor_old_log_probs.shape != rollout_old_log_probs.shape
        or actor_old_log_probs.shape != response_mask.shape
        or actor_old_log_probs.ndim != 2
        or actor_old_log_probs.size(1) < 2
    ):
        return {}

    mask = response_mask.bool()
    shift_pairs = {
        "old_t_rollout_t_plus_1": (
            actor_old_log_probs[:, :-1],
            rollout_old_log_probs[:, 1:],
            mask[:, :-1] & mask[:, 1:],
        ),
        "old_t_plus_1_rollout_t": (
            actor_old_log_probs[:, 1:],
            rollout_old_log_probs[:, :-1],
            mask[:, 1:] & mask[:, :-1],
        ),
    }

    metrics = {}
    for name, (old_values, rollout_values, pair_mask) in shift_pairs.items():
        old_probs = torch.exp(old_values)
        rollout_probs = torch.exp(rollout_values)
        valid_tokens = int(pair_mask.sum().item())
        metrics[f"training/rollout_shift/{name}/valid_tokens"] = valid_tokens
        metrics[f"training/rollout_shift/{name}/logprob_abs_diff_mean"] = _masked_mean_abs_diff(
            old_values, rollout_values, pair_mask
        )
        metrics[f"training/rollout_shift/{name}/prob_abs_diff_mean"] = _masked_mean_abs_diff(
            old_probs, rollout_probs, pair_mask
        )
        metrics[f"training/rollout_shift/{name}/logprob_pearson_corr"] = _masked_corrcoef(
            old_values, rollout_values, pair_mask
        )
        metrics[f"training/rollout_shift/{name}/prob_pearson_corr"] = _masked_corrcoef(
            old_probs, rollout_probs, pair_mask
        )
    return metrics


def _masked_value_stats(prefix: str, tensor: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    if tensor.shape != mask.shape:
        return {
            f"{prefix}/shape_mismatch": 1,
            f"{prefix}/valid_tokens": 0,
            f"{prefix}/finite_fraction": float("nan"),
            f"{prefix}/zero_fraction": float("nan"),
        }

    mask_bool = mask.bool()
    values = torch.masked_select(tensor.detach().float(), mask_bool)
    valid_tokens = int(values.numel())
    metrics: dict[str, float] = {
        f"{prefix}/shape_mismatch": 0,
        f"{prefix}/valid_tokens": valid_tokens,
    }
    if valid_tokens == 0:
        metrics.update(
            {
                f"{prefix}/finite_fraction": float("nan"),
                f"{prefix}/nonfinite_fraction": float("nan"),
                f"{prefix}/zero_fraction": float("nan"),
                f"{prefix}/near_zero_fraction": float("nan"),
            }
        )
        return metrics

    finite = torch.isfinite(values)
    finite_values = values[finite]
    metrics[f"{prefix}/finite_fraction"] = finite.float().mean().detach().item()
    metrics[f"{prefix}/nonfinite_fraction"] = (~finite).float().mean().detach().item()
    metrics[f"{prefix}/zero_fraction"] = (values == 0).float().mean().detach().item()
    metrics[f"{prefix}/near_zero_fraction"] = (values.abs() < 1e-12).float().mean().detach().item()
    if finite_values.numel() == 0:
        metrics[f"{prefix}/min"] = float("nan")
        metrics[f"{prefix}/mean"] = float("nan")
        metrics[f"{prefix}/max"] = float("nan")
    else:
        metrics[f"{prefix}/min"] = finite_values.min().detach().item()
        metrics[f"{prefix}/mean"] = finite_values.mean().detach().item()
        metrics[f"{prefix}/max"] = finite_values.max().detach().item()
    return metrics


def _maybe_log_rollout_corr_debug(
    actor_old_log_probs: torch.Tensor,
    rollout_old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    responses: torch.Tensor,
) -> None:
    raw_limit = os.environ.get("VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT", "0")
    try:
        limit = int(raw_limit)
    except ValueError:
        limit = 0
    if limit <= 0:
        return

    mask = response_mask.bool()

    def _summary(name: str, tensor: torch.Tensor) -> str:
        if tensor.shape != mask.shape:
            return f"{name}: shape={tuple(tensor.shape)} mask_shape={tuple(mask.shape)} shape_mismatch=1"
        values = torch.masked_select(tensor.detach().float(), mask)
        if values.numel() == 0:
            return f"{name}: count=0"
        finite = torch.isfinite(values)
        finite_values = values[finite]
        sample = values[:limit].cpu().tolist()
        if finite_values.numel() == 0:
            return (
                f"{name}: count={values.numel()} finite=0 "
                f"nan={(torch.isnan(values)).sum().item()} inf={(torch.isinf(values)).sum().item()} "
                f"sample={sample}"
            )
        return (
            f"{name}: count={values.numel()} "
            f"min={finite_values.min().item():.6f} "
            f"mean={finite_values.mean().item():.6f} "
            f"max={finite_values.max().item():.6f} "
            f"zero={(values == 0).sum().item()} "
            f"nan={(torch.isnan(values)).sum().item()} "
            f"inf={(torch.isinf(values)).sum().item()} "
            f"sample={sample}"
        )

    valid_lens = mask.sum(dim=-1).detach().cpu().tolist()[:limit]
    if responses.shape == mask.shape:
        response_sample = torch.masked_select(responses.detach(), mask)[:limit].cpu().tolist()
    else:
        response_sample = [f"shape_mismatch responses={tuple(responses.shape)} mask={tuple(mask.shape)}"]
    diff = actor_old_log_probs.detach().float() - rollout_old_log_probs.detach().float()
    print(
        "[RolloutCorrDebug] "
        f"limit={limit} shapes "
        f"old={tuple(actor_old_log_probs.shape)} rollout={tuple(rollout_old_log_probs.shape)} "
        f"mask={tuple(response_mask.shape)} responses={tuple(responses.shape)} "
        f"valid_lens_sample={valid_lens} response_token_sample={response_sample}"
    )
    print(f"[RolloutCorrDebug] {_summary('old_log_probs', actor_old_log_probs)}")
    print(f"[RolloutCorrDebug] {_summary('rollout_log_probs', rollout_old_log_probs)}")
    print(f"[RolloutCorrDebug] {_summary('old_minus_rollout', diff)}")
    shift_metrics = _rollout_corr_shift_metrics(actor_old_log_probs, rollout_old_log_probs, mask)
    if shift_metrics:
        printable_shift_metrics = {
            key: round(value, 6) if isinstance(value, float) and math.isfinite(value) else value
            for key, value in shift_metrics.items()
        }
        print(f"[RolloutCorrDebug] shift_metrics={printable_shift_metrics}")

    if (
        actor_old_log_probs.shape == rollout_old_log_probs.shape
        and actor_old_log_probs.shape == responses.shape
        and actor_old_log_probs.shape == mask.shape
        and actor_old_log_probs.ndim == 2
        and actor_old_log_probs.size(0) > 0
    ):
        paired_rows = []
        row_count = min(limit, actor_old_log_probs.size(0))
        token_count = min(limit, actor_old_log_probs.size(1))
        for row_idx in range(row_count):
            row_pairs = []
            for col_idx in range(token_count):
                row_pairs.append(
                    {
                        "pos": col_idx,
                        "mask": int(mask[row_idx, col_idx].item()),
                        "token": int(responses[row_idx, col_idx].item()),
                        "old": round(float(actor_old_log_probs[row_idx, col_idx].item()), 6),
                        "rollout": round(float(rollout_old_log_probs[row_idx, col_idx].item()), 6),
                        "diff": round(float(diff[row_idx, col_idx].item()), 6),
                    }
                )
            paired_rows.append(
                {
                    "row": row_idx,
                    "valid_len": int(mask[row_idx].sum().item()),
                    "rollout_zero_valid": int(((rollout_old_log_probs[row_idx] == 0) & mask[row_idx]).sum().item()),
                    "pairs": row_pairs,
                }
            )
        print(f"[RolloutCorrDebug] paired_token_logprobs={paired_rows}")


def calculate_debug_metrics(data: DataProto) -> dict:
    """
    calculate rollout vs actor logprobs diff, for debugging purpose

    Args:
        data: DataProto
            the data batch to calculate
            rollout_log_probs: log_probs record when rollout forward tokens
            old_log_probs(actor log probs): log_probs record when actor forward tokens
            loss_mask or attention_mask: to mask unrelated token
            responses: the response tokens, for calculating size
    Returns:
        dict: metrics
            "training/rollout_probs_diff_valid": 1->input is valid, 0->input is invalid
            "training/rollout_probs_diff_max": max value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_mean": mean value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_std": std value of logprob diff of rollout vs. actor
            "training/rollout_actor_probs_pearson_corr": logprob's pearson corrcoef of rollout vs. actor, reference to https://arxiv.org/pdf/2506.13585
    """

    rollout_old_log_probs = data.batch["rollout_log_probs"]
    actor_old_log_probs = data.batch["old_log_probs"]
    if "response_mask" in data.batch:
        logger.debug("response mask found, use it to mask log probs")
        log_prob_mask = data.batch["response_mask"]
    elif "attention_mask" in data.batch:
        log_prob_mask = data.batch["attention_mask"]
    else:
        logger.warning(f"no mask info found, use all log probs, {(data.batch.keys())=}")
        log_prob_mask = torch.ones_like(rollout_old_log_probs)
    responses = data.batch["responses"]
    response_length = responses.size(1)

    response_mask = log_prob_mask[:, -response_length:]
    try:
        _maybe_log_rollout_corr_debug(actor_old_log_probs, rollout_old_log_probs, response_mask, responses)
    except Exception as exc:
        print(f"[RolloutCorrDebug] failed to log rollout corr debug: {exc}")
    # calculate pearson corrcoef
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    response_mask_bool = response_mask.bool()

    # check if there are any valid tokens before computing metrics
    if not response_mask_bool.any():
        logger.warning("response_mask is all False, returning default metrics")
        return {
            "training/rollout_probs_diff_valid": 0,
            "training/rollout_probs_diff_max": float("nan"),
            "training/rollout_probs_diff_mean": float("nan"),
            "training/rollout_probs_diff_std": float("nan"),
            "training/rollout_actor_probs_pearson_corr": float("nan"),
        }

    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)
    metrics = {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
        "training/rollout_actor_logprob_pearson_corr": _masked_corrcoef(
            actor_old_log_probs, rollout_old_log_probs, response_mask_bool
        ),
        "training/rollout_actor_logprob_abs_diff_mean": _masked_mean_abs_diff(
            actor_old_log_probs, rollout_old_log_probs, response_mask_bool
        ),
    }
    metrics.update(_masked_value_stats("training/rollout_log_probs", rollout_old_log_probs, response_mask_bool))
    metrics.update(_masked_value_stats("training/actor_old_log_probs", actor_old_log_probs, response_mask_bool))
    metrics.update(_rollout_corr_shift_metrics(actor_old_log_probs, rollout_old_log_probs, response_mask_bool))
    return metrics

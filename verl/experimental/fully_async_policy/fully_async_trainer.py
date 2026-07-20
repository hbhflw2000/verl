# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import asyncio
import copy
import fcntl
import json
import logging
import math
import os
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.fully_async_policy.detach_utils import (
    MetricsAggregator,
    assemble_batch_from_rollout_samples,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.fully_async_policy.probe_utils import router_replay_tensor_metrics
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils import tensordict_utils as tu
from verl.utils.tracking import Tracking
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

logger = logging.getLogger(__name__)


class TrainingStopException(Exception):
    """Exception raised to signal training should stop"""

    pass


@ray.remote(num_cpus=10)
class FullyAsyncTrainer(SeparateRayPPOTrainer):
    """
    A fully asynchronous PPO trainer that obtains samples from a MessageQueue for training.
    Based on an improved implementation of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        device_name=None,
    ):
        # ==================== RayPPOTrainer config ====================

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert not self.hybrid_engine

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.config)

        # distillation config needed by _update_actor in ray_trainer.py
        from verl.trainer.distillation.losses import is_distillation_enabled

        if is_distillation_enabled(self.config.get("distillation")):
            self.distillation_config = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.distillation_config = None

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        # ==================== SeparateRayPPOTrainer config ====================
        self.global_steps = 0
        self.epoch = 0
        self._init_dump_executor()
        self.validation_generations_logger = None
        self.max_steps_duration = 0
        self.progress_bar = None
        self.is_last_step = False
        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False
        self.last_val_metrics = {}
        self.metrics = {}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # ==================== fully async config ====================

        self.message_queue_client = None

        # Statistics
        self.local_trigger_step = 1
        self.processed_samples = 0
        self.stale_trajectory_processed = 0
        self.current_param_version = 0
        self.total_train_steps = None
        self.progress_bar = None
        self.trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.last_ckpt_version = 0
        self.train_role = Role.ActorRollout if config.async_training.use_trainer_do_validate else Role.Actor

        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        total_gpus = (
            config.trainer.nnodes * config.trainer.n_gpus_per_node
            + config.rollout.nnodes * config.rollout.n_gpus_per_node
        )
        self.metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)

        # Reference to rollouter for parameter synchronization
        self.rollouter = None
        self.checkpoint_manager = None

        # Hybrid checkpoint manager for trainer-side validation (use_trainer_do_validate)
        # Uses naive backend to sync weights from trainer to hybrid rollout replicas.
        # Initialized in _setup_hybrid_checkpoint_manager_and_sleep() via set_rollouter().
        self.hybrid_checkpoint_manager = None

    async def _setup_checkpoint_manager(self):
        """Setup checkpoint manager after rollouter is initialized"""
        replicas = await self.rollouter.get_replicas.remote()
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config, trainer=self.actor_wg, replicas=replicas
        )
        print("[FullyAsyncTrainer] Checkpoint manager initialized")

    async def _setup_hybrid_checkpoint_manager(self):
        """Setup hybrid checkpoint manager and perform initial sleep of hybrid replicas.

        When use_trainer_do_validate is enabled:
          1. Creates a CheckpointEngineManager with naive backend for trainer-side
             weight sync to hybrid rollout replicas.
          2. Fetches hybrid replicas from the rollouter's ALM (created during
             rollouter.init_workers()).
          3. Registers them with the hybrid CP manager and calls sleep_replicas()
             to release GPU memory for training.

        Must be called AFTER set_rollouter() so that self.rollouter is available,
        and AFTER rollouter.init_workers() so that hybrid replicas exist.
        This mirrors the colocate pattern in ray_trainer.py:882-889 but fetches
        replicas from the rollouter's ALM via RPC since they live on the rollout side.
        """
        if not self.config.async_training.use_trainer_do_validate:
            return

        # --- Part 1: Create hybrid CheckpointEngineManager with naive backend ---
        print("[FullyAsyncTrainer] Setting up hybrid checkpoint manager (naive backend)")

        # Create hybrid CheckpointEngineManager with naive backend.
        checkpoint_engine_cfg = self.config.actor_rollout_ref.rollout.checkpoint_engine
        original_backend = checkpoint_engine_cfg.backend
        with open_dict(checkpoint_engine_cfg):
            checkpoint_engine_cfg.backend = "naive"
        checkpoint_engine_config = omega_conf_to_dataclass(checkpoint_engine_cfg)

        self.hybrid_checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=[],  # Start empty; will be populated below
        )

        # Restore original backend value
        with open_dict(checkpoint_engine_cfg):
            checkpoint_engine_cfg.backend = original_backend

        print("[FullyAsyncTrainer] Hybrid checkpoint manager initialized (naive backend)")

        # --- Part 2: Fetch hybrid replicas from rollouter's ALM ---
        print("[FullyAsyncTrainer] Fetching hybrid replicas from rollouter...")
        hybrid_replicas_dict = ray.get(self.rollouter.get_all_hybrid_replicas.remote())
        print(
            f"[FullyAsyncTrainer] Got {len(hybrid_replicas_dict)} hybrid replicas: {list(hybrid_replicas_dict.keys())}"
        )

        if not hybrid_replicas_dict:
            print("[FullyAsyncTrainer] No hybrid replicas found, skipping initial sleep")
            return

        # --- Part 3: Register replicas and perform initial sleep ---
        for resource_id, replica in hybrid_replicas_dict.items():
            self.hybrid_checkpoint_manager.replicas.append(replica)
            print(
                f"[FullyAsyncTrainer] Registered '{resource_id}' "
                f"(mode={getattr(replica, 'rollout_mode', '?')}, "
                f"addr={getattr(replica, '_server_address', '?')})"
            )

        # Step 3: Sleep all hybrid replicas
        print(
            f"[FullyAsyncTrainer] Calling sleep_replicas() on "
            f"{len(self.hybrid_checkpoint_manager.replicas)} replicas..."
        )
        await self.hybrid_checkpoint_manager.sleep_replicas()
        print("[FullyAsyncTrainer] Initial sleep complete, GPU memory now owned by training engine")

    def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        self.message_queue_client = message_queue_client

    async def set_rollouter(self, rollouter):
        """Set rollouter reference and initialize all checkpoint managers."""
        self.rollouter = rollouter
        # Setup checkpoint manager after rollouter is set
        await self._setup_checkpoint_manager()
        await self._setup_hybrid_checkpoint_manager()

    def set_total_train_steps(self, total_training_steps):
        self.total_train_steps = total_training_steps

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

        self.progress_bar = tqdm(total=self.total_train_steps, initial=0, desc="Training Progress")

    @staticmethod
    def _load_fixed_sequence_records(jsonl_path: str, row_limit: int) -> list[dict[str, Any]]:
        path = Path(jsonl_path)
        if not path.is_file():
            raise FileNotFoundError(f"fixed-sequence score input JSONL not found: {path}")

        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("event") != "rollout_corr_sample":
                    continue
                records.append(record)
                if row_limit > 0 and len(records) >= row_limit:
                    break
        if not records:
            raise ValueError(f"no rollout_corr_sample records found in {path}")
        return records

    @staticmethod
    def _stack_fixed_sequence_tensor(records: list[dict[str, Any]], key: str, dtype: torch.dtype) -> torch.Tensor | None:
        values = [record.get(key) for record in records]
        if any(value is None for value in values):
            return None
        return torch.as_tensor(values, dtype=dtype)

    @staticmethod
    def _masked_pair_summary(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
        if left.shape != right.shape or left.shape != mask.shape:
            return {"shape_mismatch": 1.0}
        pair_mask = mask.bool() & torch.isfinite(left.float()) & torch.isfinite(right.float())
        values_left = torch.masked_select(left.detach().float(), pair_mask)
        values_right = torch.masked_select(right.detach().float(), pair_mask)
        if values_left.numel() == 0:
            return {
                "shape_mismatch": 0.0,
                "valid_tokens": 0.0,
                "abs_diff_mean": float("nan"),
                "abs_diff_max": float("nan"),
                "pearson_corr": float("nan"),
            }
        diff = (values_left - values_right).abs()
        if values_left.numel() < 2 or torch.std(values_left) == 0 or torch.std(values_right) == 0:
            corr = float("nan")
        else:
            corr = torch.corrcoef(torch.stack([values_left, values_right], dim=0))[0][1].detach().item()
        return {
            "shape_mismatch": 0.0,
            "valid_tokens": float(values_left.numel()),
            "abs_diff_mean": diff.mean().detach().item(),
            "abs_diff_max": diff.max().detach().item(),
            "pearson_corr": corr,
        }

    @classmethod
    def _masked_shift_scan(
        cls,
        left: torch.Tensor,
        right: torch.Tensor,
        mask: torch.Tensor,
        max_shift: int = 8,
    ) -> dict[str, float]:
        if left.shape != right.shape or left.shape != mask.shape or left.ndim != 2:
            return {"shape_mismatch": 1.0}

        metrics: dict[str, float] = {"shape_mismatch": 0.0}
        best_abs_delta = 0
        best_abs_value = float("inf")
        best_corr_delta = 0
        best_corr_value = float("nan")
        for delta in range(-max_shift, max_shift + 1):
            if delta < 0:
                left_view = left[:, -delta:]
                right_view = right[:, : right.size(1) + delta]
                mask_view = mask[:, -delta:] & mask[:, : mask.size(1) + delta]
            elif delta > 0:
                left_view = left[:, : left.size(1) - delta]
                right_view = right[:, delta:]
                mask_view = mask[:, : mask.size(1) - delta] & mask[:, delta:]
            else:
                left_view = left
                right_view = right
                mask_view = mask

            summary = cls._masked_pair_summary(left_view, right_view, mask_view)
            abs_diff = float(summary.get("abs_diff_mean", float("nan")))
            corr = float(summary.get("pearson_corr", float("nan")))
            metrics[f"delta_{delta}/valid_tokens"] = float(summary.get("valid_tokens", 0.0))
            metrics[f"delta_{delta}/abs_diff_mean"] = abs_diff
            metrics[f"delta_{delta}/pearson_corr"] = corr
            if math.isfinite(abs_diff) and abs_diff < best_abs_value:
                best_abs_value = abs_diff
                best_abs_delta = delta
            if math.isfinite(corr) and (not math.isfinite(best_corr_value) or corr > best_corr_value):
                best_corr_value = corr
                best_corr_delta = delta

        metrics["best_abs_diff_delta"] = float(best_abs_delta)
        metrics["best_abs_diff_mean"] = best_abs_value
        metrics["best_corr_delta"] = float(best_corr_delta)
        metrics["best_pearson_corr"] = best_corr_value
        return metrics

    @staticmethod
    def _tensor_slice_list(tensor: torch.Tensor | None, row_idx: int, start: int, end: int) -> list:
        if tensor is None:
            return []
        row = tensor[row_idx]
        start = max(0, start)
        end = min(row.shape[-1], end)
        if start >= end:
            return []
        if row.dim() == 1:
            return row[start:end].detach().cpu().tolist()
        return row[..., start:end].detach().cpu().tolist()

    def _build_fixed_sequence_batch(
        self, jsonl_path: str, row_limit: int
    ) -> tuple[DataProto, list[dict[str, Any]], dict[str, torch.Tensor]]:
        records = self._load_fixed_sequence_records(jsonl_path, row_limit)
        tensors: dict[str, torch.Tensor] = {}
        required_int_keys = ("input_ids", "attention_mask", "responses", "response_mask")
        for key in required_int_keys:
            tensor = self._stack_fixed_sequence_tensor(records, key, torch.long)
            if tensor is None:
                raise ValueError(f"fixed-sequence record missing required key: {key}")
            tensors[key] = tensor

        prompts = self._stack_fixed_sequence_tensor(records, "prompts", torch.long)
        if prompts is None:
            prompt_length = tensors["input_ids"].shape[-1] - tensors["responses"].shape[-1]
            prompts = tensors["input_ids"][:, :prompt_length].contiguous()
        tensors["prompts"] = prompts

        optional_int_keys = ("position_ids",)
        for key in optional_int_keys:
            tensor = self._stack_fixed_sequence_tensor(records, key, torch.long)
            if tensor is not None:
                tensors[key] = tensor

        originals: dict[str, torch.Tensor] = {}
        for source_key, target_key in (
            ("rollout_log_probs", "rollout_log_probs"),
            ("actor_old_log_probs", "dump_actor_old_log_probs"),
            ("ref_log_probs", "dump_ref_log_prob"),
        ):
            tensor = self._stack_fixed_sequence_tensor(records, source_key, torch.float32)
            if tensor is not None:
                if target_key == "rollout_log_probs":
                    tensors[target_key] = tensor
                else:
                    originals[target_key] = tensor

        batch = DataProto.from_dict(
            tensors=tensors,
            non_tensors={"fixed_sequence_row": np.array([record.get("row", idx) for idx, record in enumerate(records)])},
            meta_info={
                "fixed_sequence_score_probe": True,
                "temperature": self.config.actor_rollout_ref.rollout.temperature,
            },
        )
        return batch, records, originals

    @staticmethod
    def _clone_dataproto_for_probe(batch: DataProto) -> DataProto:
        cloned_tensors = {key: value.detach().clone() for key, value in batch.batch.items()}
        cloned_non_tensors = {
            key: value.copy() if hasattr(value, "copy") else copy.deepcopy(value)
            for key, value in batch.non_tensor_batch.items()
        }
        return DataProto.from_dict(
            tensors=cloned_tensors,
            non_tensors=cloned_non_tensors,
            meta_info=copy.deepcopy(batch.meta_info),
        )

    @staticmethod
    def _sample_masked_values(tensor: torch.Tensor, mask: torch.Tensor, limit: int = 8) -> list[float]:
        if tensor.shape != mask.shape:
            return []
        values = torch.masked_select(tensor.detach().float(), mask.bool())
        return [round(float(value), 6) for value in values[:limit].cpu().tolist()]

    @staticmethod
    def _json_safe(value):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            return tensor.item() if tensor.numel() == 1 else tensor.tolist()
        if isinstance(value, dict):
            return {str(key): FullyAsyncTrainer._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [FullyAsyncTrainer._json_safe(item) for item in value]
        try:
            return int(value)
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return str(value)

    def _append_multistage_logprob_debug_jsonl(self, event: str, payload: dict[str, Any]) -> None:
        output_path = os.environ.get("VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_JSONL", "")
        if not output_path:
            return
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "event": event,
                "time": time.time(),
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "global_steps": self.global_steps,
                "local_trigger_step": self.local_trigger_step,
                "current_param_version": self.current_param_version,
                **payload,
            }
            with path.open("a", encoding="utf-8") as fout:
                fcntl.flock(fout.fileno(), fcntl.LOCK_EX)
                fout.write(json.dumps(self._json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")
                fcntl.flock(fout.fileno(), fcntl.LOCK_UN)
        except Exception as exc:
            print(f"[MultiStageLogprobDebug] failed to write {event}: {exc}", flush=True)

    @staticmethod
    def _masked_tensor_summary(tensor: torch.Tensor | None, mask: torch.Tensor | None) -> dict[str, float]:
        if tensor is None:
            return {"present": 0.0}
        values = tensor.detach().float()
        if mask is not None and mask.shape == tensor.shape:
            values = torch.masked_select(values, mask.bool())
        else:
            values = values.reshape(-1)
        finite = values[torch.isfinite(values)]
        if finite.numel() == 0:
            return {
                "present": 1.0,
                "count": 0.0,
                "finite_fraction": 0.0,
                "zero_fraction": float("nan"),
                "near_zero_fraction": float("nan"),
                "min": float("nan"),
                "mean": float("nan"),
                "max": float("nan"),
            }
        return {
            "present": 1.0,
            "count": float(values.numel()),
            "finite_fraction": float(finite.numel() / max(values.numel(), 1)),
            "zero_fraction": float((finite == 0).float().mean().item()),
            "near_zero_fraction": float((finite.abs() < 1e-5).float().mean().item()),
            "min": float(finite.min().item()),
            "mean": float(finite.mean().item()),
            "max": float(finite.max().item()),
        }

    @staticmethod
    def _values_at_positions(tensor: torch.Tensor | None, row_idx: int, positions: torch.Tensor) -> list:
        if tensor is None or tensor.dim() < 2 or row_idx >= tensor.shape[0] or positions.numel() == 0:
            return []
        row = tensor[row_idx]
        positions = positions[positions < row.shape[-1]]
        if positions.numel() == 0:
            return []
        return row.index_select(dim=-1, index=positions.to(row.device)).detach().cpu().tolist()

    def _maybe_dump_multistage_logprob_batch(self, stage: str, batch: DataProto) -> None:
        if not os.environ.get("VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_JSONL", ""):
            return
        if batch is None or batch.batch is None or "responses" not in batch.batch:
            return
        try:
            row_limit = int(os.environ.get("VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_ROWS", "4"))
        except ValueError:
            row_limit = 4
        try:
            token_limit = int(os.environ.get("VERL_OMNI_MULTISTAGE_LOGPROB_DEBUG_TOKENS", "16"))
        except ValueError:
            token_limit = 16
        if row_limit <= 0 or token_limit <= 0:
            return

        responses = batch.batch.get("responses")
        response_mask = batch.batch.get("response_mask")
        if response_mask is None and responses is not None:
            response_mask = torch.ones_like(responses, dtype=torch.bool)
        else:
            response_mask = response_mask.bool()

        fields = {
            "rollout_log_probs": batch.batch.get("rollout_log_probs"),
            "old_log_probs": batch.batch.get("old_log_probs"),
            "old_log_probs_repeat": batch.batch.get("old_log_probs_repeat"),
            "ref_log_prob": batch.batch.get("ref_log_prob"),
        }
        summaries = {name: self._masked_tensor_summary(value, response_mask) for name, value in fields.items()}
        rows = []
        max_rows = min(row_limit, responses.shape[0])
        for row_idx in range(max_rows):
            row_mask = response_mask[row_idx] if response_mask is not None else None
            if row_mask is not None:
                positions = torch.nonzero(row_mask, as_tuple=False).flatten()[:token_limit].cpu()
            else:
                positions = torch.arange(min(token_limit, responses.shape[-1]))
            row_payload = {
                "row": row_idx,
                "positions": positions.tolist(),
                "valid_tokens": int(row_mask.sum().item()) if row_mask is not None else int(responses.shape[-1]),
                "response_tokens": self._values_at_positions(responses, row_idx, positions),
            }
            for field_name, field_tensor in fields.items():
                row_payload[field_name] = self._values_at_positions(field_tensor, row_idx, positions)
            for key in ("uid", "index", "_rollout_seed_global_idx", "agent_name"):
                value = batch.non_tensor_batch.get(key)
                if value is not None and row_idx < len(value):
                    item = value[row_idx]
                    row_payload[key] = item.item() if hasattr(item, "item") else item
            rows.append(row_payload)

        self._append_multistage_logprob_debug_jsonl(
            "trainer_multistage_logprob_batch",
            {
                "stage": stage,
                "batch_size": len(batch),
                "shapes": {key: list(value.shape) for key, value in batch.batch.items()},
                "summaries": summaries,
                "rows": rows,
            },
        )

    def _write_fixed_sequence_score_output(
        self,
        output_path: str,
        records: list[dict[str, Any]],
        batch: DataProto,
        originals: dict[str, torch.Tensor],
        metrics: dict[str, float],
    ) -> None:
        if not output_path:
            return

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        response_mask = batch.batch["response_mask"].bool()
        rows = [
            {
                "event": "fixed_sequence_score_summary",
                "created_at": time.time(),
                "record_count": len(records),
                "metrics": metrics,
                "shapes": {key: list(value.shape) for key, value in batch.batch.items()},
            }
        ]

        old_log_probs = batch.batch.get("old_log_probs")
        old_log_probs_repeat = batch.batch.get("old_log_probs_repeat")
        rollout_log_probs = batch.batch.get("rollout_log_probs")
        ref_log_prob = batch.batch.get("ref_log_prob")
        input_ids = batch.batch.get("input_ids")
        attention_mask = batch.batch.get("attention_mask")
        prompts = batch.batch.get("prompts")
        position_ids = batch.batch.get("position_ids")
        response_start = None
        if input_ids is not None and batch.batch.get("responses") is not None:
            response_start = input_ids.shape[-1] - batch.batch["responses"].shape[-1]
        for row_idx, record in enumerate(records):
            row_mask = response_mask[row_idx]
            row_payload = {
                "event": "fixed_sequence_score_row",
                "created_at": time.time(),
                "input_row": record.get("row", row_idx),
                "row": row_idx,
                "valid_tokens": int(row_mask.sum().item()),
                "response_head": batch.batch["responses"][row_idx, :16].detach().cpu().tolist(),
            }
            if response_start is not None:
                boundary_start = response_start - 8
                boundary_end = response_start + 16
                row_payload["alignment_debug"] = {
                    "input_len": int(input_ids.shape[-1]) if input_ids is not None else None,
                    "prompt_len": int(prompts.shape[-1]) if prompts is not None else response_start,
                    "response_start": int(response_start),
                    "response_len": int(batch.batch["responses"].shape[-1]),
                    "valid_response_tokens": int(row_mask.sum().item()),
                    "attention_valid_tokens": (
                        int(attention_mask[row_idx].sum().item()) if attention_mask is not None else None
                    ),
                    "input_boundary": self._tensor_slice_list(input_ids, row_idx, boundary_start, boundary_end),
                    "attention_boundary": self._tensor_slice_list(attention_mask, row_idx, boundary_start, boundary_end),
                    "position_boundary": self._tensor_slice_list(position_ids, row_idx, boundary_start, boundary_end),
                    "prompt_tail": (
                        prompts[row_idx, max(0, prompts.shape[-1] - 16) :].detach().cpu().tolist()
                        if prompts is not None and prompts.dim() == 2
                        else []
                    ),
                    "response_valid_head": torch.masked_select(
                        batch.batch["responses"][row_idx].detach(), row_mask
                    )[:16].cpu().tolist(),
                }
            if old_log_probs is not None:
                row_payload["old_log_probs_sample"] = self._sample_masked_values(old_log_probs[row_idx], row_mask)
            if old_log_probs_repeat is not None:
                row_payload["old_log_probs_repeat_sample"] = self._sample_masked_values(
                    old_log_probs_repeat[row_idx], row_mask
                )
            if rollout_log_probs is not None:
                row_payload["rollout_log_probs_sample"] = self._sample_masked_values(
                    rollout_log_probs[row_idx], row_mask
                )
            if ref_log_prob is not None:
                row_payload["ref_log_prob_sample"] = self._sample_masked_values(ref_log_prob[row_idx], row_mask)
            if "dump_actor_old_log_probs" in originals:
                row_payload["dump_actor_old_log_probs_sample"] = self._sample_masked_values(
                    originals["dump_actor_old_log_probs"][row_idx], row_mask
                )
            if "dump_ref_log_prob" in originals:
                row_payload["dump_ref_log_prob_sample"] = self._sample_masked_values(
                    originals["dump_ref_log_prob"][row_idx], row_mask
                )
            rows.append(row_payload)

        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(self._json_safe(row), ensure_ascii=False, sort_keys=True) + "\n")
        print(f"[FixedSequenceScoreProbe] wrote {len(rows)} records to {path}")

    def run_fixed_sequence_score_probe(
        self,
        jsonl_path: str,
        output_path: str = "",
        row_limit: int = 8,
    ) -> dict[str, float | int | str]:
        print(
            "[FixedSequenceScoreProbe] loading fixed rollout-corr samples "
            f"from {jsonl_path}, row_limit={row_limit}"
        )
        batch, records, originals = self._build_fixed_sequence_batch(jsonl_path, row_limit)
        print(
            "[FixedSequenceScoreProbe] rebuilt DataProto "
            f"batch_size={len(batch)} keys={list(batch.batch.keys())}"
        )

        # Keep run1/run2 independent from the raw fixed batch. In earlier probes
        # run2 was computed after unioning run1 outputs back into batch, which
        # made it harder to separate scorer nondeterminism from probe-side
        # DataProto pollution.
        raw_score_batch = self._clone_dataproto_for_probe(batch)
        repeat_score_batch = self._clone_dataproto_for_probe(batch)

        direct_old_score = os.environ.get("VERL_OMNI_FIXED_SEQUENCE_SCORE_DIRECT_OLD", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if direct_old_score:
            print(
                "[FixedSequenceScoreProbe] VERL_OMNI_FIXED_SEQUENCE_SCORE_DIRECT_OLD=1; "
                "skip save_model_to_cpu/restore_model_from_cpu for old run1"
            )
            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob_allow_missing_mfu(raw_score_batch)
        else:
            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(raw_score_batch)
        if "entropys" in old_log_prob.batch:
            old_log_prob.batch.pop("entropys")
        batch = batch.union(old_log_prob)
        try:
            old_log_prob_mfu_value = float(old_log_prob_mfu)
        except (TypeError, ValueError):
            old_log_prob_mfu_value = 0.0

        replay_recorded_routes = os.environ.get(
            "VERL_OMNI_FIXED_SEQUENCE_SCORE_REPLAY_ROUTED_EXPERTS", "0"
        ).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        replay_recorded_routes_ready = replay_recorded_routes and "routed_experts" in old_log_prob.batch
        if replay_recorded_routes and not replay_recorded_routes_ready:
            raise RuntimeError(
                "VERL_OMNI_FIXED_SEQUENCE_SCORE_REPLAY_ROUTED_EXPERTS=1 requested R2 replay, "
                "but old run1 did not return routed_experts. Refusing to silently run an unreplayed repeat."
            )
        try:
            route_capture_metrics = router_replay_tensor_metrics(old_log_prob.batch.get("routed_experts"))
        except Exception as exc:
            logger.warning("Failed to summarize fixed-sequence routed_experts: %r", exc)
            route_capture_metrics = {"present": 0.0}
        if replay_recorded_routes_ready:
            repeat_score_batch.batch["routed_experts"] = old_log_prob.batch["routed_experts"]
            print("[FixedSequenceScoreProbe] replay run1 routed_experts for old run2")

        repeat_old_log_prob, repeat_old_log_prob_mfu = self._compute_old_log_prob_allow_missing_mfu(repeat_score_batch)
        batch.batch["old_log_probs_repeat"] = repeat_old_log_prob.batch["old_log_probs"]
        try:
            repeat_old_log_prob_mfu_value = float(repeat_old_log_prob_mfu)
        except (TypeError, ValueError):
            repeat_old_log_prob_mfu_value = 0.0

        if self.use_reference_policy:
            ref_log_prob = self._compute_ref_log_prob(batch)
            batch = batch.union(ref_log_prob)

        response_mask = batch.batch["response_mask"].bool()
        metrics: dict[str, float] = {
            "fixed_sequence_score/records": float(len(records)),
            "fixed_sequence_score/direct_old_score": float(direct_old_score),
            "fixed_sequence_score/replay_routed_experts": float(replay_recorded_routes_ready),
            "fixed_sequence_score/old_log_prob_mfu": old_log_prob_mfu_value,
            "fixed_sequence_score/repeat_old_log_prob_mfu": repeat_old_log_prob_mfu_value,
        }
        for name, value in route_capture_metrics.items():
            metrics[f"fixed_sequence_score/router_capture/{name}"] = value

        rollout_log_probs = batch.batch.get("rollout_log_probs")
        old_log_probs = batch.batch["old_log_probs"]
        old_log_probs_repeat = batch.batch["old_log_probs_repeat"]
        ref_log_prob = batch.batch.get("ref_log_prob")
        for name, values in self._masked_pair_summary(old_log_probs, old_log_probs_repeat, response_mask).items():
            metrics[f"fixed_sequence_score/recomputed_old_run1_vs_run2/{name}"] = values
        if rollout_log_probs is not None:
            for name, values in self._masked_pair_summary(old_log_probs, rollout_log_probs, response_mask).items():
                metrics[f"fixed_sequence_score/recomputed_old_vs_rollout/{name}"] = values
            for name, values in self._masked_shift_scan(old_log_probs, rollout_log_probs, response_mask).items():
                metrics[f"fixed_sequence_score/recomputed_old_vs_rollout_shift/{name}"] = values
        if ref_log_prob is not None:
            for name, values in self._masked_pair_summary(old_log_probs, ref_log_prob, response_mask).items():
                metrics[f"fixed_sequence_score/recomputed_old_vs_ref/{name}"] = values
            if rollout_log_probs is not None:
                for name, values in self._masked_pair_summary(ref_log_prob, rollout_log_probs, response_mask).items():
                    metrics[f"fixed_sequence_score/recomputed_ref_vs_rollout/{name}"] = values
                for name, values in self._masked_shift_scan(ref_log_prob, rollout_log_probs, response_mask).items():
                    metrics[f"fixed_sequence_score/recomputed_ref_vs_rollout_shift/{name}"] = values
        if "dump_actor_old_log_probs" in originals:
            for name, values in self._masked_pair_summary(
                old_log_probs, originals["dump_actor_old_log_probs"], response_mask
            ).items():
                metrics[f"fixed_sequence_score/recomputed_old_vs_dump_old/{name}"] = values
        if ref_log_prob is not None and "dump_ref_log_prob" in originals:
            for name, values in self._masked_pair_summary(
                ref_log_prob, originals["dump_ref_log_prob"], response_mask
            ).items():
                metrics[f"fixed_sequence_score/recomputed_ref_vs_dump_ref/{name}"] = values

        if rollout_log_probs is not None:
            try:
                from verl.utils.debug.metrics import calculate_debug_metrics

                metrics.update(calculate_debug_metrics(batch))
            except Exception as exc:
                print(f"[FixedSequenceScoreProbe] calculate_debug_metrics failed: {exc}")

        printable = {
            key: (round(value, 6) if isinstance(value, float) and math.isfinite(value) else value)
            for key, value in metrics.items()
        }
        print(f"[FixedSequenceScoreProbe] metrics={printable}")
        self._write_fixed_sequence_score_output(output_path, records, batch, originals, metrics)
        return {
            "records": len(records),
            "output_path": output_path,
            "old_vs_rollout_abs_diff_mean": metrics.get(
                "fixed_sequence_score/recomputed_old_vs_rollout/abs_diff_mean", float("nan")
            ),
            "old_vs_ref_abs_diff_mean": metrics.get(
                "fixed_sequence_score/recomputed_old_vs_ref/abs_diff_mean", float("nan")
            ),
        }

    def get_actor_wg(self):
        """Get actor worker group"""
        return self.actor_wg

    async def _get_samples_from_queue(self) -> tuple[None, None] | tuple[int, Any]:
        """
        Get samples from message queue and compose gen_batch_output
        Uses a loop to continuously collect samples until enough are gathered

        Returns:
            tuple: (epoch, batch_dict, gen_batch_output)
        """
        print(
            f"[FullyAsyncTrainer] Requesting {self.required_samples} samples from queue",
            flush=True,
        )

        # Collect samples using a simple loop calling get_sample
        consumer_start = time.time()
        queue_samples = []
        queue_len = 0
        while len(queue_samples) < self.required_samples:
            # Get a single sample and wait until there is a sample or None is received
            sample, queue_len = await self.message_queue_client.get_sample()

            if sample is None:
                print(
                    f"[FullyAsyncTrainer] Detected termination signal (None), stopping sample collection. "
                    f"Collected {len(queue_samples)}/{self.required_samples} samples"
                )
                break

            queue_samples.append(sample)

            if len(queue_samples) % 64 == 0:
                print(
                    f"[FullyAsyncTrainer] Collected {len(queue_samples)}/{self.required_samples} samples. "
                    f"mq_len: {queue_len}"
                )

        consumer_end = time.time()

        if not queue_samples or len(queue_samples) < self.required_samples:
            print("[FullyAsyncTrainer] not enough samples collected after loop")
            return None, None
        total_wait_time = consumer_end - consumer_start

        print(
            f"[FullyAsyncTrainer] Loop collection completed: {len(queue_samples)}/{self.required_samples} samples, "
            f"total wait time: {total_wait_time:.2f} seconds. "
            f"mq_len: {queue_len}"
        )

        queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]
        # Assemble batch - now working directly with RolloutSample objects
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, self._balance_batch)
        else:
            batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)

        batch.meta_info["fully_async/total_wait_time"] = total_wait_time
        return 0, batch

    def _create_actor_rollout_classes(self):
        # create actor — always use Role.Actor (not ActorRollout) even when
        # use_trainer_do_validate is enabled. Rollout capability on trainer GPUs
        # is handled by ElasticAgentLoopManager's hybrid replicas.
        for role in [self.train_role]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _create_reward_model_class(self):
        # In fully async mode, RM is managed by RewardLoopManager (standalone). Skip worker group creation for RM.
        pass

    def _init_models(self):
        if self.use_critic:
            self.critic_wg = self.all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = self.all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.actor_wg = self.all_wg[str(self.train_role)]
        self.actor_wg.init_model()
        self.actor_rollout_wg = self.actor_wg  # to be compatible with the functions that not be modified

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.
        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()

    async def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        print("[FullyAsyncTrainer] Starting FullyAsyncTrainer...")
        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")
        if self.rollouter is None:
            raise ValueError("rollouter not set. Call set_rollouter() first.")

        self.max_steps_duration = 0

        self.global_steps += 1

        self.prev_step_profile = False
        self.curr_step_profile = False
        self.next_step_profile = False

        # Use queue mode, no need for traditional dataloader iterator
        # Initialize to get the first batch of data
        while True:
            try:
                await self.fit_step()
            except TrainingStopException:
                print("[FullyAsyncTrainer] Training stopped by queue termination signal")
                break
            if (
                os.environ.get("VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS", "0").lower() in {"1", "true", "yes", "on"}
                and self.current_param_version >= self.config.trainer.total_training_steps
            ):
                print(
                    "[FullyAsyncTrainer] VERL_OMNI_STOP_AFTER_TOTAL_TRAINING_STEPS=1; "
                    f"stop at current_param_version={self.current_param_version}"
                )
                break

        self.progress_bar.close()
        if self.current_param_version % self.config.trainer.test_freq != 0 or self.local_trigger_step > 1:
            await self._fit_update_weights()
            await self._fit_validate()
        if self.metrics_aggregator.step_count > 0:
            self.logger.log(
                data=self.metrics_aggregator.get_aggregated_metrics(),
                step=self.current_param_version,
            )
            self.metrics_aggregator.reset()
        self._fit_save_checkpoint(force=True)

    async def fit_step(self, batch_dict: dict = None):
        """
        Single-step training template method. Handles all logic for one training step.

        Flow:
        1. Pre-step processing -> 2. Get batch -> 3. Generate sequences ->
        4. Compute reward -> 5. Compute log_prob -> 6. Compute reward ->
        7. Compute advantage -> 8. Update critic -> 9. Update actor -> 10. Post-step processing

        Args:
            batch_dict: Raw data dictionary
        """
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        # reward message
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        steps = self.config.global_profiler.steps
        should_profile = steps is not None and (self.current_param_version + 1) in steps
        self._fit_start_profile(should_profiler=should_profile)

        with marked_timer("step", self.timing_raw):
            batch = await self._fit_generate(None)
            batch = self._fit_compute_reward(batch)
            batch = self._fit_compute_log_prob(batch)
            batch = self._fit_compute_ref_log_prob(batch)
            batch = self._fit_compute_critic(batch)
            batch = self._fit_compute_advantage(batch)
            batch = self._fit_update_critic(batch)
            batch = self._fit_update_actor(batch)
            self._fit_update_local_step()
            await self._fit_update_weights()
            self._fit_dump_data(batch)

        await self._fit_validate()
        self._fit_save_checkpoint()
        self._fit_stop_profile(should_profiler=should_profile)
        self._fit_collect_metrics(batch)
        self._fit_postprocess_step()

    async def _fit_generate(self, batch: DataProto = None) -> DataProto | None:
        metrics = self.metrics
        timing_raw = self.timing_raw
        with marked_timer("gen", timing_raw, color="red"):
            epoch, batch = await self._get_samples_from_queue()
            if batch is None:
                raise TrainingStopException("Training terminated: queue returned None")
            self._collect_metrics_from_samples(batch, metrics)
        batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        self._maybe_dump_multistage_logprob_batch("trainer_after_queue", batch)
        return batch

    def _fit_compute_log_prob(self, batch: DataProto) -> DataProto:
        batch = super()._fit_compute_log_prob(batch)
        self._maybe_dump_multistage_logprob_batch("trainer_after_old_log_prob", batch)
        return batch

    def _fit_compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        batch = super()._fit_compute_ref_log_prob(batch)
        self._maybe_dump_multistage_logprob_batch("trainer_after_ref_log_prob", batch)
        return batch

    def _compute_old_log_prob(self, batch: DataProto):
        """
        If algorithm.rollout_correction.bypass_mode is False,
        use model engine and first version model params to re-calculate old_log_prob.

        If local_trigger_step == 1, load the training engine's parameters to the CPU
          and save a copy for subsequent MIS use.

        If local_trigger_step == 2, 3, ..., restore the parameters of version 1 to calculate the old_log_prob,
        then restore the parameters of the current version.
        """
        if self.local_trigger_step == 1:
            self.actor_rollout_wg.save_model_to_cpu(1)
            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob_allow_missing_mfu(batch)
        else:
            self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
            self.actor_rollout_wg.restore_model_from_cpu(1)
            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob_allow_missing_mfu(batch)
            self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
            self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
        return old_log_prob, old_log_prob_mfu

    def _compute_old_log_prob_allow_missing_mfu(self, batch: DataProto):
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        calculate_sum_pi_squared = self.config.actor_rollout_ref.actor.get("calculate_sum_pi_squared", False)
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=True,
            calculate_sum_pi_squared=calculate_sum_pi_squared,
            compute_loss=False,
        )
        output = self.actor_rollout_wg.compute_log_prob(batch_td)

        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        routed_experts = tu.get(output, "routed_experts")
        sum_pi_squared = tu.get(output, "sum_pi_squared") if calculate_sum_pi_squared else None
        output_metrics = tu.get(output, "metrics", {}) or {}
        old_log_prob_mfu = output_metrics.get("mfu", 0.0)

        entropy = no_padding_2_padding(entropy, batch_td)
        log_probs = no_padding_2_padding(log_probs, batch_td)
        if sum_pi_squared is not None:
            sum_pi_squared = no_padding_2_padding(sum_pi_squared, batch_td)

        result = {"old_log_probs": log_probs.float(), "entropys": entropy.float()}
        if routed_experts is not None:
            result["routed_experts"] = routed_experts
        if sum_pi_squared is not None:
            result["sum_pi_squared"] = sum_pi_squared.float()
        old_log_prob = tu.get_tensordict(result)
        old_log_prob = DataProto.from_tensordict(old_log_prob)
        return old_log_prob, old_log_prob_mfu

    def _fit_update_local_step(self):
        time_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(
            f"[FullyAsyncTrainer] global_steps: {self.global_steps} "
            f"local_trigger_step: {self.local_trigger_step} "
            f"trigger_parameter_sync_step: {self.trigger_parameter_sync_step} "
            f"{time_str}"
        )
        if self.local_trigger_step < self.trigger_parameter_sync_step:
            self.local_trigger_step += 1
        else:
            self.current_param_version += 1
            self.local_trigger_step = 1

    async def _fit_update_weights(self):
        if self.local_trigger_step != 1:
            return
        if os.environ.get("VERL_OMNI_SKIP_WEIGHT_UPDATE", "0").lower() in {"1", "true", "yes", "on"}:
            print(
                "[FullyAsyncTrainer] VERL_OMNI_SKIP_WEIGHT_UPDATE=1; "
                f"skip param sync at current_param_version={self.current_param_version}"
            )
            return

        steps = self.config.global_profiler.steps
        last_profiler_step = self.current_param_version
        if steps is not None and last_profiler_step in steps:
            await asyncio.wrap_future(self.rollouter._stop_profiling.remote().future())

        with marked_timer("timing_s/param_sync", self.timing_raw):
            await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
        print(
            f"[FullyAsyncTrainer] _fit_update_weights, "
            f"timing_s/param_sync: {self.timing_raw['timing_s/param_sync']:.4f} seconds "
            f"self.current_param_version: {self.current_param_version}"
        )

        profiler_step = last_profiler_step + 1

        if steps is not None and profiler_step in steps:
            await asyncio.wrap_future(self.rollouter._start_profiling.remote().future())

        # Reset staleness in rollouter
        timing_raw = await asyncio.wrap_future(self.rollouter.reset_staleness.remote().future())
        self.logger.log(
            data=timing_raw,
            step=self.current_param_version,
        )

        # Log aggregated training metrics
        self.logger.log(
            data=self.metrics_aggregator.get_aggregated_metrics(),
            step=self.current_param_version,
        )
        self.metrics_aggregator.reset()

    async def _fit_validate(self, val_before_train=False):
        if self.local_trigger_step != 1:
            return

        # Check if validation is needed
        need_validate = (
            self.config.trainer.test_freq > 0
            and self.current_param_version % self.config.trainer.test_freq == 0
            and self.current_param_version > 0
        )
        # Skip validation if not needed and not validation before training
        if not need_validate and not val_before_train:
            return
        # Execute validation
        if self.config.async_training.use_trainer_do_validate:
            await self._trainer_side_validate()
        else:
            val_metrics = await self.rollouter.do_validate.remote()
            self.logger.log(data=val_metrics, step=self.current_param_version)

    async def _trainer_side_validate(self):
        """Run trainer-side validation using hybrid rollout replicas."""
        print("[FullyAsyncTrainer] _trainer_side_validate === START ===")
        validate_start = time.time()
        # ================================================================
        # Phase 1: Switch ALL trainer GPUs to ROLLOUT mode
        # ================================================================
        phase_1_start = time.time()
        print("[FullyAsyncTrainer] Phase 1: Switching all GPUs to ROLLOUT mode")
        await self.hybrid_checkpoint_manager.update_weights(global_steps=self.current_param_version)
        await self.checkpoint_manager.abort_replicas()
        await self.hybrid_checkpoint_manager.abort_replicas()
        hybrid_replicas_dict = await self.rollouter.get_all_hybrid_replicas.remote()
        hybrid_resource_ids = list(hybrid_replicas_dict.keys())
        await self.rollouter.add_replicas.remote(hybrid_resource_ids)
        await self.checkpoint_manager.resume_generation_replicas()
        await self.hybrid_checkpoint_manager.resume_generation_replicas()
        print(f"[FullyAsyncTrainer] Phase 1 done ({time.time() - phase_1_start:.2f}s)")

        # ================================================================
        # Phase 2: Run validation via RPC to rollouter
        # ================================================================
        print("[FullyAsyncTrainer] Phase 2: Running validation")
        val_metrics = await self.rollouter.do_validate.remote()
        self.logger.log(data=val_metrics, step=self.current_param_version)

        # ================================================================
        # Phase 3: Switch hybrid GPUs back to TRAIN mode
        # ================================================================
        print("[FullyAsyncTrainer] Phase 3: Switching hybrid GPUs back to TRAIN mode")
        await self.checkpoint_manager.abort_replicas()
        await self.hybrid_checkpoint_manager.abort_replicas()
        # Batch remove all hybrid replicas from the load balancer in a single RPC.
        await self.rollouter.remove_replicas.remote(hybrid_resource_ids)
        await self.hybrid_checkpoint_manager.sleep_replicas()
        await self.checkpoint_manager.resume_generation_replicas()
        await self.hybrid_checkpoint_manager.resume_generation_replicas()

        total_time = time.time() - validate_start
        print(f"[FullyAsyncTrainer] _trainer_side_validate === END === (total: {total_time:.2f}s)")

    def _fit_save_checkpoint(self, force=False):
        if self.current_param_version == self.last_ckpt_version:
            return

        timing_raw = self.timing_raw
        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
        esi_close_to_expiration = should_save_ckpt_esi(
            max_steps_duration=self.max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        # Check if the conditions for saving a checkpoint are met.
        # The conditions include a mandatory condition (1) and
        # one of the following optional conditions (2/3/4):
        # 1. The save frequency is set to a positive value.
        # 2. It's the last training step.
        # 3. The current step number is a multiple of the save frequency.
        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
        if self.config.trainer.save_freq > 0 and (
            force or self.current_param_version % self.config.trainer.save_freq == 0 or esi_close_to_expiration
        ):
            if esi_close_to_expiration:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                # sleep replicas to avoid OOM during checkpoint saving
                self._save_checkpoint()
                self.last_ckpt_version = self.current_param_version

    def _fit_postprocess_step(self):
        self.global_steps += 1

        self.metrics_aggregator.add_step_metrics(
            metrics=self.metrics, sample_count=self.required_samples, timestamp=time.time()
        )

        if self.local_trigger_step == 1:
            self.progress_bar.update(1)

    def _save_checkpoint(self):
        # Warning: Currently, to align the training process and metrics of colocate,
        # we use current_param_version instead of global step.
        # This can be logically aligned with the original self.global_steps of colocate
        # and is used for metrics and ckpt. which means that the parameter synchronization
        # from trainer to rollouter will increase by 1 each time.

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.current_param_version}"
        )

        print(f"[FullyAsyncTrainer] local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", "actor"
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "[FullyAsyncTrainer] Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.current_param_version, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.current_param_version}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.current_param_version,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )
        ray.get(self.rollouter.save_checkpoint.remote(local_global_step_folder))
        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.current_param_version))

    async def load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"[FullyAsyncTrainer] Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.current_param_version = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = self.current_param_version * self.trigger_parameter_sync_step + 1
        self.last_ckpt_version = self.current_param_version
        print(
            f"[FullyAsyncTrainer] Setting global step to {self.global_steps}, "
            f"current_param_version to {self.current_param_version}"
        )
        print(f"[FullyAsyncTrainer] Resuming from  {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        return self.current_param_version

    def _collect_metrics_from_samples(self, batch, metrics):
        """
        Collect metrics from samples
        """
        if hasattr(batch, "meta_info") and batch.meta_info:
            trajectory_param_versions = batch.meta_info["trajectory_param_versions"]
            stale_traj_count = sum(1 for v in trajectory_param_versions if self.current_param_version - v >= 1)
            self.stale_trajectory_processed += stale_traj_count
            metrics.update(
                {
                    "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                    "fully_async/count/current_param_version": self.current_param_version,
                }
            )
            for key, value in batch.meta_info.items():
                if key.startswith("fully_async") or key.startswith("timing_s"):
                    metrics[key] = value

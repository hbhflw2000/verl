# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
import argparse
import logging
import os
from dataclasses import asdict
from typing import Any, Optional

import ray
import torchvision.transforms as T
import vllm_omni.entrypoints.cli.serve
from vllm import SamplingParams
from vllm.entrypoints.openai.api_server import build_app
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import omni_init_app_state
from vllm_omni.inputs.data import OmniCustomPrompt, OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.outputs import OmniRequestOutput

from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import import_external_libs
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import DiffusionModelConfig, DiffusionRolloutConfig, HFModelConfig
from verl.workers.rollout.replica import DiffusionOutput, TokenOutput
from verl.workers.rollout.utils import run_uvicorn
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class vLLMOmniHttpServer(vLLMHttpServer):
    """vLLM-Omni http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    # -----------------------------------------------------------------------
    # Initialisation hooks
    # -----------------------------------------------------------------------

    def _init_model_config(self, model_config):
        engine_kwargs = getattr(self.config, "engine_kwargs", None) or {}
        omni_kwargs = engine_kwargs.get("vllm_omni", {}) or {}
        self._ar_mode = omni_kwargs.get("output_mode", "diffusion") == "ar"

        if self._ar_mode:
            return omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        return omega_conf_to_dataclass(model_config, dataclass_type=DiffusionModelConfig)

    def _validate_configs(self) -> None:
        """Validate AR configs; diffusion models don't have max_position_embeddings."""
        if self._ar_mode and self.config.max_model_len is None:
            self.config.max_model_len = self.config.prompt_length + self.config.response_length

    def _post_init(self, cuda_visible_devices: str) -> None:
        """Omni-specific post-init: create PIL→tensor converter, then log."""
        if self._ar_mode:
            vLLMHttpServer._post_init(self, cuda_visible_devices)
        else:
            self._to_tensor = T.PILToTensor()
            super()._post_init(cuda_visible_devices)

    # -----------------------------------------------------------------------
    # launch_server hooks
    # -----------------------------------------------------------------------

    def _get_override_generation_config(self) -> dict:
        """Diffusion models have no LLM sampling params; return empty dict."""
        if self._ar_mode:
            return vLLMHttpServer._get_override_generation_config(self)
        return {}

    def _get_engine_kwargs_key(self) -> str:
        return "vllm_omni"

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        engine_kwargs.pop("output_mode", None)
        if self._ar_mode:
            engine_kwargs.pop("custom_pipeline", None)
            for underscore_key in ("stage_configs_path", "deploy_config", "stage_overrides", "async_chunk"):
                if underscore_key in engine_kwargs:
                    engine_kwargs[underscore_key.replace("_", "-")] = engine_kwargs.pop(underscore_key)
            return

        # custom_pipeline is passed directly to run_server; not supported via CLI yet
        engine_kwargs.pop("custom_pipeline", None)

    def _get_worker_extension_cls(self) -> str:
        return "verl.workers.rollout.vllm_rollout.utils.vLLMOmniColocateWorkerExtension"

    def _get_cli_modules(self) -> list:
        return [vllm_omni.entrypoints.cli.serve]

    def _get_cli_description(self) -> str:
        return "vLLM-Omni CLI"

    # -----------------------------------------------------------------------
    # Server lifecycle
    # -----------------------------------------------------------------------

    async def run_server(self, args: argparse.Namespace):
        engine_args = OmniEngineArgs.from_cli_args(args)
        engine_args = asdict(engine_args)
        for key in ("stage_init_timeout", "init_timeout"):
            value = getattr(args, key, None)
            if value is not None:
                engine_args[key] = value

        # TODO (mike): read custom_pipeline from CLI
        custom_pipeline = self.config.engine_kwargs.get("vllm_omni", {}).get("custom_pipeline", None)
        if self._ar_mode:
            if isinstance(engine_args.get("compilation_config"), dict):
                engine_args["compilation_config"] = {
                    k: v for k, v in engine_args["compilation_config"].items() if v is not None
                }
        elif custom_pipeline is not None:
            import_external_libs(self.config.external_lib)
            engine_args["enable_dummy_pipeline"] = True
            engine_args["custom_pipeline_args"] = {"pipeline_class": custom_pipeline}

        # TODO (mike): support parsing engine config from CLI
        engine_client = AsyncOmni(**engine_args)
        app = build_app(args)
        await omni_init_app_state(engine_client, app.state, args)

        self.engine = engine_client
        self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        """Run headless server in a separate thread."""
        # TODO (mike): support multi node
        raise NotImplementedError("vLLM-Omni headless mode is not implemented yet.")

    # -----------------------------------------------------------------------
    # wake_up hook: Omni does not restore KV cache on wake-up
    # -----------------------------------------------------------------------

    def _get_wake_up_tags(self) -> list[str]:
        return ["weights"]

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        negative_prompt_ids: Optional[list[int]] = None,
        priority: int = 0,
    ) -> DiffusionOutput | TokenOutput:
        if self._ar_mode:
            return await self._generate_ar(prompt_ids, sampling_params, request_id, image_data, video_data, priority)
        return await self._generate_diffusion(
            prompt_ids, sampling_params, request_id, image_data, video_data, negative_prompt_ids, priority
        )

    async def _generate_diffusion(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        negative_prompt_ids: Optional[list[int]] = None,
        priority: int = 0,
    ) -> DiffusionOutput:
        """Generate sequence with token-in-image-out."""
        prompt_ids = normalize_token_ids(prompt_ids)

        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data

        # Add lora request
        lora_request = None
        if self.lora_as_adapter:
            # Make sure we also check that the lora is already loaded in the engine
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        # Build OmniCustomPrompt with pre-tokenized IDs
        custom_prompt: OmniCustomPrompt = {"prompt_ids": prompt_ids}
        if negative_prompt_ids is not None:
            custom_prompt["negative_prompt_ids"] = negative_prompt_ids
        if multi_modal_data:
            custom_prompt["extra_args"] = {"multi_modal_data": multi_modal_data}

        # Build OmniDiffusionSamplingParams from the incoming dict
        sampling_kwargs: dict[str, Any] = {}
        extra_args: dict[str, Any] = {}
        for k, v in sampling_params.items():
            if hasattr(OmniDiffusionSamplingParams, k):
                sampling_kwargs[k] = v
            else:
                extra_args[k] = v
        sampling_kwargs["extra_args"] = extra_args
        if lora_request is not None:
            sampling_kwargs["lora_request"] = lora_request
        diffusion_sampling_params = OmniDiffusionSamplingParams(**sampling_kwargs)

        # Call AsyncOmni.generate() with the correct API
        generator = self.engine.generate(
            prompt=custom_prompt,
            request_id=request_id,
            sampling_params_list=[diffusion_sampling_params],
        )

        # Get final response
        final_res: Optional[OmniRequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        diffusion_output = self._to_tensor(final_res.images[0]).float() / 255.0

        # Extract extra data from custom_output (populated by DiffusionEngine)
        mm_output = final_res.custom_output or {}

        if sampling_params.get("logprobs", False):
            all_log_probs = mm_output.get("all_log_probs")
            log_probs = all_log_probs[0] if all_log_probs is not None else None
        else:
            log_probs = None

        all_latents = mm_output.get("all_latents")
        all_timesteps = mm_output.get("all_timesteps")
        prompt_embeds = mm_output.get("prompt_embeds")
        prompt_embeds_mask = mm_output.get("prompt_embeds_mask")
        negative_prompt_embeds = mm_output.get("negative_prompt_embeds")
        negative_prompt_embeds_mask = mm_output.get("negative_prompt_embeds_mask")

        extra_fields = {
            "all_latents": all_latents[0] if all_latents is not None else None,
            "all_timesteps": all_timesteps[0] if all_timesteps is not None else None,
            "prompt_embeds": prompt_embeds[0] if prompt_embeds is not None else None,
            "prompt_embeds_mask": prompt_embeds_mask[0] if prompt_embeds_mask is not None else None,
            "negative_prompt_embeds": negative_prompt_embeds[0] if negative_prompt_embeds is not None else None,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask[0]
            if negative_prompt_embeds_mask is not None
            else None,
            "global_steps": self.global_steps,
        }

        # Determine stop reason from finish_reason
        if final_res.request_output is not None and hasattr(final_res.request_output, "finish_reason"):
            finish_reason = final_res.request_output.finish_reason or "stop"
        else:
            finish_reason = "stop"

        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason  # for more stop reason in the future

        num_preempted = None
        if final_res.request_output is not None and hasattr(final_res.request_output, "num_preempted"):
            num_preempted = final_res.request_output.num_preempted

        return DiffusionOutput(
            diffusion_output=diffusion_output,
            log_probs=log_probs,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )

    async def _generate_ar(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        priority: int = 0,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out via vLLM-Omni."""
        prompt_ids = normalize_token_ids(prompt_ids)

        max_possible_tokens = self.config.max_model_len - len(prompt_ids)
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        if "max_tokens" in sampling_params and sampling_params["max_tokens"] is not None:
            max_tokens = sampling_params.pop("max_tokens")
        elif "max_new_tokens" in sampling_params and sampling_params["max_new_tokens"] is not None:
            max_tokens = sampling_params.pop("max_new_tokens")
        else:
            max_tokens = min(
                self.config.response_length,
                self.config.prompt_length + self.config.response_length - len(prompt_ids),
            )
        max_tokens = max(0, min(max_tokens, max_possible_tokens))

        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        if "min_tokens" in sampling_params:
            sampling_params["min_tokens"] = max(0, min(int(sampling_params["min_tokens"]), max_tokens))
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))

        debug_limit = int(os.getenv("VERL_OMNI_GENERATION_DEBUG_LIMIT", "0") or 0)
        sampling_debug_count = getattr(self, "_sampling_debug_count", 0)
        if sampling_debug_count < debug_limit:
            self._sampling_debug_count = sampling_debug_count + 1
            print(
                "[vllm_omni_ar_sampling] "
                f"request_id={request_id} prompt_len={len(prompt_ids)} response_length={self.config.response_length} "
                f"max_model_len={self.config.max_model_len} max_possible_tokens={max_possible_tokens} "
                f"max_tokens={max_tokens} min_tokens={sampling_params.get('min_tokens')} "
                f"ignore_eos={sampling_params.get('ignore_eos')}",
                flush=True,
            )

        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)

        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data

        prompt = {"prompt_token_ids": prompt_ids}
        if multi_modal_data:
            prompt["multi_modal_data"] = multi_modal_data

        lora_request = None
        if self.lora_as_adapter:
            try:
                lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            except TypeError:
                lora_loaded = True
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        generator = self.engine.generate(
            prompt=prompt,
            sampling_params_list=[sampling_params],
            request_id=request_id,
            lora_request=lora_request,
            priority=priority,
        )

        final_res: Optional[OmniRequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        req_output = final_res.request_output
        assert req_output is not None, "AR mode expects request_output with token IDs"

        extra_fields = {"global_steps": self.global_steps}
        output = req_output.outputs[0]
        token_ids = list(getattr(output, "cumulative_token_ids", None) or output.token_ids)
        debug_count = getattr(self, "_generation_debug_count", 0)
        if debug_count < debug_limit:
            self._generation_debug_count = debug_count + 1
            print(
                "[vllm_omni_ar_generation] "
                f"request_id={request_id} prompt_len={len(prompt_ids)} max_tokens={sampling_params.max_tokens} "
                f"min_tokens={getattr(sampling_params, 'min_tokens', None)} "
                f"ignore_eos={getattr(sampling_params, 'ignore_eos', None)} "
                f"finish_reason={getattr(output, 'finish_reason', None)} token_count={len(token_ids)} "
                f"delta_token_count={len(output.token_ids)} token_ids_head={token_ids[:32]} "
                f"text_head={(getattr(output, 'text', '') or '')[:256]!r}",
                flush=True,
            )
            logger.warning(
                "vLLM-Omni AR generation debug request_id=%s prompt_len=%s max_tokens=%s min_tokens=%s "
                "ignore_eos=%s finish_reason=%s token_count=%s delta_token_count=%s token_ids_head=%s text_head=%r",
                request_id,
                len(prompt_ids),
                sampling_params.max_tokens,
                getattr(sampling_params, "min_tokens", None),
                getattr(sampling_params, "ignore_eos", None),
                getattr(output, "finish_reason", None),
                len(token_ids),
                len(output.token_ids),
                token_ids[:32],
                (getattr(output, "text", "") or "")[:256],
            )
        log_probs = None
        if sampling_params.logprobs is not None:
            if output.logprobs and len(output.logprobs) == len(token_ids):
                log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(output.logprobs)]
            else:
                logger.warning(
                    "Skipping vLLM-Omni AR logprobs because token/logprob lengths differ: tokens=%s logprobs=%s",
                    len(token_ids),
                    len(output.logprobs or []),
                )

        finish_reason = output.finish_reason
        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason

        num_preempted = None
        if hasattr(output, "num_preempted"):
            num_preempted = output.num_preempted

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )


class vLLMOmniReplica(vLLMReplica):
    def __init__(
        self,
        replica_rank: int,
        config: DiffusionRolloutConfig,
        model_config: DiffusionModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(vLLMOmniHttpServer)

    def _validate_launch_requirements(self) -> None:
        """No-op: the parent check validates vllm.__version__ which is
        irrelevant for vllm-omni (a separate package)."""
        pass

    def _get_server_name_prefix(self) -> str:
        # vLLMRollout lazily looks up the colocated server using the base
        # vLLM actor name. Keep the same naming contract for vLLM-Omni.
        return "vllm_"

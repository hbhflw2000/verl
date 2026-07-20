# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import json
import os
from typing import Optional

import torch
from torch.nested._internal.nested_tensor import NestedTensor

from verl.utils.megatron_utils import unwrap_model
from verl.workers.config import MtpConfig

from .util import (
    build_vlm_attn_mask_bshd,
    build_vlm_attn_mask_thd,
    postprocess_bshd,
    postprocess_bshd_engine,
    postprocess_packed_seqs,
    postprocess_thd_engine,
    preprocess_bshd,
    preprocess_bshd_engine,
    preprocess_packed_seqs,
    preprocess_thd_engine,
)


def model_forward_gen(vision_model: bool = False):
    def model_forward(
        model,
        input_ids,
        attention_mask,
        position_ids,
        multi_modal_inputs: dict,
        logits_processor=None,
        logits_processor_args: dict = None,
        value_model=False,
        data_format: str = "thd",
        mtp_config: MtpConfig = None,
    ):
        """Forward pass for models with sequence packing."""
        assert data_format in ["thd", "bshd"], "data_format must be 'thd' or 'bshd'"
        pre_process = (
            unwrap_model(model).pre_process if not vision_model else False
        )  # vision model does not need pre_process, because we pack the input_ids to thd in the forward function
        post_process = unwrap_model(model).post_process
        sp = unwrap_model(model).config.sequence_parallel
        fp8 = unwrap_model(model).config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]

        model_kwargs = {}
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
        if "pixel_values_videos" in multi_modal_inputs:
            model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
        if "video_grid_thw" in multi_modal_inputs:
            model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

        batch_size, seq_len = attention_mask.shape[:2]
        mtp_enable_train = mtp_config and mtp_config.enable_train

        if data_format == "thd":
            input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(
                input_ids,
                attention_mask,
                pre_process=pre_process or (post_process and mtp_enable_train),
                use_fp8_padding=use_fp8_padding,
            )
            input_ids_rmpad = input_ids_rmpad.contiguous()

            # when pp > 1 and processor is not None, we need to pass the labels and loss_mask to the model
            if mtp_enable_train and post_process:
                args = {
                    k: preprocess_packed_seqs(v, attention_mask, pre_process=True, use_fp8_padding=use_fp8_padding)[0]
                    for k, v in logits_processor_args.items()
                }
                model_kwargs["labels"] = args["label"].contiguous()
                model_kwargs["loss_mask"] = args["label_mask"].contiguous()

            input_args = dict(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids if not vision_model else None,  # vision models will calculate position_ids
                packed_seq_params=packed_seq_params,
                **model_kwargs,
            )

            if vision_model:
                # workaround for supporting sequence packing with context parallelism
                # cp split with sequence packing will make model lose vision token information, so we need to keep
                # the original input_ids and pack them after vision embedding is calculated,
                # cooporate with mbridge
                input_args["input_ids"] = input_ids
                input_args["attention_mask"] = attention_mask

            output_orig = model(**input_args)

            if post_process and logits_processor is not None:
                args = {
                    k: preprocess_packed_seqs(v, attention_mask, pre_process=True, use_fp8_padding=use_fp8_padding)[0]
                    for k, v in logits_processor_args.items()
                }
                output_dict = logits_processor(output_orig, **args)
                output = {
                    k: postprocess_packed_seqs(
                        v, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
                    )
                    for k, v in output_dict.items()
                }
            else:
                output = postprocess_packed_seqs(
                    output_orig, packed_seq_params, attention_mask, batch_size, seq_len, post_process=post_process
                )
        elif data_format == "bshd":
            """
            data_format: "thd" or "bshd", default is "thd",
            why we need this?
                for some new models, GPT-OSS, the thd format is not supported, so we need to use the bshd format.
            When using the bshd format, we have to add paddings to the input_ids to meet the longest sequence length, 
            so it is recommended to disable dynamic batch size and set batch size to 1
            """
            assert fp8 is None, "fp8 is not supported for bshd format yet"

            batch_size, sequence_length = attention_mask.shape[:2]
            position_ids_for_preprocess = (
                torch.arange(sequence_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
                if vision_model
                else position_ids
            )
            pre_process_for_bshd = True if vision_model else pre_process
            new_input_ids, new_attention_mask, new_position_ids = preprocess_bshd(
                input_ids,
                attention_mask,
                position_ids_for_preprocess,
                sequence_parallel=sp,
                pre_process=pre_process_for_bshd,
            )
            output_orig = model(
                input_ids=new_input_ids,
                position_ids=None if vision_model else new_position_ids,
                attention_mask=new_attention_mask,
                **model_kwargs,
            )
            if post_process and logits_processor is not None:
                args = {
                    k: preprocess_bshd(
                        v, attention_mask, position_ids_for_preprocess, sequence_parallel=sp, pre_process=True
                    )[0]
                    for k, v in logits_processor_args.items()
                }
                output_dict = logits_processor(output_orig, **args)
                output = {
                    k: postprocess_bshd(
                        v, new_attention_mask, attention_mask, sequence_length, post_process=post_process
                    )
                    for k, v in output_dict.items()
                }
            else:
                output = postprocess_bshd(
                    output_orig, new_attention_mask, attention_mask, sequence_length, post_process=post_process
                )
        if value_model and post_process:
            output = output[..., 0]
        return output

    return model_forward


def _convert_to_nested_tensor(v, input_ids_lengths):
    """Convert regular tensor to NestedTensor, slicing according to input_ids_lengths.

    Args:
        v: Tensor to convert, shape [batch, seq_len]
        input_ids_lengths: List of valid lengths for each sample

    Returns:
        Converted NestedTensor
    """
    if isinstance(v, NestedTensor):
        return v

    batch_size = v.shape[0]
    assert len(input_ids_lengths) == batch_size, (
        f"len(input_ids_lengths)={len(input_ids_lengths)} != batch_size={batch_size}"
    )

    v_split_list = []
    for i in range(batch_size):
        vi = v[i]
        target_len = input_ids_lengths[i]
        if vi.shape[0] > target_len:
            vi = vi[:target_len]
        elif vi.shape[0] < target_len:
            vi = torch.cat([vi, torch.ones(target_len - vi.shape[0], dtype=vi.dtype, device=vi.device)])
        v_split_list.append(vi)

    v = torch.nested.nested_tensor(v_split_list, layout=torch.jagged)
    return v


def _build_mtp_loss_mask_nested(response_mask, input_ids_lengths, response_attention_mask):
    """Build a nested loss_mask aligned to ``input_ids = [prompt; response]`` for MTP.

    ``response_mask`` is response-only data. This expands it to full packed
    input length as prompt zeros followed by valid response positions.
    """
    if isinstance(response_mask, NestedTensor):
        response_offsets = response_mask.offsets().tolist()
        response_lengths = [response_offsets[i + 1] - response_offsets[i] for i in range(len(response_offsets) - 1)]
        batch_size = len(response_lengths)
        response_values = response_mask.values()
    else:
        assert response_attention_mask is not None, "response_attention_mask is required to align padded MTP loss_mask"
        assert not isinstance(response_attention_mask, NestedTensor), (
            "response_attention_mask must be a padded (bs, max_response_len) tensor, got NestedTensor"
        )
        assert response_attention_mask.shape == response_mask.shape, (
            f"response_attention_mask shape {response_attention_mask.shape} "
            f"!= response_mask shape {response_mask.shape}"
        )
        batch_size = response_mask.shape[0]
        response_lengths = response_attention_mask.to(torch.int32).sum(dim=-1).tolist()

    assert len(input_ids_lengths) == batch_size, (
        f"len(input_ids_lengths)={len(input_ids_lengths)} != batch_size={batch_size}"
    )

    pieces = []
    for i in range(batch_size):
        actual_total = int(input_ids_lengths[i])
        actual_response = int(response_lengths[i])
        actual_prompt = actual_total - actual_response
        assert actual_prompt >= 0, (
            f"sample {i}: actual_response={actual_response} > actual_total={actual_total}; "
            "loss_mask cannot be longer than input_ids"
        )
        prompt_pad = torch.zeros(actual_prompt, dtype=response_mask.dtype, device=response_mask.device)
        # Keep the whole valid response span; response_mask may contain internal zeros for tool outputs.
        if isinstance(response_mask, NestedTensor):
            response_piece = response_values[response_offsets[i] : response_offsets[i + 1]]
        else:
            response_piece = response_mask[i, :actual_response]
        full = torch.cat([prompt_pad, response_piece], dim=0)
        assert full.shape[0] == actual_total, (
            f"sample {i}: built loss_mask length {full.shape[0]} != input_ids length {actual_total}"
        )
        pieces.append(full)

    return torch.nested.nested_tensor(pieces, layout=torch.jagged)


def _model_builds_own_mrope_position_ids(model) -> bool:
    unwrapped_model = unwrap_model(model)
    model_cls = unwrapped_model.__class__
    return model_cls.__name__ == "Qwen3OmniModel" and "qwen_omni" in model_cls.__module__


_QWEN3_OMNI_BSHD_POSITION_IDS_ENV = "VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS"
_MEGATRON_BSHD_DEBUG_JSONL_ENV = "VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL"
_MEGATRON_BSHD_DEBUG_COUNTS = {}


def _debug_dist_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.getenv("RANK", "0"))


def _debug_shape(value):
    if value is None:
        return None
    try:
        return _debug_jsonable(list(value.shape))
    except Exception:
        return None


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


def _debug_tensor_slice(value, rows: int, tokens: int, tail: bool = False):
    if value is None:
        return None
    try:
        if isinstance(value, NestedTensor) or getattr(value, "is_nested", False):
            values = value.values()
            offsets = value.offsets()
            values_slice = values[-tokens:] if tail else values[:tokens]
            return {
                "nested": True,
                "values_shape": _debug_shape(values),
                "offsets": offsets[: rows + 1].detach().cpu().tolist(),
                "values": _debug_jsonable(values_slice.detach().cpu().tolist()),
            }
        tensor = value.detach()
        if tensor.dim() == 0:
            return _debug_jsonable(tensor.cpu().item())
        if tensor.dim() == 1:
            tensor_slice = tensor[-tokens:] if tail else tensor[:tokens]
            return _debug_jsonable(tensor_slice.cpu().tolist())
        tensor_slice = tensor[:rows, -tokens:] if tail else tensor[:rows, :tokens]
        return _debug_jsonable(tensor_slice.cpu().tolist())
    except Exception as exc:
        return {"error": repr(exc), "shape": _debug_shape(value)}


def _write_megatron_bshd_debug(event: str, **payload):
    base_path = os.getenv(_MEGATRON_BSHD_DEBUG_JSONL_ENV, "").strip()
    if not base_path:
        return
    limit = int(os.getenv("VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT", "2"))
    rank = _debug_dist_rank()
    key = (rank, event)
    count = _MEGATRON_BSHD_DEBUG_COUNTS.get(key, 0)
    if count >= limit:
        return
    _MEGATRON_BSHD_DEBUG_COUNTS[key] = count + 1

    rows = int(os.getenv("VERL_OMNI_MEGATRON_BSHD_DEBUG_ROWS", "2"))
    tokens = int(os.getenv("VERL_OMNI_MEGATRON_BSHD_DEBUG_TOKENS", "16"))
    record = {
        "event": event,
        "rank": rank,
        "local_rank": os.getenv("LOCAL_RANK"),
        "count": count,
    }
    for name, value in payload.items():
        record[name] = {
            "shape": _debug_shape(value),
            "head": _debug_tensor_slice(value, rows=rows, tokens=tokens),
            "tail": _debug_tensor_slice(value, rows=rows, tokens=tokens, tail=True),
        }

    path = f"{base_path}.rank{rank}.jsonl"
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_debug_jsonable(record), ensure_ascii=True) + "\n")
    except Exception as exc:
        print(f"[MegatronBSHDDebug] failed to write {event} debug record: {exc}", flush=True)


def _select_bshd_position_ids_for_engine(model, vision_model: bool, position_ids_bshd):
    if vision_model:
        return None
    if _model_builds_own_mrope_position_ids(model):
        mode = os.getenv(_QWEN3_OMNI_BSHD_POSITION_IDS_ENV, "model").strip().lower()
        if mode in {"model", "auto", "none", "0", "false", "no"}:
            return None
        if mode in {"explicit", "precomputed", "pass", "1", "true", "yes"}:
            return position_ids_bshd
        raise ValueError(
            f"{_QWEN3_OMNI_BSHD_POSITION_IDS_ENV} must be one of "
            "'model'/'auto'/'none' or 'explicit'/'precomputed'/'pass', got {mode!r}"
        )
    return position_ids_bshd


def gptmodel_forward_model_engine(
    model,
    input_ids,
    multi_modal_inputs: dict,
    logits_processor=None,
    logits_processor_args: dict = None,
    value_model=False,
    vision_model=False,
    pad_token_id=None,
    data_format: str = "thd",
    mtp_enable_train: bool = False,
    local_cp_size: Optional[int] = None,
):
    """Default forward pass for GPT models with optional sequence packing."""

    assert data_format in ["thd", "bshd"], "data_format must be 'thd' or 'bshd'"
    pre_process = unwrap_model(model).pre_process
    post_process = unwrap_model(model).post_process

    fp8 = unwrap_model(model).config.fp8
    use_fp8_padding = fp8 in ["e4m3", "hybrid"]

    model_kwargs = {}
    if "pixel_values" in multi_modal_inputs:
        model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
    if "image_grid_thw" in multi_modal_inputs:
        model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
    if "pixel_values_videos" in multi_modal_inputs:
        model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
    if "video_grid_thw" in multi_modal_inputs:
        model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

    batch_size = input_ids.shape[0]
    if data_format == "thd":
        input_ids_rmpad, packed_seq_params, position_ids_rmpad = preprocess_thd_engine(
            input_ids,
            pre_process=pre_process or (post_process and mtp_enable_train),
            use_fp8_padding=use_fp8_padding,
            local_cp_size=local_cp_size,
        )
        input_ids_rmpad = input_ids_rmpad.contiguous()

        args = {}
        if mtp_enable_train and post_process:
            # Use input_ids sequence length to ensure label and loss_mask alignment
            input_ids_offsets = input_ids.offsets()
            input_ids_lengths = input_ids_offsets.diff().tolist()
            response_attention_mask = logits_processor_args.get("response_attention_mask", None)

            for k in ["label", "loss_mask"]:
                v = logits_processor_args[k]
                if k == "loss_mask":
                    v = _build_mtp_loss_mask_nested(v, input_ids_lengths, response_attention_mask)
                else:
                    v = _convert_to_nested_tensor(v, input_ids_lengths)
                logits_processor_args[k] = v
                args[k] = preprocess_thd_engine(
                    v,
                    pre_process=True,
                    need_roll=True,
                    use_fp8_padding=use_fp8_padding,
                    local_cp_size=local_cp_size,
                )[0]

            model_kwargs["labels"] = args["label"].contiguous()
            model_kwargs["loss_mask"] = args["loss_mask"].contiguous()

        if logits_processor_args and "loss_mask" in logits_processor_args:
            logits_processor_args.pop("loss_mask")
        if logits_processor_args and "response_attention_mask" in logits_processor_args:
            logits_processor_args.pop("response_attention_mask")

        # For VLM model, need to pass bshd format `input_ids` and `attention_mask`.
        attention_mask = None
        if vision_model:
            input_ids_rmpad, attention_mask = build_vlm_attn_mask_thd(input_ids, pad_token_id)

        output_orig = model(
            input_ids=input_ids_rmpad,
            attention_mask=attention_mask,
            position_ids=position_ids_rmpad if mtp_enable_train else None,  # position_ids is only needed for MTP
            packed_seq_params=packed_seq_params,
            **model_kwargs,
        )

        if post_process and logits_processor is not None:
            args = {
                k: preprocess_thd_engine(
                    v,
                    pre_process=True,
                    need_roll=(k == "label"),
                    use_fp8_padding=use_fp8_padding,
                    local_cp_size=local_cp_size,
                )[0]
                for k, v in logits_processor_args.items()
            }
            output_dict = logits_processor(output_orig, **args)
            output = {
                k: postprocess_thd_engine(
                    v, packed_seq_params, input_ids, batch_size, post_process=post_process, local_cp_size=local_cp_size
                )
                for k, v in output_dict.items()
            }
        else:
            output = postprocess_thd_engine(
                output_orig,
                packed_seq_params,
                input_ids,
                batch_size,
                post_process=post_process,
                local_cp_size=local_cp_size,
            )
    else:
        """
        data_format: "thd" or "bshd", default is "thd",
        why we need this?
            for some new models, GPT-OSS, the thd format is not supported, so we need to use the bshd format.
        When using the bshd format, we have to add paddings to the input_ids to meet the longest sequence length, 
        so it is recommended to disable dynamic batch size and set batch size to 1
        """
        assert local_cp_size is None, "dynamic_CP is not supported for bshd format"

        input_ids_bshd, attention_mask_bshd, position_ids_bshd = preprocess_bshd_engine(
            input_ids, pre_process=pre_process or (post_process and mtp_enable_train), use_fp8_padding=use_fp8_padding
        )

        if mtp_enable_train and post_process:
            args = {}
            # Use input_ids sequence length to ensure label and loss_mask alignment
            input_ids_offsets = input_ids.offsets()
            input_ids_lengths = input_ids_offsets.diff().tolist()
            response_attention_mask = logits_processor_args.get("response_attention_mask", None)

            for k in ["label", "loss_mask"]:
                v = logits_processor_args[k]
                if k == "loss_mask":
                    v = _build_mtp_loss_mask_nested(v, input_ids_lengths, response_attention_mask)
                else:
                    v = _convert_to_nested_tensor(v, input_ids_lengths)
                logits_processor_args[k] = v
                args[k] = preprocess_bshd_engine(v, pre_process=True, need_roll=True, use_fp8_padding=use_fp8_padding)[
                    0
                ]
            model_kwargs["labels"] = args["label"].contiguous()
            model_kwargs["loss_mask"] = args["loss_mask"].contiguous()

        if logits_processor_args and "loss_mask" in logits_processor_args:
            logits_processor_args.pop("loss_mask")
        if logits_processor_args and "response_attention_mask" in logits_processor_args:
            logits_processor_args.pop("response_attention_mask")

        # For VLM model, need to pass bshd format `input_ids` and `attention_mask`.
        if vision_model:
            input_ids_bshd, attention_mask = build_vlm_attn_mask_bshd(input_ids, batch_size, pad_token_id)
        else:
            attention_mask = attention_mask_bshd

        output_orig = model(
            input_ids=input_ids_bshd,
            attention_mask=attention_mask,
            position_ids=_select_bshd_position_ids_for_engine(model, vision_model, position_ids_bshd),
            **model_kwargs,
        )
        if post_process and logits_processor is not None:
            args = {}
            for key, value in logits_processor_args.items():
                if key == "audit_input_ids":
                    # Already in BSHD's unpadded order; processing it again
                    # would change the sequence identifier used by the audit.
                    args[key] = input_ids_bshd
                elif key == "audit_attention_mask":
                    args[key] = attention_mask_bshd
                elif key == "audit_response_lengths":
                    args[key] = value
                else:
                    args[key] = preprocess_bshd_engine(
                        value, pre_process=True, need_roll=(key == "label"), use_fp8_padding=use_fp8_padding
                    )[0]
            _write_megatron_bshd_debug(
                "before_logits_processor",
                input_ids_bshd=input_ids_bshd,
                attention_mask_bshd=attention_mask_bshd,
                position_ids_bshd=position_ids_bshd,
                label=args.get("label"),
                temperature=args.get("temperature"),
            )
            output_dict = logits_processor(output_orig, **args)
            _write_megatron_bshd_debug(
                "after_logits_processor",
                label=args.get("label"),
                log_probs=output_dict.get("log_probs"),
                entropy=output_dict.get("entropy"),
                sum_pi_squared=output_dict.get("sum_pi_squared"),
            )
            output = {
                k: postprocess_bshd_engine(v, attention_mask_bshd, post_process=post_process)
                for k, v in output_dict.items()
            }
            _write_megatron_bshd_debug(
                "after_postprocess",
                log_probs=output.get("log_probs"),
                entropy=output.get("entropy"),
                sum_pi_squared=output.get("sum_pi_squared"),
            )
        else:
            output = postprocess_bshd_engine(output_orig, attention_mask_bshd, post_process=post_process)

    if value_model and post_process:
        # output = output[..., 0]
        # while using nested tensor, the advanced indexing operation above will result in an error at backward, i.e.
        # ValueError: NestedTensor _nested_select_backward_default(grad_output: t, self: jt_all, dim: any, index: any)
        # so we use `squeeze` to remove the last dimension
        output = output.squeeze(-1)

    return output

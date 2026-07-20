import pytest
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner


def _make_config(*, use_rollout_log_probs=True, calculate_log_probs=True, logprobs_mode="raw_logprobs"):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"use_rollout_log_probs": use_rollout_log_probs},
                "rollout": {
                    "calculate_log_probs": calculate_log_probs,
                    "logprobs_mode": logprobs_mode,
                },
            },
        }
    )


def test_fully_async_accepts_raw_rollout_logprobs_for_training_ratio():
    FullyAsyncTaskRunner._validate_rollout_logprob_semantics(
        _make_config(logprobs_mode="raw_logprobs")
    )


def test_fully_async_rejects_processed_rollout_logprobs_for_training_ratio():
    with pytest.raises(ValueError, match="raw full-vocab policy logprobs"):
        FullyAsyncTaskRunner._validate_rollout_logprob_semantics(
            _make_config(logprobs_mode="processed_logprobs")
        )


def test_fully_async_rejects_missing_rollout_logprob_calculation():
    with pytest.raises(ValueError, match="calculate_log_probs=True"):
        FullyAsyncTaskRunner._validate_rollout_logprob_semantics(
            _make_config(calculate_log_probs=False)
        )


def test_fully_async_allows_processed_logprobs_when_not_used_for_training_ratio():
    FullyAsyncTaskRunner._validate_rollout_logprob_semantics(
        _make_config(use_rollout_log_probs=False, logprobs_mode="processed_logprobs")
    )


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

import unittest

import torch

from verl.protocol import DataProto
from verl.utils.debug.metrics import calculate_debug_metrics


class TestMetrics(unittest.TestCase):
    def test_calculate_debug_metrics(self):
        data = DataProto.from_dict(
            {
                "rollout_log_probs": torch.tensor(
                    [
                        [-1.5085, -0.1200, -0.6650, -0.4823, -0.1426, -1.5557, -2.8532, -0.3919, -0.4294, -0.4700],
                        [-0.0585, -0.0573, -0.4681, -0.5187, -0.7451, -1.2737, -0.0682, -0.4284, -0.5754, -0.0611],
                    ]
                ),
                "old_log_probs": torch.tensor(
                    [
                        [-1.8636, -0.7863, -0.2136, -0.4376, -2.0257, -0.2579, -1.1547, -0.5203, -0.3802, -0.9872],
                        [-0.3507, -0.5426, -0.2725, -0.4637, -0.3577, -0.3733, -1.7560, -1.9542, -0.4229, -1.3098],
                    ]
                ),
                "loss_mask": torch.tensor([[1, 0, 0, 0, 1, 1, 0, 1, 1, 0], [1, 0, 1, 0, 1, 1, 1, 0, 1, 1]]),
                "responses": torch.zeros((2, 10)),
            }
        )
        metrics = calculate_debug_metrics(data)
        print(metrics)
        assert metrics["training/rollout_probs_diff_valid"] == 1

    def test_calculate_debug_metrics_reports_shift_alignment(self):
        rollout_log_probs = torch.tensor([[-9.0, -1.0, -2.0, -3.0]])
        old_log_probs = torch.tensor([[-1.0, -2.0, -3.0, -8.0]])
        data = DataProto.from_dict(
            {
                "rollout_log_probs": rollout_log_probs,
                "old_log_probs": old_log_probs,
                "response_mask": torch.ones_like(rollout_log_probs, dtype=torch.long),
                "responses": torch.zeros((1, 4), dtype=torch.long),
            }
        )

        metrics = calculate_debug_metrics(data)

        assert metrics["training/rollout_shift/old_t_rollout_t_plus_1/valid_tokens"] == 3
        assert metrics["training/rollout_shift/old_t_rollout_t_plus_1/logprob_abs_diff_mean"] == 0
        assert metrics["training/rollout_shift/old_t_plus_1_rollout_t/logprob_abs_diff_mean"] > 0

    def test_calculate_debug_metrics_reports_rollout_logprob_pathologies(self):
        rollout_log_probs = torch.tensor([[0.0, -1.0, float("inf"), -3.0]])
        old_log_probs = torch.tensor([[-0.5, -1.0, -2.0, -3.5]])
        data = DataProto.from_dict(
            {
                "rollout_log_probs": rollout_log_probs,
                "old_log_probs": old_log_probs,
                "response_mask": torch.ones_like(rollout_log_probs, dtype=torch.long),
                "responses": torch.zeros((1, 4), dtype=torch.long),
            }
        )

        metrics = calculate_debug_metrics(data)

        assert metrics["training/rollout_log_probs/valid_tokens"] == 4
        assert metrics["training/rollout_log_probs/zero_fraction"] == 0.25
        assert metrics["training/rollout_log_probs/nonfinite_fraction"] == 0.25
        assert metrics["training/actor_old_log_probs/nonfinite_fraction"] == 0
        assert "training/rollout_actor_logprob_pearson_corr" in metrics


if __name__ == "__main__":
    unittest.main()

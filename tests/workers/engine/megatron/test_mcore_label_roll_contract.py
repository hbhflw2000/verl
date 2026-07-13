import pytest
import torch

pytest.importorskip("megatron.core")

from verl.models.mcore import util as mcore_util


def test_thd_preprocess_rolls_labels_once(monkeypatch):
    monkeypatch.setattr(mcore_util.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mcore_util.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mcore_util.mpu, "get_context_parallel_rank", lambda: 0)

    labels = torch.nested.nested_tensor([[1, 2, 3, 4], [10, 11, 12]], layout=torch.jagged)

    unrolled, _, _ = mcore_util.preprocess_thd_engine(labels, pre_process=True, need_roll=False)
    rolled, _, _ = mcore_util.preprocess_thd_engine(labels, pre_process=True, need_roll=True)

    assert unrolled.squeeze(0).tolist() == [1, 2, 3, 4, 10, 11, 12]
    assert rolled.squeeze(0).tolist() == [2, 3, 4, 10, 11, 12, 1]

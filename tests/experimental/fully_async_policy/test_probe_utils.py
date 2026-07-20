import torch

from verl.experimental.fully_async_policy.probe_utils import router_replay_tensor_metrics


def test_router_replay_fingerprint_is_order_sensitive():
    routes = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
    reordered = torch.tensor([[2, 1], [3, 4]], dtype=torch.int64)

    original = router_replay_tensor_metrics(routes)
    changed = router_replay_tensor_metrics(reordered)

    assert original["present"] == 1.0
    assert original["numel"] == 4.0
    assert original["sum"] == changed["sum"]
    assert original["sample_hash"] != changed["sample_hash"]


def test_router_replay_fingerprint_handles_missing_routes():
    assert router_replay_tensor_metrics(None) == {"present": 0.0}

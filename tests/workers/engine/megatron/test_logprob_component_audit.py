import pytest
import torch

from verl.utils.debug.logprob_audit import response_score_positions
from verl.workers.engine.megatron.transformer_impl import _first_tensor_argument


def test_response_score_positions_exclude_the_wrapped_causal_label():
    # [prompt..., response...] has labels shifted left. The final model
    # position wraps to token zero and must not be treated as a response score.
    assert list(response_score_positions(10, 3)) == [6, 7, 8]


@pytest.mark.parametrize("sequence_length,response_length", [(10, 0), (10, 10), (1, 1)])
def test_response_score_positions_reject_invalid_ranges(sequence_length, response_length):
    assert list(response_score_positions(sequence_length, response_length)) == []


def test_decoder_audit_reads_keyword_hidden_states():
    hidden_states = torch.ones(2, 3, 4)
    assert _first_tensor_argument((), {"hidden_states": hidden_states}, ("hidden_states", "input_")) is hidden_states

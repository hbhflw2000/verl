"""Small pure helpers shared by log-probability diagnostic probes."""


def response_score_positions(sequence_length: int, response_length: int) -> range:
    """Return causal-logit positions that score the final response tokens."""
    if response_length <= 0 or response_length >= sequence_length:
        return range(0)
    return range(sequence_length - response_length - 1, sequence_length - 1)

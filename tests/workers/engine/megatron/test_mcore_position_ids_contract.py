import pytest

from verl.models.mcore.model_forward import _select_bshd_position_ids_for_engine


class Qwen3OmniModel:
    __module__ = "megatron.bridge.models.qwen_omni.modeling_qwen3_omni.model"


class PlainGPTModel:
    __module__ = "megatron.core.models.gpt.gpt_model"


def test_qwen3_omni_bshd_lets_model_build_mrope_position_ids(monkeypatch):
    monkeypatch.delenv("VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS", raising=False)
    sentinel = object()

    assert _select_bshd_position_ids_for_engine(Qwen3OmniModel(), False, sentinel) is None


@pytest.mark.parametrize("mode", ["explicit", "precomputed", "pass", "1", "true", "yes"])
def test_qwen3_omni_bshd_can_probe_precomputed_position_ids(monkeypatch, mode):
    monkeypatch.setenv("VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS", mode)
    sentinel = object()

    assert _select_bshd_position_ids_for_engine(Qwen3OmniModel(), False, sentinel) is sentinel


@pytest.mark.parametrize("mode", ["model", "auto", "none", "0", "false", "no"])
def test_qwen3_omni_bshd_can_force_model_owned_position_ids(monkeypatch, mode):
    monkeypatch.setenv("VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS", mode)
    sentinel = object()

    assert _select_bshd_position_ids_for_engine(Qwen3OmniModel(), False, sentinel) is None


def test_qwen3_omni_bshd_rejects_unknown_position_id_mode(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS", "surprise")
    sentinel = object()

    with pytest.raises(ValueError, match="VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS"):
        _select_bshd_position_ids_for_engine(Qwen3OmniModel(), False, sentinel)


def test_plain_bshd_keeps_precomputed_position_ids():
    sentinel = object()

    assert _select_bshd_position_ids_for_engine(PlainGPTModel(), False, sentinel) is sentinel


def test_vision_bshd_still_lets_model_build_position_ids():
    sentinel = object()

    assert _select_bshd_position_ids_for_engine(PlainGPTModel(), True, sentinel) is None

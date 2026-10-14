from __future__ import annotations

import pytest
import torch

from sounddiffsep.models.deftmamba.checkpoint import (
    RELATIVE_MIXTURE_SCHEMA,
    build_from_config,
    load_relative_mixture_checkpoint,
    wrapper_state_dict,
)

CONFIG = {
    "model": {
        "model_type": "deftmamba_official_2ch_1src",
        "n_mics": 2,
        "n_srcs": 1,
        "n_layers": 1,
        "emb_dim": 16,
        "n_heads": 2,
        "emb_kernel_size": 4,
        "emb_hop_size": 1,
        "hidden_dim": 64,
        "dropout": 0.1,
        "eps": 1e-5,
    },
    "audio": {
        "sample_rate": 16000,
        "duration_seconds": 1.0,
        "n_fft": 256,
        "win_length": 256,
        "hop_length": 128,
        "window": "hamming",
        "center": True,
        "periodic": True,
        "normalized": False,
        "ri_packing": "channel_major_interleaved",
    },
}


def test_prefix_strip() -> None:
    state = {"ss_model._orig_mod.a.weight": 1, "ss_model.b.bias": 2}
    assert wrapper_state_dict(state) == {"a.weight": 1, "b.bias": 2}
    with pytest.raises(KeyError):
        wrapper_state_dict({"other.weight": 0})


@pytest.mark.cuda
def test_roundtrip_relative_mixture_checkpoint(
    deftmamba_cuda_device: torch.device, tmp_path
) -> None:
    torch.manual_seed(3)
    model = build_from_config(CONFIG).to(deftmamba_cuda_device).eval()
    state = {f"ss_model._orig_mod.{k}": v for k, v in model.state_dict().items()}
    path = tmp_path / "step.pt"
    torch.save(
        {"schema": RELATIVE_MIXTURE_SCHEMA, "global_step": 7,
         "contract": {"config": CONFIG}, "state_dict": state},
        path,
    )
    loaded, info = load_relative_mixture_checkpoint(path, deftmamba_cuda_device)
    assert info["global_step"] == 7 and not loaded.training
    x = torch.randn(2, 2, 4000, device=deftmamba_cuda_device)
    with torch.no_grad():
        expected, actual = model(x), loaded(x)
    assert actual.shape == (2, 1, 4000)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("extra", [{}, {"mask": {"reference_channel": 1},
                                       "model_contract": {"reference_channel": 1}}])
def test_build_from_config_forwards_mask_sections(monkeypatch, extra) -> None:
    import sounddiffsep.models.deftmamba as deftmamba

    seen = {}
    monkeypatch.setattr(deftmamba, "create_deftmamba_waveform_model", lambda **kw: seen.update(kw))
    build_from_config({**CONFIG, **extra})
    assert seen["mask_config"] == extra.get("mask")
    assert seen["model_contract"] == extra.get("model_contract")
    assert "model_type" not in seen["model_config"]

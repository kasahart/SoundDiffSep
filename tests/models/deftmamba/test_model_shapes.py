from __future__ import annotations

import subprocess
import sys

import pytest
import torch

from sounddiffsep.models.deftmamba.model import DeFTMambaCore


def make_core(device: torch.device, **overrides: object) -> DeFTMambaCore:
    settings: dict[str, object] = {
        "n_mics": 2,
        "n_srcs": 1,
        "n_layers": 1,
        "emb_dim": 64,
        "n_heads": 4,
        "emb_kernel_size": 4,
        "emb_hop_size": 1,
        "hidden_dim": 256,
        "dropout": 0.0,
    }
    settings.update(overrides)
    return DeFTMambaCore(**settings).to(device)


def test_normal_sounddiffsep_import_does_not_load_optional_cuda_modules() -> None:
    statement = (
        "import sys; import sounddiffsep; "
        "assert 'mamba_ssm' not in sys.modules; "
        "assert 'flash_attn' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", statement],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.cuda
def test_core_maps_four_ri_channels_to_two(
    deftmamba_cuda_device: torch.device,
) -> None:
    model = make_core(deftmamba_cuda_device).eval()
    model_input = torch.randn(2, 12, 10, 4, device=deftmamba_cuda_device)

    with torch.no_grad():
        output = model(model_input)

    assert output.shape == (2, 12, 10, 2)
    assert torch.isfinite(output).all()


@pytest.mark.cuda
def test_one_block_parameter_count_matches_upstream_minfix(
    deftmamba_cuda_device: torch.device,
) -> None:
    model = make_core(deftmamba_cuda_device)

    assert sum(parameter.numel() for parameter in model.parameters()) == 631_042
    assert sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    ) == 631_042
    assert len(model.state_dict()) == 84


@pytest.mark.cuda
def test_core_rejects_wrong_ri_channel_count(
    deftmamba_cuda_device: torch.device,
) -> None:
    model = make_core(deftmamba_cuda_device)

    with pytest.raises(ValueError, match=r"2 \* n_mics = 4"):
        model(torch.randn(1, 8, 8, 2, device=deftmamba_cuda_device))

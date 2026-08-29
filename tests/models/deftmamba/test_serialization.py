from __future__ import annotations

import io

import pytest
import torch

from sounddiffsep.models.deftmamba.model import DeFTMambaCore


def build_model(device: torch.device) -> DeFTMambaCore:
    return DeFTMambaCore(
        n_mics=2,
        n_srcs=1,
        n_layers=1,
        emb_dim=64,
        n_heads=4,
        emb_kernel_size=4,
        emb_hop_size=1,
        hidden_dim=256,
        dropout=0.0,
    ).to(device)


@pytest.mark.cuda
def test_state_dict_reload_reproduces_output(
    deftmamba_cuda_device: torch.device,
) -> None:
    torch.manual_seed(29)
    original = build_model(deftmamba_cuda_device).eval()
    model_input = torch.randn(1, 8, 8, 4, device=deftmamba_cuda_device)
    buffer = io.BytesIO()
    torch.save(original.state_dict(), buffer)
    buffer.seek(0)
    restored = build_model(deftmamba_cuda_device).eval()
    restored.load_state_dict(
        torch.load(buffer, map_location=deftmamba_cuda_device, weights_only=True),
        strict=True,
    )

    with torch.no_grad():
        expected = original(model_input)
        actual = restored(model_input)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

from __future__ import annotations

import pytest
import torch

from sounddiffsep.models.deftmamba.model import DeFTMambaCore


def make_core(device: torch.device, *, dropout: float = 0.0) -> DeFTMambaCore:
    return DeFTMambaCore(
        n_mics=2,
        n_srcs=1,
        n_layers=1,
        emb_dim=64,
        n_heads=4,
        emb_kernel_size=4,
        emb_hop_size=1,
        hidden_dim=256,
        dropout=dropout,
    ).to(device)


@pytest.mark.cuda
def test_forward_backward_has_finite_output_and_gradients(
    deftmamba_cuda_device: torch.device,
) -> None:
    model = make_core(deftmamba_cuda_device).train()
    model_input = torch.randn(
        1, 10, 9, 4, device=deftmamba_cuda_device, requires_grad=True
    )

    output = model(model_input)
    output.abs().mean().backward()

    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert output.shape == (1, 10, 9, 2)
    assert torch.isfinite(output).all()
    assert model_input.grad is not None and torch.isfinite(model_input.grad).all()
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.cuda
def test_zero_input_remains_finite(deftmamba_cuda_device: torch.device) -> None:
    model = make_core(deftmamba_cuda_device).eval()

    with torch.no_grad():
        output = model(torch.zeros(1, 8, 8, 4, device=deftmamba_cuda_device))

    assert torch.isfinite(output).all()


@pytest.mark.cuda
def test_eval_disables_attention_dropout(
    deftmamba_cuda_device: torch.device,
) -> None:
    """eval() output must not depend on the torch RNG state (attention dropout off)."""
    model = make_core(deftmamba_cuda_device, dropout=0.1).eval()
    model_input = torch.randn(1, 8, 8, 4, device=deftmamba_cuda_device)

    outputs = []
    with torch.no_grad():
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            outputs.append(model(model_input))

    assert all(torch.isfinite(output).all() for output in outputs)
    assert torch.equal(outputs[0], outputs[1]) and torch.equal(outputs[0], outputs[2])


@pytest.mark.cuda
def test_train_mode_keeps_attention_dropout(
    deftmamba_cuda_device: torch.device,
) -> None:
    model = make_core(deftmamba_cuda_device, dropout=0.1).train()
    model_input = torch.randn(1, 8, 8, 4, device=deftmamba_cuda_device)

    with torch.no_grad():
        torch.manual_seed(0)
        first = model(model_input)
        torch.manual_seed(1)
        second = model(model_input)
        torch.manual_seed(0)
        repeated = model(model_input)

    assert not torch.equal(first, second)
    assert torch.equal(first, repeated)

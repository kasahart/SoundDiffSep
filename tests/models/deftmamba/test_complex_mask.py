from __future__ import annotations

import io
import math

import pytest
import torch
from torch import nn

from sounddiffsep.models.deftmamba import (
    BoundedCartesianComplexMask,
    DeFTMambaCore,
    DeFTMambaComplexMaskWaveformWrapper,
    DeFTMambaWaveformWrapper,
    initialize_complex_mask_output_head,
)


class ZeroCore(nn.Module):
    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        return packed.new_zeros((*packed.shape[:3], 2))


class ConstantLogitCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(*packed.shape[:3], 2)


def build_wrapper(core: nn.Module | None = None) -> DeFTMambaComplexMaskWaveformWrapper:
    return DeFTMambaComplexMaskWaveformWrapper(
        core or ZeroCore(),
        n_mics=2,
        n_srcs=1,
        reference_channel=0,
        component_bound=2.0,
        identity_initialization=True,
        n_fft=32,
        win_length=32,
        hop_length=16,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_logits_produce_identity_mask(dtype: torch.dtype) -> None:
    parameterization = BoundedCartesianComplexMask(component_bound=2.0)
    raw_mask = torch.zeros(2, 5, 7, 2, dtype=dtype, requires_grad=True)

    mask = parameterization(raw_mask)

    assert torch.equal(mask.real, torch.ones_like(mask.real))
    assert torch.equal(mask.imag, torch.zeros_like(mask.imag))
    (mask.real.sum() + mask.imag.sum()).backward()
    assert bool(torch.isfinite(raw_mask.grad).all())
    assert bool((raw_mask.grad != 0).all())


def test_component_and_complex_magnitude_bounds() -> None:
    bound = 2.0
    parameterization = BoundedCartesianComplexMask(component_bound=bound)
    raw_mask = torch.linspace(-100.0, 100.0, 2 * 3 * 4 * 2).reshape(2, 3, 4, 2)

    mask = parameterization(raw_mask)

    assert bool((mask.real >= -bound).all())
    assert bool((mask.real <= bound).all())
    assert bool((mask.imag >= -bound).all())
    assert bool((mask.imag <= bound).all())
    assert float(mask.abs().max()) <= math.sqrt(2.0) * bound + 1.0e-6


def test_complex_mask_application_matches_manual_complex_multiplication() -> None:
    wrapper = build_wrapper()
    input_spectrum = torch.complex(
        torch.randn(2, 2, 17, 5), torch.randn(2, 2, 17, 5)
    )
    complex_mask = torch.complex(
        torch.randn(2, 17, 5), torch.randn(2, 17, 5)
    )

    actual = wrapper.apply_complex_mask(input_spectrum, complex_mask)
    expected = complex_mask.unsqueeze(1) * input_spectrum[:, 0:1]

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_identity_mask_estimate_equals_reference_spectrum() -> None:
    wrapper = build_wrapper()
    input_spectrum = torch.complex(
        torch.randn(2, 2, 17, 5), torch.randn(2, 2, 17, 5)
    )

    estimate, raw_mask, complex_mask = wrapper.estimate_spectrogram(input_spectrum)

    assert raw_mask.shape == (2, 17, 5, 2)
    torch.testing.assert_close(
        complex_mask, torch.ones_like(complex_mask), rtol=1.0e-6, atol=1.0e-6
    )
    torch.testing.assert_close(
        estimate, input_spectrum[:, 0:1], rtol=1.0e-6, atol=1.0e-6
    )


def test_estimate_plus_spectral_residual_equals_reference() -> None:
    core = ConstantLogitCore()
    core.logits.data.copy_(torch.tensor([0.3, -0.4]))
    wrapper = build_wrapper(core)
    input_spectrum = torch.complex(
        torch.randn(2, 2, 17, 5), torch.randn(2, 2, 17, 5)
    )

    estimate, _, _ = wrapper.estimate_spectrogram(input_spectrum)
    residual = input_spectrum[:, 0:1] - estimate

    torch.testing.assert_close(
        estimate + residual, input_spectrum[:, 0:1], rtol=1.0e-6, atol=1.0e-6
    )


def test_waveform_estimate_plus_residual_reconstructs_reference() -> None:
    torch.manual_seed(3)
    core = ConstantLogitCore()
    core.logits.data.copy_(torch.tensor([0.3, -0.4]))
    wrapper = build_wrapper(core)
    waveform = torch.randn(2, 2, 257)

    estimate, residual = wrapper.forward_with_residual(waveform)

    assert estimate.shape == residual.shape == (2, 1, 257)
    torch.testing.assert_close(
        estimate + residual, waveform[:, 0:1], rtol=1.0e-5, atol=2.0e-5
    )


def test_all_cartesian_quadrants_and_full_phase_range_are_representable() -> None:
    parameterization = BoundedCartesianComplexMask(component_bound=2.0)
    raw_mask = torch.tensor(
        [[[[10.0, 10.0], [-10.0, 10.0], [-10.0, -10.0], [10.0, -10.0]]]]
    )

    mask = parameterization(raw_mask).reshape(-1)
    signs = {(bool(value.real > 0), bool(value.imag > 0)) for value in mask}

    assert signs == {(True, True), (False, True), (False, False), (True, False)}
    phases = torch.angle(mask)
    assert bool((phases > math.pi / 2).any())
    assert bool((phases < -math.pi / 2).any())


def test_tanh_saturation_rate_is_reported_from_latest_forward() -> None:
    parameterization = BoundedCartesianComplexMask(component_bound=2.0)
    assert parameterization.saturation_rate(torch.zeros(1, 2, 3, 2)).item() == 0.0
    assert parameterization.saturation_rate(torch.full((1, 2, 3, 2), 100.0)).item() == 1.0

    wrapper = build_wrapper()
    input_spectrum = torch.complex(
        torch.randn(1, 2, 17, 4), torch.randn(1, 2, 17, 4)
    )
    wrapper.estimate_spectrogram(input_spectrum)
    assert wrapper.last_mask_saturation_rate is not None
    assert wrapper.last_mask_saturation_rate.item() == 0.0


def test_direct_ri_wrapper_path_remains_direct_mapping() -> None:
    class DirectCore(nn.Module):
        def forward(self, packed: torch.Tensor) -> torch.Tensor:
            return 0.25 * packed[..., :2]

    direct = DeFTMambaWaveformWrapper(
        DirectCore(), n_mics=2, n_srcs=1, n_fft=32, win_length=32, hop_length=16
    )
    waveform = torch.randn(2, 2, 257)

    actual = direct(waveform)
    expected_spectrum = direct.waveform_to_spectrogram(waveform)[:, 0:1] * 0.25
    expected = direct.spectrogram_to_waveform(expected_spectrum, length=257)

    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)


def test_direct_ri_state_dict_is_rejected_by_complex_mask_model() -> None:
    direct = DeFTMambaWaveformWrapper(
        ZeroCore(), n_mics=2, n_srcs=1, n_fft=32, win_length=32, hop_length=16
    )
    masked = build_wrapper()

    with pytest.raises(RuntimeError, match="direct-RI checkpoint"):
        masked.load_state_dict(direct.state_dict(), strict=True)


def test_complex_mask_state_dict_round_trip_preserves_contract() -> None:
    original = build_wrapper(ConstantLogitCore())
    original.core_model.logits.data.copy_(torch.tensor([0.25, -0.5]))
    restored = build_wrapper(ConstantLogitCore())
    serialized = io.BytesIO()
    torch.save(original.state_dict(), serialized)
    serialized.seek(0)

    restored.load_state_dict(torch.load(serialized, weights_only=True), strict=True)

    assert restored.model_contract == original.model_contract
    torch.testing.assert_close(
        restored.core_model.logits, original.core_model.logits, rtol=0, atol=0
    )


def test_complex_mask_state_dict_rejects_different_bound_contract() -> None:
    source = build_wrapper()
    incompatible = DeFTMambaComplexMaskWaveformWrapper(
        ZeroCore(),
        n_mics=2,
        n_srcs=1,
        reference_channel=0,
        component_bound=3.0,
        n_fft=32,
        win_length=32,
        hop_length=16,
    )

    with pytest.raises(RuntimeError, match="model_contract mismatch"):
        incompatible.load_state_dict(source.state_dict(), strict=True)


def test_one_batch_overfit_reduces_loss_and_gradients_stay_finite() -> None:
    torch.manual_seed(11)
    core = ConstantLogitCore()
    wrapper = build_wrapper(core)
    waveform = 0.1 * torch.randn(2, 2, 257)
    target = 0.4 * waveform[:, 0:1]
    optimizer = torch.optim.Adam(wrapper.parameters(), lr=5.0e-2)

    with torch.no_grad():
        initial_loss = torch.nn.functional.mse_loss(wrapper(waveform), target)

    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(wrapper(waveform), target)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in wrapper.parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        optimizer.step()

    with torch.no_grad():
        final_output = wrapper(waveform)
        final_loss = torch.nn.functional.mse_loss(final_output, target)

    assert bool(torch.isfinite(final_output).all())
    assert final_loss < initial_loss * 0.01


@pytest.mark.cuda
def test_real_deftmamba_mask_one_batch_overfit_has_finite_gradients(
    deftmamba_cuda_device: torch.device,
) -> None:
    torch.manual_seed(23)
    core = DeFTMambaCore(
        n_mics=2,
        n_srcs=1,
        n_layers=1,
        emb_dim=64,
        n_heads=4,
        emb_kernel_size=4,
        emb_hop_size=1,
        hidden_dim=256,
        dropout=0.0,
    )
    initialize_complex_mask_output_head(core, std=1.0e-3)
    wrapper = DeFTMambaComplexMaskWaveformWrapper(
        core,
        n_mics=2,
        n_srcs=1,
        reference_channel=0,
        component_bound=2.0,
        n_fft=64,
        win_length=64,
        hop_length=32,
    ).to(deftmamba_cuda_device)
    waveform = torch.randn(
        1, 2, 256, device=deftmamba_cuda_device, requires_grad=True
    )
    target = 0.8 * waveform[:, 0:1].detach()
    output_projection = wrapper.core_model.output_projection
    head_parameters = {
        id(output_projection.weight),
        id(output_projection.bias),
    }
    body_parameters = [
        parameter
        for parameter in wrapper.parameters()
        if id(parameter) not in head_parameters
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": body_parameters, "lr": 1.0e-5},
            {"params": [output_projection.weight], "lr": 1.0e-3},
            {"params": [output_projection.bias], "lr": 1.0e-2},
        ]
    )
    expected_gradient_count = sum(
        parameter.requires_grad for parameter in wrapper.parameters()
    )

    with torch.no_grad():
        initial_estimate = wrapper(waveform)
        initial_loss = torch.nn.functional.mse_loss(initial_estimate, target)

    gradients_finite = True
    populated_gradient_counts: list[int] = []
    for _ in range(16):
        optimizer.zero_grad(set_to_none=True)
        estimate = wrapper(waveform)
        loss = torch.nn.functional.mse_loss(estimate, target)
        assert bool(torch.isfinite(loss))
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in wrapper.parameters()
            if parameter.grad is not None
        ]
        populated_gradient_counts.append(len(gradients))
        assert len(gradients) == expected_gradient_count
        gradients_finite = gradients_finite and bool(
            gradients
            and all(torch.isfinite(gradient).all().item() for gradient in gradients)
        )
        torch.nn.utils.clip_grad_norm_(
            wrapper.parameters(), max_norm=5.0, error_if_nonfinite=True
        )
        optimizer.step()

    with torch.no_grad():
        final_estimate = wrapper(waveform)
        final_loss = torch.nn.functional.mse_loss(final_estimate, target)

    assert final_estimate.shape == (1, 1, 256)
    assert bool(torch.isfinite(final_estimate).all())
    assert gradients_finite
    assert populated_gradient_counts
    assert min(populated_gradient_counts) > 0
    assert final_loss < initial_loss * 0.8
    assert wrapper.last_mask_saturation_rate is not None
    assert bool(torch.isfinite(wrapper.last_mask_saturation_rate))


def test_bfloat16_logits_build_a_float32_complex_mask() -> None:
    parameterization = BoundedCartesianComplexMask(component_bound=2.0)
    raw = torch.randn(2, 5, 3, 2).to(torch.bfloat16)

    mask = parameterization(raw)

    assert mask.dtype == torch.complex64
    expected = parameterization(raw.float())
    torch.testing.assert_close(mask, expected, rtol=1e-2, atol=1e-2)

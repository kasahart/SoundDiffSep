import pytest
import torch
from torch import nn

from sounddiffsep.models.deftmamba.complex_io import pack_complex_spectrogram
from sounddiffsep.models.deftmamba.waveform_wrapper import (
    DeFTMambaWaveformWrapper,
)


class FirstMicrophoneCore(nn.Module):
    """Return mic 0 RI values as the one-source direct estimate."""

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        return packed[..., :2]


class FixedSpectralCore(nn.Module):
    """Return a stored estimate that is independent of the mixture."""

    def __init__(self, estimate: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("estimate", estimate)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        expected = (*packed.shape[:3], 2)
        if self.estimate.shape[1:] != expected[1:]:
            raise AssertionError(
                f"test estimate shape {self.estimate.shape} does not match {expected}"
            )
        return self.estimate.expand(packed.shape[0], -1, -1, -1)


class WrongOutputCore(nn.Module):
    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        return packed


def make_wrapper(core: nn.Module, **kwargs: object) -> DeFTMambaWaveformWrapper:
    return DeFTMambaWaveformWrapper(
        core,
        n_mics=2,
        n_srcs=1,
        n_fft=64,
        win_length=64,
        hop_length=32,
        **kwargs,
    )


@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("center", [False, True])
def test_stft_istft_round_trip_preserves_two_channels_and_length(
    center: bool, normalized: bool
) -> None:
    generator = torch.Generator().manual_seed(17)
    waveform = torch.randn(2, 2, 512, generator=generator)
    wrapper = make_wrapper(
        FirstMicrophoneCore(), center=center, normalized=normalized, periodic=False
    )

    spectrogram = wrapper.waveform_to_spectrogram(waveform)
    restored = wrapper.spectrogram_to_waveform(spectrogram, length=waveform.shape[-1])

    assert spectrogram.shape[:3] == (2, 2, 33)
    assert torch.is_complex(spectrogram)
    assert restored.shape == waveform.shape
    torch.testing.assert_close(restored, waveform, rtol=2e-5, atol=2e-6)


def test_waveform_wrapper_maps_two_mics_to_one_source() -> None:
    generator = torch.Generator().manual_seed(31)
    waveform = torch.randn(2, 2, 511, generator=generator)
    wrapper = make_wrapper(FirstMicrophoneCore())

    estimate = wrapper(waveform)

    assert estimate.shape == (2, 1, 511)
    torch.testing.assert_close(estimate, waveform[:, :1], rtol=2e-5, atol=2e-6)


def test_core_output_is_direct_mapping_not_an_input_mask() -> None:
    generator = torch.Generator().manual_seed(99)
    target = torch.randn(1, 1, 512, generator=generator)
    transform = make_wrapper(FirstMicrophoneCore())
    target_spectrogram = transform.waveform_to_spectrogram(target)
    fixed_core = FixedSpectralCore(pack_complex_spectrogram(target_spectrogram))
    wrapper = make_wrapper(fixed_core)

    first_mixture = torch.randn(1, 2, 512, generator=generator)
    second_mixture = 5.0 * torch.randn(1, 2, 512, generator=generator)
    first_estimate = wrapper(first_mixture)
    second_estimate = wrapper(second_mixture)

    torch.testing.assert_close(first_estimate, target, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(second_estimate, target, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(first_estimate, second_estimate, rtol=0, atol=0)


def test_hamming_window_is_non_persistent_configuration_state() -> None:
    wrapper = make_wrapper(FirstMicrophoneCore(), periodic=False)

    expected = torch.hamming_window(64, periodic=False)
    torch.testing.assert_close(wrapper._window, expected)
    assert "_window" in dict(wrapper.named_buffers())
    assert "_window" not in wrapper.state_dict()
    assert wrapper.window_name == "hamming"


def test_wrapper_rejects_wrong_waveform_channel_count() -> None:
    wrapper = make_wrapper(FirstMicrophoneCore())

    with pytest.raises(ValueError, match="expected 2, received 1"):
        wrapper(torch.randn(1, 1, 512))


def test_wrapper_reports_wrong_core_output_shape() -> None:
    wrapper = make_wrapper(WrongOutputCore())

    with pytest.raises(ValueError, match="core_model output shape mismatch"):
        wrapper(torch.randn(1, 2, 512))


def test_wrapper_rejects_unsupported_window() -> None:
    with pytest.raises(ValueError, match="only 'hamming'"):
        make_wrapper(FirstMicrophoneCore(), window="hann")


@pytest.mark.parametrize("samples", [40, 64, 95, 511])
def test_uncentered_forward_keeps_the_trailing_partial_frame(samples: int) -> None:
    generator = torch.Generator().manual_seed(23)
    waveform = torch.randn(2, 2, samples, generator=generator)
    wrapper = make_wrapper(FirstMicrophoneCore(), center=False, periodic=True)

    with torch.no_grad():
        estimate = wrapper(waveform)

    assert estimate.shape == (2, 1, samples)
    # Only the first half-window is under-covered without centering.
    torch.testing.assert_close(
        estimate[..., 32:], waveform[:, :1, 32:], rtol=2e-5, atol=2e-6
    )

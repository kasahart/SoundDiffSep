"""Waveform/STFT adapter for bounded Cartesian complex-mask DeFT-Mamba."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import torch
from torch import nn

from .complex_io import pack_complex_spectrogram
from .complex_mask import BOUNDED_CARTESIAN, BoundedCartesianComplexMask
from .waveform_wrapper import (
    DeFTMambaWaveformWrapper,
    _validate_complex_spectrogram,
    _validate_waveform,
)


COMPLEX_MASK_CONTRACT_VERSION: Final = 1
COMPLEX_MASK_OUTPUT_MODE: Final = "complex_mask"


def complex_mask_model_contract(
    *, reference_channel: int = 0, component_bound: float = 2.0
) -> dict[str, Any]:
    if isinstance(reference_channel, bool) or not isinstance(reference_channel, int):
        raise TypeError("reference_channel must be int")
    if reference_channel < 0:
        raise ValueError("reference_channel must be non-negative")
    # Contract version 1 implies identity initialization and therefore the
    # same component-bound domain as the parameterization.
    validated_parameterization = BoundedCartesianComplexMask(
        component_bound=component_bound,
        identity_initialization=True,
    )
    return {
        "output_mode": COMPLEX_MASK_OUTPUT_MODE,
        "mask_parameterization": BOUNDED_CARTESIAN,
        "reference_channel": int(reference_channel),
        "component_bound": validated_parameterization.component_bound,
        "contract_version": COMPLEX_MASK_CONTRACT_VERSION,
    }


class DeFTMambaComplexMaskWaveformWrapper(DeFTMambaWaveformWrapper):
    """Estimate a complex mask and apply it to one reference microphone.

    The core sees both microphone spectra packed as ``[B,F,T,4]`` and emits
    real Cartesian logits ``[B,F,T,2]``.  Microphone 1 is therefore available
    to mask inference, while only ``reference_channel`` is multiplied by the
    estimated mask.  No polar gain/phase parameterization is used.

    Real and imaginary mask components are each in
    ``[-component_bound, component_bound]``.  Therefore the complex magnitude
    is bounded by ``sqrt(2) * component_bound``, not ``component_bound``.
    """

    def __init__(
        self,
        core_model: nn.Module,
        *,
        n_mics: int = 2,
        n_srcs: int = 1,
        reference_channel: int = 0,
        component_bound: float = 2.0,
        identity_initialization: bool = True,
        **audio_settings: Any,
    ) -> None:
        super().__init__(
            core_model,
            n_mics=n_mics,
            n_srcs=n_srcs,
            **audio_settings,
        )
        if n_srcs != 1:
            raise ValueError("complex-mask wrapper currently requires n_srcs=1")
        if identity_initialization is not True:
            raise ValueError(
                "complex-mask checkpoint contract version 1 requires "
                "identity_initialization=true"
            )
        if isinstance(reference_channel, bool) or not isinstance(
            reference_channel, int
        ):
            raise TypeError("reference_channel must be int")
        if not 0 <= reference_channel < n_mics:
            raise ValueError(
                f"reference_channel must be in [0, {n_mics}); "
                f"received {reference_channel}"
            )
        self.reference_channel = reference_channel
        self.mask_parameterization = BoundedCartesianComplexMask(
            component_bound=component_bound,
            identity_initialization=identity_initialization,
        )
        self._model_contract = complex_mask_model_contract(
            reference_channel=reference_channel,
            component_bound=component_bound,
        )
        self._last_mask_saturation_rate: torch.Tensor | None = None

    @property
    def component_bound(self) -> float:
        return self.mask_parameterization.component_bound

    @property
    def model_contract(self) -> dict[str, Any]:
        return dict(self._model_contract)

    @property
    def last_mask_saturation_rate(self) -> torch.Tensor | None:
        """Detached saturation fraction from the latest forward call."""

        return self._last_mask_saturation_rate

    def raw_mask_to_complex(self, raw_mask: torch.Tensor) -> torch.Tensor:
        return self.mask_parameterization(raw_mask)

    def apply_complex_mask(
        self, input_spectrum: torch.Tensor, complex_mask: torch.Tensor
    ) -> torch.Tensor:
        _validate_complex_spectrogram(
            input_spectrum, expected_frequencies=self.n_fft // 2 + 1
        )
        if input_spectrum.shape[1] != self.n_mics:
            raise ValueError(
                "input spectrum channel mismatch: "
                f"expected {self.n_mics}, received {input_spectrum.shape[1]}"
            )
        expected_mask_shape = (
            input_spectrum.shape[0],
            input_spectrum.shape[2],
            input_spectrum.shape[3],
        )
        if not isinstance(complex_mask, torch.Tensor) or not torch.is_complex(
            complex_mask
        ):
            raise TypeError("complex_mask must be a complex torch.Tensor")
        if tuple(complex_mask.shape) != expected_mask_shape:
            raise ValueError(
                "complex_mask must have shape [B,F,T]: expected "
                f"{expected_mask_shape}, received {tuple(complex_mask.shape)}"
            )
        reference_spectrum = input_spectrum[
            :, self.reference_channel : self.reference_channel + 1
        ]
        return complex_mask.unsqueeze(1) * reference_spectrum

    def estimate_spectrogram(
        self, input_spectrum: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return estimate, raw logits, and complex mask for a complex STFT."""

        _validate_complex_spectrogram(
            input_spectrum, expected_frequencies=self.n_fft // 2 + 1
        )
        if input_spectrum.shape[1] != self.n_mics:
            raise ValueError(
                "input spectrum channel mismatch: "
                f"expected {self.n_mics}, received {input_spectrum.shape[1]}"
            )
        core_input = pack_complex_spectrogram(input_spectrum)
        raw_mask = self.core_model(core_input)
        self._validate_core_output(raw_mask, core_input=core_input)
        complex_mask = self.raw_mask_to_complex(raw_mask)
        self._last_mask_saturation_rate = (
            self.mask_parameterization.saturation_rate(raw_mask).detach()
        )
        estimate = self.apply_complex_mask(input_spectrum, complex_mask)
        return estimate, raw_mask, complex_mask

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Map ``[B,2,N]`` to a masked reference estimate ``[B,1,N]``."""

        _validate_waveform(waveform, expected_channels=self.n_mics)
        input_length = waveform.shape[-1]
        input_spectrum = self.waveform_to_spectrogram(waveform)
        estimate, _, _ = self.estimate_spectrogram(input_spectrum)
        return self.spectrogram_to_waveform(estimate, length=input_length)

    def forward_with_residual(
        self, waveform: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return estimated and complementary residual reference waveforms."""

        _validate_waveform(waveform, expected_channels=self.n_mics)
        input_length = waveform.shape[-1]
        input_spectrum = self.waveform_to_spectrogram(waveform)
        estimate_spectrum, _, _ = self.estimate_spectrogram(input_spectrum)
        reference_spectrum = input_spectrum[
            :, self.reference_channel : self.reference_channel + 1
        ]
        residual_spectrum = reference_spectrum - estimate_spectrum
        estimate = self.spectrogram_to_waveform(
            estimate_spectrum, length=input_length
        )
        residual = self.spectrogram_to_waveform(
            residual_spectrum, length=input_length
        )
        return estimate, residual

    def get_extra_state(self) -> dict[str, Any]:
        """Persist the semantic model contract beside otherwise compatible weights."""

        return self.model_contract

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, Mapping):
            raise RuntimeError(
                "complex-mask checkpoint has no valid model_contract; "
                "direct-RI checkpoints cannot be loaded into this model"
            )
        actual = dict(state)
        if actual != self._model_contract:
            raise RuntimeError(
                "complex-mask checkpoint model_contract mismatch: "
                f"expected={self._model_contract}, actual={actual}"
            )

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: Mapping[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        contract_key = prefix + "_extra_state"
        if contract_key not in state_dict:
            raise RuntimeError(
                "checkpoint is missing the complex-mask model_contract; "
                "a direct-RI checkpoint cannot be loaded into a complex-mask model"
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def extra_repr(self) -> str:
        parent = super().extra_repr()
        return (
            f"{parent}, output_mode={COMPLEX_MASK_OUTPUT_MODE!r}, "
            f"reference_channel={self.reference_channel}, "
            f"component_bound={self.component_bound}, "
            "magnitude_bound=sqrt(2)*component_bound"
        )


__all__ = [
    "COMPLEX_MASK_CONTRACT_VERSION",
    "COMPLEX_MASK_OUTPUT_MODE",
    "DeFTMambaComplexMaskWaveformWrapper",
    "complex_mask_model_contract",
]

"""Bounded Cartesian complex-mask parameterization.

The real and imaginary components are bounded independently.  Consequently,
``abs(mask) <= sqrt(2) * component_bound``; ``component_bound`` is not a bound
on the complex magnitude.
"""

from __future__ import annotations

import math
from typing import Final

import torch
from torch import nn


BOUNDED_CARTESIAN: Final = "bounded_cartesian"


class BoundedCartesianComplexMask(nn.Module):
    """Convert ``[..., 2]`` real logits into a bounded complex ratio mask.

    ``raw[..., 0]`` and ``raw[..., 1]`` are transformed independently with
    real-valued :func:`torch.tanh`.  The identity offset is applied only to the
    real logit, so zero logits produce exactly ``1 + 0j`` when identity
    initialization is enabled. Applying ``tanh`` to a complex tensor is
    intentionally not supported by this parameterization.
    """

    def __init__(
        self,
        *,
        component_bound: float = 2.0,
        identity_initialization: bool = True,
        saturation_level: float = 0.99,
    ) -> None:
        super().__init__()
        if isinstance(component_bound, bool) or not isinstance(
            component_bound, (int, float)
        ):
            raise TypeError("component_bound must be a real number")
        if not math.isfinite(component_bound) or component_bound <= 1.0:
            raise ValueError("component_bound must be finite and greater than 1.0")
        if not isinstance(identity_initialization, bool):
            raise TypeError("identity_initialization must be bool")
        if isinstance(saturation_level, bool) or not isinstance(
            saturation_level, (int, float)
        ):
            raise TypeError("saturation_level must be a real number")
        if not 0.0 < saturation_level < 1.0:
            raise ValueError("saturation_level must be between 0 and 1")

        self.component_bound = float(component_bound)
        self.identity_initialization = identity_initialization
        self.saturation_level = float(saturation_level)
        identity_bias = (
            math.atanh(1.0 / self.component_bound)
            if identity_initialization
            else 0.0
        )
        self.register_buffer(
            "identity_bias",
            torch.tensor(identity_bias, dtype=torch.float64),
            persistent=False,
        )

    def transformed_components(
        self, raw_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_raw_mask(raw_mask)
        raw_real = raw_mask[..., 0]
        raw_imag = raw_mask[..., 1]
        identity_bias = self.identity_bias.to(
            device=raw_real.device, dtype=raw_real.dtype
        )
        # Deliberately apply real-valued tanh component by component.
        transformed_real = torch.tanh(raw_real + identity_bias)
        transformed_imag = torch.tanh(raw_imag)
        return transformed_real, transformed_imag

    def forward(self, raw_mask: torch.Tensor) -> torch.Tensor:
        transformed_real, transformed_imag = self.transformed_components(raw_mask)
        mask_real = self.component_bound * transformed_real
        mask_imag = self.component_bound * transformed_imag
        # The analytical expression is exactly one at raw_real=0, but libm
        # round-off can leave a one-ULP error in float64. Correct only that
        # numerical invariant while retaining the analytical gradient.
        if self.identity_initialization:
            identity_correction = torch.where(
                raw_mask[..., 0] == 0,
                (torch.ones_like(mask_real) - mask_real).detach(),
                torch.zeros_like(mask_real),
            )
            mask_real = mask_real + identity_correction
        if mask_real.dtype not in (torch.float16, torch.float32, torch.float64):
            # torch.complex has no bfloat16 kernel (BF16 autocast).
            mask_real, mask_imag = mask_real.float(), mask_imag.float()
        return torch.complex(mask_real, mask_imag)

    def saturation_rate(self, raw_mask: torch.Tensor) -> torch.Tensor:
        """Return the fraction of real/imag tanh activations near saturation."""

        transformed_real, transformed_imag = self.transformed_components(raw_mask)
        saturated = torch.cat(
            (
                transformed_real.abs().reshape(-1),
                transformed_imag.abs().reshape(-1),
            )
        ) >= self.saturation_level
        return saturated.to(dtype=raw_mask.dtype).mean()

    @staticmethod
    def _validate_raw_mask(raw_mask: object) -> None:
        if not isinstance(raw_mask, torch.Tensor):
            raise TypeError(
                "raw_mask must be a torch.Tensor; "
                f"received {type(raw_mask).__name__}"
            )
        if raw_mask.ndim != 4 or raw_mask.shape[-1] != 2:
            raise ValueError(
                "raw_mask must have shape [B, F, T, 2]; "
                f"received {tuple(raw_mask.shape)}"
            )
        if torch.is_complex(raw_mask) or not raw_mask.is_floating_point():
            raise TypeError(
                "raw_mask must be a real floating-point tensor; "
                f"received {raw_mask.dtype}"
            )

    def extra_repr(self) -> str:
        return (
            f"component_bound={self.component_bound}, "
            f"identity_initialization={self.identity_initialization}, "
            f"saturation_level={self.saturation_level}"
        )


def initialize_complex_mask_output_head(
    core_model: nn.Module, *, std: float = 1.0e-3
) -> None:
    """Initialize only a complex-mask core's final projection near zero."""

    if isinstance(std, bool) or not isinstance(std, (int, float)):
        raise TypeError("head initialization std must be a real number")
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("head initialization std must be finite and positive")
    output_projection = getattr(core_model, "output_projection", None)
    if not isinstance(output_projection, nn.Conv2d):
        raise TypeError(
            "complex-mask core must expose output_projection as torch.nn.Conv2d"
        )
    with torch.no_grad():
        output_projection.weight.normal_(mean=0.0, std=float(std))
        if output_projection.bias is not None:
            output_projection.bias.zero_()


__all__ = [
    "BOUNDED_CARTESIAN",
    "BoundedCartesianComplexMask",
    "initialize_complex_mask_output_head",
]

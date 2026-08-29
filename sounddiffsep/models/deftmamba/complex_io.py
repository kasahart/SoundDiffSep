"""Complex spectrogram conversion for the DeFT-Mamba core interface.

The public DeFT-Mamba code does not establish an unambiguous real/imaginary
packing order.  This reproduction uses channel-major, interleaved components:

``[channel0_real, channel0_imag, channel1_real, channel1_imag, ...]``.
"""

from __future__ import annotations

import torch


def pack_complex_spectrogram(spectrogram: torch.Tensor) -> torch.Tensor:
    """Pack ``[B, C, F, T]`` complex values as ``[B, F, T, 2 * C]``.

    Real and imaginary components are adjacent for each channel.  The
    operation is differentiable and does not change the numerical values.

    Args:
        spectrogram: A rank-four complex tensor ordered as batch, channel,
            frequency, and time.

    Returns:
        A real-valued tensor whose last dimension follows the reproduction's
        channel-major interleaved RI convention.

    Raises:
        TypeError: If ``spectrogram`` is not a complex tensor.
        ValueError: If its rank or non-batch dimensions are invalid.
    """

    _validate_tensor(spectrogram, name="spectrogram")
    if spectrogram.ndim != 4:
        raise ValueError(
            "spectrogram must have shape [B, C, F, T]; "
            f"received rank {spectrogram.ndim} with shape {tuple(spectrogram.shape)}"
        )
    if not torch.is_complex(spectrogram):
        raise TypeError(
            "spectrogram must use a complex dtype so real and imaginary "
            f"components are explicit; received {spectrogram.dtype}"
        )

    batch, channels, frequencies, frames = spectrogram.shape
    _validate_spectrogram_dimensions(
        channels=channels, frequencies=frequencies, frames=frames
    )

    components = torch.stack((spectrogram.real, spectrogram.imag), dim=-1)
    return components.permute(0, 2, 3, 1, 4).reshape(
        batch, frequencies, frames, 2 * channels
    )


def unpack_complex_spectrogram(
    packed: torch.Tensor, *, n_channels: int | None = None
) -> torch.Tensor:
    """Unpack interleaved RI values to a ``[B, C, F, T]`` complex tensor.

    Args:
        packed: A real floating-point tensor with shape ``[B, F, T, 2 * C]``.
        n_channels: Expected channel/source count.  When omitted, it is
            inferred from the final dimension after checking that it is even.

    Returns:
        A complex tensor ordered as batch, channel, frequency, and time.

    Raises:
        TypeError: If ``packed`` is not a real floating-point tensor.
        ValueError: If its shape is invalid or conflicts with ``n_channels``.
    """

    _validate_tensor(packed, name="packed")
    if packed.ndim != 4:
        raise ValueError(
            "packed must have shape [B, F, T, 2 * C]; "
            f"received rank {packed.ndim} with shape {tuple(packed.shape)}"
        )
    if torch.is_complex(packed) or not packed.is_floating_point():
        raise TypeError(
            "packed must be a real floating-point tensor; "
            f"received {packed.dtype}"
        )

    batch, frequencies, frames, packed_channels = packed.shape
    if packed_channels <= 0 or packed_channels % 2 != 0:
        raise ValueError(
            "packed's final dimension must be a positive even number "
            f"(real/imaginary pairs); received {packed_channels}"
        )

    inferred_channels = packed_channels // 2
    if n_channels is None:
        n_channels = inferred_channels
    else:
        _validate_positive_int(n_channels, name="n_channels")
        if packed_channels != 2 * n_channels:
            raise ValueError(
                "packed's final dimension does not match n_channels: "
                f"expected {2 * n_channels}, received {packed_channels}"
            )

    _validate_spectrogram_dimensions(
        channels=n_channels, frequencies=frequencies, frames=frames
    )
    components = packed.reshape(batch, frequencies, frames, n_channels, 2)
    spectrogram = torch.complex(components[..., 0], components[..., 1])
    return spectrogram.permute(0, 3, 1, 2).contiguous()


def _validate_tensor(value: object, *, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor; received {type(value).__name__}"
        )


def _validate_positive_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; received {value!r}")


def _validate_spectrogram_dimensions(
    *, channels: int, frequencies: int, frames: int
) -> None:
    if channels <= 0:
        raise ValueError(f"channel dimension must be positive; received {channels}")
    if frequencies <= 0:
        raise ValueError(
            f"frequency dimension must be positive; received {frequencies}"
        )
    if frames <= 0:
        raise ValueError(f"time dimension must be positive; received {frames}")


__all__ = ["pack_complex_spectrogram", "unpack_complex_spectrogram"]

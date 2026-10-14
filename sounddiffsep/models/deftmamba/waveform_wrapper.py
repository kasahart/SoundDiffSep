"""Waveform/STFT adapter for a direct-mapping DeFT-Mamba core."""

from __future__ import annotations

import math
from typing import Final

import torch
from torch import nn

from .complex_io import pack_complex_spectrogram, unpack_complex_spectrogram


_SUPPORTED_WINDOW: Final = "hamming"


class DeFTMambaWaveformWrapper(nn.Module):
    """Adapt a spectral DeFT-Mamba core to ``[B, M, N]`` waveforms.

    The wrapped core receives ``[B, F, T, 2 * n_mics]`` and must directly
    produce ``[B, F, T, 2 * n_srcs]``.  Its output is interpreted as a complex
    source estimate and passed straight to ISTFT.  In particular, this wrapper
    never treats the core output as a mask and never multiplies it by an input
    spectrogram.

    ``hamming`` is the only window currently exposed because it is the chosen
    reproduction setting.  Keeping ``window`` as a named argument makes later
    extension explicit without silently changing this baseline.
    """

    def __init__(
        self,
        core_model: nn.Module,
        *,
        n_mics: int = 2,
        n_srcs: int = 1,
        sample_rate: int = 16_000,
        n_fft: int = 512,
        win_length: int = 512,
        hop_length: int = 256,
        window: str = _SUPPORTED_WINDOW,
        center: bool = True,
        periodic: bool = True,
        normalized: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(core_model, nn.Module):
            raise TypeError(
                "core_model must be a torch.nn.Module; "
                f"received {type(core_model).__name__}"
            )

        for name, value in (
            ("n_mics", n_mics),
            ("n_srcs", n_srcs),
            ("sample_rate", sample_rate),
            ("n_fft", n_fft),
            ("win_length", win_length),
            ("hop_length", hop_length),
        ):
            _validate_positive_int(value, name=name)
        if win_length > n_fft:
            raise ValueError(
                "win_length cannot exceed n_fft; "
                f"received win_length={win_length}, n_fft={n_fft}"
            )
        for name, value in (
            ("center", center),
            ("periodic", periodic),
            ("normalized", normalized),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool; received {type(value).__name__}")
        if not center and win_length < n_fft:
            # torch.stft zero-pads a short window to n_fft; uncentered, the
            # first samples then get a zero overlap-add envelope (ISTFT NOLA).
            raise ValueError(
                "center=False requires win_length == n_fft; "
                f"received win_length={win_length}, n_fft={n_fft}"
            )
        if not isinstance(window, str):
            raise TypeError(
                f"window must be a string; received {type(window).__name__}"
            )
        window_name = window.lower()
        if window_name != _SUPPORTED_WINDOW:
            raise ValueError(
                f"unsupported window {window!r}; only {_SUPPORTED_WINDOW!r} is "
                "defined for this reproduction"
            )

        self.core_model = core_model
        self.n_mics = n_mics
        self.n_srcs = n_srcs
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.window_name = window_name
        self.center = center
        self.periodic = periodic
        self.normalized = normalized

        analysis_window = torch.hamming_window(
            win_length, periodic=periodic, dtype=torch.float32
        )
        # The window is deterministic configuration-derived state, not a model
        # parameter/checkpoint payload.
        self.register_buffer("_window", analysis_window, persistent=False)

    def waveform_to_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute a complex STFT for a real ``[B, C, N]`` waveform tensor."""

        _validate_waveform(waveform)
        batch, channels, samples = waveform.shape
        flattened = waveform.reshape(batch * channels, samples)
        if not self.center:
            # Without centering, torch.stft drops a trailing partial frame;
            # zero-pad so every input sample is analysed (ISTFT crops back).
            frames = max(0, math.ceil((samples - self.n_fft) / self.hop_length))
            padded = self.n_fft + frames * self.hop_length
            flattened = torch.nn.functional.pad(flattened, (0, padded - samples))
        elif samples <= self.n_fft // 2:
            # Reflection padding needs more than n_fft // 2 samples; zero-pad
            # short clips (ISTFT crops back to the input length).
            flattened = torch.nn.functional.pad(
                flattened, (0, self.n_fft // 2 + 1 - samples)
            )
        window = self._window_for(flattened)
        spectrogram = torch.stft(
            flattened,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            onesided=True,
            return_complex=True,
        )
        return spectrogram.reshape(
            batch, channels, self.n_fft // 2 + 1, spectrogram.shape[-1]
        )

    def spectrogram_to_waveform(
        self, spectrogram: torch.Tensor, *, length: int
    ) -> torch.Tensor:
        """Invert a complex ``[B, C, F, T]`` spectrogram to exact ``length``."""

        _validate_complex_spectrogram(
            spectrogram, expected_frequencies=self.n_fft // 2 + 1
        )
        _validate_positive_int(length, name="length")
        batch, channels, frequencies, frames = spectrogram.shape
        flattened = spectrogram.reshape(batch * channels, frequencies, frames)
        window = self._window_for(flattened.real)
        waveform = torch.istft(
            flattened,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
            normalized=self.normalized,
            onesided=True,
            length=length,
            return_complex=False,
        )
        return waveform.reshape(batch, channels, length)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Map ``[B, n_mics, N]`` directly to ``[B, n_srcs, N]``."""

        _validate_waveform(waveform, expected_channels=self.n_mics)
        input_length = waveform.shape[-1]
        input_spectrogram = self.waveform_to_spectrogram(waveform)
        core_input = pack_complex_spectrogram(input_spectrogram)

        core_output = self.core_model(core_input)
        self._validate_core_output(core_output, core_input=core_input)
        estimated_spectrogram = unpack_complex_spectrogram(
            core_output, n_channels=self.n_srcs
        )
        return self.spectrogram_to_waveform(
            estimated_spectrogram, length=input_length
        )

    def _validate_core_output(
        self, core_output: object, *, core_input: torch.Tensor
    ) -> None:
        if not isinstance(core_output, torch.Tensor):
            raise TypeError(
                "core_model must return a torch.Tensor with shape "
                f"[B, F, T, {2 * self.n_srcs}]; received "
                f"{type(core_output).__name__}"
            )
        expected_shape = (*core_input.shape[:3], 2 * self.n_srcs)
        if tuple(core_output.shape) != expected_shape:
            raise ValueError(
                "core_model output shape mismatch: expected "
                f"{expected_shape}, received {tuple(core_output.shape)}"
            )
        if torch.is_complex(core_output) or not core_output.is_floating_point():
            raise TypeError(
                "core_model output must be a real floating-point RI tensor; "
                f"received {core_output.dtype}"
            )
        if core_output.device != core_input.device:
            raise ValueError(
                "core_model output must remain on the input device: "
                f"expected {core_input.device}, received {core_output.device}"
            )

    def _window_for(self, reference: torch.Tensor) -> torch.Tensor:
        return self._window.to(device=reference.device, dtype=reference.dtype)

    def extra_repr(self) -> str:
        return (
            f"n_mics={self.n_mics}, n_srcs={self.n_srcs}, "
            f"sample_rate={self.sample_rate}, n_fft={self.n_fft}, "
            f"win_length={self.win_length}, hop_length={self.hop_length}, "
            f"window={self.window_name!r}, center={self.center}, "
            f"periodic={self.periodic}, normalized={self.normalized}"
        )


def _validate_waveform(
    waveform: object, *, expected_channels: int | None = None
) -> None:
    if not isinstance(waveform, torch.Tensor):
        raise TypeError(
            "waveform must be a torch.Tensor; "
            f"received {type(waveform).__name__}"
        )
    if waveform.ndim != 3:
        raise ValueError(
            "waveform must have shape [B, C, N]; "
            f"received rank {waveform.ndim} with shape {tuple(waveform.shape)}"
        )
    if torch.is_complex(waveform) or not waveform.is_floating_point():
        raise TypeError(
            "waveform must be a real floating-point tensor; "
            f"received {waveform.dtype}"
        )
    _, channels, samples = waveform.shape
    if channels <= 0:
        raise ValueError(f"waveform channel dimension must be positive; got {channels}")
    if samples <= 0:
        raise ValueError(f"waveform sample dimension must be positive; got {samples}")
    if expected_channels is not None and channels != expected_channels:
        raise ValueError(
            "waveform channel mismatch: "
            f"expected {expected_channels}, received {channels}"
        )


def _validate_complex_spectrogram(
    spectrogram: object, *, expected_frequencies: int
) -> None:
    if not isinstance(spectrogram, torch.Tensor):
        raise TypeError(
            "spectrogram must be a torch.Tensor; "
            f"received {type(spectrogram).__name__}"
        )
    if spectrogram.ndim != 4:
        raise ValueError(
            "spectrogram must have shape [B, C, F, T]; "
            f"received rank {spectrogram.ndim} with shape {tuple(spectrogram.shape)}"
        )
    if not torch.is_complex(spectrogram):
        raise TypeError(
            "spectrogram must use a complex dtype; "
            f"received {spectrogram.dtype}"
        )
    _, channels, frequencies, frames = spectrogram.shape
    if channels <= 0:
        raise ValueError(
            f"spectrogram channel dimension must be positive; got {channels}"
        )
    if frequencies != expected_frequencies:
        raise ValueError(
            "spectrogram frequency mismatch: "
            f"expected {expected_frequencies}, received {frequencies}"
        )
    if frames <= 0:
        raise ValueError(f"spectrogram frame dimension must be positive; got {frames}")


def _validate_positive_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; received {value!r}")


__all__ = ["DeFTMambaWaveformWrapper"]

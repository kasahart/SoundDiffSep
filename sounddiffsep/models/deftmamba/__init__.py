"""Experimental DeFT-Mamba 2-channel/1-source baseline.

Importing this module does not import ``mamba_ssm`` or ``flash_attn``.  Those
optional CUDA dependencies are resolved only when :class:`DeFTMambaCore` is
constructed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .complex_io import pack_complex_spectrogram, unpack_complex_spectrogram
from .model import DeFTMambaCore, create_official_2ch_1src_core
from .waveform_wrapper import DeFTMambaWaveformWrapper


DEFTMAMBA_OFFICIAL_2CH_1SRC = "deftmamba_official_2ch_1src"

_MODEL_CONFIG_KEYS = frozenset(
    {
        "name",
        "n_mics",
        "n_srcs",
        "n_layers",
        "emb_dim",
        "n_heads",
        "emb_kernel_size",
        "emb_hop_size",
        "hidden_dim",
        "dropout",
        "eps",
    }
)
_AUDIO_CONFIG_KEYS = frozenset(
    {
        "sample_rate",
        "duration_seconds",
        "n_fft",
        "win_length",
        "hop_length",
        "window",
        "center",
        "periodic",
        "normalized",
        "ri_packing",
    }
)


def create_deftmamba_waveform_model(
    model_name: str = DEFTMAMBA_OFFICIAL_2CH_1SRC,
    *,
    model_config: Mapping[str, Any] | None = None,
    audio_config: Mapping[str, Any] | None = None,
) -> DeFTMambaWaveformWrapper:
    """Create the named experimental direct-RI waveform model.

    The mappings can be passed directly from the ``model`` and ``audio``
    sections of either provided YAML configuration.  Configuration is checked
    before core construction, so invalid names and fields do not trigger
    optional CUDA dependency imports.  ``duration_seconds`` describes the
    benchmark/training example length and is intentionally not forwarded: the
    waveform wrapper preserves whatever sample length it receives.

    Calling this function constructs the DeFT-Mamba core and is therefore the
    point at which the optional ``mamba-ssm`` and ``flash-attn`` runtimes may be
    imported.
    """

    if model_name != DEFTMAMBA_OFFICIAL_2CH_1SRC:
        raise ValueError(
            f"unknown experimental DeFT-Mamba model {model_name!r}; expected "
            f"{DEFTMAMBA_OFFICIAL_2CH_1SRC!r}"
        )

    model_settings = _validated_config_mapping(
        model_config, section="model", allowed_keys=_MODEL_CONFIG_KEYS
    )
    configured_name = model_settings.pop("name", model_name)
    if configured_name != model_name:
        raise ValueError(
            f"model.name {configured_name!r} does not match requested model "
            f"{model_name!r}"
        )

    n_mics = model_settings.get("n_mics", 2)
    n_srcs = model_settings.get("n_srcs", 1)
    if n_mics != 2 or isinstance(n_mics, bool):
        raise ValueError(
            f"{model_name!r} requires model.n_mics=2; received {n_mics!r}"
        )
    if n_srcs != 1 or isinstance(n_srcs, bool):
        raise ValueError(
            f"{model_name!r} requires model.n_srcs=1; received {n_srcs!r}"
        )

    audio_settings = _validated_config_mapping(
        audio_config, section="audio", allowed_keys=_AUDIO_CONFIG_KEYS
    )
    duration_seconds = audio_settings.pop("duration_seconds", None)
    if duration_seconds is not None and (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or duration_seconds <= 0
    ):
        raise ValueError(
            "audio.duration_seconds must be a positive number; received "
            f"{duration_seconds!r}"
        )
    ri_packing = audio_settings.pop(
        "ri_packing", "channel_major_interleaved"
    )
    if ri_packing != "channel_major_interleaved":
        raise ValueError(
            "this reproduction supports only "
            "audio.ri_packing='channel_major_interleaved'; received "
            f"{ri_packing!r}"
        )

    core_model = create_official_2ch_1src_core(**model_settings)
    return DeFTMambaWaveformWrapper(
        core_model,
        n_mics=n_mics,
        n_srcs=n_srcs,
        **audio_settings,
    )


def _validated_config_mapping(
    config: Mapping[str, Any] | None,
    *,
    section: str,
    allowed_keys: frozenset[str],
) -> dict[str, Any]:
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise TypeError(
            f"{section}_config must be a mapping; received {type(config).__name__}"
        )
    settings = dict(config)
    unexpected = sorted(set(settings) - allowed_keys)
    if unexpected:
        raise ValueError(
            f"unsupported {section} configuration field(s): {', '.join(unexpected)}"
        )
    return settings


__all__ = [
    "DEFTMAMBA_OFFICIAL_2CH_1SRC",
    "DeFTMambaCore",
    "DeFTMambaWaveformWrapper",
    "create_deftmamba_waveform_model",
    "create_official_2ch_1src_core",
    "pack_complex_spectrogram",
    "unpack_complex_spectrogram",
]

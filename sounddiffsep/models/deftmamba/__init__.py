"""Experimental DeFT-Mamba 2-channel/1-source baseline.

Importing this module does not import ``mamba_ssm`` or ``flash_attn``.  Those
optional CUDA dependencies are resolved only when :class:`DeFTMambaCore` is
constructed.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from .complex_mask import (
    BOUNDED_CARTESIAN,
    BoundedCartesianComplexMask,
    initialize_complex_mask_output_head,
)
from .complex_io import pack_complex_spectrogram, unpack_complex_spectrogram
from .mask_waveform_wrapper import (
    COMPLEX_MASK_CONTRACT_VERSION,
    COMPLEX_MASK_OUTPUT_MODE,
    DeFTMambaComplexMaskWaveformWrapper,
    complex_mask_model_contract,
)
from .model import DeFTMambaCore, create_official_2ch_1src_core
from .waveform_wrapper import DeFTMambaWaveformWrapper


DEFTMAMBA_OFFICIAL_2CH_1SRC = "deftmamba_official_2ch_1src"
DEFTMAMBA_COMPLEX_MASK_2CH_1SRC = "deftmamba_complex_mask_2ch_1src"

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
_MASK_CONFIG_KEYS = frozenset(
    {
        "parameterization",
        "reference_channel",
        "component_bound",
        "head_init_std",
        "identity_initialization",
    }
)
_MODEL_CONTRACT_KEYS = frozenset(
    {
        "output_mode",
        "mask_parameterization",
        "reference_channel",
        "component_bound",
        "contract_version",
    }
)


def create_deftmamba_waveform_model(
    model_name: str = DEFTMAMBA_OFFICIAL_2CH_1SRC,
    *,
    model_config: Mapping[str, Any] | None = None,
    audio_config: Mapping[str, Any] | None = None,
    mask_config: Mapping[str, Any] | None = None,
    model_contract: Mapping[str, Any] | None = None,
) -> DeFTMambaWaveformWrapper | DeFTMambaComplexMaskWaveformWrapper:
    """Create a named experimental direct-RI or complex-mask waveform model.

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

    supported_names = {
        DEFTMAMBA_OFFICIAL_2CH_1SRC,
        DEFTMAMBA_COMPLEX_MASK_2CH_1SRC,
    }
    if model_name not in supported_names:
        raise ValueError(
            f"unknown experimental DeFT-Mamba model {model_name!r}; expected one "
            f"of {sorted(supported_names)!r}"
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

    if model_name == DEFTMAMBA_OFFICIAL_2CH_1SRC:
        if mask_config is not None:
            raise ValueError("direct-RI model does not accept a mask configuration")
        if model_contract is not None:
            raise ValueError("direct-RI model does not accept a complex-mask contract")
        core_model = create_official_2ch_1src_core(**model_settings)
        return DeFTMambaWaveformWrapper(
            core_model,
            n_mics=n_mics,
            n_srcs=n_srcs,
            **audio_settings,
        )

    mask_settings = _validated_config_mapping(
        mask_config, section="mask", allowed_keys=_MASK_CONFIG_KEYS
    )
    parameterization = mask_settings.pop("parameterization", BOUNDED_CARTESIAN)
    if parameterization != BOUNDED_CARTESIAN:
        raise ValueError(
            "complex-mask model requires "
            f"mask.parameterization={BOUNDED_CARTESIAN!r}; "
            f"received {parameterization!r}"
        )
    reference_channel = mask_settings.pop("reference_channel", 0)
    component_bound = mask_settings.pop("component_bound", 2.0)
    head_init_std = mask_settings.pop("head_init_std", 1.0e-3)
    identity_initialization = mask_settings.pop("identity_initialization", True)
    if identity_initialization is not True:
        raise ValueError(
            "complex-mask checkpoint contract version 1 requires "
            "mask.identity_initialization=true"
        )
    expected_contract = complex_mask_model_contract(
        reference_channel=reference_channel,
        component_bound=component_bound,
    )
    if reference_channel >= n_mics:
        raise ValueError(
            f"mask.reference_channel must be in [0, {n_mics}); "
            f"received {reference_channel}"
        )
    if (
        isinstance(head_init_std, bool)
        or not isinstance(head_init_std, (int, float))
        or not math.isfinite(head_init_std)
        or head_init_std <= 0.0
    ):
        raise ValueError("mask.head_init_std must be finite and positive")
    if model_contract is not None:
        configured_contract = _validated_config_mapping(
            model_contract,
            section="model_contract",
            allowed_keys=_MODEL_CONTRACT_KEYS,
        )
        if configured_contract != expected_contract:
            raise ValueError(
                "model_contract does not match the requested complex-mask model: "
                f"expected={expected_contract}, actual={configured_contract}"
            )

    core_model = create_official_2ch_1src_core(**model_settings)
    initialize_complex_mask_output_head(core_model, std=head_init_std)
    return DeFTMambaComplexMaskWaveformWrapper(
        core_model,
        n_mics=n_mics,
        n_srcs=n_srcs,
        reference_channel=reference_channel,
        component_bound=component_bound,
        identity_initialization=identity_initialization,
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
    "BOUNDED_CARTESIAN",
    "COMPLEX_MASK_CONTRACT_VERSION",
    "COMPLEX_MASK_OUTPUT_MODE",
    "DEFTMAMBA_COMPLEX_MASK_2CH_1SRC",
    "DEFTMAMBA_OFFICIAL_2CH_1SRC",
    "BoundedCartesianComplexMask",
    "DeFTMambaCore",
    "DeFTMambaComplexMaskWaveformWrapper",
    "DeFTMambaWaveformWrapper",
    "complex_mask_model_contract",
    "create_deftmamba_waveform_model",
    "create_official_2ch_1src_core",
    "initialize_complex_mask_output_head",
    "pack_complex_spectrogram",
    "unpack_complex_spectrogram",
]

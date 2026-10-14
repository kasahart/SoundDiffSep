"""Load DeFT-Mamba training checkpoints into the waveform wrapper.

Supported schema: ``"relative_mixture_training_v1"`` (2chssDNN relative-mixture
trainer).  Such a checkpoint is a mapping with ``contract["config"]`` (the full
training config) and ``state_dict`` (or ``model_state_dict``) whose keys are
prefixed by the Lightning system attribute ``ss_model.`` (``ss_model._orig_mod.``
when the model was ``torch.compile``-d).

The wrapper is rebuilt exactly as the trainer's ``build_system`` does:
``create_deftmamba_waveform_model(model_type, model_config=config["model"] minus
"model_type", audio_config=config["audio"])`` plus ``mask`` / ``model_contract`` when
present (complex-mask models); the prefix is stripped and the
state dict is loaded with ``strict=True``.  The trainer itself is not imported.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

RELATIVE_MIXTURE_SCHEMA = "relative_mixture_training_v1"
_PREFIXES = ("ss_model._orig_mod.", "ss_model.")


def wrapper_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    """Strip the Lightning ``ss_model.`` (and compile ``_orig_mod.``) prefix."""

    out = {}
    for key, value in state.items():
        key = str(key)
        for prefix in _PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        else:
            raise KeyError(f"unexpected state-dict key without ss_model prefix: {key}")
        out[key] = value
    return out


def build_from_config(config: Mapping[str, Any]) -> torch.nn.Module:
    """Build the (untrained) waveform wrapper from a training config mapping."""

    from . import create_deftmamba_waveform_model

    model_conf = dict(config["model"])
    model_type = model_conf.pop("model_type")
    return create_deftmamba_waveform_model(
        model_name=model_type,
        model_config=model_conf,
        audio_config=dict(config["audio"]),
        mask_config=config.get("mask"),
        model_contract=config.get("model_contract"),
    )


def load_relative_mixture_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Return ``(wrapper.eval(), info)`` for a relative-mixture checkpoint.

    ``wrapper(x)`` maps ``[B, 2, T]`` to ``[B, 1, T]``.  ``info`` holds the
    schema, global step and the training config.  Parameters are frozen.
    """

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    schema = str(checkpoint.get("schema"))
    if schema != RELATIVE_MIXTURE_SCHEMA:
        raise ValueError(f"unsupported checkpoint schema {schema!r}")
    config = checkpoint["contract"]["config"]
    field = "state_dict" if isinstance(checkpoint.get("state_dict"), Mapping) else "model_state_dict"
    model = build_from_config(config)
    model.load_state_dict(wrapper_state_dict(checkpoint[field]), strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    info = {
        "schema": schema,
        "global_step": int(checkpoint.get("global_step", -1)),
        "config": config,
    }
    return model, info
